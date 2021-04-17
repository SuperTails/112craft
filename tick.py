"""Provides the `tick` function, which does all of the game's updates.

At regular intervals, the AI of all entities are processed, gravity is applied,
new entities are spawned, other entities are removed, collisions occur, etc.
"""

from entity import Entity
from player import Player
from client import ClientState
from util import BlockPos, roundHalfUp, ChunkPos
import world
import time
import math
import config
from math import cos, sin
import copy
import random
import network
import resources
from inventory import Stack
from typing import Optional

def sendPlayerDigging(app, action: network.DiggingAction, location: BlockPos, face: int):
    if action == network.DiggingAction.START_DIGGING:
        app.breakingBlock = 0.0
        app.breakingBlockPos = location
    elif action == network.DiggingAction.CANCEL_DIGGING:
        app.breakingBlock = 0.0
    elif action == network.DiggingAction.FINISH_DIGGING:
        app.breakingBlock = 1000.0
    else:
        # TODO:
        print(f"Ignoring other action {action}")

    network.c2sQueue.put(network.PlayerDiggingC2S(action, location, face))

def sendPlayerLook(app, yaw: float, pitch: float, onGround: bool):
    app.cameraYaw = yaw
    app.cameraPitch = pitch

    # TODO:
    #app.mode.player.onGround = onGround

    network.c2sQueue.put(network.PlayerLookC2S(yaw, pitch, onGround))

def sendPlayerPosition(app, x, y, z, onGround):
    # TODO:
    #app.mode.player.pos = [x, y, z]
    #app.mode.player.onGround = onGround

    network.c2sQueue.put(network.PlayerPositionC2S(x, y, z, onGround))

def sendClickWindow(app, windowId: int, slotIdx: int, button: int, actionNum: int, mode: int, item, count):
    # TODO:

    print(f'Sending window ID {windowId} and slot {slotIdx}, action {actionNum}')

    network.c2sQueue.put(network.ClickWindowC2S(windowId, slotIdx, button, actionNum, mode, item, count))

def sendCloseWindow(app, windowId: int):
    network.c2sQueue.put(network.CloseWindowC2S(windowId))

def sendUseItem(app, hand: int):
    network.c2sQueue.put(network.UseItemC2S(hand))

def sendTeleportConfirm(app, teleportId: int):
    network.c2sQueue.put(network.TeleportConfirmC2S(teleportId))

def sendPlayerMovement(app, onGround: bool):
    # TODO:
    #app.mode.player.onGround = onGround

    network.c2sQueue.put(network.PlayerMovementC2S(onGround))

def sendPlayerPlacement(app, hand: int, location: BlockPos, face: int, cx: float, cy: float, cz: float, insideBlock: bool):
    network.c2sQueue.put(network.PlayerPlacementC2S(hand, location, face, cx, cy, cz, insideBlock))

def sendClientStatus(app, status: int):
    network.c2sQueue.put(network.ClientStatusC2S(status))

def sendHeldItemChange(app, newSlot: int):
    # TODO:
    #app.mode.player.hotbarIdx = newSlot

    network.c2sQueue.put(network.HeldItemChangeC2S(newSlot))


def updateBlockBreaking(app):
    if not app.world.local:
        return

    pos = app.breakingBlockPos

    if app.breakingBlock == 0.0:
        return

    blockId = app.world.getBlock(pos)

    toolStack = app.mode.player.inventory[app.mode.player.hotbarIdx].stack
    if toolStack.isEmpty():
        tool = ''
    else:
        tool = toolStack.item

    hardness = resources.getHardnessAgainst(blockId, tool)

    # TODO: Sound effect packets

    if app.breakingBlock >= hardness:
        mcBlockId = app.world.registry.encode_block({ 'name': 'minecraft:' + blockId })

        network.s2cQueue.put(network.AckPlayerDiggingS2C(
            pos,
            mcBlockId,
            network.DiggingAction.FINISH_DIGGING,
            True
        ))

        droppedItem = resources.getBlockDrop(app, blockId, tool)

        resources.getDigSound(app, blockId).play()

        world.removeBlock(app, pos)

        app.breakingBlock = 0.0

        if droppedItem is not None:
            stack = Stack(droppedItem, 1)

            entityId = getNextEntityId()

            xVel = ((random.random() - 0.5) * 0.1)
            yVel = ((random.random() - 0.5) * 0.1)
            zVel = ((random.random() - 0.5) * 0.1)

            # TODO: UUID
            network.s2cQueue.put(network.SpawnEntityS2C(entityId, None, 37,
                pos.x, pos.y, pos.z, 0.0, 0.0, 1,
                int(xVel * 8000), int(yVel * 8000), int(zVel * 8000)))

            itemId = app.world.registry.encode('minecraft:item', 'minecraft:' + stack.item)

            network.s2cQueue.put(network.EntityMetadataS2C(
                entityId, { (6, 7): { 'item': itemId, 'count': stack.amount } }
            ))
            
            ent = Entity(app, 'item', pos.x, pos.y, pos.z)
            ent.extra.stack = stack
            ent.velocity = [xVel, yVel, zVel]
            app.entities.append(ent)

entityIdNum = 10_000

def getNextEntityId() -> int:
    global entityIdNum
    entityIdNum += 1
    return entityIdNum


def clientTick(client: ClientState, instData):
    startTime = time.time()

    client.time += 1

    client.world.loadUnloadChunks(client.cameraPos, instData)
    client.world.addChunkDetails(instData)

    player: Player = client.player

    playerChunkPos = world.toChunkLocal(player.getBlockPos())[0]
    playerChunkPos = ChunkPos(playerChunkPos.x, 0, playerChunkPos.z)

    # W makes the player go forward, S makes them go backwards,
    # and pressing both makes them stop!
    z = float(client.w) - float(client.s)
    # Likewise for side to side movement
    x = float(client.d) - float(client.a)

    if playerChunkPos in client.world.chunks and client.world.chunks[playerChunkPos].isTicking:
        if x != 0.0 or z != 0.0:
            mag = math.sqrt(x*x + z*z)
            x /= mag
            z /= mag

            newX = math.cos(client.cameraYaw) * x - math.sin(client.cameraYaw) * z
            newZ = math.sin(client.cameraYaw) * x + math.cos(client.cameraYaw) * z

            x, z = newX, newZ

            x *= player.walkSpeed 
            z *= player.walkSpeed

        #player.tick(app, app.world, app.entities, 0.0, 0.0)
        
        collideY(client, player)
        if player.onGround:
            player.velocity[0] = x
            player.velocity[2] = z
        else:
            player.velocity[0] += x / 10.0
            player.velocity[2] += z / 10.0
        collideXZ(client, player)

    client.cameraPos = copy.copy(player.pos)
    client.cameraPos[1] += player.height

    for entity in client.entities:
        entChunkPos = world.toChunkLocal(entity.getBlockPos())[0]
        entChunkPos = ChunkPos(entChunkPos.x, 0, entChunkPos.z)

        if entChunkPos not in client.world.chunks or not client.world.chunks[entChunkPos].isTicking:
            continue
    
        collide(client, entity)

    endTime = time.time()
    client.tickTimes[client.tickTimeIdx] = (endTime - startTime)
    client.tickTimeIdx += 1
    client.tickTimeIdx %= len(client.tickTimes)


def tick(app):
    startTime = time.time()

    app.time += 1

    instData = (app.textures, app.cube, app.textureIndices)

    app.world.loadUnloadChunks(app.cameraPos, (app.textures, app.cube, app.textureIndices))
    app.world.addChunkDetails(instData)
    app.world.tickChunks((app.textures, app.cube, app.textureIndices))

    updateBlockBreaking(app)

    doMobSpawning(app)
    doMobDespawning(app)

    # Ticking is done in stages so that collision detection works as expected:
    # First we update the player's Y position and resolve Y collisions,
    # then we update the player's X position and resolve X collisions,
    # and finally update the player's Z position and resolve Z collisions.

    # TODO: USE MOVE PACKETS

    # W makes the player go forward, S makes them go backwards,
    # and pressing both makes them stop!
    z = float(app.client.w) - float(app.client.s)
    # Likewise for side to side movement
    x = float(app.client.d) - float(app.client.a)

    player: Player = app.mode.player

    playerChunkPos = world.toChunkLocal(player.getBlockPos())[0]
    playerChunkPos = ChunkPos(playerChunkPos.x, 0, playerChunkPos.z)

    if playerChunkPos in app.world.chunks and app.world.chunks[playerChunkPos].isTicking:
        if x != 0.0 or z != 0.0:
            mag = math.sqrt(x*x + z*z)
            x /= mag
            z /= mag

            newX = math.cos(app.cameraYaw) * x - math.sin(app.cameraYaw) * z
            newZ = math.sin(app.cameraYaw) * x + math.cos(app.cameraYaw) * z

            x, z = newX, newZ

            x *= player.walkSpeed 
            z *= player.walkSpeed

        player.tick(app, app.world, app.entities, 0.0, 0.0)
        
        #player.pos = copy.copy(app.cameraPos)
        #player.pos[1] -= player.height

        collideY(app, player)
        if player.onGround:
            player.velocity[0] = x
            player.velocity[2] = z
        else:
            player.velocity[0] += x / 10.0
            player.velocity[2] += z / 10.0
        collideXZ(app, player)

    app.cameraPos = copy.copy(player.pos)
    app.cameraPos[1] += player.height

    entities = app.entities + [player]

    for entity in app.entities:
        entChunkPos = world.toChunkLocal(entity.getBlockPos())[0]
        entChunkPos = ChunkPos(entChunkPos.x, 0, entChunkPos.z)

        if entChunkPos not in app.world.chunks or not app.world.chunks[entChunkPos].isTicking:
            continue

        againstWall = collide(app, entity)

        if againstWall and entity.onGround:
            entity.velocity[1] = 0.40
        
        entity.tick(app, app.world, entities, player.pos[0], player.pos[2])

        if not config.UGLY_HACK:
            if entity.kind.name == 'item':
                dx = (player.pos[0] - entity.pos[0])**2
                dy = (player.pos[1] - entity.pos[1])**2
                dz = (player.pos[2] - entity.pos[2])**2
                if math.sqrt(dx + dy + dz) < 2.0 and entity.extra.pickupDelay == 0:
                    player.pickUpItem(app, entity.extra.stack)
                    entity.health = 0.0
            
        if entity.pos[1] < -64.0:
            entity.hit(app, 10.0, (0.0, 0.0))
    
    if player.pos[1] < -64.0:
        player.hit(app, 10.0, (0.0, 0.0))
    
    syncClient(app)
    
    endTime = time.time()
    app.tickTimes[app.tickTimeIdx] = (endTime - startTime)
    app.tickTimeIdx += 1
    app.tickTimeIdx %= len(app.tickTimes)

def syncClient(app):
    # TODO: Copy
    client: ClientState = app.client
    client.world = app.world
    client.entities = app.entities
    client.player = app.mode.player
    client.time = app.time

    #client.tickTimes = app.tickTimes
    #client.breakingBlock = app.breakingBlock
    #client.breakingBlockPos = app.breakingBlockPos

def doMobDespawning(app):
    player = app.mode.player

    idx = 0
    while idx < len(app.entities):
        [x, y, z] = app.entities[idx].pos
        dist = math.sqrt((x-player.pos[0])**2 + (y-player.pos[1])**2 + (z-player.pos[2])**2)

        maxDist = 128.0

        if dist > maxDist or app.entities[idx].health <= 0.0:
            app.entities.pop(idx)
        else:
            idx += 1

def doMobSpawning(app):
    mobCap = len(app.world.chunks) / 4

    random.seed(time.time())

    player = app.mode.player

    for (chunkPos, chunk) in app.world.chunks.items():
        chunk: world.Chunk
        if chunk.isTicking:
            if len(app.entities) > mobCap:
                return

            # FIXME: Random tick speed?
            x = random.randrange(0, 16) + chunkPos.x * 16
            y = random.randrange(0, world.CHUNK_HEIGHT) + chunkPos.y * world.CHUNK_HEIGHT
            z = random.randrange(0, 16) + chunkPos.z * 16

            dist = math.sqrt((x-player.pos[0])**2 + (y-player.pos[1])**2 + (z-player.pos[2])**2)

            minSpawnDist = 24.0
            
            if dist < minSpawnDist:
                continue

            if not isValidSpawnLocation(app, BlockPos(x, y, z)): continue

            mob = random.choice(['creeper', 'zombie', 'skeleton'])

            packSize = 4
            for _ in range(packSize):
                x += random.randint(-2, 2)
                z += random.randint(-2, 2)
                if isValidSpawnLocation(app, BlockPos(x, y, z)):
                    app.entities.append(Entity(app, mob, x, y, z))

def isValidSpawnLocation(app, pos: BlockPos):
    floor = BlockPos(pos.x, pos.y - 1, pos.z)
    feet = pos
    head = BlockPos(pos.x, pos.y + 1, pos.z)

    light = app.world.getTotalLight(app.time, pos)

    isOk = (app.world.coordsOccupied(floor)
        and not app.world.coordsOccupied(feet)
        and not app.world.coordsOccupied(head)
        and light < 3)
    
    return isOk

def collideY(client: ClientState, entity: Entity):
    entity.pos[1] += entity.velocity[1]

    if entity.onGround:
        if not client.world.hasBlockBeneath(entity):
            entity.onGround = False
    else:
        #if not hasattr(entity, 'flying') or not entity.flying: #type:ignore
        entity.velocity[1] -= client.gravity
        [_, yPos, _] = entity.pos
        #yPos -= entity.height
        yPos -= 0.1
        feetPos = roundHalfUp(yPos)
        if client.world.hasBlockBeneath(entity): 
            entity.onGround = True
            if hasattr(entity, 'flying'): entity.flying = False #type:ignore
            entity.velocity[1] = 0.0
            #app.cameraPos[1] = (feetPos + 0.5) + entity.height
            entity.pos[1] = feetPos + 0.5

    if not entity.onGround:
        for x in [entity.pos[0] - entity.radius * 0.99, entity.pos[0] + entity.radius * 0.99]:
            for z in [entity.pos[2] - entity.radius * 0.99, entity.pos[2] + entity.radius * 0.99]:
                hiYCoord = roundHalfUp(entity.pos[1] + entity.height)

                if client.world.coordsOccupied(BlockPos(round(x), hiYCoord, round(z))):
                    yEdge = hiYCoord - 0.55
                    entity.pos[1] = yEdge - entity.height
                    if entity.velocity[1] > 0.0:
                        entity.velocity[1] = 0.0


def collide(client: ClientState, entity: Entity):
    collideY(client, entity)
    return collideXZ(client, entity)

def collideXZ(client: ClientState, entity: Entity):
    hitWall = False

    minY = roundHalfUp((entity.pos[1]))
    maxY = roundHalfUp((entity.pos[1] + entity.height))

    entity.pos[0] += entity.velocity[0]

    for y in range(minY, maxY + 1):
        for z in [entity.pos[2] - entity.radius * 0.99, entity.pos[2] + entity.radius * 0.99]:
            x = entity.pos[0]

            hiXBlockCoord = round((x + entity.radius))
            loXBlockCoord = round((x - entity.radius))

            if client.world.coordsOccupied(BlockPos(hiXBlockCoord, y, round(z))):
                # Collision on the right, so move to the left
                xEdge = (hiXBlockCoord - 0.5)
                entity.pos[0] = xEdge - entity.radius
                hitWall = True
            elif client.world.coordsOccupied(BlockPos(loXBlockCoord, y, round(z))):
                # Collision on the left, so move to the right
                xEdge = (loXBlockCoord + 0.5)
                entity.pos[0] = xEdge + entity.radius
                hitWall = True
    
    entity.pos[2] += entity.velocity[2]

    for y in range(minY, maxY + 1):
        for x in [entity.pos[0] - entity.radius * 0.99, entity.pos[0] + entity.radius * 0.99]:
            z = entity.pos[2]

            hiZBlockCoord = round((z + entity.radius))
            loZBlockCoord = round((z - entity.radius))

            if client.world.coordsOccupied(BlockPos(round(x), y, hiZBlockCoord)):
                zEdge = (hiZBlockCoord - 0.5)
                entity.pos[2] = zEdge - entity.radius
                hitWall = True
            elif client.world.coordsOccupied(BlockPos(round(x), y, loZBlockCoord)):
                zEdge = (loZBlockCoord + 0.5)
                entity.pos[2] = zEdge + entity.radius
                hitWall = True
    
    return hitWall

