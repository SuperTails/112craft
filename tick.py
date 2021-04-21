"""Provides the `tick` function, which does all of the game's updates.

At regular intervals, the AI of all entities are processed, gravity is applied,
new entities are spawned, other entities are removed, collisions occur, etc.
"""

from entity import Entity
from player import Player
from client import ClientState
from server import ServerState
from util import BlockPos, roundHalfUp, ChunkPos
import util
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
from quarry.types.uuid import UUID

def sendPlayerDigging(app, action: network.DiggingAction, location: BlockPos, face: int):
    if hasattr(app, 'server'):
        server: ServerState = app.server
        if action == network.DiggingAction.START_DIGGING:
            server.breakingBlock = 0.0
            server.breakingBlockPos = location
        elif action == network.DiggingAction.CANCEL_DIGGING:
            server.breakingBlock = 0.0
        elif action == network.DiggingAction.FINISH_DIGGING:
            server.breakingBlock = 1000.0
        else:
            print(f"Ignoring other action {action}")
    else:
        network.c2sQueue.put(network.PlayerDiggingC2S(action, location, face))

def sendPlayerLook(app, yaw: float, pitch: float, onGround: bool):
    if hasattr(app, 'server'):
        player = app.server.getLocalPlayer()

        player.headYaw = yaw
        player.headPitch = pitch
        player.onGround = onGround
    else:
        network.c2sQueue.put(network.PlayerLookC2S(yaw, pitch, onGround))

def sendPlayerPosition(app, x, y, z, onGround):
    if hasattr(app, 'server'):
        player = app.server.getLocalPlayer()

        player.pos = [x, y, z]
        player.onGround = onGround
    else:
        network.c2sQueue.put(network.PlayerPositionC2S(x, y, z, onGround))

def sendClickWindow(app, windowId: int, slotIdx: int, button: int, actionNum: int, mode: int, item, count):
    print(f'Sending window ID {windowId} and slot {slotIdx}, action {actionNum}')

    if hasattr(app, 'server'):
        # TODO:
        pass
    else:
        network.c2sQueue.put(network.ClickWindowC2S(windowId, slotIdx, button, actionNum, mode, item, count))

def sendCloseWindow(app, windowId: int):
    if hasattr(app, 'server'):
        # TODO:
        pass
    else:
        network.c2sQueue.put(network.CloseWindowC2S(windowId))

def sendUseItem(app, hand: int):
    if hasattr(app, 'server'):
        server: ServerState = app.server
        player: Player = server.getLocalPlayer()
        
        heldSlot = player.inventory[player.hotbarIdx]
        if heldSlot.stack.isEmpty():
            return

        # FIXME:
        cameraPos = (player.pos[0], player.pos[1] + player.height, player.pos[2])

        if heldSlot.stack.item == 'bucket':
            block = server.world.lookedAtBlock(player.reach, cameraPos,
                player.headPitch, player.headYaw, useFluids=True)

            if block is not None:
                (pos, _) = block

                blockId = server.world.getBlock(pos)
                blockState = server.world.getBlockState(pos)

                if blockId in ['water', 'flowing_water'] and blockState['level'] == '0':
                    server.world.setBlock((app.textures, app.cube, app.textureIndices), pos, 'air', {})

                    heldSlot.stack.item = 'water_bucket'
        elif heldSlot.stack.item == 'water_bucket':
            block = server.world.lookedAtBlock(player.reach, cameraPos,
                player.headPitch, player.headYaw, useFluids=False)
            
            if block is not None:
                (pos, face) = block
                faceIdx = ['left', 'right', 'back', 'front', 'bottom', 'top'].index(face) * 2
                pos2 = world.adjacentBlockPos(pos, faceIdx)

                server.world.setBlock((app.textures, app.cube, app.textureIndices), pos2, 'flowing_water', { 'level': '0' })

                heldSlot.stack.item = 'bucket'
    else:
        network.c2sQueue.put(network.UseItemC2S(hand))

def sendTeleportConfirm(app, teleportId: int):
    if hasattr(app, 'server'):
        pass
    else:
        network.c2sQueue.put(network.TeleportConfirmC2S(teleportId))

def sendPlayerMovement(app, onGround: bool):
    if hasattr(app, 'server'):
        player = app.server.getLocalPlayer()
        player.onGround = True
    else:
        network.c2sQueue.put(network.PlayerMovementC2S(onGround))

def sendPlayerPlacement(app, hand: int, location: BlockPos, face: int, cx: float, cy: float, cz: float, insideBlock: bool):
    if hasattr(app, 'server'):
        player: Player = app.server.getLocalPlayer()
        if not player.creative:
            stack = player.inventory[player.hotbarIdx].stack
            if not stack.isInfinite():
                stack.amount -= 1
    else:
        network.c2sQueue.put(network.PlayerPlacementC2S(hand, location, face, cx, cy, cz, insideBlock))

def sendChatMessage(app, text: str):
    if hasattr(app, 'server'):
        if text.startswith('/'):
            text = text.removeprefix('/')

            parts = text.split()

            server: ServerState = app.server

            print(f"COMMAND {text}")

            if parts[0] == 'pathfind':
                player: Player = server.getLocalPlayer()
                target = player.getBlockPos()
                for ent in server.entities:
                    ent.updatePath(server.world, target)
            elif parts[0] == 'give':
                itemId = parts[1]
                if len(parts) == 3:
                    amount = int(parts[2])
                else:
                    amount = 1
                server.getLocalPlayer().pickUpItem(app, Stack(itemId, amount))
            elif parts[0] == 'time':
                if parts[1] == 'set':
                    if parts[2] == 'day':
                        server.time = 1000
                    elif parts[2] == 'night':
                        server.time = 13000
                    elif parts[2] == 'midnight':
                        server.time = 18000
                    else:
                        server.time = int(parts[2])
                elif parts[1] == 'add':
                    server.time += int(parts[2])
            elif parts[0] == 'gamemode':
                # TODO:
                '''
                if parts[1] == 'creative':
                    server.getLocalPlayer().creative = True
                elif parts[1] == 'survival':
                    app.mode.player.creative = False
                '''
            elif parts[0] == 'summon':
                player = server.getLocalPlayer()
                ent = Entity(app, server.getEntityId(), parts[1],
                    player.pos[0]+0.5, player.pos[1]+0.5, player.pos[2]+0.5)
                app.entities.append(ent)
            elif parts[0] == 'tp':
                # TODO:
                '''
                player = server.getLocalPlayer()
                player.pos[0] = float(parts[1])
                player.pos[1] = float(parts[2])
                player.pos[2] = float(parts[3])
                '''
    else:
        network.c2sQueue.put(network.ChatMessageC2S(text))


def sendClientStatus(app, status: int):
    # TODO: This isn't *really* a player joining packet, buuuut...

    if hasattr(app, 'server'):
        server: ServerState = app.server
        player = server.getLocalPlayer()

        network.s2cQueue.put(network.PlayerPositionAndLookS2C(
            player.pos[0], player.pos[1], player.pos[2], 0.0, 0.0,
            False, False, False, True, True, server.teleportId
        ))

        server.teleportId += 1

        '''
        for ent in server.entities:
            # TODO:
            uuid = UUID.random()

            kind = util.REGISTRY.encode('minecraft:entity_type', entity.)

            network.s2cQueue.put(network.SpawnMobS2C(
                ent.entityId, uuid, 
            ))
        '''
    else:
        network.c2sQueue.put(network.ClientStatusC2S(status))

def sendHeldItemChange(app, newSlot: int):
    if hasattr(app, 'server'):
        player = app.server.getLocalPlayer()
        player.hotbarIdx = newSlot
    else:
        network.c2sQueue.put(network.HeldItemChangeC2S(newSlot))

def sendInteractEntity(app, entityId, kind, *, x=None, y=None, z=None, hand=None, sneaking):
    if hasattr(app, 'server'):
        server: ServerState = app.server
        player = server.getLocalPlayer()

        for ent in server.entities:
            if ent.entityId == entityId:
                if kind == network.InteractKind.ATTACK:
                    knockX = ent.pos[0] - player.pos[0]
                    knockZ = ent.pos[2] - player.pos[2]
                    mag = math.sqrt(knockX**2 + knockZ**2)
                    knockX /= mag
                    knockZ /= mag

                    slot = player.inventory[player.hotbarIdx]
                    
                    if slot.isEmpty():
                        dmg = 1.0
                    else:
                        dmg = 1.0 + resources.getAttackDamage(app, slot.stack.item)

                    ent.hit(app, dmg, (knockX, knockZ))
                    break
                else:
                    # TODO:
                    pass
    else:
        network.c2sQueue.put(network.InteractEntityC2S(
            entityId, kind, x=x, y=y, z=z, hand=hand, sneaking=sneaking
        ))


def updateBlockBreaking(app, server: ServerState):
    pos = server.breakingBlockPos

    if server.breakingBlock == 0.0:
        return

    blockId = server.world.getBlock(pos)
    blockState = server.world.getBlockState(pos)

    # HACK:
    player = server.getLocalPlayer()

    toolStack = player.inventory[player.hotbarIdx].stack
    if toolStack.isEmpty():
        tool = ''
    else:
        tool = toolStack.item

    hardness = resources.getHardnessAgainst(blockId, tool)

    # TODO: Sound effect packets

    if server.breakingBlock >= hardness:
        if blockId == 'oak_log':
            blockState['axis'] = 'y'
        mcBlockId = util.REGISTRY.encode_block({ 'name': 'minecraft:' + blockId } | blockState)

        network.s2cQueue.put(network.AckPlayerDiggingS2C(
            pos,
            mcBlockId,
            network.DiggingAction.FINISH_DIGGING,
            True
        ))

        droppedItem = resources.getBlockDrop(app, blockId, tool)

        resources.getDigSound(app, blockId).play()

        server.world.setBlock((app.textures, app.cube, app.textureIndices), pos, 'air')

        server.breakingBlock = 0.0

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

            itemId = util.REGISTRY.encode('minecraft:item', 'minecraft:' + stack.item)

            network.s2cQueue.put(network.EntityMetadataS2C(
                entityId, { (6, 7): { 'item': itemId, 'count': stack.amount } }
            ))
            
            # FIXME: IDs
            ent = Entity(app, 1, 'item', pos.x, pos.y, pos.z)
            ent.extra.stack = stack
            ent.velocity = [xVel, yVel, zVel]
            server.entities.append(ent)

entityIdNum = 10_000

def getNextEntityId() -> int:
    global entityIdNum
    entityIdNum += 1
    return entityIdNum

def clientTick(client: ClientState, instData):
    startTime = time.time()

    client.time += 1

    if not client.local:
        client.world.loadUnloadChunks(client.player.pos, instData)
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
    
        entity.clientTick()
    
        #collide(client, entity)

    endTime = time.time()
    client.tickTimes[client.tickTimeIdx] = (endTime - startTime)
    client.tickTimeIdx += 1
    client.tickTimeIdx %= len(client.tickTimes)

    client.lastTickTime = endTime

def serverTick(app, server: ServerState):
    startTime = time.time()

    server.time += 1

    instData = (app.textures, app.cube, app.textureIndices)

    server.world.loadUnloadChunks(server.getLocalPlayer().pos, (app.textures, app.cube, app.textureIndices))
    server.world.addChunkDetails(instData)
    #server.world.tickChunks((app.textures, app.cube, app.textureIndices))
    server.world.tickChunks(app)

    updateBlockBreaking(app, server)

    doMobSpawning(app, server)
    doMobDespawning(app, server)

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

    # FIXME:
    player: Player = server.getLocalPlayer()

    playerChunkPos = world.toChunkLocal(player.getBlockPos())[0]
    playerChunkPos = ChunkPos(playerChunkPos.x, 0, playerChunkPos.z)

    player.tick(app, server.world, server.entities, 0.0, 0.0)
    
    '''
    collideY(app, player)
    if player.onGround:
        player.velocity[0] = x
        player.velocity[2] = z
    else:
        player.velocity[0] += x / 10.0
        player.velocity[2] += z / 10.0
    collideXZ(app, player)
    '''

    # FIXME: types???
    entities = server.entities + server.players #type:ignore

    for entity in server.entities:
        entChunkPos = world.toChunkLocal(entity.getBlockPos())[0]
        entChunkPos = ChunkPos(entChunkPos.x, 0, entChunkPos.z)

        if entChunkPos not in server.world.chunks or not server.world.chunks[entChunkPos].isTicking:
            continue

        againstWall = collide(server, entity) #type:ignore

        if againstWall and entity.onGround:
            entity.velocity[1] = 0.40
        
        entity.tick(app, server.world, entities, player.pos[0], player.pos[2])

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
    
    network.s2cQueue.put(network.TimeUpdateS2C(0, server.time))
    
    # HACK:
    for ent1, ent2 in zip(server.entities, app.client.entities):
        ent1.variables = copy.copy(ent2.variables)
    app.client.entities = copy.deepcopy(server.entities)
    app.client.player.inventory = copy.deepcopy(server.getLocalPlayer().inventory)
    
    endTime = time.time()
    server.tickTimes[server.tickTimeIdx] = (endTime - startTime)
    server.tickTimeIdx += 1
    server.tickTimeIdx %= len(server.tickTimes)

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

def doMobDespawning(app, server: ServerState):
    # HACK:
    player = server.players[0]

    toDelete = []

    idx = 0
    while idx < len(server.entities):
        [x, y, z] = server.entities[idx].pos
        dist = math.sqrt((x-player.pos[0])**2 + (y-player.pos[1])**2 + (z-player.pos[2])**2)

        maxDist = 128.0

        if dist > maxDist or server.entities[idx].health <= 0.0:
            toDelete.append(server.entities[idx].entityId)
            server.entities.pop(idx)
        else:
            idx += 1
    
    network.s2cQueue.put(network.DestroyEntitiesS2C(toDelete))

def doMobSpawning(app, server: ServerState):
    mobCap = len(server.world.chunks) / 4

    random.seed(time.time())

    # HACK:
    player = server.players[0]

    for (chunkPos, chunk) in server.world.chunks.items():
        chunk: world.Chunk
        if chunk.isTicking:
            if len(server.entities) > mobCap:
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
                    # FIXME: IDs
                    server.entities.append(Entity(app, 1, mob, x, y, z))

def isValidSpawnLocation(app, pos: BlockPos):
    server: ServerState = app.server

    floor = BlockPos(pos.x, pos.y - 1, pos.z)
    feet = pos
    head = BlockPos(pos.x, pos.y + 1, pos.z)

    light = server.world.getTotalLight(app.time, pos)

    isOk = (server.world.coordsOccupied(floor)
        and not server.world.coordsOccupied(feet)
        and not server.world.coordsOccupied(head)
        and light < 8)
    
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

                if client.world.coordsOccupied(BlockPos(round(x), hiYCoord, round(z)), world.isSolid):
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

    lastX = entity.pos[0]
    lastZ = entity.pos[2]

    entity.pos[0] += entity.velocity[0]

    for y in range(minY, maxY + 1):
        for z in [entity.pos[2] - entity.radius * 0.99, entity.pos[2] + entity.radius * 0.99]:
            x = entity.pos[0]

            hiXBlockCoord = round((x + entity.radius))
            loXBlockCoord = round((x - entity.radius))

            if client.world.coordsOccupied(BlockPos(hiXBlockCoord, y, round(z)), world.isSolid):
                # Collision on the right, so move to the left
                xEdge = (hiXBlockCoord - 0.5)
                entity.pos[0] = xEdge - entity.radius
                hitWall = True
            elif client.world.coordsOccupied(BlockPos(loXBlockCoord, y, round(z)), world.isSolid):
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

            if client.world.coordsOccupied(BlockPos(round(x), y, hiZBlockCoord), world.isSolid):
                zEdge = (hiZBlockCoord - 0.5)
                entity.pos[2] = zEdge - entity.radius
                hitWall = True
            elif client.world.coordsOccupied(BlockPos(round(x), y, loZBlockCoord), world.isSolid):
                zEdge = (loZBlockCoord + 0.5)
                entity.pos[2] = zEdge + entity.radius
                hitWall = True
    
    entity.distanceMoved = math.sqrt((entity.pos[0] - lastX)**2 + (entity.pos[2] - lastZ)**2)
    
    return hitWall

