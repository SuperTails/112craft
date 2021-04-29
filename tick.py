"""Provides the `tick` function, which does all of the game's updates.

At regular intervals, the AI of all entities are processed, gravity is applied,
new entities are spawned, other entities are removed, collisions occur, etc.
"""

from entity import Entity
from player import Player
from client import ClientState
from server import ServerState, Window
from dimension import Dimension
from util import BlockPos, roundHalfUp, ChunkPos
import util
import world
from world import World, isSolid
import time
import math
import config
from math import cos, sin
import copy
import random
import network
import resources
from inventory import Stack, Slot
import inventory
from typing import Optional, List, Tuple
from quarry.types.uuid import UUID
from quarry.types.chat import Message

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
        elif action == network.DiggingAction.DROP_ITEM:
            # TODO: Spawn the item
            player = server.getLocalPlayer()
            slot = player.inventory[player.hotbarIdx]
            if not slot.stack.isEmpty() and not slot.stack.isInfinite():
                slot.stack.amount -= 1
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

def getSlotsInWindow(server: ServerState, windowId: int) -> Tuple[Stack, List[Slot]]:
    if windowId == 0:
        player = server.getLocalPlayer()

        if player.entityId not in server.craftSlots:
            server.craftSlots[player.entityId] = [Slot(canInput=False)] + [Slot() for _ in range(4)]
        
        # TODO: Armor slots
        baseSlots = server.craftSlots[player.entityId] + [Slot() for _ in range(4)]
    else:
        window = server.openWindows[windowId]

        player = None
        for p in server.players:
            if p.entityId == window.playerId:
                player = p
                break
        
        if player is None:
            raise Exception(f'Window click from nonexistent player {window.playerId}')
        
        if window.kind == 'furnace':
            (chunk, localPos) = server.getLocalDimension().world.getChunk(window.pos)
            furnace: world.Furnace = chunk.tileEntities[localPos]

            baseSlots = [furnace.inputSlot, furnace.fuelSlot, furnace.outputSlot] 
        else:
            raise Exception(f'Unknown window kind {window.kind}')

    slots = baseSlots + player.inventory[9:36] + player.inventory[0:9]
    
    if player.entityId not in server.heldItems:
        server.heldItems[player.entityId] = Stack('', 0)
    
    return (server.heldItems[player.entityId], slots)
    
def sendClickWindow(app, windowId: int, slotIdx: int, button: int, actionNum: int, mode: int, item, count):
    print(f'Sending window ID {windowId} and slot {slotIdx}, action {actionNum}')

    if hasattr(app, 'server'):
        server: ServerState = app.server

        heldItem, slots = getSlotsInWindow(server, windowId)

        if windowId == 0 or server.openWindows[windowId].kind == 'crafting':
            prevOutput = copy.deepcopy(slots[0].stack)
        else:
            prevOutput = None

        if mode == 0:
            if button == 0:
                isRight = False
            elif button == 1:
                isRight = True
            else:
                raise Exception(f'Invalid button {button} in mode 0')

            inventory.onSlotClicked(heldItem, app, isRight, slots[slotIdx])
        else:
            raise Exception(f'Invalid mode {mode} and button {button}')
        
        if windowId == 0:
            assert(prevOutput is not None)
            craftingGuiPostClick(slots, False, app, slotIdx, prevOutput)
        elif server.openWindows[windowId].kind == 'crafting':
            assert(prevOutput is not None)
            craftingGuiPostClick(slots, True, app, slotIdx, prevOutput)
        
    else:
        network.c2sQueue.put(network.ClickWindowC2S(windowId, slotIdx, button, actionNum, mode, item, count))

def craftingGuiPostClick(slots: List[Slot], is3x3: bool, app, slotIdx, prevOutput: Stack):
    if is3x3:
       totalCraftSlots = 9+1
    else:
        totalCraftSlots = 4+1

    if slotIdx == 0 and prevOutput != slots[0].stack:
        # Something was crafted
        for slot in slots[1:totalCraftSlots]:
            if slot.stack.amount > 0:
                slot.stack.amount -= 1

    def toid(s): return None if s.isEmpty() else s.item

    rowLen = round(math.sqrt(totalCraftSlots - 1))

    c = []

    for rowIdx in range(rowLen):
        row = []
        for colIdx in range(rowLen):
            row.append(toid(slots[1 + rowIdx * rowLen + colIdx].stack))
        c.append(row)
    
    slots[0].stack = Stack('', 0)

    for r in app.recipes:
        if r.isCraftedBy(c):
            slots[0].stack = copy.copy(r.outputs)
            break
    
def sendCloseWindow(app, windowId: int):
    if hasattr(app, 'server'):
        server: ServerState = app.server

        if windowId == 0:
            player = server.getLocalPlayer()
            if player.entityId in server.craftSlots:
                craftSlots = server.craftSlots.pop(player.entityId)
                for craftSlot in craftSlots[1:]:
                    player.pickUpItem(app, craftSlot.stack)
        else:
            server.openWindows.pop(windowId)
    else:
        network.c2sQueue.put(network.CloseWindowC2S(windowId))

def sendUseItem(app, hand: int):
    if hasattr(app, 'server'):
        server: ServerState = app.server
        player: Player = server.getLocalPlayer()

        wld = server.getLocalDimension().world
        
        heldSlot = player.inventory[player.hotbarIdx]
        if heldSlot.stack.isEmpty():
            return

        # FIXME:
        cameraPos = (player.pos[0], player.pos[1] + player.height, player.pos[2])

        if heldSlot.stack.item == 'bucket':
            block = wld.lookedAtBlock(player.reach, cameraPos,
                player.headPitch, player.headYaw, useFluids=True)

            if block is not None:
                (pos, _) = block

                blockId = wld.getBlock(pos)
                blockState = wld.getBlockState(pos)

                if blockId in ('water', 'flowing_water') and blockState['level'] == '0':
                    wld.setBlock((app.textures, app.cube, app.textureIndices), pos, 'air', {})
                    if not player.creative:
                        heldSlot.stack.item = 'water_bucket'
                elif blockId in ('lava', 'flowing_lava') and blockState['level'] == '0':
                    wld.setBlock((app.textures, app.cube, app.textureIndices), pos, 'air', {})
                    if not player.creative:
                        heldSlot.stack.item = 'lava_bucket'
        elif heldSlot.stack.item == 'water_bucket' or heldSlot.stack.item == 'lava_bucket':
            block = wld.lookedAtBlock(player.reach, cameraPos,
                player.headPitch, player.headYaw, useFluids=False)
            
            if block is not None:
                (pos, face) = block
                faceIdx = ['left', 'right', 'back', 'front', 'bottom', 'top'].index(face) * 2
                pos2 = world.adjacentBlockPos(pos, faceIdx)

                if heldSlot.stack.item == 'water_bucket':
                    blockId = 'flowing_water'
                else:
                    blockId = 'flowing_lava'

                wld.setBlock((app.textures, app.cube, app.textureIndices), pos2, blockId, { 'level': '0' })

                if not player.creative:
                    heldSlot.stack.item = 'bucket'
        elif heldSlot.stack.item == 'flint_and_steel':
            block = wld.lookedAtBlock(player.reach, cameraPos,
                player.headPitch, player.headYaw)

            if block is not None:
                (pos, face) = block
                faceIdx = ['left', 'right', 'back', 'front', 'bottom', 'top'].index(face) * 2
                pos2 = world.adjacentBlockPos(pos, faceIdx)

                portals = findPortalFrame(server, server.getLocalDimension(), pos2)
                if portals is not None:
                    portals, axis = portals
                    instData = (app.textures, app.cube, app.textureIndices)
                    for p in portals:
                        server.getLocalDimension().world.setBlock(instData, p, 'nether_portal', { 'axis': axis }, doBlockUpdates=False)
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

def syncWindowSlots(server: ServerState, windowId: int):
    if windowId == 0:
        # TODO:
        raise Exception()
    else:
        for slotIdx, slot in enumerate(getSlotsInWindow(server, windowId)[1]):
            if not slot.stack.isEmpty():
                itemId = util.REGISTRY.encode('minecraft:item', 'minecraft:' + slot.stack.item)
                network.s2cQueue.put(network.SetSlotS2C(windowId, slotIdx, itemId, slot.stack.amount))


def sendPlayerPlacement(app, hand: int, location: BlockPos, face: int, cx: float, cy: float, cz: float, insideBlock: bool):
    if hasattr(app, 'server'):
        server: ServerState = app.server
        player: Player = server.getLocalPlayer()

        blockId = server.getLocalDimension().world.getBlock(location)
        if blockId == 'crafting_table':
            windowId = server.getWindowId()
            kind = util.REGISTRY.encode('minecraft:menu', 'minecraft:crafting')
            title = Message.from_string('Crafting Table')

            server.openWindows[windowId] = Window(player.entityId, location, 'crafting')
            network.s2cQueue.put(network.OpenWindowS2C(windowId, kind, title))
            syncWindowSlots(server, windowId)
        elif blockId == 'furnace':
            windowId = server.getWindowId()
            kind = util.REGISTRY.encode('minecraft:menu', 'minecraft:furnace')
            title = Message.from_string('Furnace')

            server.openWindows[windowId] = Window(player.entityId, location, 'furnace')
            network.s2cQueue.put(network.OpenWindowS2C(windowId, kind, title))
            syncWindowSlots(server, windowId)
        elif not player.creative and blockId == 'air':
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

            dim = server.getLocalDimension()
            wld, entities = dim.world, dim.entities

            print(f"COMMAND {text}")

            if parts[0] == 'pathfind':
                player: Player = server.getLocalPlayer()
                target = player.getBlockPos()
                for ent in entities:
                    ent.updatePath(wld, target)
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
                if parts[1] == 'creative':
                    server.getLocalPlayer().creative = True
                elif parts[1] == 'survival':
                    server.getLocalPlayer().creative = False
            elif parts[0] == 'summon':
                player = server.getLocalPlayer()
                ent = Entity(app, server.getEntityId(), parts[1],
                    player.pos[0]+0.5, player.pos[1]+0.5, player.pos[2]+0.5)
                app.entities.append(ent)
            elif parts[0] == 'explode':
                power = int(parts[1])

                player = server.getLocalPlayer()

                pos = world.nearestBlockPos(player.pos[0], player.pos[1], player.pos[2])

                wld.explodeAt(pos, power, (app.textures, app.cube, app.textureIndices))
            elif parts[0] == 'dimension':
                player = server.getLocalPlayer()
                
                if player.dimension == 'minecraft:overworld':
                    player.dimension = 'minecraft:the_nether'
                elif player.dimension == 'minecraft:the_nether':
                    player.dimension = 'minecraft:overworld'
                else:
                    raise Exception(player.dimension)
                
                import quarry.types.nbt as quarrynbt

                player.portalCooldown = 80
                
                # TODO:
                network.s2cQueue.put(network.RespawnS2C(
                    quarrynbt.TagCompound({}), player.dimension,
                    0, 0, None, False, False, True
                ))
            elif parts[0] == 'chunkstates':
                for dim in server.dimensions:
                    print(f'== DIMENSION {dim} ==')
                    for pos, chunk in dim.world.chunks.items():
                        print(f'{pos} - {chunk.worldgenStage}')

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
            False, False, False, True, True, server.getTeleportId()
        ))

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

        entities = server.getLocalDimension().entities

        for ent in entities:
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
    
    wld = server.getLocalDimension().world

    blockId = wld.getBlock(pos)
    blockState = wld.getBlockState(pos)

    if blockId == 'air':
        print(f'Invalid mining at {server.breakingBlockPos}')
        return

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

        wld.setBlock((app.textures, app.cube, app.textureIndices), pos, 'air')

        server.breakingBlock = 0.0

        if droppedItem is not None:
            stack = Stack(droppedItem, 1)

            entityId = server.getEntityId()

            xVel = ((random.random() - 0.5) * 0.1)
            yVel = ((random.random() - 0.5) * 0.1)
            zVel = ((random.random() - 0.5) * 0.1)

            # TODO: UUID
            network.s2cQueue.put(network.SpawnEntityS2C(entityId, None, 37,
                float(pos.x), float(pos.y), float(pos.z), 0.0, 0.0, 1,
                int(xVel * 8000), int(yVel * 8000), int(zVel * 8000)))

            itemId = util.REGISTRY.encode('minecraft:item', 'minecraft:' + stack.item)

            network.s2cQueue.put(network.EntityMetadataS2C(
                entityId, { (6, 7): { 'item': itemId, 'count': stack.amount } }
            ))
            
            ent = Entity(app, entityId, 'item', float(pos.x), float(pos.y), float(pos.z))
            ent.extra.stack = stack
            ent.velocity = [xVel, yVel, zVel]

            server.getLocalDimension().entities.append(ent)

def findPortalFrame(server: ServerState, dim: Dimension, pos: BlockPos) -> Optional[Tuple[List[BlockPos], str]]:
    bottomPos1 = None
    topPos1 = None

    for i in range(1, 4):
        p = BlockPos(pos.x, pos.y - i, pos.z)
        if dim.world.getBlock(p) == 'obsidian':
            bottomPos1 = p
            break
    
    if bottomPos1 is None:
        return None

    for i in range(1, 4):
        p = BlockPos(pos.x, pos.y + i, pos.z)
        if dim.world.getBlock(p) == 'obsidian':
            topPos1 = p
            break
    
    if topPos1 is None:
        return None
    
    if topPos1.y - bottomPos1.y != 4:
        return None
    
    for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        bottomPos2 = BlockPos(bottomPos1.x + dx, bottomPos1.y, bottomPos1.z + dz) 
        topPos2 = BlockPos(topPos1.x + dx, topPos1.y, topPos1.z + dz)

        # Need 2 blocks on top, 2 blocks on bottom
        if dim.world.getBlock(bottomPos2) != 'obsidian':
            continue
        if dim.world.getBlock(topPos2) != 'obsidian':
            continue
    
        answer = []

        ok = True

        for i in range(3):
            backSidePos = BlockPos(bottomPos1.x - dx, bottomPos1.y + 1 + i, bottomPos1.z - dz)
            backMidPos = BlockPos(bottomPos1.x, bottomPos1.y + 1 + i, bottomPos1.z)
            frontMidPos = BlockPos(bottomPos2.x, bottomPos2.y + 1 + i, bottomPos2.z)
            frontSidePos = BlockPos(bottomPos2.x + dx, bottomPos2.y + 1 + i, bottomPos2.z + dz)

            if (dim.world.getBlock(backSidePos) != 'obsidian'
                or dim.world.getBlock(frontSidePos) != 'obsidian'
                or dim.world.getBlock(backMidPos) != 'air'
                or dim.world.getBlock(frontMidPos) != 'air'):

                ok = False
                break

            answer.append(backMidPos)
            answer.append(frontMidPos)
        
        if ok:
            return (answer, 'x' if dx != 0 else 'z')
    
    return None

def getDestination(app, world: World, searchPos: BlockPos, maxHeight: int) -> BlockPos:
    existing = findPortalNear(world, searchPos, maxHeight)
    if existing is not None:
        print(f'Found existing portal at {existing}')
        return existing
    
    spot = findSpaceForPortal(world, searchPos, maxHeight)

    if spot is not None:
        print(f'Found space for portal at {spot}')
        createPortalAt(app, world, spot, clearNearby=False)
        return BlockPos(spot.x, spot.y + 1, spot.z)
    
    forcedPos = BlockPos(searchPos.x, maxHeight - 20, searchPos.z)
    
    createPortalAt(app, world, forcedPos, clearNearby=True)
    return BlockPos(forcedPos.x, forcedPos.y + 1, forcedPos.z)
    
def findPortalNear(world: World, blockPos: BlockPos, maxHeight: int) -> Optional[BlockPos]:
    for totalDist in range(32):
        for xDist in range(totalDist):
            zDist = totalDist - xDist
            
            for y in range(maxHeight):
                if world.getBlock(BlockPos(blockPos.x + xDist, y, blockPos.z + zDist)) == 'nether_portal':
                    return BlockPos(blockPos.x + xDist, y, blockPos.z + zDist)
                if world.getBlock(BlockPos(blockPos.x + xDist, y, blockPos.z - zDist)) == 'nether_portal':
                    return BlockPos(blockPos.x + xDist, y, blockPos.z + zDist)
                if world.getBlock(BlockPos(blockPos.x - xDist, y, blockPos.z + zDist)) == 'nether_portal':
                    return BlockPos(blockPos.x + xDist, y, blockPos.z + zDist)
                if world.getBlock(BlockPos(blockPos.x - xDist, y, blockPos.z - zDist)) == 'nether_portal':
                    return BlockPos(blockPos.x + xDist, y, blockPos.z + zDist)
    
    return None

def createPortalAt(app, world: World, blockPos: BlockPos, clearNearby: bool):
    instData = (app.textures, app.cube, app.textureIndices)

    for dx in (-1, 0, 1, 2):
        world.setBlock(instData, BlockPos(blockPos.x + dx, blockPos.y, blockPos.z), 'obsidian', {})
        world.setBlock(instData, BlockPos(blockPos.x + dx, blockPos.y + 4, blockPos.z), 'obsidian', {})
    
    for dy in (1, 2, 3):
        world.setBlock(instData, BlockPos(blockPos.x - 1, blockPos.y + dy, blockPos.z), 'obsidian', {})
        world.setBlock(instData, BlockPos(blockPos.x + 2, blockPos.y + dy, blockPos.z), 'obsidian', {})
    
    for dx in (0, 1):
        for dy in (1, 2, 3):
            world.setBlock(instData, BlockPos(blockPos.x + dx, blockPos.y + dy, blockPos.z), 'nether_portal', { 'axis': 'x' }, doBlockUpdates=False)
        
    if clearNearby:
        for dz in (-1, 1):
            for dx in (-1, 0, 1, 2):
                for dy in (0, 1, 2, 3):
                    if dx in (0, 1) and dy == 0:
                        blockId = 'obsidian'
                        blockState = {}
                    else:
                        blockId = 'air'
                        blockState = {}

                    world.setBlock(instData, BlockPos(blockPos.x + dx, blockPos.y + dy, blockPos.z + dz), blockId, blockState)

def findSpaceForPortal(world: World, blockPos: BlockPos, maxHeight: int) -> Optional[BlockPos]:
    def isValidPos(pos: BlockPos):
        for dy in range(0, 4):
            for dx in range(-1, 4):
                blockId = world.getBlock(BlockPos(pos.x + dx, pos.y + dy, pos.z))
                if dy == 0:
                    if not isSolid(blockId):
                        return False
                else:
                    if blockId != 'air':
                        return False
        
        return True

    for totalDist in range(16):
        for xDist in range(totalDist):
            zDist = totalDist - xDist
            
            for y in range(10, maxHeight - 4):
                if isValidPos(BlockPos(blockPos.x + xDist, y, blockPos.z + zDist)):
                    return BlockPos(blockPos.x + xDist, y, blockPos.z + zDist)
                    
                if isValidPos(BlockPos(blockPos.x + xDist, y, blockPos.z - zDist)):
                    return BlockPos(blockPos.x + xDist, y, blockPos.z - zDist)
                    
                if isValidPos(BlockPos(blockPos.x - xDist, y, blockPos.z + zDist)):
                    return BlockPos(blockPos.x - xDist, y, blockPos.z + zDist)

                if isValidPos(BlockPos(blockPos.x - xDist, y, blockPos.z - zDist)):
                    return BlockPos(blockPos.x - xDist, y, blockPos.z - zDist)
    
    return None

def clientTick(client: ClientState, instData):
    startTime = time.time()

    client.time += 1

    if not client.local:
        chunkPos, _ = world.toChunkLocal(client.player.getBlockPos())

        client.world.tickets[chunkPos] = 1

        client.world.loadUnloadChunks(instData)
        client.world.addChunkDetails(instData, needUrgent=False)
    
    client.world.flushLightChanges()

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
        
        collideY(client.world, player)
        if player.onGround:
            player.velocity[0] = x
            player.velocity[2] = z
        else:
            player.velocity[0] += x / 10.0
            player.velocity[2] += z / 10.0
        collideXZ(client.world, player)
    
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

    for i, dim in enumerate(server.dimensions):
        player = server.getLocalPlayer()
        chunkPos, _ = world.toChunkLocal(player.getBlockPos())
        if i == 0 and player.dimension == 'minecraft:overworld':
            dim.world.addTicket(chunkPos, 1)
        elif i == 1 and player.dimension == 'minecraft:the_nether':
            dim.world.addTicket(chunkPos, 1)

        dim.world.loadUnloadChunks((app.textures, app.cube, app.textureIndices))
        dim.world.addChunkDetails(instData)
        dim.world.tickChunks(app)

    updateBlockBreaking(app, server)

    for dim in server.dimensions:
        doMobSpawning(app, server, dim)
        doMobDespawning(app, server, dim)

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

    for player in server.players:
        playerChunkPos = world.toChunkLocal(player.getBlockPos())[0]
        playerChunkPos = ChunkPos(playerChunkPos.x, 0, playerChunkPos.z)

        dim = server.getDimensionOf(player)

        player.tick(app, dim.world, dim.entities, 0.0, 0.0)

        if dim.world.getBlock(player.getBlockPos()) == 'nether_portal':
            if player.portalCooldown == 0:
                import quarry.types.nbt as quarrynbt

                if player.dimension == 'minecraft:overworld':
                    player.dimension = 'minecraft:the_nether'
                elif player.dimension == 'minecraft:the_nether':
                    player.dimension = 'minecraft:overworld'
                else:
                    raise Exception(player.dimension)
        
                player.portalCooldown = 80

                destDim = server.getDimension(player.dimension)
                destDim.world.addTicket(world.toChunkLocal(player.getBlockPos())[0], 300)
                destDim.world.loadUnloadChunks(instData)
                destDim.world.addChunkDetails(instData)

                dest = getDestination(app, destDim.world, player.getBlockPos(), destDim.world.dimTy.logicalHeight)

                # TODO:
                network.s2cQueue.put(network.RespawnS2C(
                    quarrynbt.TagCompound({}), player.dimension,
                    0, 0, None, False, False, True
                ))

                network.s2cQueue.put(network.PlayerPositionAndLookS2C(
                    dest.x, dest.y, dest.z, 0.0, 0.0, False, False, False, False, False, server.getTeleportId()
                ))
            else:
                player.portalCooldown = 80

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
    entities = server.getLocalDimension().entities + server.players #type:ignore

    # FIXME:
    player = server.getLocalPlayer()

    for dim in server.dimensions:
        for entity in dim.entities:
            entChunkPos = world.toChunkLocal(entity.getBlockPos())[0]
            entChunkPos = ChunkPos(entChunkPos.x, 0, entChunkPos.z)

            if entChunkPos not in dim.world.chunks or not dim.world.chunks[entChunkPos].isTicking:
                continue

            againstWall = collide(dim.world, entity)

            if againstWall and entity.onGround:
                entity.velocity[1] = 0.40
            
            entity.tick(app, dim.world, entities, player.pos[0], player.pos[2])

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
    
    for dim in server.dimensions:
        # HACK:
        for ent1, ent2 in zip(dim.entities, app.client.entities):
            ent1.variables = copy.copy(ent2.variables)

    app.client.entities = copy.deepcopy(server.getLocalDimension().entities)
    app.client.player.inventory = copy.deepcopy(server.getLocalPlayer().inventory)
    app.client.player.creative = server.getLocalPlayer().creative
    
    endTime = time.time()
    server.tickTimes[server.tickTimeIdx] = (endTime - startTime)
    server.tickTimeIdx += 1
    server.tickTimeIdx %= len(server.tickTimes)

    app.client.serverTickTimes = server.tickTimes
    app.client.serverTickTimeIdx = server.tickTimeIdx

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

def doMobDespawning(app, server: ServerState, dim: Dimension):
    # HACK:
    player = server.players[0]

    toDelete = []

    idx = 0
    while idx < len(dim.entities):
        [x, y, z] = dim.entities[idx].pos
        dist = math.sqrt((x-player.pos[0])**2 + (y-player.pos[1])**2 + (z-player.pos[2])**2)

        maxDist = 128.0

        if dist > maxDist or dim.entities[idx].health <= 0.0:
            toDelete.append(dim.entities[idx].entityId)
            dim.entities.pop(idx)
        else:
            idx += 1
    
    network.s2cQueue.put(network.DestroyEntitiesS2C(toDelete))

def doMobSpawning(app, server: ServerState, dim: Dimension):
    mobCap = len(dim.world.chunks) / 4

    random.seed(time.time())

    # HACK:
    player = server.players[0]

    for (chunkPos, chunk) in dim.world.chunks.items():
        chunk: world.Chunk
        if chunk.isTicking:
            if len(dim.entities) > mobCap:
                return

            # FIXME: Random tick speed?
            x = random.randrange(0, 16) + chunkPos.x * 16
            y = random.randrange(0, world.CHUNK_HEIGHT) + chunkPos.y * world.CHUNK_HEIGHT
            z = random.randrange(0, 16) + chunkPos.z * 16

            dist = math.sqrt((x-player.pos[0])**2 + (y-player.pos[1])**2 + (z-player.pos[2])**2)

            minSpawnDist = 24.0
            
            if dist < minSpawnDist:
                continue

            if not isValidSpawnLocation(app, dim, BlockPos(x, y, z)): continue

            mob = random.choice(['creeper', 'zombie', 'skeleton'])

            packSize = 4
            for _ in range(packSize):
                x += random.randint(-2, 2)
                z += random.randint(-2, 2)
                if isValidSpawnLocation(app, dim, BlockPos(x, y, z)):
                    dim.entities.append(Entity(app, server.getEntityId(), mob, x, y, z))

def isValidSpawnLocation(app, dim: Dimension, pos: BlockPos):
    server: ServerState = app.server

    floor = BlockPos(pos.x, pos.y - 1, pos.z)
    feet = pos
    head = BlockPos(pos.x, pos.y + 1, pos.z)

    light = dim.world.getTotalLight(app.time, pos)

    isOk = (dim.world.coordsOccupied(floor)
        and not dim.world.coordsOccupied(feet)
        and not dim.world.coordsOccupied(head)
        and light < 8)
    
    return isOk

GRAVITY = 0.1

def collideY(wld: World, entity: Entity):
    entity.pos[1] += entity.velocity[1]

    if entity.onGround:
        if not wld.hasBlockBeneath(entity):
            entity.onGround = False
    else:
        #if not hasattr(entity, 'flying') or not entity.flying: #type:ignore
        entity.velocity[1] -= GRAVITY
        [_, yPos, _] = entity.pos
        #yPos -= entity.height
        yPos -= 0.1
        feetPos = roundHalfUp(yPos)
        if wld.hasBlockBeneath(entity): 
            entity.onGround = True
            if hasattr(entity, 'flying'): entity.flying = False #type:ignore
            entity.velocity[1] = 0.0
            #app.cameraPos[1] = (feetPos + 0.5) + entity.height
            entity.pos[1] = feetPos + 0.5

    if not entity.onGround:
        for x in [entity.pos[0] - entity.radius * 0.99, entity.pos[0] + entity.radius * 0.99]:
            for z in [entity.pos[2] - entity.radius * 0.99, entity.pos[2] + entity.radius * 0.99]:
                hiYCoord = roundHalfUp(entity.pos[1] + entity.height)

                if wld.coordsOccupied(BlockPos(round(x), hiYCoord, round(z)), world.isSolid):
                    yEdge = hiYCoord - 0.55
                    entity.pos[1] = yEdge - entity.height
                    if entity.velocity[1] > 0.0:
                        entity.velocity[1] = 0.0


def collide(wld: World, entity: Entity):
    collideY(wld, entity)
    return collideXZ(wld, entity)

def collideXZ(wld: World, entity: Entity):
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

            if wld.coordsOccupied(BlockPos(hiXBlockCoord, y, round(z)), world.isSolid):
                # Collision on the right, so move to the left
                xEdge = (hiXBlockCoord - 0.5)
                entity.pos[0] = xEdge - entity.radius
                hitWall = True
            elif wld.coordsOccupied(BlockPos(loXBlockCoord, y, round(z)), world.isSolid):
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

            if wld.coordsOccupied(BlockPos(round(x), y, hiZBlockCoord), world.isSolid):
                zEdge = (hiZBlockCoord - 0.5)
                entity.pos[2] = zEdge - entity.radius
                hitWall = True
            elif wld.coordsOccupied(BlockPos(round(x), y, loZBlockCoord), world.isSolid):
                zEdge = (loZBlockCoord + 0.5)
                entity.pos[2] = zEdge + entity.radius
                hitWall = True
    
    entity.distanceMoved = math.sqrt((entity.pos[0] - lastX)**2 + (entity.pos[2] - lastZ)**2)
    
    return hitWall

