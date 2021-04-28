"""This module is used for representing the game's terrain and its generation.

# Important classes #

Emu - Used during worldgen to generate cave shapes.

Chunk - A small box of terrain that can be individually loaded or unloaded

World - A collection of chunks pieced together to form coherent terrain.

# Worldgen #

Each chunk is generated in stages, which are, in order:

Not started - The chunk is completely empty.

Generated - The chunk is now a barren landscape, (maybe with some caves).

Populated - Trees, ores, and other interesting features have been added.

Optimized - The chunk has been lit and hidden faces have been marked.

Complete - The chunk has a full mesh built and is ready to render.

The "optimize" and "mesh" steps are also run for chunks loaded from disk.
"""

import numpy as np
import math
import heapq
import render
import time
import perlin
import random
import config
import anvil
import os
import copy
import itertools
from collections import deque
from nbt import nbt
from inventory import Slot, Stack
from enum import IntEnum
from math import cos, sin
from numpy import ndarray
from abc import ABC, abstractmethod
from typing import NamedTuple, List, Any, Tuple, Optional, Union, Iterable
import util
from util import *
from OpenGL.GL import * #type:ignore
from network import ChunkDataS2C
from dimregistry import BiomeEntry

# Places a tree with its bottommost log at the given position in the world.
# If `doUpdates` is True, this recalculates the lighting and block visibility.
# Normally that's a good thing, but during worldgen it's redundant.
def generateTree(world: 'World', instData, basePos: BlockPos, doUpdates=True):
    x = basePos.x
    y = basePos.y
    z = basePos.z

    l = doUpdates
    b = doUpdates

    # Place bottom logs
    world.setBlock(instData, basePos, 'oak_log', doUpdateLight=l, doUpdateBuried=b)
    y += 1
    world.setBlock(instData, BlockPos(x, y, z), 'oak_log', doUpdateLight=l, doUpdateBuried=b)
    # Place log and oak_leaves around it
    for _ in range(2):
        y += 1
        world.setBlock(instData, BlockPos(x, y, z), 'oak_log', doUpdateLight=l, doUpdateBuried=b)
        for xOffset in range(-2, 2+1):
            for zOffset in range(-2, 2+1):
                if abs(xOffset) == 2 and abs(zOffset) == 2: continue
                if xOffset == 0 and zOffset == 0: continue

                world.setBlock(instData, BlockPos(x + xOffset, y, z + zOffset), 'oak_leaves', doUpdateLight=l, doUpdateBuried=b)
    
    # Narrower top part
    y += 1
    world.setBlock(instData, BlockPos(x - 1, y, z), 'oak_leaves', doUpdateLight=l, doUpdateBuried=b)
    world.setBlock(instData, BlockPos(x + 1, y, z), 'oak_leaves', doUpdateLight=l, doUpdateBuried=b)
    world.setBlock(instData, BlockPos(x, y, z - 1), 'oak_leaves', doUpdateLight=l, doUpdateBuried=b)
    world.setBlock(instData, BlockPos(x, y, z + 1), 'oak_leaves', doUpdateLight=l, doUpdateBuried=b)
    world.setBlock(instData, BlockPos(x, y, z), 'oak_log', doUpdateLight=l, doUpdateBuried=b)

    # Top cap of just oak_leaves
    y += 1
    world.setBlock(instData, BlockPos(x - 1, y, z), 'oak_leaves', doUpdateLight=l, doUpdateBuried=b)
    world.setBlock(instData, BlockPos(x + 1, y, z), 'oak_leaves', doUpdateLight=l, doUpdateBuried=b)
    world.setBlock(instData, BlockPos(x, y, z - 1), 'oak_leaves', doUpdateLight=l, doUpdateBuried=b)
    world.setBlock(instData, BlockPos(x, y, z + 1), 'oak_leaves', doUpdateLight=l, doUpdateBuried=b)
    world.setBlock(instData, BlockPos(x, y, z), 'oak_leaves', doUpdateLight=l, doUpdateBuried=b)


class WorldgenStage(IntEnum):
    NOT_STARTED = 0,
    GENERATED = 1,
    POPULATED = 2,
    OPTIMIZED = 3,
    COMPLETE = 4,

# -34, 70, -89
# 2, 6, -6
# at y = 70ish

seen = set()

CHUNK_HEIGHT = 256
MESH_HEIGHT = 16

MAX_CAVE_DISP = 100
MAX_CAVE_LENGTH = 200

class Emu:
    """A class used during worldgen to represent a random walk.

    Named after the 'drunken emu' method from the talk linked below.
    Each step it moves forward and then turns slightly. The path the 'emu'
    takes carves out a cave. 

    # https://www.gdcvault.com/play/1025191/Math-for-Game-Programmers-Digging
    """

    start: BlockPos 

    x: float
    y: float
    z: float

    count: int

    yaw: float
    pitch: float

    def __init__(self, pos: BlockPos):
        self.start = pos

        self.x = pos.x
        self.y = pos.y
        self.z = pos.z

        self.yaw = 0.0
        self.pitch = math.radians(-35.0)

        self.count = 0

    def step(self, seed: int) -> bool:
        random.seed(hash((self.count, seed)))

        self.yaw += ((random.random() * 2.0) - 1.0) * math.radians(20.0)

        # Caves should not intersect with the bottom of the world,
        # so make them go back upwards if they're too low
        if self.y > 100 and self.pitch > 0.0:
            pitchOffset = -math.exp(-0.1 * (128 - self.y))
        if self.y < 10 and self.pitch < 0.0:
            pitchOffset = math.exp(-0.5 * self.y)
        else:
            pitchOffset = 0.0

        self.pitch += ((random.random() * 2.0) - 1.0 + pitchOffset) * math.radians(20.0)

        self.x += cos(self.pitch) * cos(self.yaw)
        self.y += sin(self.pitch)
        self.z += cos(self.pitch) * sin(-self.yaw)

        self.count += 1

        dist = abs(self.x - self.start.x) + abs(self.y - self.start.y) + abs(self.z - self.start.z)

        return self.count > MAX_CAVE_LENGTH or dist > MAX_CAVE_DISP
    
    def clearNearby(self, world: 'World', instData):
        blockPos = nearestBlockPos(self.x, self.y, self.z)

        for xOffset in range(-1, 2):
            for yOffset in range(-1, 2):
                for zOffset in range(-1, 2):
                    bx = blockPos.x + xOffset
                    by = blockPos.y + yOffset
                    bz = blockPos.z + zOffset
                    bPos = BlockPos(bx, by, bz)

                    (cp, cl) = toChunkLocal(bPos)

                    world.chunks[cp].setBlock(world, instData, cl, 'air')

def generateCaveCenter(startPos: BlockPos, seed: int) -> List[BlockPos]:
    emu = Emu(startPos)
    positions = []
    while True:
        positions.append(nearestBlockPos(emu.x, emu.y, emu.z))
        if emu.step(seed):
            break
    positions.sort()
    return positions

def timed(count=3):
    def timedDecor(f):
        i = 0
        durations = [0.0] * count

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            f(*args, **kwargs)
            end = time.perf_counter()

            nonlocal i, durations
            durations[i] = end - start
            i += 1
            i %= len(durations)

            avg = sum(durations) / len(durations)
            print(f"average time for {f.__name__} is {avg}")

        return wrapper

    return timedDecor

def binarySearchIdx(L: List[BlockPos], x: int) -> Optional[int]:
    startIdx = 0
    endIdx = len(L)

    while True:
        idx = (endIdx + startIdx) // 2

        if idx == len(L):
            return None
        if L[idx].x == x:
            return idx
        elif endIdx == startIdx:
            return None
        elif L[idx].x > x:
            endIdx = idx
        else:
            startIdx = idx + 1

def binarySearchMin(L: List[BlockPos], x: int) -> Optional[int]:
    upperBound = binarySearchIdx(L, x)
    if upperBound is None:
        return None
    else:
        for i in range(upperBound, -1, -1):
            if L[i].x != x:
                return i + 1
        
        return 0

def binarySearchMax(L: List[BlockPos], x: int) -> Optional[int]:
    lowerBound = binarySearchIdx(L, x)
    if lowerBound is None:
        return None
    else:
        for i in range(lowerBound, len(L)):
            if L[i].x != x:
                return i - 1
        
        return len(L) - 1

CUBE_MESH_VERTICES = np.array([
    # Left face
    -0.5,  0.5,  0.5,  1.0, 1.0, # top-right
    -0.5,  0.5, -0.5,  0.0, 1.0, # top-left
    -0.5, -0.5, -0.5,  0.0, 0.0, # bottom-left
    -0.5, -0.5, -0.5,  0.0, 0.0, # bottom-left
    -0.5, -0.5,  0.5,  1.0, 0.0, # bottom-right
    -0.5,  0.5,  0.5,  1.0, 1.0, # top-right
    # Right face
    0.5,  0.5,  0.5,  0.0, 1.0, # top-left
    0.5, -0.5, -0.5,  1.0, 0.0, # bottom-right
    0.5,  0.5, -0.5,  1.0, 1.0, # top-right         
    0.5, -0.5, -0.5,  1.0, 0.0, # bottom-right
    0.5,  0.5,  0.5,  0.0, 1.0, # top-left
    0.5, -0.5,  0.5,  0.0, 0.0, # bottom-left     
    # Back face
    -0.5, -0.5, -0.5,  0.0, 0.0, # Bottom-left
    0.5,  0.5, -0.5,  1.0, 1.0, # top-right
    0.5, -0.5, -0.5,  1.0, 0.0, # bottom-right         
    0.5,  0.5, -0.5,  1.0, 1.0, # top-right
    -0.5, -0.5, -0.5,  0.0, 0.0, # bottom-left
    -0.5,  0.5, -0.5,  0.0, 1.0, # top-left
    # Front face
    -0.5, -0.5,  0.5,  0.0, 0.0, # bottom-left
    0.5, -0.5,  0.5,  1.0, 0.0, # bottom-right
    0.5,  0.5,  0.5,  1.0, 1.0, # top-right
    0.5,  0.5,  0.5,  1.0, 1.0, # top-right
    -0.5,  0.5,  0.5,  0.0, 1.0, # top-left
    -0.5, -0.5,  0.5,  0.0, 0.0, # bottom-left
    # Bottom face
    -0.5, -0.5, -0.5,  1.0, 1.0, # top-right
    0.5, -0.5, -0.5,  0.0, 1.0, # top-left
    0.5, -0.5,  0.5,  0.0, 0.0, # bottom-left
    0.5, -0.5,  0.5,  0.0, 0.0, # bottom-left
    -0.5, -0.5,  0.5,  1.0, 0.0, # bottom-right
    -0.5, -0.5, -0.5,  1.0, 1.0, # top-right
    # Top face
    -0.5,  0.5, -0.5,  0.0, 1.0, # top-left
    0.5,  0.5,  0.5,  1.0, 0.0, # bottom-right
    0.5,  0.5, -0.5,  1.0, 1.0, # top-right     
    0.5,  0.5,  0.5,  1.0, 0.0, # bottom-right
    -0.5,  0.5, -0.5,  0.0, 1.0, # top-left
    -0.5,  0.5,  0.5,  0.0, 0.0, # bottom-left
    ], dtype='float32')

class Furnace:
    pos: BlockPos

    inputSlot: Slot
    fuelSlot: Slot
    outputSlot: Slot

    fuelLeft: int
    progress: int

    def __init__(self, pos: BlockPos):
        self.pos = pos

        self.inputSlot = Slot()
        self.fuelSlot = Slot(itemFilter='fuel')
        self.outputSlot = Slot(canInput=False)

        self.fuelLeft = 0
        self.progress = 0
    
    def toNbt(self) -> nbt.TAG_Compound:
        tag = nbt.TAG_Compound()

        tag.tags.append(nbt.TAG_Int(self.pos.x, 'x'))
        tag.tags.append(nbt.TAG_Int(self.pos.y, 'y'))
        tag.tags.append(nbt.TAG_Int(self.pos.z, 'z'))

        tag.tags.append(nbt.TAG_String('furnace', 'id'))

        tag.tags.append(nbt.TAG_Short(self.fuelLeft, 'BurnTime'))
        tag.tags.append(nbt.TAG_Short(self.progress, 'CookTime'))
        tag.tags.append(nbt.TAG_Short(200, 'CookTimeTotal'))
        tag.tags.append(nbt.TAG_Byte(0, 'keepPacked'))

        items = nbt.TAG_List(nbt.TAG_Compound, name='Items')
        for (idx, slot) in enumerate((self.inputSlot, self.fuelSlot, self.outputSlot)):
            if not slot.stack.isEmpty():
                items.append(slot.stack.toNbt(idx))
        tag.tags.append(items)

        return tag
    
    @classmethod
    def fromNbt(cls, tag: nbt.TAG_Compound):
        assert(tag['id'].value == 'furnace')

        x, y, z = tag['x'].value, tag['y'].value, tag['z'].value

        furnace = cls(BlockPos(x, y, z))
        furnace.fuelLeft = tag['BurnTime'].value
        furnace.progress = tag['CookTime'].value

        for stackTag in tag['Items']:
            (stack, slotIdx) = Stack.fromNbt(stackTag, getSlot=True)
            if slotIdx == 0:
                furnace.inputSlot.stack = stack
            elif slotIdx == 1:
                furnace.fuelSlot.stack = stack
            elif slotIdx == 2:
                furnace.outputSlot.stack = stack
            else:
                raise Exception(f"Invalid furnace slot {slotIdx}")
        
        return furnace

    def tick(self, app):
        if self.fuelLeft > 0:
            self.fuelLeft -= 1

            if self.inputSlot.isEmpty():
                self.progress = 0
            else:
                self.progress += 1
            
            if self.progress == 200:
                self.progress = 0

                if self.outputSlot.isEmpty():
                    self.outputSlot.stack.item = app.furnaceRecipes[self.inputSlot.stack.item]
                    self.outputSlot.stack.amount = 1
                else:
                    self.outputSlot.stack.amount += 1
                self.inputSlot.stack.amount -= 1
        
        if self.fuelLeft == 0 and not self.inputSlot.isEmpty() and not self.fuelSlot.isEmpty():
            self.fuelSlot.stack.amount -= 1
            self.fuelLeft = 1600

def blockEntityFromNbt(tag: nbt.TAG_Compound) -> Union[Furnace]:
    kind = tag['id'].value
    if kind == 'furnace':
        return Furnace.fromNbt(tag)
    else:
        raise Exception(f"Unknown block entity kind {kind}")

def convertBlock(block, instData):
    global seen

    if block == 'grass_block':
        block = 'grass'
    elif block in ['smooth_stone_slab', 'gravel']:
        block = 'stone'
    elif block in ['mossy_cobblestone']:
        block = 'cobblestone'
    elif block in ['birch_leaves', 'spruce_leaves', 'jungle_leaves']:
        block = 'oak_leaves'
    elif block in ['birch_log', 'jungle_wood']:
        block = 'oak_log'
    elif block in ['dandelion', 'poppy', 'fern', 'grass', 'brown_mushroom', 'red_mushroom']:
        block = 'air'
    elif block in ['glowstone', 'dispenser', 'red_bed', 'chest', 'note_block', 'gold_block']:
        block = 'oak_planks'
    elif block.endswith('_wool'):
        block = 'crafting_table'
    elif 'piston' in block:
        block = 'crafting_table'
    elif block == 'cave_air':
        block = 'air'
    elif block != 'air' and block not in instData[0]:
        if block not in seen:
            print(f"UNKNOWN BLOCK {block}")
            seen.add(block)
        block = 'bedrock'
    
    return block

def getWireConnections(pos: BlockPos, world: 'World'):
    def connectable(blockId):
        return blockId in ('redstone_wire', 'redstone_torch', 'redstone_wall_torch')

    def connectableVert(blockId):
        return blockId == 'redstone_wire'

    state = {}
    for dirName, dx, dz in (('west', -1, 0), ('east', 1, 0), ('south', 0, -1), ('north', 0, 1)):
        conn = 'none'

        horizPos = BlockPos(pos.x + dx, pos.y, pos.z + dz)

        if world.coordsOccupied(horizPos, connectable):
            conn = 'side'
        else:
            downPos = BlockPos(horizPos.x, horizPos.y - 1, horizPos.z)
            if not world.coordsOccupied(horizPos) and world.coordsOccupied(downPos, connectableVert):
                conn = 'down'
            else:
                corner = BlockPos(pos.x, pos.y + 1, pos.z)
                upPos = BlockPos(horizPos.x, horizPos.y + 1, horizPos.z)
                if not world.coordsOccupied(corner) and world.coordsOccupied(upPos, connectableVert):
                    conn = 'up'
                
        state[dirName] = conn
    
    cnt = (state['east'] != 'none') + (state['west'] != 'none') + (state['north'] != 'none') + (state['south'] != 'none')
    if cnt == 1:
        if state['east'] != 'none':
            state['west'] = 'side'
        elif state['west'] != 'none':
            state['east'] = 'side'
        elif state['north'] != 'none':
            state['south'] = 'side'
        elif state['south'] != 'none':
            state['north'] = 'side'
    
    return state
    
def updateRedstoneWire(pos: BlockPos, world: 'World', instData):
    state = world.getBlockState(pos)

    newConns = getWireConnections(pos, world)

    connsChanged = False

    for name, val in newConns.items():
        if val == 'down':
            val = 'side'

        if state[name] != val:
            connsChanged = True
            state[name] = val

    powerChanged = False
    
    if connsChanged or powerChanged:
        world.setBlock(instData, pos, 'redstone_wire', state, doBlockUpdates=powerChanged)

class Chunk:
    pos: ChunkPos
    blocks: ndarray
    blockStates: ndarray
    lightLevels: ndarray
    blockLightLevels: ndarray
    instances: List[Any]

    biomes: ndarray

    tileEntities: dict[BlockPos, Any]

    meshVaos: List[int]
    meshVbos: List[int]
    meshVertexCounts: List[int]
    meshDirtyFlags: List[bool]

    worldgenStage: WorldgenStage = WorldgenStage.NOT_STARTED

    tickIdx: int
    scheduledTicks: deque[Tuple[int, BlockPos]]

    isTicking: bool
    isVisible: bool

    def __init__(self, pos: ChunkPos):
        self.pos = pos

        self.blocks = np.full((16, CHUNK_HEIGHT, 16), 'air', dtype=object)
        self.blockStates = np.full((16, CHUNK_HEIGHT, 16), None, dtype=object)
        for (x, y, z) in np.ndindex(16, CHUNK_HEIGHT, 16):
            self.blockStates[x, y, z] = {}
        self.lightLevels = np.full((16, CHUNK_HEIGHT, 16), 0)
        self.blockLightLevels = np.full((16, CHUNK_HEIGHT, 16), 0)
        self.instances = [None] * self.blocks.size

        self.biomes = np.full((16, CHUNK_HEIGHT // 4, 16), util.DIMENSION_CODEC.getBiome('minecraft:the_void'), dtype=object)

        self.tickIdx = 0
        self.scheduledTicks = deque()

        self.isTicking = False
        self.isVisible = False

        self.tileEntities = dict()

        self.meshVaos = [0] * (CHUNK_HEIGHT // MESH_HEIGHT)
        self.meshVbos = [0] * (CHUNK_HEIGHT // MESH_HEIGHT)
        self.meshVertexCounts = [0] * (CHUNK_HEIGHT // MESH_HEIGHT)
        self.meshDirtyFlags = [True] * (CHUNK_HEIGHT // MESH_HEIGHT)
    
    def tick(self, app, world: 'World'):
        for (pos, entity) in self.tileEntities.items():
            entity.tick(app)

        total = 0
        
        while len(self.scheduledTicks) != 0 and self.scheduledTicks[0][0] <= self.tickIdx:
            (_, pos) = self.scheduledTicks.popleft()
            total += 1
            self.doScheduledTick(app, world, pos)
        
        if total != 0:
            print(f'Did {total} scheduled ticks')
        
        self.tickIdx += 1
    
    def requestScheduledTick(self, blockPos: BlockPos, delay: int):
        self.scheduledTicks.append((self.tickIdx + delay, blockPos))
    
    def doBlockUpdate(self, instData, world: 'World', blockPos: BlockPos, priority: int):
        blockId = self.blocks[blockPos.x, blockPos.y, blockPos.z]
        if priority == -1:
            if blockId in ('water', 'flowing_water'):
                for (_, p) in self.scheduledTicks:
                    if p == blockPos:
                        return
                
                self.requestScheduledTick(blockPos, 5)
            elif blockId in ('lava', 'flowing_lava'):
                for (_, p) in self.scheduledTicks:
                    if p == blockPos:
                        return
                
                globalPos = self._globalBlockPos(blockPos)
                for faceIdx in range(0, 12, 2):
                    adjPos = adjacentBlockPos(globalPos, faceIdx)
                    adjBlockId = world.getBlock(adjPos)
                    if adjBlockId in ('water', 'flowing_water'):
                        blockState = self.blockStates[blockPos.x, blockPos.y, blockPos.z]
                        if blockState['level'] == '0':
                            self.setBlock(world, instData, blockPos, 'obsidian')
                        else:
                            self.setBlock(world, instData, blockPos, 'cobblestone')
                
                self.requestScheduledTick(blockPos, 30)
            elif blockId == 'redstone_wire':
                updateRedstoneWire(self._globalBlockPos(blockPos), world, instData)
                world.updateRedstone([self._globalBlockPos(blockPos)], instData)
            elif blockId == 'nether_portal':
                blockState = self.blockStates[blockPos.x, blockPos.y, blockPos.z]

                validBlocks = 0

                for da, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if blockState['axis'] == 'x':
                        dx = da
                        dz = 0
                    else:
                        dx = 0
                        dz = da

                    adjPos = self._globalBlockPos(BlockPos(blockPos.x + dx, blockPos.y + dy, blockPos.z + dz))

                    if world.getBlock(adjPos) in ('obisidian', 'nether_portal'):
                        validBlocks += 1
                
                if validBlocks != 4:
                    self.setBlock(world, instData, blockPos, 'air', {})
        elif priority == 3:
            if blockId in ('redstone_torch', 'redstone_wall_torch'):
                for (_, p) in self.scheduledTicks:
                    if p == blockPos:
                        return
 
                self.requestScheduledTick(blockPos, 2)
        
    def doScheduledTick(self, app, world: 'World', blockPos: BlockPos):
        instData = (app.textures, app.cube, app.textureIndices)

        gp = self._globalBlockPos(blockPos)
        sideHigher = False

        blockId = copy.deepcopy(self.blocks[blockPos.x, blockPos.y, blockPos.z])
        blockState = copy.deepcopy(self.blockStates[blockPos.x, blockPos.y, blockPos.z])

        if blockId in ('water', 'flowing_water', 'lava', 'flowing_lava'):
            if blockId in ('water', 'flowing_water'):
                levelChange = 1
                flowingName = 'flowing_water'
                liquid = ('water', 'flowing_water')
                liquidOrAir = ('water', 'flowing_water', 'air')
            else:
                levelChange = 2
                flowingName = 'flowing_lava'
                liquid = ('lava', 'flowing_lava')
                liquidOrAir = ('lava', 'flowing_lava', 'air')
            
            def setIfChanged(pos: BlockPos, blockId: BlockId, blockState, isLocal):
                if isLocal:
                    oldId = self.blocks[pos.x, pos.y, pos.z]
                    oldState = self.blockStates[pos.x, pos.y, pos.z]

                    if blockId != oldId or oldState != blockState:
                        print(f'Changed {oldId}[{oldState}] to {blockId}[{blockState}] (local) at {self._globalBlockPos(pos)}')
                        self.setBlock(world, instData, pos, blockId, blockState)
                else:
                    oldId = world.getBlock(pos)
                    oldState = world.getBlockState(pos)

                    if blockId != oldId or oldState != blockState:
                        print(f'Changed {oldId}[{oldState}] to {blockId}[{blockState}] (global) at {pos}')
                        world.setBlock(instData, pos, blockId, blockState)
            
            
            if blockPos.y == 0:
                canFlowDown = False
            else:
                belowPos = BlockPos(blockPos.x, blockPos.y - 1, blockPos.z)
                
                belowId = self.blocks[belowPos.x, belowPos.y, belowPos.z]
                belowState = self.blockStates[belowPos.x, belowPos.y, belowPos.z]

                if belowId == 'air':
                    canFlowDown = True
                elif belowId in liquid:
                    canFlowDown = belowState['level'] != '0'
                else:
                    canFlowDown = False

            level = int(blockState['level'])
                    
            if canFlowDown:
                setIfChanged(BlockPos(blockPos.x, blockPos.y - 1, blockPos.z),
                    flowingName, { 'level': '8' }, isLocal=True)
            else:
                if (level + levelChange) // 8 == level // 8:
                    for faceIdx in range(0, 8, 2):
                        adjPos = adjacentBlockPos(gp, faceIdx)
                        blockId = world.getBlock(adjPos)
                        shouldFlow = False
                        nextLevel = (level % 8) + levelChange
                        if blockId in liquid:
                            blockState = world.getBlockState(adjPos)
                            oldLevel = int(blockState['level'])
                            if (oldLevel % 8) > nextLevel:
                                shouldFlow = True
                        elif blockId == 'air':
                            shouldFlow = True
                        else:
                            shouldFlow = False
                        
                        if shouldFlow:
                            setIfChanged(adjPos, flowingName, { 'level': str(nextLevel) }, isLocal=False)
            
            for faceIdx in range(0, 8, 2):
                adjPos = adjacentBlockPos(gp, faceIdx)
                blockId = world.getBlock(adjPos)
                if blockId in liquid:
                    blockState = world.getBlockState(adjPos)
                    oldLevel = int(blockState['level'])
                    
                    if (oldLevel % 8) < (level % 8):
                        sideHigher = True
                        break
            
            aboveHigher = self.blocks[blockPos.x, blockPos.y+1, blockPos.z] in liquid

            if level != 0:
                if level == 8 and not aboveHigher:
                    setIfChanged(blockPos, flowingName, { 'level': '1' }, isLocal=False)
                elif (level + levelChange) // 8 != level // 8:
                    setIfChanged(blockPos, 'air', {}, isLocal=False)
                elif not sideHigher:
                    setIfChanged(blockPos, flowingName, { 'level': str((level % 8) + 1) }, isLocal=False)

        elif blockId in ('redstone_torch', 'redstone_wall_torch'):
            if blockId == 'redstone_torch':
                otherPos = BlockPos(blockPos.x, blockPos.y - 1, blockPos.z)
            elif blockState['facing'] == 'west':
                otherPos = BlockPos(blockPos.x + 1, blockPos.y, blockPos.z)
            elif blockState['facing'] == 'east':
                otherPos = BlockPos(blockPos.x - 1, blockPos.y, blockPos.z)
            elif blockState['facing'] == 'north':
                otherPos = BlockPos(blockPos.x, blockPos.y, blockPos.z - 1)
            else:
                otherPos = BlockPos(blockPos.x, blockPos.y, blockPos.z + 1)
            
            otherPos = self._globalBlockPos(otherPos)

            lit = not world.isWeaklyPowered(otherPos)
            
            newLit = 'true' if lit else 'false'

            if newLit != blockState['lit']:
                blockState['lit'] = newLit 
                self.setBlock(world, instData, blockPos, blockId, blockState)
            
    def save(self, path):
        file = nbt.NBTFile()
        file.name = 'data'

        l = nbt.TAG_List(type=nbt.TAG_Compound, name='TileEntities')
        for te in self.tileEntities.values():
            l.append(te.toNbt())
        file.tags.append(l)
    
        file.write_file(path+'.nbt')

        np.savez(path,
            blocks=self.blocks,
            blockStates=self.blockStates,
            lightLevels=self.lightLevels,
            blockLightLevels=self.blockLightLevels,
            biomes=self.biomes,
        )
    
    @timed()
    def loadFromPacket(self, world: 'World', instData, packet: ChunkDataS2C):
        for (idx, section) in enumerate(packet.sections):
            if section is None:
                self.blocks[:, (idx*16):(idx*16)+1, :] = 'air'

                for (x, y, z) in np.ndindex(16, 16, 16):
                    self.blockStates[x, y+(idx*16), z] = {}
            else:
                section[0].registry = util.REGISTRY
                for (blockIdx, block) in enumerate(section[0]):
                    blockId = block.pop('name').removeprefix('minecraft:')
                    blockId = convertBlock(blockId, instData)

                    y = blockIdx // 256 + 16 * idx
                    z = 15 - ((blockIdx // 16) % 16)
                    x = blockIdx % 16

                    self.blocks[x, y, z] = blockId
                    self.blockStates[x, y, z] = block
        
        if packet.biomes is not None:
            for idx, biome in enumerate(packet.biomes):
                xIdx = idx // ((CHUNK_HEIGHT // 4) * 4)
                zIdx = (idx // (CHUNK_HEIGHT // 4)) % 4
                yIdx = idx % (CHUNK_HEIGHT // 4)

                self.biomes[xIdx, yIdx, zIdx] = util.DIMENSION_CODEC.getBiome(biome)

        self.setAllBlocks(world, instData)

        self.worldgenStage = WorldgenStage.POPULATED

    @timed()
    def loadFromAnvilChunk(self, world, instData, chunk):
        for (i, block) in enumerate(chunk.stream_chunk()):
            y = (i // (16 * 16))
            z = (i // 16) % 16
            x = i % 16

            block: str = convertBlock(block.id, instData) #type:ignore

            if y == 0:
                block = 'bedrock'

            self.setBlock(world, instData, BlockPos(x, y, z), block, doUpdateLight=False, doUpdateBuried=False)

            '''
            if block != 'air':
                thisIdx = self._coordsToIdx(BlockPos(x, y, z))
                thisInst = self.instances[thisIdx][0]

                if x > 0:
                    thatIdx = thisIdx - 16
                    #thatIdx = self._coordsToIdx(BlockPos(x - 1, y, z))

                    if self.instances[thatIdx] is not None:
                        self.instances[thatIdx][0].visibleFaces[2] = False
                        self.instances[thatIdx][0].visibleFaces[3] = False

                        thisInst.visibleFaces[0] = False
                        thisInst.visibleFaces[1] = False
                
                if z > 0:
                    thatIdx = thisIdx - 1
                    #thatIdx = self._coordsToIdx(BlockPos(x, y, z - 1))

                    if self.instances[thatIdx] is not None:
                        self.instances[thatIdx][0].visibleFaces[6] = False
                        self.instances[thatIdx][0].visibleFaces[7] = False

                        thisInst.visibleFaces[4] = False
                        thisInst.visibleFaces[5] = False
                
                
                if y > 0:
                    #thatIdx = self._coordsToIdx(BlockPos(x, y - 1, z))
                    thatIdx = thisIdx - 256

                    if self.instances[thatIdx] is not None:
                        self.instances[thatIdx][0].visibleFaces[10] = False
                        self.instances[thatIdx][0].visibleFaces[11] = False

                        thisInst.visibleFaces[8] = False
                        thisInst.visibleFaces[9] = False
            '''
        
        self.worldgenStage = WorldgenStage.POPULATED
    
    def load(self, world, instData, path):
        # TODO: See if I can serialize strings some other way
        with np.load(path, allow_pickle=True) as npz:
            self.blocks = npz['blocks']
            self.blockStates = npz['blockStates']
            self.lightLevels = npz['lightLevels']
            self.blockLightLevels = npz['blockLightLevels']
            self.biomes = npz['biomes']
            self.setAllBlocks(world, instData)
        
        file = nbt.NBTFile(path+'.nbt')
        for te in file['TileEntities']:
            te = blockEntityFromNbt(te)
            self.tileEntities[te.pos] = te

        self.worldgenStage = WorldgenStage.POPULATED

    
    @timed()
    def doFirstLighting(self, world: 'World'):
        needUpdates = set()

        anyOccupied = False

        if world.hasSkyLight:
            for yIdx in range(CHUNK_HEIGHT - 1, -1, -1):
                allAreDark = True

                if yIdx == CHUNK_HEIGHT - 1:
                    self.lightLevels[:, yIdx, :] = 15
                else:
                    self.lightLevels[:, yIdx, :] = self.lightLevels[:, yIdx + 1, :]

                for xIdx in range(16):
                    for zIdx in range(16):
                        thisPos = BlockPos(xIdx, yIdx, zIdx)
                        if self.coordsOccupied(thisPos, isOpaque):
                            anyOccupied = True
                            self.lightLevels[xIdx, yIdx, zIdx] = 0
                        elif self.lightLevels[xIdx, yIdx, zIdx] != 0:
                            allAreDark = False
                            if anyOccupied:
                                if xIdx == 0 or xIdx == 15 or zIdx == 0 or zIdx == 15:
                                    needUpdates.add(self._globalBlockPos(thisPos))
                                else:
                                    for faceIdx in range(0, 8, 2):
                                        adjPos = adjacentBlockPos(BlockPos(xIdx, yIdx, zIdx), faceIdx)

                                        if not self.coordsOccupied(adjPos, isOpaque) and self.lightLevels[adjPos.x, adjPos.y, adjPos.z] != 15:
                                            needUpdates.add(self._globalBlockPos(thisPos))
                                            break
                        
                if allAreDark:
                    break
        
        skyCount = len(needUpdates)
            
        for xIdx in range(16):
            for yIdx in range(256):
                for zIdx in range(16):
                    blockId = self.blocks[xIdx, yIdx, zIdx]
                    if getLuminance(blockId) != 0:
                        needUpdates.add(self._globalBlockPos(BlockPos(xIdx, yIdx, zIdx)))
    
        print(f'Found {len(needUpdates)} ({skyCount} sky, {len(needUpdates) - skyCount} block) blocks that needed updates')

        world.dirtySources |= needUpdates
    
    @timed()
    def updateAllBuried(self, world: 'World'):
        def okFilter(p):
            return p[1] is not None

        for i, thisInst in filter(okFilter, enumerate(self.instances)):
            thisInst[1] = True
            thisInst[0].visibleFaces = [True] * 12
        
        for i, thisInst in filter(okFilter, enumerate(self.instances)):
            x, y, z = self._coordsFromIdx(i)
            thisInst = thisInst[0]

            if x > 0:
                thatIdx = i - 16
                #thatIdx = self._coordsToIdx(BlockPos(x - 1, y, z))

                if self.instances[thatIdx] is not None:
                    self.instances[thatIdx][0].visibleFaces[2] = False
                    self.instances[thatIdx][0].visibleFaces[3] = False

                    thisInst.visibleFaces[0] = False
                    thisInst.visibleFaces[1] = False
            
            if z > 0:
                thatIdx = i - 1
                #thatIdx = self._coordsToIdx(BlockPos(x, y, z - 1))

                if self.instances[thatIdx] is not None:
                    self.instances[thatIdx][0].visibleFaces[6] = False
                    self.instances[thatIdx][0].visibleFaces[7] = False

                    thisInst.visibleFaces[4] = False
                    thisInst.visibleFaces[5] = False
            
            
            if y > 0:
                #thatIdx = self._coordsToIdx(BlockPos(x, y - 1, z))
                thatIdx = i - 256

                if self.instances[thatIdx] is not None:
                    self.instances[thatIdx][0].visibleFaces[10] = False
                    self.instances[thatIdx][0].visibleFaces[11] = False

                    thisInst.visibleFaces[8] = False
                    thisInst.visibleFaces[9] = False
        
        for i, thisInst in filter(okFilter, enumerate(self.instances)):
            thisInst[1] = any(thisInst[0].visibleFaces)
        
        for y in range(CHUNK_HEIGHT):
            for foo in range(16):
                self.updateBuriedStateAt(world, BlockPos(0, y, foo))
                self.updateBuriedStateAt(world, BlockPos(15, y, foo))

                self.updateBuriedStateAt(world, BlockPos(foo, y, 0))
                self.updateBuriedStateAt(world, BlockPos(foo, y, 15))
    
    def lightAndOptimize(self, world: 'World'):
        print(f"Lighting and optimizing chunk at {self.pos}")
        
        self.updateAllBuried(world)
        
        self.doFirstLighting(world)
            
        self.worldgenStage = WorldgenStage.OPTIMIZED
    
    def createNextMesh(self, world: 'World', instData) -> bool:
        """Returns True if any meshes were actually changed"""

        for i in range(len(self.meshVaos)):
            if self.meshDirtyFlags[i]:
                self.createOneMesh(i, world, instData)
                return True
            
        return False

    def createMesh(self, world: 'World', instData):
        for i in range(len(self.meshVaos)):
            self.createOneMesh(i, world, instData)
        
        self.worldgenStage = WorldgenStage.COMPLETE
    
    def createOneMesh(self, i: int, world: 'World', instData):
        if self.meshDirtyFlags[i]:
            self.createOneMeshUncached(i, world, instData)

        if not any(self.meshDirtyFlags):
            self.worldgenStage = WorldgenStage.COMPLETE
    
    def createOneMeshUncached(self, meshIdx: int, world: 'World', instData):
        self.meshDirtyFlags[meshIdx] = False

        if not config.USE_OPENGL_BACKEND:
            return

        usedVertices = []

        for i in range(meshIdx * 16 * 16 * MESH_HEIGHT, (meshIdx + 1) * 16 * 16 * MESH_HEIGHT):
            inst = self.instances[i]

            if inst is None: continue

            inst, unburied = inst
            if not unburied: continue

            by = (i // (16 * 16))
            bx = (i // 16) % 16
            bz = i % 16

            blockId = self.blocks[bx, by, bz]
            blockState = self.blockStates[bx, by, bz]

            # left, right, near, far, bottom, top
            for faceIdx in range(0, 12, 2):
                if not inst.visibleFaces[faceIdx]: continue

                faceVertices = np.zeros((6, 12))

                for l in range(6):
                    vert = CUBE_MESH_VERTICES[((faceIdx // 2) * 6 + l) * 5:][:5]

                    tint = (1.0, 1.0, 1.0)

                    if blockId in ('torch', 'redstone_torch'):
                        if faceIdx < 8:
                            uSize, vSize = 2/16, 10/16
                            uOffset, vOffset = 7/16, 0/16
                        else:
                            uSize, vSize = 2/16, 2/16
                            uOffset, vOffset = 7/16, 8/16
                        vert = (vert * [2/16, 10/16, 2/16, uSize, vSize]) + [0.0, -3/16, 0.0, uOffset, vOffset]
                    elif blockId in ('wall_torch', 'redstone_wall_torch'):
                        if faceIdx < 8:
                            uSize, vSize = 2/16, 10/16
                            uOffset, vOffset = 7/16, 0/16
                        else:
                            uSize, vSize = 2/16, 2/16
                            uOffset, vOffset = 7/16, 8/16

                        facing = blockState['facing']

                        vert = (vert * [2/16, 10/16, 2/16, uSize, vSize]) + [0.0, 0.0, 0.0, uOffset, vOffset]

                        if facing == 'west':
                            newX = math.cos(math.radians(20.0)) * vert[0] - math.sin(math.radians(20.0)) * vert[1] + 0.4
                            newY = math.cos(math.radians(20.0)) * vert[1] + math.sin(math.radians(20.0)) * vert[0]
                            newZ = vert[2]
                        elif facing == 'east':
                            newX = math.cos(math.radians(-20.0)) * vert[0] - math.sin(math.radians(-20.0)) * vert[1] - 0.4
                            newY = math.cos(math.radians(-20.0)) * vert[1] + math.sin(math.radians(-20.0)) * vert[0]
                            newZ = vert[2]
                        elif facing == 'south':
                            newX = vert[0]
                            newZ = math.cos(math.radians(20.0)) * vert[2] - math.sin(math.radians(20.0)) * vert[1] + 0.4
                            newY = math.cos(math.radians(20.0)) * vert[1] + math.sin(math.radians(20.0)) * vert[2]
                        elif facing == 'north':
                            newX = vert[0]
                            newZ = math.cos(math.radians(-20.0)) * vert[2] - math.sin(math.radians(-20.0)) * vert[1] - 0.4
                            newY = math.cos(math.radians(-20.0)) * vert[1] + math.sin(math.radians(-20.0)) * vert[2]
                        else:
                            raise Exception(facing)

                        vert[0] = newX
                        vert[1] = newY
                        vert[2] = newZ
                    elif blockId in ('water', 'flowing_water', 'lava', 'flowing_lava'):
                        level = int(blockState['level']) if 'level' in blockState else 0

                        if level > 8:
                            print(f'level: {level}')

                        if level < 8:
                            blockHeight = (8 - level) / 9
                        else:
                            blockHeight = 1.0
                        
                        vert = (vert * [1.0, blockHeight, 1.0, 1.0, 1.0] - [0.0, (1 - blockHeight) / 2, 0.0, 0.0, 0.0])
                    elif blockId == 'redstone_wire':
                        power = int(blockState['power'])
                        tintStr = (
                            '#4B0000',
                            '#6F0000',
                            '#790000',
                            '#820000',
                            '#8C0000',
                            '#970000',
                            '#A10000',
                            '#AB0000',
                            '#B50000',
                            '#BF0000',
                            '#CA0000',
                            '#D30000',
                            '#DD0000',
                            '#E70600',
                            '#F11B00',
                            '#FC3100',
                        )[power]

                        r = int(tintStr[1:3], 16) / 0xFF
                        g = int(tintStr[3:5], 16) / 0xFF
                        b = int(tintStr[5:7], 16) / 0xFF

                        tint = (r, g, b)

                        vert = (vert * [1.0, 1/16, 1.0, 1.0, 1.0] - [0.0, (1 - 1/16) / 2, 0.0, 0.0, 0.0])

                    faceVertices[l, :5] = vert
                    faceVertices[l, 9:12] = tint

                adjBlockPos = adjacentBlockPos(BlockPos(bx, by, bz), faceIdx)
                (ckPos, ckLocal) = toChunkLocal(self._globalBlockPos(adjBlockPos))
                if self.pos == ckPos:
                    lightLevel = self.lightLevels[ckLocal.x, ckLocal.y, ckLocal.z]
                    blockLightLevel = self.blockLightLevels[ckLocal.x, ckLocal.y, ckLocal.z]
                else:
                    if ckPos in world.chunks:
                        lightLevel = world.chunks[ckPos].lightLevels[ckLocal.x, ckLocal.y, ckLocal.z]
                        blockLightLevel = world.chunks[ckPos].blockLightLevels[ckLocal.x, ckLocal.y, ckLocal.z]
                    else:
                        lightLevel = 15
                        blockLightLevel = 0

                atlasIdx = instData[2][blockId][faceIdx // 2]
                if blockId == 'redstone_wire':
                    east = blockState['east'] != 'none'
                    north = blockState['north'] != 'none'
                    south = blockState['south'] != 'none'
                    west = blockState['west'] != 'none'

                    atlasIdx += east * 8 + north * 4 + south * 2 + west
                elif blockId in ('redstone_torch', 'redstone_wall_torch'):
                    if blockState['lit'] == 'false':
                        atlasIdx += 1

                offsetArr = [
                    bx + self.pos.x * 16, by + self.pos.y * CHUNK_HEIGHT, bz + self.pos.z * 16,
                    0.0, 0.0,
                    lightLevel, blockLightLevel,
                    # Block index
                    bx * 16 * MESH_HEIGHT + (by % MESH_HEIGHT) * 16 + bz,
                    # Texture atlas index
                    atlasIdx,
                    # Tint
                    0.0,
                    0.0,
                    0.0
                ]

                faceVertices += offsetArr

                faceVertices.shape = (faceVertices.size, )

                usedVertices += list(faceVertices)
        
        usedVertices = np.array(usedVertices, dtype='float32')

        self.setMesh(meshIdx, usedVertices)

        if not any(self.meshDirtyFlags):
            self.worldgenStage = WorldgenStage.COMPLETE

        #vao: int = glGenVertexArrays(1) #type:ignore
        #vbo: int = glGenBuffers(1)

    def setMesh(self, meshIdx: int, usedVertices: ndarray):
        # FIXME: MEMORY LEAK
        '''
        if self.meshVaos[meshIdx] != 0:
            glDeleteVertexArrays(1, np.array([self.meshVaos[meshIdx]])) #type:ignore
            glDeleteBuffers(1, np.array([self.meshVbos[meshIdx]])) #type:ignore

            self.meshVaos[meshIdx] = 0
        '''

        vao = self.meshVaos[meshIdx]
        vbo = self.meshVbos[meshIdx]

        if vao == 0:
            vao: int = glGenVertexArrays(1) #type:ignore
            vbo: int = glGenBuffers(1) #type:ignore

        glBindVertexArray(vao)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, usedVertices.nbytes, usedVertices, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 12 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 12 * 4, ctypes.c_void_p(5 * 4))
        glEnableVertexAttribArray(2)

        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 12 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(3)

        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 12 * 4, ctypes.c_void_p(7 * 4))
        glEnableVertexAttribArray(4)

        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, 12 * 4, ctypes.c_void_p(8 * 4))
        glEnableVertexAttribArray(5)

        glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, 12 * 4, ctypes.c_void_p(9 * 4))
        glEnableVertexAttribArray(6)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(0)

        self.meshVaos[meshIdx] = vao
        self.meshVbos[meshIdx] = vbo
        self.meshVertexCounts[meshIdx] = len(usedVertices) // 7


    def _coordsToIdx(self, pos: BlockPos) -> int:
        (xw, yw, zw) = self.blocks.shape
        (x, y, z) = pos
        return y * xw * zw + x * zw + z

    def _coordsFromIdx(self, idx: int) -> BlockPos:
        (x, y, z) = self.blocks.shape
        yIdx = (idx // (z * x))
        xIdx = (idx // z) % x
        zIdx = (idx % z)
        return BlockPos(xIdx, yIdx, zIdx)
    
    def _globalBlockPos(self, blockPos: BlockPos) -> BlockPos:
        (x, y, z) = blockPos
        x += 16 * self.pos[0]
        y += CHUNK_HEIGHT * self.pos[1]
        z += 16 * self.pos[2]
        return BlockPos(x, y, z)

    def updateBuriedStateAt(self, world: 'World', blockPos: BlockPos):
        idx = self._coordsToIdx(blockPos)
        if self.instances[idx] is None:
            return

        globalPos = self._globalBlockPos(blockPos)

        inst = self.instances[idx][0]

        uncovered = False
        for faceIdx in range(0, 12, 2):
            adjPos = adjacentBlockPos(blockPos, faceIdx)

            isVisible = not world.coordsOccupied(self._globalBlockPos(adjPos), isOpaque)

            '''
            isVisible = (
                adjPos.x < 0 or 16 <= adjPos.x or
                adjPos.y < 0 or CHUNK_HEIGHT <= adjPos.y or
                adjPos.z < 0 or 16 <= adjPos.z or
                not self.coordsOccupied(adjPos))
            '''

            inst.visibleFaces[faceIdx] = isVisible
            inst.visibleFaces[faceIdx + 1] = isVisible

            if isVisible:
                uncovered = True
            
        self.instances[idx][1] = uncovered

    def coordsOccupied(self, pos: BlockPos, predicate=None) -> bool:
        (x, y, z) = pos
        block = self.blocks[x, y, z]
        return block != 'air' and (predicate is None or predicate(block))
    
    def setAllBlocks(self, world, instData):
        self.meshDirtyFlags = [True] * len(self.meshDirtyFlags)

        (textures, cube, _) = instData

        self.tileEntities = {}

        if config.USE_OPENGL_BACKEND:
            for ((x, y, z), block) in np.ndenumerate(self.blocks):
                blockPos = BlockPos(x, y, z)
                idx = self._coordsToIdx(blockPos)
                if block == 'air':
                    self.instances[idx] = None
                else:
                    texture = textures[block]
                    #[modelX, modelY, modelZ] = blockToWorld(self._globalBlockPos(blockPos))
                    #self.instances[idx] = [render.Instance(cube, np.array([[modelX], [modelY], [modelZ]]), texture), True]
                    self.instances[idx] = [render.Instance(None, None, texture), True] #type:ignore
                    self.instances[idx][0].visibleFaces = [True] * 12
                    if block == 'furnace':
                        self.tileEntities[blockPos] = Furnace(blockPos)
        else:
            for ((x, y, z), block) in np.ndenumerate(self.blocks):
                blockPos = BlockPos(x, y, z)
                idx = self._coordsToIdx(blockPos)
                if block == 'air':
                    self.instances[idx] = None
                else:
                    texture = textures[block]
                    [modelX, modelY, modelZ] = blockToWorld(self._globalBlockPos(blockPos))
                    self.instances[idx] = [render.Instance(cube, np.array([[modelX], [modelY], [modelZ]]), texture), True]
                    #self.instances[idx] = [render.Instance(None, None, texture), True]
                    #self.instances[idx][0].visibleFaces = [True] * 12
                    if block == 'furnace':
                        self.tileEntities[blockPos] = Furnace(blockPos)

    def setBlock(self, world, instData, blockPos: BlockPos, blockId: BlockId, blockState: Optional[BlockState] = None, doUpdateLight=True, doUpdateBuried=True, doUpdateMesh=False, doBlockUpdates=True):
        (x, y, z) = blockPos

        if doBlockUpdates:
            prevId = self.blocks[x, y, z]
            prevState = copy.deepcopy(self.blockStates[x, y, z])

        meshIdx = blockPos.y // MESH_HEIGHT
        self.meshDirtyFlags[meshIdx] = True

        (textures, cube, _) = instData
        self.blocks[x, y, z] = blockId
        self.blockStates[x, y, z] = {} if blockState is None else blockState
        idx = self._coordsToIdx(blockPos)
        if blockId == 'air':
            self.instances[idx] = None
        else:
            texture = textures[blockId]

            [modelX, modelY, modelZ] = blockToWorld(self._globalBlockPos(blockPos))

            self.instances[idx] = [render.Instance(cube, np.array([[modelX], [modelY], [modelZ]]), texture), True]
            if doUpdateBuried:
                self.updateBuriedStateAt(world, blockPos)

        if blockPos in self.tileEntities:
            self.tileEntities.pop(blockPos)
        
        if blockId == 'furnace':
            self.tileEntities[blockPos] = Furnace(blockPos)

        if doBlockUpdates:
            if blockId == 'redstone_wire':
                if prevId == blockId: #type:ignore
                    priority = 3
                else:
                    priority = -1

                self.doBlockUpdate(instData, world, blockPos, priority)

                toUpdate = set()

                for faceIdx1 in range(0, 12, 2):
                    adjPos1 = adjacentBlockPos(blockPos, faceIdx1)
                    toUpdate.add(adjPos1)
                    for faceIdx2 in range(0, 12, 2):
                        adjPos2 = adjacentBlockPos(adjPos1, faceIdx2)
                        if adjPos2 != blockPos:
                            toUpdate.add(adjPos2)
                
                for adjPos2 in toUpdate:
                    globalPos = self._globalBlockPos(adjPos2)

                    (chunk, localPos) = world.getChunk(globalPos)
                    chunk.doBlockUpdate(instData, world, localPos, priority)
            else:
                self.doBlockUpdate(instData, world, blockPos, -1)
                
                globalPos = self._globalBlockPos(blockPos)
                for faceIdx in range(0, 12, 2):
                    adjPos = adjacentBlockPos(globalPos, faceIdx)
                    
                    (chunk, localPos) = world.getChunk(adjPos)
                    chunk.doBlockUpdate(instData, world, localPos, -1)
        
        '''
        for faceIdx in range(0, 12, 2):
            adjPos = adjacentBlockPos(blockPos, faceIdx)
            if adjPos.x < 0 or 16 <= adjPos.x: continue
            if adjPos.y < 0 or CHUNK_HEIGHT <= adjPos.y: continue
            if adjPos.z < 0 or 16 <= adjPos.z: continue

            adjIdx = self._coordsToIdx(adjPos)
            otherFaceIdx = [2, 0, 6, 4, 10, 8][faceIdx // 2]
            if self.instances[adjIdx] is not None:
                if id == 'air':
                    self.instances[adjIdx][0].visibleFaces[otherFaceIdx] = True
                    self.instances[adjIdx][0].visibleFaces[otherFaceIdx + 1] = True
                else:
                    self.instances[adjIdx][0].visibleFaces[otherFaceIdx] = False
                    self.instances[adjIdx][0].visibleFaces[otherFaceIdx + 1] = False
        '''

        '''       
        uncovered = False
        for faceIdx in range(0, 12, 2):
            adjPos = adjacentBlockPos(globalPos, faceIdx)
            if world.coordsOccupied(adjPos):
                inst.visibleFaces[faceIdx] = False
                inst.visibleFaces[faceIdx + 1] = False
            else:
                inst.visibleFaces[faceIdx] = True
                inst.visibleFaces[faceIdx + 1] = True
                uncovered = True
            
        self.instances[idx][1] = uncovered
        '''

        if doUpdateBuried:
            globalPos = self._globalBlockPos(blockPos)
            updateBuriedStateNear(world, globalPos)
        
        if doUpdateLight:
            globalPos = self._globalBlockPos(blockPos)
            #world.propogateLight([globalPos], isSky=False, addOne=True)
            world.updateLight2((globalPos, ), isSky=True)
            world.updateLight2((globalPos, ), isSky=False)
            #world.updateLight(globalPos, isSky=True)
            #world.updateLight(globalPos, isSky=False)
        
        if doUpdateMesh:
            self.createMesh(world, instData)

class TerrainGen(ABC):
    @abstractmethod
    def generate(self, chunk: Chunk, world: 'World', instData, seed):
        pass

    @abstractmethod
    def populate(self, chunk: Chunk, world: 'World', instData, seed):
        pass

    @abstractmethod
    def checkCavesAround(self, centerPos: ChunkPos, seed):
        pass

class NullGen(TerrainGen):
    def generate(self, chunk: Chunk, world: 'World', instData, seed):
        pass

    def populate(self, chunk: Chunk, world: 'World', instData, seed):
        pass

    def checkCavesAround(self, centerPos: ChunkPos, seed):
        pass

class OverworldGen(TerrainGen):
    caveChecked: set[ChunkPos]
    caves: dict[ChunkPos, List[BlockPos]]

    def __init__(self):
        self.caveChecked = set()
        self.caves = {}

    @timed()
    def generate(self, chunk: Chunk, world: 'World', instData,  seed):
        chunk.biomes[:, :, :] = util.DIMENSION_CODEC.getBiome('minecraft:forest')

        cavePositions = self.caves.values()

        positions = []

        for cave in cavePositions:
            minIdx = binarySearchMin(cave, chunk.pos.x * 16 - 1)
            maxIdx = binarySearchMax(cave, (chunk.pos.x + 1) * 16)

            if minIdx is not None or maxIdx is not None:
                if minIdx is None:
                    minIdx = 0
                if maxIdx is None:
                    maxIdx = len(cave) - 1

                for posIdx in range(minIdx, maxIdx + 1):
                    pos = cave[posIdx]
                    for xOff in range(-1, 2):
                        for yOff in range(-1, 2):
                            for zOff in range(-1, 2):
                                pos2 = BlockPos(pos.x + xOff, pos.y + yOff, pos.z + zOff)
                                (ckPos, ckLocal) = toChunkLocal(pos2)
                                if ckPos == chunk.pos:
                                    positions.append(ckLocal)
                
        print(f"{len(positions)}-many positions")

        for xIdx in range(0, 16):
            for zIdx in range(0, 16):
                globalPos = chunk._globalBlockPos(BlockPos(xIdx, 0, zIdx))

                noise = perlin.getPerlinFractal(globalPos.x, globalPos.z, 1.0 / 256.0, 4, seed)

                factor = 20
                center = 72

                topY = int(noise * factor + center)

                for yIdx in range(0, topY):
                    if BlockPos(xIdx, yIdx, zIdx) in positions:
                        blockId = 'air'
                    elif yIdx == 0:
                        blockId = 'bedrock'
                    elif yIdx == topY - 1:
                        blockId = 'grass'
                    elif topY - yIdx < 3:
                        blockId = 'dirt'
                    else:
                        blockId = 'stone'
                    chunk.blocks[xIdx, yIdx, zIdx] = blockId
                    chunk.blockStates[xIdx, yIdx, zIdx] = {}
            
        chunk.setAllBlocks(world, instData)
        #chunk.setBlock(world, instData, BlockPos(xIdx, yIdx, zIdx), blockId, doUpdateLight=False, doUpdateBuried=False, doBlockUpdates=False)
        
        chunk.worldgenStage = WorldgenStage.GENERATED
    
    def populate(self, chunk: Chunk, world: 'World', instData, seed):
        random.seed(hash((chunk.pos, seed)))

        self.disperseOre(chunk, world, instData, seed, 'coal_ore', 20, CHUNK_HEIGHT // 2)
        self.disperseOre(chunk, world, instData, seed, 'iron_ore', 20, CHUNK_HEIGHT // 4)
        self.disperseOre(chunk, world, instData, seed, 'diamond_ore', 1, CHUNK_HEIGHT // 16)

        treePos = []

        for treeIdx in range(4):
            treeX = random.randint(0, 15)
            treeZ = random.randint(0, 15)

            for prevPos in treePos:
                dist = abs(prevPos[0] - treeX) + abs(prevPos[1] - treeZ)
                if dist <= 4:
                    continue
            
            treePos.append((treeX, treeZ))

            baseY = CHUNK_HEIGHT - 1
            for yIdx in range(CHUNK_HEIGHT - 1, 0, -1):
                block = chunk.blocks[treeX, yIdx, treeZ]
                if block == 'grass':
                    baseY = yIdx + 1
                elif block == 'stone':
                    break
            
            globalPos = chunk._globalBlockPos(BlockPos(treeX, baseY, treeZ))
        
            if globalPos.y < CHUNK_HEIGHT - 6:
                generateTree(world, instData, globalPos, doUpdates=False)
        
        chunk.worldgenStage = WorldgenStage.POPULATED
    
    def disperseOre(self, chunk: Chunk, world: 'World', instData, seed, ore: BlockId, frequency: int, maxHeight: int):
        for _ in range(frequency):
            x = random.randrange(0, 16)
            y = random.randrange(0, maxHeight)
            z = random.randrange(0, 16)

            for x in range(x, min(x + 2, 16)):
                for y in range(y, min(y + 2, CHUNK_HEIGHT)):
                    for z in range(z, min(z + 2, 16)):
                        if chunk.blocks[x, y, z] == 'stone':
                            chunk.setBlock(world, instData, BlockPos(x, y, z), ore, doUpdateLight=False, doUpdateBuried=False, doUpdateMesh=False, doBlockUpdates=False)
    
    def checkCavesAround(self, centerPos: ChunkPos, seed):
        dist = math.ceil(MAX_CAVE_DISP / 16)
        for pos in adjacentChunks(centerPos, dist):
            if pos not in self.caveChecked:
                random.seed(hash((pos, seed)))

                self.caveChecked.add(pos)

                if random.random() < 0.1:
                    caveX = pos.x * 16 + random.randrange(0, 16)
                    caveY = pos.y * CHUNK_HEIGHT + random.randrange(40, 100)
                    caveZ = pos.z * 16 + random.randrange(0, 16)

                    caveStart = BlockPos(caveX, caveY, caveZ)

                    self.caves[pos] = generateCaveCenter(caveStart, seed)

import simplex

class NetherGen(TerrainGen):
    def generate(self, chunk: Chunk, world: 'World', instData, seed):
        for xIdx in range(16):
            for yIdx in range(128):
                for zIdx in range(16):
                    gpos = chunk._globalBlockPos(BlockPos(xIdx, yIdx, zIdx))
                    s = simplex.getSimplexFractal(gpos.x, gpos.y, gpos.z, 1.0 / 256.0, 4, 0)

                    if s < 0.0:
                        if yIdx < 30:
                            chunk.blocks[xIdx, yIdx, zIdx] = 'flowing_lava'
                            chunk.blockStates[xIdx, yIdx, zIdx] = { 'level': '0' }
                        else:
                            chunk.blocks[xIdx, yIdx, zIdx] = 'air'
                    else:
                        chunk.blocks[xIdx, yIdx, zIdx] = 'netherrack'

        chunk.blocks[:, 0, :] = 'bedrock'
        chunk.blocks[:, 128, :] = 'bedrock'

        chunk.setAllBlocks(world, instData)

        chunk.worldgenStage = WorldgenStage.GENERATED
        
    def populate(self, chunk: Chunk, world: 'World', instData, seed):
        chunk.worldgenStage = WorldgenStage.POPULATED

    def checkCavesAround(self, centerPos: ChunkPos, seed):
        pass

    
def getRegionCoords(pos: ChunkPos) -> Tuple[int, int]:
    return (math.floor(pos.x / 32), math.floor(pos.z / 32))

class World:
    chunks: dict[ChunkPos, Chunk]
    seed: int

    hasSkyLight: bool

    regions: dict[Tuple[int, int], anvil.Region]
    importPath: str

    savePath: str
    
    local: bool

    serverChunks: dict[ChunkPos, ChunkDataS2C]

    dirtyLights: set[BlockPos]
    dirtySources: set[BlockPos]

    generator: TerrainGen

    def getHighestBlock(self, x: int, z: int) -> int:
        for y in range(CHUNK_HEIGHT - 1, -1, -1):
            if self.getBlock(BlockPos(x, y, z)) != 'air':
                return y
        return 0
    
    def __init__(self, savePath: str, generator: TerrainGen, seed=None, importPath=''):
        self.chunks = {}
        self.importPath = importPath
        self.regions = {}

        self.savePath = savePath

        self.generator = generator

        self.serverChunks = {}

        self.dirtyLights = set()

        self.dirtySources = set()

        self.local = True

        if self.importPath != '' and not self.importPath.endswith('/'):
            self.importPath += '/'

        self.emu = None

        os.makedirs(self.savePath, exist_ok=True)

        if seed is not None:
            if not isinstance(seed, int):
                seed = hash(seed)

            self.seed = seed

        try:
            with open("saves/{self.name}/meta.txt", "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    [key, value] = line.split('=')
                    if key == 'seed':
                        self.seed = int(value)
                    elif key == 'importPath':
                        self.importPath = importPath
                    else:
                        raise Exception(f"Unknown meta {key}={value}")

        except FileNotFoundError:
            self.saveMetaFile()
        
        print(f"Opened world with seed {self.seed}")

    def saveMetaFile(self):
        with open(f"{self.savePath}/meta.txt", "w") as f:
            f.write(f"seed={self.seed}\n")
            if self.importPath != '':
                f.write(f"importPath={self.importPath}\n")
    
    def setBlock(self, instData, blockPos: BlockPos, blockId: BlockId, blockState: Optional[BlockState] = None, doUpdateLight=True, doUpdateBuried=True, doUpdateMesh=False, doBlockUpdates=True):
        (chunk, ckLocal) = self.getChunk(blockPos)

        #chunk.setBlock(self, instData, ckLocal, blockId, blockState, doUpdateLight, doUpdateBuried, doUpdateMesh)
        chunk.setBlock(self, instData, ckLocal, blockId, blockState, False, doUpdateBuried, doUpdateMesh, doBlockUpdates)
        self.dirtyLights.add(blockPos)

    def getBlock(self, blockPos: BlockPos) -> str:
        (chunkPos, localPos) = toChunkLocal(blockPos)
        return self.chunks[chunkPos].blocks[localPos.x, localPos.y, localPos.z]
    
    def getBlockState(self, blockPos: BlockPos) -> dict[str, Any]:
        (chunkPos, localPos) = toChunkLocal(blockPos)
        return self.chunks[chunkPos].blockStates[localPos.x, localPos.y, localPos.z]
    
    def explodeAt(self, blockPos: BlockPos, power: int, instData):
        exploded = set()

        # https://minecraft.fandom.com/wiki/Explosion
        for d1 in range(-8, 8):
            for d2 in range(-8, 8):
                exploded |= self.explodeRay(blockPos, d1, d2, -8, power)
                exploded |= self.explodeRay(blockPos, d1, d2,  8, power)

                exploded |= self.explodeRay(blockPos, d1, -8, d2, power)
                exploded |= self.explodeRay(blockPos, d1,  8, d2, power)

                exploded |= self.explodeRay(blockPos, -8, d1, d2, power)
                exploded |= self.explodeRay(blockPos,  8, d1, d2, power)

        for exp in exploded:
            self.setBlock(instData, exp, 'air', {})
    
    def explodeRay(self, startPos: BlockPos, dx, dy, dz, power: int) -> set[BlockPos]:
        strength = power * (0.7 + random.random() * 0.6)

        mag = math.sqrt(dx**2 + dy**2 + dz**2)

        x, y, z = startPos.x, startPos.y, startPos.z

        xChange = dx * 0.3 / mag
        yChange = dy * 0.3 / mag
        zChange = dz * 0.3 / mag

        exploded = set()

        while strength > 0.0:
            if strength <= 0.0:
                break
        
            x += xChange
            y += yChange
            z += zChange

            strength -= 0.3 * 0.75

            curPos = nearestBlockPos(x, y, z)

            blockId = self.getBlock(curPos)
            if blockId != 'air':
                resist = getBlastResistance(blockId)

                strength -= (resist + 0.3) * 0.3

                if strength > 0.0:
                    exploded.add(curPos)
        
        return exploded
        
    def hasBlockBeneath(self, entity):
        [xPos, yPos, zPos] = entity.pos
        yPos -= 0.1

        for x in [xPos - entity.radius * 0.99, xPos + entity.radius * 0.99]:
            for z in [zPos - entity.radius * 0.99, zPos + entity.radius * 0.99]:
                feetPos = nearestBlockPos(x, yPos, z)
                if self.coordsOccupied(feetPos, isSolid):
                    return True
        
        return False
    
    def addChunkDetails(self, instData, maxTime=0.030):
        startTime = time.perf_counter()

        keepGoing = True

        for chunkPos in self.chunks:
            chunk: Chunk = self.chunks[chunkPos]
            (adj, gen, pop, opt, com) = self.countLoadedAdjacentChunks(chunkPos, 1)
            if chunk.worldgenStage == WorldgenStage.GENERATED and gen == 8:
                self.generator.populate(chunk, self, instData, self.seed)
                if time.perf_counter() - startTime > maxTime:
                    keepGoing = False
                
            if keepGoing and chunk.worldgenStage == WorldgenStage.POPULATED and pop == 8:
                chunk.lightAndOptimize(self)
                if time.perf_counter() - startTime > maxTime:
                    keepGoing = False

            if keepGoing and chunk.worldgenStage == WorldgenStage.OPTIMIZED and opt == 8:
                while keepGoing and chunk.createNextMesh(self, instData):
                    if time.perf_counter() - startTime > maxTime:
                        keepGoing = False
            
            if keepGoing and chunk.worldgenStage == WorldgenStage.COMPLETE:
                while keepGoing and chunk.createNextMesh(self, instData):
                    if time.perf_counter() - startTime > maxTime:
                        keepGoing = False

            chunk.isVisible = chunk.worldgenStage == WorldgenStage.COMPLETE
            chunk.isTicking = chunk.isVisible and adj == 8

            if not keepGoing:
                break
    
    def flushLightChanges(self):
        if self.dirtySources != set():
            try:
                self.propogateLight(self.dirtySources, False, False)
                if self.hasSkyLight:
                    self.propogateLight(self.dirtySources, True, False)
            except Exception as e:
                print(f'Ignoring exception in flushLightChanges {e}')
            
            self.dirtySources = set()

        if self.dirtyLights != set():
            try:
                self.updateLight2(self.dirtyLights, False)
                if self.hasSkyLight:
                    self.updateLight2(self.dirtyLights, True)
            except Exception as e:
                print(f'Ignoring exception in flushLightChanges {e}')
                
            self.dirtyLights = set()
        
    def tickChunks(self, app):
        for chunk in self.chunks.values():
            if chunk.isTicking:
                chunk.tick(app, self)
        
        self.flushLightChanges()
        
    def loadUnloadChunks(self, centerPos, instData):
        (chunkPos, _) = toChunkLocal(nearestBlockPos(centerPos[0], centerPos[1], centerPos[2]))
        (x, _, z) = chunkPos

        chunkLoadDistance = math.ceil(config.CHUNK_LOAD_DISTANCE / 16)

        # Unload chunks
        shouldUnload = []
        for unloadChunkPos in self.chunks:
            (ux, _, uz) = unloadChunkPos
            dist = max(abs(ux - x), abs(uz - z))
            if dist > chunkLoadDistance + 1:
                # Unload chunk
                shouldUnload.append(unloadChunkPos)

        for unloadChunkPos in shouldUnload:
            self.unloadChunk(unloadChunkPos)

        loadedChunks = 0

        #queuedForLoad = []

        for loadChunkPos in itertools.chain(adjacentChunks(chunkPos, chunkLoadDistance), (chunkPos, )):
            if loadChunkPos not in self.chunks:
                (ux, _, uz) = loadChunkPos
                dist = max(abs(ux - x), abs(uz - z))

                urgent = dist <= 2

                if (urgent or (loadedChunks < 1)) and self.canLoadChunk(loadChunkPos):
                    #queuedForLoad.append((app.world, (app.textures, app.cube), loadChunkPos))
                    loadedChunks += 1
                    self.loadChunk(instData, loadChunkPos)

    def countLoadedAdjacentChunks(self, chunkPos: ChunkPos, dist: int) -> Tuple[int, int, int, int, int]:
        totalCount = 0
        genCount = 0
        popCount = 0
        optCount = 0
        comCount = 0
        for pos in adjacentChunks(chunkPos, dist):
            if pos in self.chunks:
                totalCount += 1
                chunk = self.chunks[pos]
                if chunk.worldgenStage >= WorldgenStage.GENERATED: genCount += 1
                if chunk.worldgenStage >= WorldgenStage.POPULATED: popCount += 1
                if chunk.worldgenStage >= WorldgenStage.OPTIMIZED: optCount += 1
                if chunk.worldgenStage >= WorldgenStage.COMPLETE:  comCount += 1
                
        return (totalCount, genCount, popCount, optCount, comCount)

    def chunkFileName(self, pos: ChunkPos) -> str:
        return f'{self.savePath}/c_{pos.x}_{pos.y}_{pos.z}.npz'
    
    def save(self):
        print("Saving world... ", end='')
        for pos in self.chunks:
            self.saveChunk(pos)
        
        self.saveMetaFile()
        print("Done!")
    
    def saveChunk(self, pos: ChunkPos):
        self.chunks[pos].save(self.chunkFileName(pos))
    
    def createChunk(self, instData, pos: ChunkPos):
        ck = Chunk(pos)
        try:
            ck.load(self, instData, self.chunkFileName(pos))
        except FileNotFoundError:
            if self.importPath != '':
                pos2 = ChunkPos(pos.x + 2, pos.y, pos.z - 6)
                regionPos = getRegionCoords(pos2)
                if regionPos not in self.regions:
                    path = self.importPath + f'region/r.{regionPos[0]}.{regionPos[1]}.mca'
                    self.regions[regionPos] = anvil.Region.from_file(path)

                chunk = anvil.Chunk.from_region(self.regions[regionPos], pos2.x, pos2.z)
                ck.loadFromAnvilChunk(self, instData, chunk)
            else:
                self.generator.generate(ck, self, instData, self.seed)

        return ck
    
    def canLoadChunk(self, pos: ChunkPos):
        if self.local:
            return pos.y == 0
        else:
            return pos in self.serverChunks
    
    def loadChunk(self, instData, pos: ChunkPos):
        print(f"Loading chunk at {pos}")
        if self.local:
            self.generator.checkCavesAround(pos, self.seed)
            self.chunks[pos] = self.createChunk(instData, pos)
        else:
            self.chunks[pos] = Chunk(pos)
            self.chunks[pos].loadFromPacket(self, instData, self.serverChunks[pos]) #type:ignore
    
    def unloadChunk(self, pos: ChunkPos):
        print(f"Unloading chunk at {pos}")
        saveFile = self.chunkFileName(pos)
        self.chunks[pos].save(saveFile)
        self.chunks.pop(pos)
    
    def getChunk(self, pos: BlockPos) -> Tuple[Chunk, BlockPos]:
        (cx, cy, cz) = pos
        cx //= 16
        cy //= CHUNK_HEIGHT
        cz //= 16

        chunk = self.chunks[ChunkPos(cx, cy, cz)]
        [x, y, z] = pos
        x %= 16
        y %= CHUNK_HEIGHT
        z %= 16
        return (chunk, BlockPos(x, y, z))

    def coordsOccupied(self, pos: BlockPos, predicate=None) -> bool:
        if not self.coordsInBounds(pos):
            return False

        (chunk, innerPos) = self.getChunk(pos)

        return chunk.coordsOccupied(innerPos, predicate)

    def coordsInBounds(self, pos: BlockPos) -> bool:
        (chunkPos, _) = toChunkLocal(pos)
        return chunkPos in self.chunks

    def blockIsBuried(self, blockPos: BlockPos):
        for faceIdx in range(0, 12, 2):
            if not self.coordsOccupied(adjacentBlockPos(blockPos, faceIdx)):
                # Unburied
                return False
        # All spaces around are occupied, this block is buried
        return True

        '''
        if pos not in app.world.chunks:
            return False
        
        (x, y, z) = pos
        (xw, yw, zw) = app.blocks.shape
        if x < 0 or xw <= x: return False
        if y < 0 or yw <= y: return False
        if z < 0 or zw <= z: return False
        return True
        '''
    
    def getTotalLight(self, gameTime: float, blockPos: BlockPos) -> int:
        sky = self.getLightLevel(blockPos)
        block = self.getBlockLightLevel(blockPos)

        sky = roundHalfUp(sky * getSkylightFactor(gameTime))

        return max(sky, block)
    
    def getLightLevel(self, blockPos: BlockPos) -> int:
        (chunk, (x, y, z)) = self.getChunk(blockPos)
        return chunk.lightLevels[x, y, z]
    
    def getBlockLightLevel(self, blockPos: BlockPos) -> int:
        (chunk, (x, y, z)) = self.getChunk(blockPos)
        return chunk.blockLightLevels[x, y, z]

    def setLightLevel(self, blockPos: BlockPos, level: int):
        (chunk, (x, y, z)) = self.getChunk(blockPos)
        chunk.lightLevels[x, y, z] = level
        # FIXME:
        chunk.meshDirtyFlags[y // MESH_HEIGHT] = True

    def setBlockLightLevel(self, blockPos: BlockPos, level: int):
        (chunk, (x, y, z)) = self.getChunk(blockPos)
        chunk.blockLightLevels[x, y, z] = level
        # FIXME:
        chunk.meshDirtyFlags[y // MESH_HEIGHT] = True
    
    def isWeaklyPowered(self, pos: BlockPos) -> bool:
        if self.isStronglyPowered(pos):
            return True
        
        above = BlockPos(pos.x, pos.y + 1, pos.z)
        if self.getBlock(above) == 'redstone_wire' and int(self.getBlockState(above)['power']) > 0:
            return True
        else:
            for face, neededDir in ((0, 'east'), (2, 'west'), (4, 'north'), (6, 'south')):
                adjPos = adjacentBlockPos(pos, face)
                adjId = self.getBlock(adjPos)
                adjState = self.getBlockState(adjPos)
                if adjId == 'redstone_wire' and int(adjState['power']) > 0:
                    return True
            
            return False
        
    def isStronglyPowered(self, pos: BlockPos) -> bool:
        if (self.getBlock(pos) in ('redstone_torch', 'redstone_wall_torch')
            and self.getBlockState(pos)['lit'] == 'true'):

            return True
        elif (self.getBlock(BlockPos(pos.x, pos.y - 1, pos.z)) in ('redstone_torch', 'redstone_wall_torch')
            and self.getBlockState(BlockPos(pos.x, pos.y - 1, pos.z))['lit'] == 'true'):

            return True
        else:
            return False
    
    def updateRedstone(self, changes: Iterable[BlockPos], instData):
        def isWire(pos):
            return self.getBlock(pos) == 'redstone_wire'

        changes = list(filter(isWire, changes))

        wires = self.findWires(changes)

        for wire in wires:
            blockState = self.getBlockState(wire)
            blockState['power'] = '0'
            self.setBlock(instData, wire, 'redstone_wire', blockState, doUpdateLight=False, doUpdateBuried=False, doBlockUpdates=False)

        self.propogateRedstone(wires, False, instData)
    
    def findWires(self, changes: Iterable[BlockPos]):
        visited = set()

        toVisit = list(changes)

        while len(toVisit) > 0:
            currentPos = toVisit.pop()

            visited.add(currentPos)

            edges = []

            if self.getBlock(currentPos) == 'redstone_wire':
                conns = getWireConnections(currentPos, self)

                dirs = { 'west':  (-1, 0), 'east': (1, 0), 'south': (0, -1), 'north': (0, 1) }

                for dirName, val in conns.items():
                    dx, dz = dirs[dirName]
                    if val == 'up':
                        dy = 1
                    elif val == 'down':
                        dy = -1
                    elif val == 'side':
                        dy = 0
                    else:
                        continue

                    nextPos = BlockPos(currentPos.x + dx, currentPos.y + dy, currentPos.z + dz)

                    if self.getBlock(nextPos) != 'redstone_wire':
                        continue

                    edges.append(nextPos)
            else:
                for faceIdx in range(0, 12, 2):
                    nextPos = adjacentBlockPos(currentPos, faceIdx)
                    if self.getBlock(nextPos) != 'redstone_wire':
                        continue
                    edges.append(nextPos)

            for nextPos in edges:
                if nextPos not in visited:
                    toVisit.append(nextPos)
        
        return visited
 
    def propogateRedstone(self, changes: Iterable[BlockPos], isEx, instData):
        def getPowerWithSources(pos):
            blockId = self.getBlock(pos)
            blockState = self.getBlockState(pos)
            if blockId == 'redstone_wire':
                for faceIdx in range(0, 12, 2):
                    adjPos = adjacentBlockPos(pos, faceIdx)
                    if self.isStronglyPowered(adjPos):
                        return 15

                return int(blockState['power'])
            else:
                return 0
            
        def getPowerLevel(pos):
            blockId = self.getBlock(pos)
            blockState = self.getBlockState(pos)
            if blockId == 'redstone_wire':
                return int(blockState['power'])
            else:
                raise Exception(pos)
                return 0
        
        def setPowerLevel(pos, level):
            doBlockUpdates = not isEx

            blockId = self.getBlock(pos)
            blockState = self.getBlockState(pos)

            if blockId == 'redstone_wire':
                blockState['power'] = str(level)
                self.setBlock(instData, pos, blockId, blockState, doBlockUpdates=True)
            else:
                print("Ignoring power level at {level}")
        
        fringe = set()
        visited = set()

        if isEx:
            queue = [(-(getPowerLevel(pos) + 1), pos) for pos in changes]
        else:
            queue = [(-getPowerWithSources(pos), pos) for pos in changes]
        heapq.heapify(queue)

        while len(queue) > 0:
            (currentLight, currentPos) = heapq.heappop(queue)
            currentLight *= -1

            if currentPos in visited:
                continue

            visited.add(currentPos)

            posLightLevel = getPowerLevel(currentPos)

            if posLightLevel > 0 and posLightLevel > currentLight or (isEx and posLightLevel == currentLight):
                fringe.add(currentPos)
                continue

            if not isEx:
                setPowerLevel(currentPos, currentLight)

            if currentLight == 0:
                continue
    
            edges = []
        
            if self.getBlock(currentPos) == 'redstone_wire':
                conns = getWireConnections(currentPos, self)

                dirs = { 'west':  (-1, 0), 'east': (1, 0), 'south': (0, -1), 'north': (0, 1) }

                for dirName, val in conns.items():
                    dx, dz = dirs[dirName]
                    if val == 'up':
                        dy = 1
                    elif val == 'down':
                        dy = -1
                    elif val == 'side':
                        dy = 0
                    else:
                        continue

                    nextPos = BlockPos(currentPos.x + dx, currentPos.y + dy, currentPos.z + dz)
                    cost = 1

                    if self.getBlock(nextPos) != 'redstone_wire':
                        continue

                    edges.append((nextPos, cost))
            else:
                for faceIdx in range(0, 12, 2):
                    nextPos = adjacentBlockPos(currentPos, faceIdx)
                    if self.getBlock(nextPos) != 'redstone_wire':
                        continue
                    cost = 0
                    edges.append((nextPos, cost))

            for nextPos, cost in edges:
                nextLight = max(currentLight - cost, 0)

                heapq.heappush(queue, (-nextLight, nextPos))
        
        if isEx:
            for v in visited:
                if v not in fringe:
                    setPowerLevel(v, 0)
        
        return visited 

    
    def updateLight2(self, changes: Iterable[BlockPos], isSky: bool):
        fringe = self.propogateLight(changes, isSky, True)
        self.propogateLight(fringe | set(changes), isSky, False)

    def propogateLight(self, changes: Iterable[BlockPos], isSky: bool, isEx):
        def getLightWithLum(pos):
            if isSky:
                return self.getLightLevel(pos)
            else:
                blockId = self.getBlock(pos)
                return max(self.getBlockLightLevel(pos), getLuminance(blockId))
            
        def getLightLevel(pos):
            if isSky:
                return self.getLightLevel(pos)
            else:
                return self.getBlockLightLevel(pos)
        
        def setLightLevel(pos, level):
            if isSky:
                self.setLightLevel(pos, level)
            else:
                self.setBlockLightLevel(pos, level)
        
        fringe = set()
        visited = set()

        if isEx:
            queue = [(-(getLightLevel(pos) + 1), pos) for pos in changes]
        else:
            queue = [(-getLightWithLum(pos), pos) for pos in changes]
        heapq.heapify(queue)

        while len(queue) > 0:
            (currentLight, currentPos) = heapq.heappop(queue)
            currentLight *= -1

            if currentPos in visited:
                continue

            visited.add(currentPos)

            posLightLevel = getLightLevel(currentPos)

            if posLightLevel > 0 and posLightLevel > currentLight or (isEx and posLightLevel == currentLight):
                fringe.add(currentPos)
                continue

            if not isEx:
                setLightLevel(currentPos, currentLight)

            if currentLight == 0:
                continue

            for faceIdx in range(0, 12, 2):
                nextPos = adjacentBlockPos(currentPos, faceIdx)

                if nextPos.y >= CHUNK_HEIGHT or nextPos.y < 0:
                    continue
                elif self.coordsOccupied(nextPos, isOpaque):
                    cost = 15
                elif faceIdx == 8 and isSky:
                    cost = 0
                else:
                    cost = 1

                nextLight = max(currentLight - cost, 0)

                heapq.heappush(queue, (-nextLight, nextPos))
        
        if isEx:
            for v in visited:
                if v not in fringe:
                    setLightLevel(v, 0)
        
        return fringe
        
    def updateLight(self, blockPos: BlockPos, isSky: bool):
        added = self.coordsOccupied(blockPos)

        (chunk, localPos) = self.getChunk(blockPos)

        block = chunk.blocks[localPos.x, localPos.y, localPos.z]
        prevBlockLight = chunk.blockLightLevels[localPos.x, localPos.y, localPos.z]

        skyExposed = True
        for y in range(localPos.y + 1, CHUNK_HEIGHT):
            checkPos = BlockPos(localPos.x, y, localPos.z)
            if chunk.coordsOccupied(checkPos):
                skyExposed = False
                break

        if isSky:
            decreased = added
        else:
            if added:
                lum = getLuminance(block)
                decreased = False
                for faceIdx in range(0, 12, 2):
                    gPos = adjacentBlockPos(blockPos, faceIdx)

                    if self.coordsOccupied(gPos, isOpaque):
                        continue

                    if not self.coordsInBounds(gPos):
                        continue

                    if self.getBlockLightLevel(gPos) > lum:
                        decreased = True
                        break
            else:
                decreased = True
                for faceIdx in range(0, 12, 2):
                    gPos = adjacentBlockPos(blockPos, faceIdx)

                    if self.coordsOccupied(gPos, isOpaque):
                        continue

                    if not self.coordsInBounds(gPos):
                        continue

                    if self.getBlockLightLevel(gPos) > prevBlockLight:
                        decreased = False
                        break


        if decreased:
            # When a block is ADDED:
            # If the block is directly skylit:
            #   Mark all blocks visibly beneath as ex-sources
            # Mark every block adjacent to the change as an ex-source
            # Propogate "negative light" from the ex-sources
            # Mark any "overpowering" lights as actual sources
            # Reset all "negative lights" to 0
            # Propogate from the actual sources

            exSources = []

            # FIXME: If I ever add vertical chunks this needs to change
            if skyExposed and isSky:
                for y in range(localPos.y - 1, -1, -1):
                    checkPos = BlockPos(localPos.x, y, localPos.z)
                    if chunk.coordsOccupied(checkPos):
                        break

                    heapq.heappush(exSources, (-15, BlockPos(blockPos.x, y, blockPos.z)))
                
            for faceIdx in range(0, 12, 2):
                gPos = adjacentBlockPos(blockPos, faceIdx)

                if self.coordsOccupied(gPos, isOpaque):
                    continue

                if not self.coordsInBounds(gPos):
                    continue

                lightLevel = self.getLightLevel(gPos) if isSky else self.getBlockLightLevel(gPos)

                heapq.heappush(exSources, (-lightLevel, gPos))
            
            self.setBlockLightLevel(blockPos, getLuminance(block))

            exVisited = []

            queue = []
            
            while len(exSources) > 0:
                (neglight, pos) = heapq.heappop(exSources)
                neglight *= -1
                if pos in exVisited:
                    continue
                exVisited.append(pos)

                existingLight = self.getLightLevel(pos) if isSky else self.getBlockLightLevel(pos)

                if isSky:
                    self.setLightLevel(pos, 0)
                else:
                    self.setBlockLightLevel(pos, 0)

                if existingLight > neglight:
                    heapq.heappush(queue, (-existingLight, pos))
                    continue
                
                if neglight == 0:
                    continue

                nextLight = max(neglight - 1, 0)
                for faceIdx in range(0, 12, 2):
                    nextPos = adjacentBlockPos(pos, faceIdx)
                    if nextPos in exVisited:
                        continue
                    if not self.coordsInBounds(nextPos):
                        continue
                    if self.coordsOccupied(nextPos, isOpaque):
                        continue

                    heapq.heappush(exSources, (-nextLight, nextPos))
            
            if isSky:
                chunk.lightLevels[localPos.x, localPos.y, localPos.z] = 0
            else:
                chunk.blockLightLevels[localPos.x, localPos.y, localPos.z] = 0

        else:
            # When a block is REMOVED:
            # Note that this can only ever *increase* the light level of a block! So:
            # If the block is directly skylit:
            #   Propogate light downwards
            #   Add every block visibly beneath the change to the queue
            # 
            # Add every block adjacent to the change to the queue

            queue = []

            # FIXME: If I ever add vertical chunks this needs to change
            if skyExposed and isSky:
                for y in range(localPos.y, -1, -1):
                    checkPos = BlockPos(localPos.x, y, localPos.z)
                    if chunk.coordsOccupied(checkPos):
                        break

                    heapq.heappush(queue, (-15, BlockPos(blockPos.x, y, blockPos.z)))
            
            for faceIdx in range(0, 12, 2):
                gPos = adjacentBlockPos(blockPos, faceIdx)

                if not self.coordsInBounds(gPos):
                    continue

                lightLevel = self.getLightLevel(gPos) if isSky else max(getLuminance(block) - 1, 0)

                heapq.heappush(queue, (-lightLevel, gPos))


            if not isSky:
                self.setBlockLightLevel(blockPos, getLuminance(block))
        
        visited = []
        
        while len(queue) > 0:
            (light, pos) = heapq.heappop(queue)
            light *= -1
            if pos in visited:
                continue
            visited.append(pos)

            if isSky:
                self.setLightLevel(pos, light)
            else:
                self.setBlockLightLevel(pos, light)

            for faceIdx in range(0, 12, 2):
                if isSky and faceIdx == 8 and light == 15:
                    nextLight = 15
                else:
                    nextLight = max(light - 1, 0)
                nextPos = adjacentBlockPos(pos, faceIdx)
                if nextPos in visited:
                    continue
                if not self.coordsInBounds(nextPos):
                    continue
                if self.coordsOccupied(nextPos, isOpaque):
                    continue

                existingLight = self.getLightLevel(nextPos) if isSky else self.getBlockLightLevel(nextPos)

                if nextLight > existingLight:
                    heapq.heappush(queue, (-nextLight, nextPos))

    def lookedAtBlock(self, reach: float, cameraPos: Tuple[float, float, float], pitch: float, yaw: float, useFluids: bool = False) -> Optional[Tuple[BlockPos, str]]:
        lookX = cos(pitch)*sin(-yaw)
        lookY = sin(pitch)
        lookZ = cos(pitch)*cos(-yaw)

        if lookX == 0.0:
            lookX = 1e-6
        if lookY == 0.0:
            lookY = 1e-6
        if lookZ == 0.0:
            lookZ = 1e-6

        mag = math.sqrt(lookX**2 + lookY**2 + lookZ**2)
        lookX /= mag
        lookY /= mag
        lookZ /= mag

        # From the algorithm/code shown here:
        # http://www.cse.yorku.ca/~amana/research/grid.pdf

        x = nearestBlockCoord(cameraPos[0])
        y = nearestBlockCoord(cameraPos[1])
        z = nearestBlockCoord(cameraPos[2])

        stepX = 1 if lookX > 0.0 else -1
        stepY = 1 if lookY > 0.0 else -1
        stepZ = 1 if lookZ > 0.0 else -1

        tDeltaX = 1.0 / abs(lookX)
        tDeltaY = 1.0 / abs(lookY)
        tDeltaZ = 1.0 / abs(lookZ)

        nextXWall = x + 0.5 if stepX == 1 else x - 0.5
        nextYWall = y + 0.5 if stepY == 1 else y - 0.5
        nextZWall = z + 0.5 if stepZ == 1 else z - 0.5

        tMaxX = (nextXWall - cameraPos[0]) / lookX
        tMaxY = (nextYWall - cameraPos[1]) / lookY
        tMaxZ = (nextZWall - cameraPos[2]) / lookZ

        blockPos = None
        lastMaxVal = 0.0
        
        def blockFilter(blockId):
            if blockId in ('water', 'flowing_water', 'lava', 'flowing_lava'):
                return useFluids
            else:
                return blockId != 'air'

        while 1:
            if self.coordsOccupied(BlockPos(x, y, z), blockFilter):
                blockPos = BlockPos(x, y, z)
                break

            minVal = min(tMaxX, tMaxY, tMaxZ)

            if minVal == tMaxX:
                x += stepX
                # FIXME: if outside...
                lastMaxVal = tMaxX
                tMaxX += tDeltaX
            elif minVal == tMaxY:
                y += stepY
                lastMaxVal = tMaxY
                tMaxY += tDeltaY
            else:
                z += stepZ
                lastMaxVal = tMaxZ
                tMaxZ += tDeltaZ
            
            if lastMaxVal > reach:
                break
        
        if blockPos is None:
            return None
        else:
            pointX = cameraPos[0] + lastMaxVal * lookX
            pointY = cameraPos[1] + lastMaxVal * lookY
            pointZ = cameraPos[2] + lastMaxVal * lookZ

            pointX -= blockPos.x
            pointY -= blockPos.y
            pointZ -= blockPos.z

            if abs(pointX) > abs(pointY) and abs(pointX) > abs(pointZ):
                face = 'right' if pointX > 0.0 else 'left'
            elif abs(pointY) > abs(pointX) and abs(pointY) > abs(pointZ):
                face = 'top' if pointY > 0.0 else 'bottom'
            else:
                face = 'front' if pointZ > 0.0 else 'back'
            
            return (blockPos, face)


    # app.instances[idx] = [Instance(app.cube, np.array([[modelX], [modelY], [modelZ]]), texture), False]

def getBlastResistance(block: BlockId):
    if block == 'grass':
        return 0.6
    elif block == 'dirt':
        return 0.5
    elif block in ('stone', 'cobblestone'):
        return 6.0
    elif block.endswith('_leaves'):
        return 0.2
    elif block.endswith('_log'):
        return 2.0
    elif block.endswith('_planks'):
        return 3.0
    elif block in ('water', 'lava', 'flowing_water', 'flowing_lava'):
        return 100.0
    elif block == 'bedrock':
        return 1e6
    elif block == 'crafting_table':
        return 2.5
    elif block == 'furnace':
        return 3.5
    elif block.endswith('_ore'):
        return 3.0
    else:
        return 6.0
    

def isSolid(block: BlockId):
    return block not in ['torch', 'wall_torch', 'redstone_torch', 'redstone_wall_torch', 'redstone_wire',
        'air', 'water', 'flowing_water', 'lava', 'flowing_lava', 'nether_portal']

def isOpaque(block: BlockId):
    return block not in ['torch', 'wall_torch', 'redstone_torch', 'redstone_wall_torch', 'redstone_wire',
        'air', 'water', 'flowing_water', 'lava', 'flowing_lava', 'nether_portal']

def getLuminance(block: BlockId):
    if block in ('glowstone', 'lava', 'flowing_lava'):
        return 15
    elif block in ('torch', 'wall_torch'):
        return 14
    else:
        return 0

def updateBuriedStateAt(world: World, pos: BlockPos):
    (chunk, innerPos) = world.getChunk(pos)
    chunk.updateBuriedStateAt(world, innerPos)

def setBlock(app, pos: BlockPos, id: BlockId, doUpdateLight=True, doUpdateBuried=True, doUpdateMesh=False) -> None:
    (chunk, innerPos) = app.world.getChunk(pos)
    chunk.setBlock(app.world, (app.textures, app.cube, app.textureIndices), innerPos, id, doUpdateLight, doUpdateBuried, doUpdateMesh)

def toChunkLocal(pos: BlockPos) -> Tuple[ChunkPos, BlockPos]:
    (x, y, z) = pos
    cx = math.floor(x / 16)
    cy = math.floor(y / CHUNK_HEIGHT)
    cz = math.floor(z / 16)

    chunkPos = ChunkPos(cx, cy, cz)

    x %= 16
    y %= CHUNK_HEIGHT
    z %= 16

    blockPos = BlockPos(x, y, z)

    return (chunkPos, blockPos)

def nearestBlockCoord(coord: float) -> int:
    return roundHalfUp(coord)

def nearestBlockPos(x: float, y: float, z: float) -> BlockPos:
    blockX: int = nearestBlockCoord(x)
    blockY: int = nearestBlockCoord(y)
    blockZ: int = nearestBlockCoord(z)
    return BlockPos(blockX, blockY, blockZ)

# Returns the position of the center of the block
def blockToWorld(pos: BlockPos) -> Tuple[float, float, float]:
    (x, y, z) = pos
    return (x, y, z)

def updateBuriedStateNear(world: World, blockPos: BlockPos):
    for faceIdx in range(0, 12, 2):
        pos = adjacentBlockPos(blockPos, faceIdx)
        if world.coordsInBounds(pos):
            updateBuriedStateAt(world, pos)

# length is dist * 2

def adjacentChunks(chunkPos, dist):
    for r in range(1, dist + 1):
        length = r * 2
        corners = (
            ((-1,  1), ( 1,  0)),
            (( 1,  1), ( 0, -1)),
            (( 1, -1), (-1,  0)),
            ((-1, -1), ( 0,  1)),
        )
        for ((cx, cz), (dx, dz)) in corners:
            for i in range(length):
                xOffset = cx * r + dx * i
                zOffset = cz * r + dz * i

                (x, y, z) = chunkPos
                x += xOffset
                z += zOffset

                yield ChunkPos(x, y, z)
            
def adjacentChunks2(chunkPos, dist):
    for xOffset in range(-dist, dist+1):
        for zOffset in range(-dist, dist+1):
            if xOffset == 0 and zOffset == 0: continue

            (x, y, z) = chunkPos
            x += xOffset
            z += zOffset

            newChunkPos = ChunkPos(x, y, z)
            yield newChunkPos

print(list(adjacentChunks(ChunkPos(0, 0, 0), 1)))
        
import pickle

def pickle_trick(obj, max_depth=10):
    output = {}

    if max_depth <= 0:
        return output

    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError) as e:
        failing_children = []

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                result = pickle_trick(v, max_depth=max_depth - 1)
                if result:
                    failing_children.append(result)

        output = {
            "fail": obj, 
            "err": e, 
            "depth": max_depth, 
            "failing_children": failing_children
        }

    return output

def mapMultiFunc(pair):
    (world, instData, chunkPos) = pair
    return (chunkPos, world.createChunk(instData, chunkPos))

    
def removeBlock(app, blockPos: BlockPos):
    setBlock(app, blockPos, 'air', doUpdateMesh=True)

def addBlock(app, blockPos: BlockPos, id: BlockId):
    setBlock(app, blockPos, id, doUpdateMesh=True)

def adjacentBlockPos(blockPos: BlockPos, faceIdx: int) -> BlockPos:
    [x, y, z] = blockPos

    faceIdx //= 2

    # Left, right, near, far, bottom, top
    (a, b, c) = [(-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0)][faceIdx]

    x += a
    y += b
    z += c

    return BlockPos(x, y, z)

