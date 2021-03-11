import numpy as np
import math
import heapq
import render
import time
from math import cos, sin
from numpy import ndarray
from typing import NamedTuple, List, Any, Tuple, Optional

class ChunkPos(NamedTuple):
    x: int
    y: int
    z: int

class BlockPos(NamedTuple):
    x: int
    y: int
    z: int

BlockId = str

class Chunk:
    pos: ChunkPos
    blocks: ndarray
    lightLevels: ndarray
    instances: List[Any]

    isFinalized: bool = False
    isTicking: bool = False
    isVisible: bool = False

    def __init__(self, pos: ChunkPos):
        self.pos = pos

    def generate(self, app):
        # x and y and z
        self.blocks = np.full((16, 16, 16), 'air')
        self.lightLevels = np.full((16, 16, 16), 7)
        self.instances = [None] * self.blocks.size

        for xIdx in range(0, 16):
            for zIdx in range(0, 16):
                for yIdx in range(0, 8):
                    self.lightLevels[xIdx, yIdx, zIdx] = 0
                    blockId = 'grass' if yIdx == 7 else 'stone'
                    self.setBlock(app, BlockPos(xIdx, yIdx, zIdx), blockId, doUpdateLight=False, doUpdateBuried=False)
    
    def lightAndOptimize(self, app):
        print(f"Lighting and optimizing chunk at {self.pos}")
        for xIdx in range(0, 16):
            for yIdx in range(0, 8):
                for zIdx in range(0, 16):
                    self.updateBuriedStateAt(app, BlockPos(xIdx, yIdx, zIdx))

        self.isFinalized = True
    
    def iterInstances(self):
        if self.isFinalized and self.isVisible:
            for (i, instance) in enumerate(self.instances):
                if instance is not None:
                    wx = self.pos.x * 16 + (i // 256)
                    wy = self.pos.y * 16 + (i // 16) % 16
                    wz = self.pos.z * 16 + (i % 16)
                    yield (BlockPos(wx, wy, wz), instance)

    def _coordsToIdx(self, pos: BlockPos) -> int:
        (xw, yw, zw) = self.blocks.shape
        (x, y, z) = pos
        return x * yw * zw + y * zw + z

    def _coordsFromIdx(self, idx: int) -> BlockPos:
        (x, y, z) = self.blocks.shape
        xIdx = idx // (y * z)
        yIdx = (idx // z) % y
        zIdx = (idx % z)
        return BlockPos(xIdx, yIdx, zIdx)
    
    def _globalBlockPos(self, blockPos: BlockPos) -> BlockPos:
        (x, y, z) = blockPos
        x += 16 * self.pos[0]
        y += 16 * self.pos[1]
        z += 16 * self.pos[2]
        return BlockPos(x, y, z)

    def updateBuriedStateAt(self, app, blockPos: BlockPos):
        idx = self._coordsToIdx(blockPos)
        if self.instances[idx] is None:
            return

        globalPos = self._globalBlockPos(blockPos)

        uncovered = False
        for faceIdx in range(0, 12, 2):
            adjPos = adjacentBlockPos(globalPos, faceIdx)
            if coordsOccupied(app, adjPos):
                self.instances[idx][0].visibleFaces[faceIdx] = False
                self.instances[idx][0].visibleFaces[faceIdx + 1] = False
                pass
            else:
                self.instances[idx][0].visibleFaces[faceIdx] = True
                self.instances[idx][0].visibleFaces[faceIdx + 1] = True
                uncovered = True
            
        self.instances[idx][1] = uncovered

    def coordsOccupied(self, pos: BlockPos) -> bool:
        (x, y, z) = pos
        return self.blocks[x, y, z] != 'air'

    def setBlock(self, app, blockPos: BlockPos, id: BlockId, doUpdateLight=True, doUpdateBuried=True):
        (x, y, z) = blockPos
        self.blocks[x, y, z] = id
        idx = self._coordsToIdx(blockPos)
        if id == 'air':
            self.instances[idx] = None
        else:
            texture = app.textures[id]

            [modelX, modelY, modelZ] = blockToWorld(self._globalBlockPos(blockPos))

            self.instances[idx] = [render.Instance(app.cube, np.array([[modelX], [modelY], [modelZ]]), texture), False]
            if doUpdateBuried:
                self.updateBuriedStateAt(app, blockPos)
        
        globalPos = self._globalBlockPos(blockPos)

        if doUpdateBuried:
            updateBuriedStateNear(app, globalPos)
        
        if doUpdateLight:
            updateLight(app, globalPos)

    # app.instances[idx] = [Instance(app.cube, np.array([[modelX], [modelY], [modelZ]]), texture), False]

def updateBuriedStateAt(app, pos: BlockPos):
    (chunk, innerPos) = getChunk(app, pos)
    chunk.updateBuriedStateAt(app, innerPos)

def getChunk(app, pos: BlockPos) -> Tuple[Chunk, BlockPos]:
    (cx, cy, cz) = pos
    cx //= 16
    cy //= 16
    cz //= 16

    chunk = app.chunks[ChunkPos(cx, cy, cz)]
    [x, y, z] = pos
    x %= 16
    y %= 16
    z %= 16
    return (chunk, BlockPos(x, y, z))

def coordsOccupied(app, pos: BlockPos) -> bool:
    if not coordsInBounds(app, pos):
        return False

    (chunk, innerPos) = getChunk(app, pos)
    return chunk.coordsOccupied(innerPos)

def setBlock(app, pos: BlockPos, id: BlockId, doUpdateLight=True) -> None:
    (chunk, innerPos) = getChunk(app, pos)
    chunk.setBlock(app, innerPos, id, doUpdateLight)

def toChunkLocal(pos: BlockPos) -> Tuple[ChunkPos, BlockPos]:
    (x, y, z) = pos
    cx = x // 16
    cy = y // 16
    cz = z // 16

    chunkPos = ChunkPos(cx, cy, cz)

    x %= 16
    y %= 16
    z %= 16

    blockPos = BlockPos(x, y, z)

    return (chunkPos, blockPos)

def coordsInBounds(app, pos: BlockPos) -> bool:
    (chunkPos, _) = toChunkLocal(pos)
    return chunkPos in app.chunks
    '''
    if pos not in app.chunks:
        return False
    
    (x, y, z) = pos
    (xw, yw, zw) = app.blocks.shape
    if x < 0 or xw <= x: return False
    if y < 0 or yw <= y: return False
    if z < 0 or zw <= z: return False
    return True
    '''

def nearestBlockCoord(coord: float) -> int:
    return round(coord)

def nearestBlockPos(x: float, y: float, z: float) -> BlockPos:
    blockX: int = nearestBlockCoord(x)
    blockY: int = nearestBlockCoord(y)
    blockZ: int = nearestBlockCoord(z)
    return BlockPos(blockX, blockY, blockZ)

# Returns the position of the center of the block
def blockToWorld(pos: BlockPos) -> Tuple[float, float, float]:
    (x, y, z) = pos
    return (x, y, z)

def blockIsBuried(app, blockPos: BlockPos):
    for faceIdx in range(0, 12, 2):
        if not coordsOccupied(app, adjacentBlockPos(blockPos, faceIdx)):
            # Unburied
            return False
    # All spaces around are occupied, this block is buried
    return True

def updateBuriedStateNear(app, blockPos: BlockPos):
    for faceIdx in range(0, 12, 2):
        pos = adjacentBlockPos(blockPos, faceIdx)
        if coordsInBounds(app, pos):
            updateBuriedStateAt(app, pos)

def adjacentChunks(chunkPos, dist):
    for xOffset in range(-dist, dist+1):
        for zOffset in range(-dist, dist+1):
            if xOffset == 0 and zOffset == 0:
                continue
        
            (x, y, z) = chunkPos
            x += xOffset
            z += zOffset

            newChunkPos = ChunkPos(x, y, z)
            yield newChunkPos
        
def unloadChunk(app, pos: ChunkPos):
    print(f"Unloading chunk at {pos}")
    app.chunks.pop(pos)

def loadChunk(app, pos: ChunkPos):
    print(f"Loading chunk at {pos}")
    app.chunks[pos] = Chunk(pos)
    app.chunks[pos].generate(app)

def loadUnloadChunks(app):
    (chunkPos, _) = toChunkLocal(nearestBlockPos(app.cameraPos[0], app.cameraPos[1], app.cameraPos[2]))
    (x, _, z) = chunkPos

    # Unload chunks
    shouldUnload = []
    for unloadChunkPos in app.chunks:
        (ux, _, uz) = unloadChunkPos
        dist = max(abs(ux - x), abs(uz - z))
        if dist > 2:
            # Unload chunk
            shouldUnload.append(unloadChunkPos)

    for unloadChunkPos in shouldUnload:
        unloadChunk(app, unloadChunkPos)

    loadedChunks = 0

    for loadChunkPos in adjacentChunks(chunkPos, 2):
        if loadChunkPos not in app.chunks:
            (ux, _, uz) = loadChunkPos
            dist = max(abs(ux - x), abs(uz - z))

            urgent = dist <= 1

            if urgent or (loadedChunks < 1):
                loadedChunks += 1
                loadChunk(app, loadChunkPos)

def countLoadedAdjacentChunks(app, chunkPos: ChunkPos, dist: int) -> int:
    count = 0
    for pos in adjacentChunks(chunkPos, dist):
        if pos in app.chunks:
            count += 1
    return count

def tickChunks(app):
    for chunkPos in app.chunks:
        chunk = app.chunks[chunkPos]
        adjacentChunks = countLoadedAdjacentChunks(app, chunkPos, 1)
        if not chunk.isFinalized and adjacentChunks == 8:
            chunk.lightAndOptimize(app)
        chunk.isVisible = adjacentChunks == 8
        chunk.isTicking = adjacentChunks == 8

def tick(app):
    startTime = time.time()

    loadUnloadChunks(app)

    tickChunks(app)

    # Ticking is done in stages so that collision detection works as expected:
    # First we update the player's Y position and resolve Y collisions,
    # then we update the player's X position and resolve X collisions,
    # and finally update the player's Z position and resolve Z collisions.

    app.cameraPos[1] += app.playerVel[1]

    if app.playerOnGround:
        if not hasBlockBeneath(app):
            app.playerOnGround = False
    else:
        app.playerVel[1] -= app.gravity
        [_, yPos, _] = app.cameraPos
        yPos -= app.playerHeight
        yPos -= 0.1
        feetPos = round(yPos)
        if hasBlockBeneath(app):
            app.playerOnGround = True
            app.playerVel[1] = 0.0
            app.cameraPos[1] = (feetPos + 0.5) + app.playerHeight
    
    # W makes the player go forward, S makes them go backwards,
    # and pressing both makes them stop!
    z = float(app.w) - float(app.s)
    # Likewise for side to side movement
    x = float(app.d) - float(app.a)

    if x != 0.0 or z != 0.0:
        mag = math.sqrt(x*x + z*z)
        x /= mag
        z /= mag

        newX = math.cos(app.cameraYaw) * x - math.sin(app.cameraYaw) * z
        newZ = math.sin(app.cameraYaw) * x + math.cos(app.cameraYaw) * z

        x, z = newX, newZ

        x *= app.playerWalkSpeed 
        z *= app.playerWalkSpeed

    xVel = x
    zVel = z

    minY = round((app.cameraPos[1] - app.playerHeight + 0.1))
    maxY = round((app.cameraPos[1]))

    app.cameraPos[0] += xVel

    for y in range(minY, maxY):
        for z in [app.cameraPos[2] - app.playerRadius * 0.99, app.cameraPos[2] + app.playerRadius * 0.99]:
            x = app.cameraPos[0]

            hiXBlockCoord = round((x + app.playerRadius))
            loXBlockCoord = round((x - app.playerRadius))

            if coordsOccupied(app, BlockPos(hiXBlockCoord, y, round(z))):
                # Collision on the right, so move to the left
                xEdge = (hiXBlockCoord - 0.5)
                app.cameraPos[0] = xEdge - app.playerRadius
            elif coordsOccupied(app, BlockPos(loXBlockCoord, y, round(z))):
                # Collision on the left, so move to the right
                xEdge = (loXBlockCoord + 0.5)
                app.cameraPos[0] = xEdge + app.playerRadius
    
    app.cameraPos[2] += zVel

    for y in range(minY, maxY):
        for x in [app.cameraPos[0] - app.playerRadius * 0.99, app.cameraPos[0] + app.playerRadius * 0.99]:
            z = app.cameraPos[2]

            hiZBlockCoord = round((z + app.playerRadius))
            loZBlockCoord = round((z - app.playerRadius))

            if coordsOccupied(app, BlockPos(round(x), y, hiZBlockCoord)):
                zEdge = (hiZBlockCoord - 0.5)
                app.cameraPos[2] = zEdge - app.playerRadius
            elif coordsOccupied(app, BlockPos(round(x), y, loZBlockCoord)):
                zEdge = (loZBlockCoord + 0.5)
                app.cameraPos[2] = zEdge + app.playerRadius
    
    endTime = time.time()
    app.tickTimes[app.tickTimeIdx] = (endTime - startTime)
    app.tickTimeIdx += 1
    app.tickTimeIdx %= len(app.tickTimes)

def getLightLevel(app, blockPos: BlockPos) -> int:
    (chunk, (x, y, z)) = getChunk(app, blockPos)
    return chunk.lightLevels[x, y, z]

def setLightLevel(app, blockPos: BlockPos, level: int):
    (chunk, (x, y, z)) = getChunk(app, blockPos)
    chunk.lightLevels[x, y, z] = level

def updateLight(app, blockPos: BlockPos):
    added = coordsOccupied(app, blockPos)

    (chunk, localPos) = getChunk(app, blockPos)

    skyExposed = True
    for y in range(localPos.y + 1, 16):
        checkPos = BlockPos(localPos.x, y, localPos.z)
        if chunk.coordsOccupied(checkPos):
            skyExposed = False
            break

    if added:
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
        if skyExposed:
            for y in range(localPos.y - 1, -1, -1):
                checkPos = BlockPos(localPos.x, y, localPos.z)
                if chunk.coordsOccupied(checkPos):
                    break

                heapq.heappush(exSources, (-7, BlockPos(blockPos.x, y, blockPos.z)))
        
        for faceIdx in range(0, 12, 2):
            gPos = adjacentBlockPos(blockPos, faceIdx)

            if coordsOccupied(app, gPos):
                continue

            lightLevel = getLightLevel(app, gPos)

            heapq.heappush(exSources, (-lightLevel, gPos))

        exVisited = []

        queue = []
        
        while len(exSources) > 0:
            (neglight, pos) = heapq.heappop(exSources)
            neglight *= -1
            if pos in exVisited:
                continue
            exVisited.append(pos)

            existingLight = getLightLevel(app, pos)
            if existingLight > neglight:
                heapq.heappush(queue, (-existingLight, pos))
                continue

            setLightLevel(app, pos, 0)

            nextLight = max(neglight - 1, 0)
            for faceIdx in range(0, 12, 2):
                nextPos = adjacentBlockPos(pos, faceIdx)
                if nextPos in exVisited:
                    continue
                if not coordsInBounds(app, nextPos):
                    continue
                if coordsOccupied(app, nextPos):
                    continue

                heapq.heappush(exSources, (-nextLight, nextPos))
        
        chunk.lightLevels[localPos.x, localPos.y, localPos.z] = 0
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
        if skyExposed:
            for y in range(localPos.y, -1, -1):
                checkPos = BlockPos(localPos.x, y, localPos.z)
                if chunk.coordsOccupied(checkPos):
                    break

                heapq.heappush(queue, (-7, BlockPos(blockPos.x, y, blockPos.z)))
        
        for faceIdx in range(0, 12, 2):
            gPos = adjacentBlockPos(blockPos, faceIdx)

            lightLevel = getLightLevel(app, gPos)

            heapq.heappush(queue, (-lightLevel, gPos))


    visited = []
    
    while len(queue) > 0:
        (light, pos) = heapq.heappop(queue)
        light *= -1
        if pos in visited:
            continue
        visited.append(pos)
        setLightLevel(app, pos, light)

        nextLight = max(light - 1, 0)
        for faceIdx in range(0, 12, 2):
            nextPos = adjacentBlockPos(pos, faceIdx)
            if nextPos in visited:
                continue
            if not coordsInBounds(app, nextPos):
                continue
            if coordsOccupied(app, nextPos):
                continue

            existingLight = getLightLevel(app, nextPos)

            if nextLight > existingLight:
                heapq.heappush(queue, (-nextLight, nextPos))
    
def removeBlock(app, blockPos: BlockPos):
    setBlock(app, blockPos, 'air')

def addBlock(app, blockPos: BlockPos, id: BlockId):
    setBlock(app, blockPos, id)

def hasBlockBeneath(app):
    [xPos, yPos, zPos] = app.cameraPos
    yPos -= app.playerHeight
    yPos -= 0.1

    for x in [xPos - app.playerRadius * 0.99, xPos + app.playerRadius * 0.99]:
        for z in [zPos - app.playerRadius * 0.99, zPos + app.playerRadius * 0.99]:
            feetPos = nearestBlockPos(x, yPos, z)
            if coordsOccupied(app, feetPos):
                return True
    
    return False

def adjacentBlockPos(blockPos: BlockPos, faceIdx: int) -> BlockPos:
    [x, y, z] = blockPos

    faceIdx //= 2

    # Left, right, near, far, bottom, top
    (a, b, c) = [(-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0)][faceIdx]

    x += a
    y += b
    z += c

    return BlockPos(x, y, z)

def lookedAtBlock(app) -> Optional[Tuple[BlockPos, str]]:
    lookX = cos(app.cameraPitch)*sin(-app.cameraYaw)
    lookY = sin(app.cameraPitch)
    lookZ = cos(app.cameraPitch)*cos(-app.cameraYaw)

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

    x = nearestBlockCoord(app.cameraPos[0])
    y = nearestBlockCoord(app.cameraPos[1])
    z = nearestBlockCoord(app.cameraPos[2])

    stepX = 1 if lookX > 0.0 else -1
    stepY = 1 if lookY > 0.0 else -1
    stepZ = 1 if lookZ > 0.0 else -1

    tDeltaX = 1.0 / abs(lookX)
    tDeltaY = 1.0 / abs(lookY)
    tDeltaZ = 1.0 / abs(lookZ)

    nextXWall = x + 0.5 if stepX == 1 else x - 0.5
    nextYWall = y + 0.5 if stepY == 1 else y - 0.5
    nextZWall = z + 0.5 if stepZ == 1 else z - 0.5

    tMaxX = (nextXWall - app.cameraPos[0]) / lookX
    tMaxY = (nextYWall - app.cameraPos[1]) / lookY
    tMaxZ = (nextZWall - app.cameraPos[2]) / lookZ

    print(f"{tMaxX}, {tMaxY}, {tMaxZ}")

    blockPos = None
    lastMaxVal = 0.0

    while 1:
        print(x, y, z)

        if coordsOccupied(app, BlockPos(x, y, z)):
            blockPos = BlockPos(x, y, z)
            break

        print(f"tmaxes: {tMaxX}, {tMaxY}, {tMaxZ}")

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
        
        if lastMaxVal > app.playerReach:
            break
    
    if blockPos is None:
        return None
    else:
        pointX = app.cameraPos[0] + lastMaxVal * lookX
        pointY = app.cameraPos[1] + lastMaxVal * lookY
        pointZ = app.cameraPos[2] + lastMaxVal * lookZ

        pointX -= blockPos.x
        pointY -= blockPos.y
        pointZ -= blockPos.z

        if abs(pointX) > abs(pointY) and abs(pointX) > abs(pointZ):
            face = 'right' if x > 0.0 else 'left'
        elif abs(pointY) > abs(pointX) and abs(pointY) > abs(pointZ):
            face = 'top' if y > 0.0 else 'bottom'
        else:
            face = 'front' if z > 0.0 else 'back'
        
        return (blockPos, face)