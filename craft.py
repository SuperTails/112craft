from cmu_112_graphics import *
import numpy as np
import math
import heapq
import time
from collections import namedtuple
from collections import deque
from math import cos, sin
from numpy import infty, ndarray
from typing import Optional, NamedTuple, List, Tuple, Any, Union
import copy
import perlin_noise

# =========================================================================== #
# ----------------------------- THE APP ------------------------------------- #
# =========================================================================== #

def appStarted(app):
    vertices = [
        np.array([[-1.0], [-1.0], [-1.0]]) / 2.0,
        np.array([[-1.0], [-1.0], [1.0]]) / 2.0,
        np.array([[-1.0], [1.0], [-1.0]]) / 2.0,
        np.array([[-1.0], [1.0], [1.0]]) / 2.0,
        np.array([[1.0], [-1.0], [-1.0]]) / 2.0,
        np.array([[1.0], [-1.0], [1.0]]) / 2.0,
        np.array([[1.0], [1.0], [-1.0]]) / 2.0,
        np.array([[1.0], [1.0], [1.0]]) / 2.0
    ]

    grassTexture = [
        '#FF0000', '#FF0000',
        '#A52A2A', '#A52A2A',
        '#A52A2A', '#A52A2A',
        '#A52A2A', '#A52A2A',
        '#A52A2A', '#A52A2A',
        '#00FF00', '#00EE00']
    
    stoneTexture = [
        '#AAAAAA', '#AAAABB',
        '#AAAACC', '#AABBBB',
        '#AACCCC', '#88AAAA',
        '#AA88AA', '#888888',
        '#AA88CC', '#778888',
        '#BBCCAA', '#BBBBBB'
    ]

    app.textures = {
        'grass': grassTexture,
        'stone': stoneTexture,
    }

    # Vertices in CCW order
    faces: List[Face] = [
        # Left
        (0, 2, 1),
        (1, 2, 3),
        # Right
        (4, 5, 6),
        (6, 5, 7),
        # Near
        (0, 4, 2),
        (2, 4, 6),
        # Far
        (5, 1, 3),
        (5, 3, 7),
        # Bottom
        (0, 1, 4),
        (4, 1, 5),
        # Top
        (3, 2, 6),
        (3, 6, 7),
    ]

    app.lowNoise = perlin_noise.PerlinNoise(octaves=3)

    app.cube = Model(vertices, faces)

    app.chunks = {
        ChunkPos(0, 0, 0): Chunk(ChunkPos(0, 0, 0))
    }

    app.chunks[ChunkPos(0, 0, 0)].generate(app)

    app.playerHeight = 1.5
    app.playerWidth = 0.6
    app.playerRadius = app.playerWidth / 2
    app.playerOnGround = False
    app.playerVel = [0.0, 0.0, 0.0]
    app.playerWalkSpeed = 0.2
    app.selectedBlock = 'air'
    app.gravity = 0.10
    app.renderDistance = 6.0

    app.cameraYaw = 0
    app.cameraPitch = 0
    app.cameraPos = [4.0, 10.0 + app.playerHeight, 4.0]

    app.cameraPitch = 0.0
    app.cameraYaw = 0.0

    app.vpDist = 0.25
    app.vpWidth = 3.0 / 4.0
    app.vpHeight = app.vpWidth * app.height / app.width 

    app.timerDelay = 50

    app.w = False
    app.s = False
    app.a = False
    app.d = False

    app.prevMouse = None

    app.captureMouse = False

    app.wireframe = False

    app.csToCanvasMat = csToCanvasMat(app.vpDist, app.vpWidth, app.vpHeight,
                        app.width, app.height)
    
def sizeChanged(app):
    app.csToCanvasMat = csToCanvasMat(app.vpDist, app.vpWidth, app.vpHeight,
                        app.width, app.height)

def mousePressed(app, event):
    block = lookedAtBlock(app)
    if block is not None:
        (pos, face) = block
        if app.selectedBlock == 'air':
            removeBlock(app, pos)
        else:
            [x, y, z] = pos
            if face == 'left':
                x -= 1
            elif face == 'right':
                x += 1
            elif face == 'bottom':
                y -= 1
            elif face == 'top':
                y += 1
            elif face == 'back':
                z -= 1
            elif face == 'front':
                z += 1
            
            addBlock(app, BlockPos(x, y, z), app.selectedBlock)

def mouseMoved(app, event):
    if not app.captureMouse:
        app.prevMouse = None

    if app.prevMouse is not None:
        xChange = -(event.x - app.prevMouse[0])
        yChange = -(event.y - app.prevMouse[1])

        app.cameraPitch += yChange * 0.01

        if app.cameraPitch < -math.pi / 2 * 0.95:
            app.cameraPitch = -math.pi / 2 * 0.95
        elif app.cameraPitch > math.pi / 2 * 0.95:
            app.cameraPitch = math.pi / 2 * 0.95

        app.cameraYaw += xChange * 0.01

    if app.captureMouse:
        x = app.width / 2
        y = app.height / 2
        app._theRoot.event_generate('<Motion>', warp=True, x=x, y=y)
        app.prevMouse = (x, y)

def keyReleased(app, event):
    if event.key == 'w':
        app.w = False
    elif event.key == 's':
        app.s = False 
    elif event.key == 'a':
        app.a = False
    elif event.key == 'd':
        app.d = False

def timerFired(app):
    tick(app)

def keyPressed(app, event):
    if event.key == '1':
        app.selectedBlock = 'air'
    elif event.key == '2':
        app.selectedBlock = 'grass'
    elif event.key == '3':
        app.selectedBlock = 'stone'
    elif event.key == 'w':
        app.w = True
    elif event.key == 's':
        app.s = True
    elif event.key == 'a':
        app.a = True
    elif event.key == 'd':
        app.d = True
    elif event.key == 'Space' and app.playerOnGround:
        app.playerVel[1] = 0.35
    elif event.key == 'Escape':
        app.captureMouse = not app.captureMouse
        if app.captureMouse:
            app._theRoot.config(cursor="none")
        else:
            app._theRoot.config(cursor="")

# =========================================================================== #
# ------------------------- IDK WHAT TO NAME THIS --------------------------- #
# =========================================================================== #

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
                # globalPos = self._globalBlockPos(BlockPos(xIdx, 0, zIdx))
                # val = app.lowNoise([globalPos[0] / 16.0, globalPos[1] / 16.0, globalPos[2] / 16.0])

                # h = 6 + int(val * 4)

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
                    wx = self.pos[0] * 16 + (i // 256)
                    wy = self.pos[1] * 16 + (i // 16) % 16
                    wz = self.pos[2] * 16 + (i % 16)
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
        #if not coordsInBounds(self, pos):
        #    raise Exception("outside of chunk")
        
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

            self.instances[idx] = [Instance(app.cube, np.array([[modelX], [modelY], [modelZ]]), texture), False]
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

#@delegateToChunk
#def coordsOccupied(app, pos: BlockPos): pass
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
    (chunkPos, blockPos) = toChunkLocal(pos)
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

            if urgent or (loadedChunks < 2):
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
    # Ticking is done in stages so that collision detection works as expected:
    # First we update the player's Y position and resolve Y collisions,
    # then we update the player's X position and resolve X collisions,
    # and finally update the player's Z position and resolve Z collisions.

    loadUnloadChunks(app)

    tickChunks(app)

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

def setLightLevel(app, blockPos: BlockPos, level: int):
    (chunk, (x, y, z)) = getChunk(app, blockPos)
    chunk.lightLevels[x, y, z] = level

def updateLight(app, blockPos: BlockPos):
    # FIXME: Lighting changes need to propogate across chunk boundaries
    # But that's REALLY slow

    (chunk, blockPos) = getChunk(app, blockPos)
    chunk.lightLevels = np.full_like(chunk.blocks, 0, int)

    shape = chunk.blocks.shape

    visited = []
    queue = []

    for x in range(shape[0]):
        for z in range(shape[2]):
            y = shape[1] - 1
            heapq.heappush(queue, (-7, BlockPos(x, y, z)))
    
    while len(queue) > 0:
        (light, pos) = heapq.heappop(queue)
        light *= -1
        if pos in visited:
            continue
        visited.append(pos)
        #setLightLevel(app, pos, light)
        (x, y, z) = pos
        chunk.lightLevels[x, y, z] = light

        for faceIdx in range(0, 10, 2):
            nextPos = adjacentBlockPos(pos, faceIdx)
            globalPos = chunk._globalBlockPos(nextPos)
            if nextPos in visited:
                continue
            if not coordsInBounds(app, globalPos):
                continue
            if coordsOccupied(app, globalPos):
                continue
            if nextPos[0] < 0 or 16 <= nextPos[0]:
                continue
            if nextPos[1] < 0 or 16 <= nextPos[1]:
                continue
            if nextPos[2] < 0 or 16 <= nextPos[2]:
                continue

            if light == 7 and faceIdx == 8:
                nextLight = 7
            else:
                nextLight = max(light - 1, 0)
            
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

# =========================================================================== #
# ---------------------------- RENDERING ------------------------------------ #
# =========================================================================== #

Color = str

# Always holds indices into the model's list of vertices
Face = Tuple[int, int, int]

class Model:
    vertices: List[ndarray]
    faces: List[Face]

    def __init__(self, vertices: List[ndarray], faces: List[Face]):
        self.vertices = vertices

        self.faces = []
        for face in faces:
            if len(face) == 4:
                # FIXME
                1 / 0
            elif len(face) == 3:
                self.faces.append(face)
            else:
                raise Exception("Invalid number of vertices for face")

class Instance:
    model: Model
    trans: ndarray
    texture: List[Color]
    visibleFaces: List[bool]

    _worldSpaceVertices: List[ndarray]

    def __init__(self, model: Model, trans: ndarray, texture: List[Color]):
        self.model = model
        self.trans = trans
        self.texture = texture

        self._worldSpaceVertices = list(map(toHomogenous, self.worldSpaceVerticesUncached()))
        self.visibleFaces = [True] * len(model.faces)

    def worldSpaceVertices(self) -> List[ndarray]:
        return self._worldSpaceVertices
    
    def worldSpaceVerticesUncached(self) -> List[ndarray]:
        result = []
        for vertex in self.model.vertices:
            result.append(vertex + self.trans)
        return result
    
def toHomogenous(cartesian: ndarray) -> ndarray:
    assert(cartesian.shape[1] == 1)

    #return np.vstack((cartesian, np.array([[1]])))

    # This one line change makes the world load *twice as fast*
    return np.array([[cartesian[0, 0]], [cartesian[1, 0]], [cartesian[2, 0]], [1.0]])


def toCartesian(cartesian: ndarray) -> ndarray:
    assert(cartesian.shape[1] == 1)

    cart = cartesian.ravel()

    return cart[:-1] / cart[-1]

def rotateX(ang):
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, math.cos(ang), -math.sin(ang), 0.0],
        [0.0, math.sin(ang), math.cos(ang), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotateY(ang):
    return np.array([
        [math.cos(ang), 0.0, math.sin(ang), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-math.sin(ang), 0.0, math.cos(ang), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotateZ(ang):
    return np.array([
        [math.cos(ang), -math.sin(ang), 0.0, 0.0],
        [math.sin(ang), math.cos(ang), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def translationMat(x, y, z):
    return np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0]])


def wsToCanvasMat(camPos, yaw, pitch, vpDist, vpWidth, vpHeight,
    canvWidth, canvHeight):
    vpToCanv = vpToCanvasMat(vpWidth, vpHeight, canvWidth, canvHeight)
    return vpToCanv @ camToVpMat(vpDist) @ wsToCamMat(camPos, yaw, pitch)

def csToCanvasMat(vpDist, vpWidth, vpHeight, canvWidth, canvHeight):
    vpToCanv = vpToCanvasMat(vpWidth, vpHeight, canvWidth, canvHeight)
    return vpToCanv @ camToVpMat(vpDist)

def wsToCamMat(camPos, yaw, pitch):
    # Original technique from
    # https://gamedev.stackexchange.com/questions/168542/camera-view-matrix-from-position-yaw-pitch-worldup
    # (My axes are oriented differently, so the matrix is different)

    # I think I made a mistake in the calculations but this fixes it lol
    yaw = -yaw

    '''
    Here is the camera matrix:

    x = camPos[0]
    y = camPos[1]
    z = camPos[2]

    cam = [
        [cos(yaw),  -sin(pitch)*sin(yaw), cos(pitch)*sin(yaw), x],
        [0.0,       cos(pitch),           sin(pitch),          y],
        [-sin(yaw), -sin(pitch)*cos(yaw), cos(pitch)*cos(yaw), z],
        [0.0,       0.0,                  0.0,                 1.0]
    ]

    cam = np.linalg.inv(cam)
    '''

    y = yaw
    p = pitch

    a = camPos[0]
    b = camPos[1]
    c = camPos[2]

    # This is the manually-calculated inverse of the matrix shown above
    cam = np.array([
        [cos(y), 0.0, -sin(y), c*sin(y) - a*cos(y)],
        [-sin(p)*sin(y), cos(p), -sin(p)*cos(y), c*sin(p)*cos(y) + a*sin(p)*sin(y) - b*cos(p)],
        [cos(p)*sin(y), sin(p), cos(p)*cos(y), -b*sin(p) - a*sin(y)*cos(p) - c*cos(y)*cos(p)],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return cam

def vpToCanvasMat(vpWidth, vpHeight, canvWidth, canvHeight):
    w = canvWidth / vpWidth
    h = -canvHeight / vpHeight

    x = canvWidth * 0.5
    y = canvHeight * 0.5

    return np.array([
        [w, 0.0, x],
        [0.0, h, y],
        [0.0, 0.0, 1.0]])

def wsToCam(point, camPos):
    x = point[0] - camPos[0]
    y = point[1] - camPos[1]
    z = point[2] - camPos[2]

    return [x, y, z]

def camToVpMat(vpDist):
    vpd = vpDist

    return np.array([
        [vpd, 0.0, 0.0, 0.0],
        [0.0, vpd, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])

def camToVp(point, vpDist):
    vpX = point[0] * vpDist / point[2]
    vpY = point[1] * vpDist / point[2]

    return [vpX, vpY]

def vpToCanvas(point, vpWidth, vpHeight, canvWidth, canvHeight):
    canvX = (point[0] / vpWidth + 0.5) * canvWidth
    canvY = (-point[1] / vpHeight + 0.5) * canvHeight

    return [canvX, canvY]

def lookedAtBlock(app) -> Optional[Tuple[BlockPos, str]]:
    # TODO: Optimize

    lookX = cos(app.cameraPitch)*sin(-app.cameraYaw)
    lookY = sin(app.cameraPitch)
    lookZ = cos(app.cameraPitch)*cos(-app.cameraYaw)

    mag = math.sqrt(lookX**2 + lookY**2 + lookZ**2)
    lookX /= mag
    lookY /= mag
    lookZ /= mag

    step = 0.1
    lookX *= step
    lookY *= step
    lookZ *= step

    [x, y, z] = app.cameraPos

    maxDist = 6.0

    blockPos = None

    for _ in range(int(maxDist / step)):
        x += lookX
        y += lookY
        z += lookZ

        tempBlockPos = nearestBlockPos(x, y, z)

        if coordsOccupied(app, tempBlockPos):
            blockPos = tempBlockPos
            break
    
    if blockPos is None:
        return None
    
    [centerX, centerY, centerZ] = blockPos

    x -= centerX
    y -= centerY
    z -= centerZ

    if abs(x) > abs(y) and abs(x) > abs(z):
        face = 'right' if x > 0.0 else 'left'
    elif abs(y) > abs(x) and abs(y) > abs(z):
        face = 'top' if y > 0.0 else 'bottom'
    else:
        face = 'front' if z > 0.0 else 'back'
    
    return (blockPos, face)

def wsToCanvas(app, point):
    point = toHomogenous(point)
    mat = wsToCanvasMat(app.cameraPos, app.cameraYaw, app.cameraPitch,
        app.vpDist, app.vpWidth, app.vpHeight, app.width, app.height)

    point = mat @ point

    point = toCartesian(point)

    # point = wsToCam(point, app.cameraPos)
    # point = camToVp(point, app.vpDist)
    #point = vpToCanvas(point, app.vpWidth, app.vpHeight, app.width, app.height)
    return point

def faceNormal(v0, v1, v2):
    v0 = toCartesian(v0)
    v1 = toCartesian(v1)
    v2 = toCartesian(v2)

    a = v1 - v0
    b = v2 - v0
    cross = np.cross(a, b)
    return cross

# Vertices must be in camera space
def isBackFace(v0, v1, v2) -> bool:
    # From https://en.wikipedia.org/wiki/Back-face_culling

    normal = faceNormal(v0, v1, v2)
    v0 = toCartesian(v0)

    return -np.dot(v0, normal) >= 0

def adjacentBlockPos(blockPos: BlockPos, faceIdx: int) -> BlockPos:
    [x, y, z] = blockPos

    faceIdx //= 2

    # Left, right, near, far, bottom, top
    (a, b, c) = [(-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0)][faceIdx]

    x += a
    y += b
    z += c

    return BlockPos(x, y, z)

def blockFaceVisible(app, blockPos: BlockPos, faceIdx: int) -> bool:
    (x, y, z) = adjacentBlockPos(blockPos, faceIdx)

    if coordsOccupied(app, BlockPos(x, y, z)):
        return False

    return True

def blockFaceLight(app, blockPos: BlockPos, faceIdx: int) -> int:
    pos = adjacentBlockPos(blockPos, faceIdx)
    (chunk, (x, y, z)) = getChunk(app, pos)
    return chunk.lightLevels[x, y, z]

def isBackBlockFace(app, blockPos: BlockPos, faceIdx: int) -> bool:
    # Left 
    faceIdx //= 2
    (x, y, z) = blockToWorld(blockPos)
    xDiff = app.cameraPos[0] - x
    yDiff = app.cameraPos[1] - y
    zDiff = app.cameraPos[2] - z
    # Left, right, near, far, bottom, top
    if faceIdx == 0:
        # Left
        return xDiff > -0.5
    elif faceIdx == 1:
        # Right
        return xDiff < 0.5
    elif faceIdx == 2:
        # Near
        return zDiff > -0.5
    elif faceIdx == 3:
        # Far
        return zDiff < 0.5
    elif faceIdx == 4:
        # Bottom
        return yDiff > -0.5
    else:
        # Top
        return yDiff < 0.5
    
# FIXME: This does NOT preserve the CCW vertex ordering!
# And also adds stuff to `vertices`
def clip(app, vertices: List[Any], face: Face) -> List[Face]:
    outOfView = lambda idx: vertices[idx][2] < app.vpDist

    numVisible = (not outOfView(face[0])) + (
        (not outOfView(face[1])) + (not outOfView(face[2])))

    if numVisible == 0:
        return []
    elif numVisible == 3:
        return [face]

    [v0, v1, v2] = sorted(face, key=outOfView)
    
    [[x0], [y0], [z0], _] = vertices[v0]
    [[x1], [y1], [z1], _] = vertices[v1]
    [[x2], [y2], [z2], _] = vertices[v2]

    if numVisible == 2:
        xd = (x2 - x0) * (app.vpDist - z0) / (z2 - z0) + x0
        yd = (y2 - y0) * (app.vpDist - z0) / (z2 - z0) + y0

        xc = (x2 - x1) * (app.vpDist - z1) / (z2 - z1) + x1
        yc = (y2 - y1) * (app.vpDist - z1) / (z2 - z1) + y1

        dIdx = len(vertices)
        vertices.append(np.array([[xd], [yd], [app.vpDist], [1.0]]))
        cIdx = len(vertices)
        vertices.append(np.array([[xc], [yc], [app.vpDist], [1.0]]))

        face0: Face = (v0, v1, dIdx)
        face1: Face = (v0, v1, cIdx)

        return [face0, face1]
    else:
        xa = (x1 - x0) * (app.vpDist - z0) / (z1 - z0) + x0
        ya = (y1 - y0) * (app.vpDist - z0) / (z1 - z0) + y0

        xb = (x2 - x0) * (app.vpDist - z0) / (z2 - z0) + x0
        yb = (y2 - y0) * (app.vpDist - z0) / (z2 - z0) + y0

        aIdx = len(vertices)
        vertices.append(np.array([[xa], [ya], [app.vpDist], [1.0]]))
        bIdx = len(vertices)
        vertices.append(np.array([[xb], [yb], [app.vpDist], [1.0]]))


        clippedFace: Face = (v0, aIdx, bIdx)

        return [clippedFace]

# This converts the instance's vertices to points in camera space, and then:
# For all blocks, the following happens:
#       - Faces pointing away from the camera are removed
#       - Faces that are hidden 'underground' are removed
#       - The color of each face is adjusted based on lighting
#       - ~~A "fog" is applied~~ NOT IMPLEMENTED!
# For anything else:
#       - Normal back face culling is applied
# 
# Then, the faces are clipped, which may remove, modify, or split faces
# Then a list of faces, their vertices, and their colors are returned
def cullInstance(app, toCamMat: ndarray, inst: Instance, blockPos: Optional[BlockPos]) -> List[Tuple[Any, Face, Color]]:
    # FIXME: Remove homogenous
    vertices = list(map(lambda v: toCamMat @ v, inst.worldSpaceVertices()))

    faces = []

    skipNext = False

    for (faceIdx, (face, color)) in enumerate(zip(inst.model.faces, inst.texture)):
        if skipNext:
            skipNext = False
            continue 

        if blockPos is not None:
            if not inst.visibleFaces[faceIdx]:
                continue
            
            if isBackBlockFace(app, blockPos, faceIdx):
                skipNext = True
                continue

            if not blockFaceVisible(app, blockPos, faceIdx):
                skipNext = True
                continue

            light = blockFaceLight(app, blockPos, faceIdx)
            r = int(color[1:3], base=16)
            g = int(color[3:5], base=16)
            b = int(color[5:7], base=16)

            brightness = (light + 1) / 8
            r *= brightness
            g *= brightness
            b *= brightness

            '''
            avg = (r + g + b) / 3.0
            desaturation = min(vertices[face[0]][2] / 8.0, 1.0)**2
            r += (0.0 - r) * desaturation
            g += (128.0 - g) * desaturation
            b += (255.0 - b) * desaturation
            '''

            r = max(0.0, min(255.0, r))
            g = max(0.0, min(255.0, g))
            b = max(0.0, min(255.0, b))

            color = '#{:02X}{:02X}{:02X}'.format(int(r), int(g), int(b))
        else:
            # Backface culling (surprisingly expensive)
            backFace = isBackFace(
                vertices[face[0]], 
                vertices[face[1]],
                vertices[face[2]]
            )
            if backFace:
                continue

        for clippedFace in clip(app, vertices, face):
            faces.append([vertices, clippedFace, color])

    return faces

def blockPosIsVisible(app, pos: BlockPos):
    lookX = cos(app.cameraPitch)*sin(-app.cameraYaw)
    lookY = sin(app.cameraPitch)
    lookZ = cos(app.cameraPitch)*cos(-app.cameraYaw)

    [camX, camY, camZ] = app.cameraPos
    [blockX, blockY, blockZ] = blockToWorld(pos)

    # This is only a conservative estimate, so we move the camera "back"
    # to make sure we don't miss blocks behind us
    camX -= lookX
    camY -= lookY
    camZ -= lookZ

    dot = lookX * (blockX - camX) + lookY * (blockY - camY) + lookZ * (blockZ - camZ)

    return dot >= 0

def renderInstances(app, canvas):
    faces = drawToFaces(app)

    zCoord = lambda d: -(d[0][d[1][0]][2] + d[0][d[1][1]][2] + d[0][d[1][2]][2])
    
    faces.sort(key=zCoord)

    drawToCanvas(app, canvas, faces)

def drawToFaces(app):
    toCamMat = wsToCamMat(app.cameraPos, app.cameraYaw, app.cameraPitch)
    faces = []
    for chunkPos in app.chunks:
        for (blockPos, inst) in app.chunks[chunkPos].iterInstances():
            (inst, unburied) = inst
            if unburied and blockPosIsVisible(app, blockPos):
                (x, y, z) = blockPos
                x -= app.cameraPos[0]
                y -= app.cameraPos[1]
                z -= app.cameraPos[2]
                if x**2 + y**2 + z**2 <= app.renderDistance**2:
                    faces += cullInstance(app, toCamMat, inst, blockPos)
    return faces

def drawToCanvas(app, canvas, faces):
    mat = app.csToCanvasMat

    for i in range(len(faces)):
        if type(faces[i][0]) != type((0, 0)):
            verts = list(map(lambda v: toCartesian(mat @ v), faces[i][0]))
            faces[i][0] = (verts, True)

        ((vertices, _), face, color) = faces[i]

        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        if app.wireframe:
            edges = [(v0, v1), (v0, v2), (v1, v2)]

            for (v0, v1) in edges:            
                canvas.create_line(v0[0], v0[1], v1[0], v1[1], fill=color)
        else:
            canvas.create_polygon(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], fill=color)
            '''
            vs = [v0, v1, v2]

            if vs[1][1] <= vs[0][1] and vs[1][1] <= vs[2][1]:
                vs[0], vs[1] = vs[1], vs[0]
            elif vs[2][1] <= vs[0][1] and vs[2][1] <= vs[1][1]:
                vs[0], vs[2] = vs[2], vs[0]
            
            if vs[2][1] <= vs[1][1]:
                vs[1], vs[2] = vs[2], vs[1]

            yMin = int(vs[0][1])
            yMid = int(vs[1][1])
            yMax = int(vs[2][1])

            print(yMin, yMid, yMax)

            if yMin == yMid or yMid == yMax:
                continue

            tallSlope = (vs[2][0] - vs[0][0]) / (yMax - yMin)
            shortSlope1 = (vs[1][0] - vs[0][0]) / (yMid - yMin)
            shortSlope2 = (vs[2][0] - vs[1][0]) / (yMax - yMid)

            for y in range(yMin, yMax + 1):
                tallX = vs[0][0] + tallSlope * (y - yMin)

                if y < yMid:
                    shortX = vs[0][0] + shortSlope1 * (y - yMin)
                else:
                    shortX = vs[1][0] + shortSlope2 * (y - yMin)
                
                minX = int(min(tallX, shortX))
                maxX = int(max(tallX, shortX))

                #for x in range(minX, maxX + 1):
                canvas.create_rectangle(minX, y, maxX, y, outline=color)
            '''

frameTimes = [0.0] * 10
frameTimeIdx = 0

def redrawAll(app, canvas):
    startTime = time.time()

    canvas.create_rectangle(0.0, 0.0, app.width, app.height, fill='#0080FF')

    renderInstances(app, canvas)

    origin = wsToCanvas(app, np.array([[0.0], [0.0], [0.0]]))
    xAxis = wsToCanvas(app, np.array([[1.0], [0.0], [0.0]]))
    yAxis = wsToCanvas(app, np.array([[0.0], [1.0], [0.0]]))
    zAxis = wsToCanvas(app, np.array([[0.0], [0.0], [1.0]]))

    xpoint = wsToCamMat(app.cameraPos, app.cameraYaw, app.cameraPitch) @ toHomogenous(np.array([[1.0], [0.0], [0.0]]))
    xpoint = toCartesian(xpoint)
    # print(f"x point: {xpoint}")

    canvas.create_line(origin[0], origin[1], xAxis[0], xAxis[1], fill='red')
    canvas.create_line(origin[0], origin[1], yAxis[0], yAxis[1], fill='green')
    canvas.create_line(origin[0], origin[1], zAxis[0], zAxis[1], fill='blue')

    canvas.create_oval(app.width / 2 - 1, app.height / 2 - 1, 
        app.width / 2 + 1, app.height / 2 + 1)
    
    global frameTimes
    global frameTimeIdx

    endTime = time.time()
    frameTimes[frameTimeIdx] = (endTime - startTime)
    frameTimeIdx += 1
    frameTimeIdx %= len(frameTimes)
    frameTime = sum(frameTimes) / len(frameTimes) * 1000.0

    # This makes it more easily legible on both dark and light backgrounds
    canvas.create_text(11, 11, text=f'Frame Time: {frameTime:.2f}ms', anchor='nw')
    canvas.create_text(10, 10, text=f'Frame Time: {frameTime:.2f}ms', anchor='nw', fill='white')


# P'A = |PB| * |OA| / (|OB|) 

def main():
    runApp(width=600, height=400, mvcCheck=False)

if __name__ == '__main__':
    main()