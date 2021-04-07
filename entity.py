from os import X_OK
from typing import List, Tuple, Optional
import json
import numpy as np
import heapq
import math
from util import BlockPos
from dataclasses import dataclass
from OpenGL.GL import * #type:ignore

EntityKind = str

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

@dataclass
class Cube:
    origin: Tuple[float, float, float]
    size: Tuple[int, int, int]
    uv: Tuple[int, int]

    def toVertices(self) -> np.ndarray:
        vertices = CUBE_MESH_VERTICES.reshape((-1, 5))

        factor = np.array([self.size[0], self.size[1], self.size[2], 1.0, 1.0])

        offset = np.array([self.origin[0], self.origin[1], self.origin[2], 0.0, 0.0])

        result = (vertices + np.array([0.5, 0.5, 0.5, 0.0, 0.0])) * factor + offset

        for row in range(vertices.shape[0]):
            face = ['left', 'right', 'back', 'front', 'bottom', 'top'][row // 6]

            if face == 'left':
                uOffset = 0
                uSize = self.size[2]

                vOffset = 0
                vSize = self.size[1]
            elif face == 'front':
                uOffset = self.size[2]
                uSize = self.size[0]

                vOffset = 0
                vSize = self.size[1]
            elif face == 'top':
                uOffset = self.size[2]
                uSize = self.size[0]

                vOffset = self.size[1]
                vSize = self.size[2]
            elif face == 'right':
                uOffset = self.size[2] + self.size[0]
                uSize = self.size[2]

                vOffset = 0
                vSize = self.size[1]
            elif face == 'back':
                uOffset = 2*self.size[2] + self.size[0]
                uSize = self.size[0]

                vOffset = 0
                vSize = self.size[1]
            else: # face == 'bottom'
                uOffset = self.size[2] + self.size[0]
                uSize = self.size[0]

                vOffset = self.size[1]
                vSize = self.size[2]

            uFrac = result[row, 3]
            vFrac = result[row, 4]

            result[row, 3] = self.uv[0] + uOffset + uFrac * uSize
            result[row, 4] = (32.0 - (self.uv[1] + self.size[1] + self.size[2])) + vOffset + vFrac * vSize
        
        return result

@dataclass
class Bone:
    name: str
    pivot: Tuple[float, float, float]
    cubes: List[Cube]

    def toVertices(self) -> np.ndarray:
        self.innerVertices = []
        for cube in self.cubes:
            self.innerVertices.append(cube.toVertices())
        return np.vstack(self.innerVertices)

    def toVao(self) -> Tuple[int, int]:
        #if self.meshVaos[meshIdx] != 0:
        #    glDeleteVertexArrays(1, np.array([self.meshVaos[meshIdx]])) #type:ignore
        #    glDeleteBuffers(1, np.array([self.meshVbos[meshIdx]])) #type:ignore
        #
        #    self.meshVaos[meshIdx] = 0

        vertices = self.toVertices().flatten().astype('float32')

        print(len(vertices))
        print(vertices.nbytes / len(vertices))

        print(vertices)

        vao: int = glGenVertexArrays(1) #type:ignore
        vbo: int = glGenBuffers(1) #type:ignore

        glBindVertexArray(vao)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(0)

        return (vao, len(vertices) // 5)


@dataclass
class EntityModel:
    bones: List[Bone]
    vaos: List[Tuple[int, int]]

    def __init__(self, bones):
        self.bones = bones
        self.vaos = []
        for bone in self.bones:
            if bone.cubes != []:
                self.vaos.append(bone.toVao())

def parseCube(j) -> Cube:
    origin = tuple(j['origin'])
    size = tuple(j['size'])
    uv = tuple(j['uv'])
    return Cube(origin, size, uv) #type:ignore
    
def parseBone(j) -> Bone:
    name = j['name']
    if 'pivot' in j:
        pivot = tuple(j['pivot'])
    else:
        #FIXME: ???
        pivot = (0.0, 0.0, 0.0)
    if 'cubes' in j:
        cubes = list(map(parseCube, j['cubes']))
    else:
        cubes = []
    return Bone(name, pivot, cubes) #type:ignore

def parseModel(j) -> EntityModel:
    bones = list(map(parseBone, j['bones']))
    return EntityModel(bones)

def openModels(path) -> dict[str, EntityModel]:
    with open(path) as f:
        j = json.load(f)
    
    result = {}
    
    for (name, model) in j.items():
        if name.startswith('geometry.'):
            name = name.removeprefix('geometry.')
            result[name] = parseModel(model)
    
    return result

#print(models['fox'].bones[3].cubes[0].toVertices())

class Ai:
    target: BlockPos
    path: List[BlockPos]

    def __init__(self, target: BlockPos):
        self.target = target
        self.path = []

class Entity:
    pos: List[float]
    velocity: List[float]
    kind: EntityKind
    radius: float
    onGround: bool
    walkSpeed: float

    path: List[BlockPos]

    def __init__(self, kind: EntityKind, x: float, y: float, z: float):
        self.kind = kind
        self.pos = [x, y, z]
        self.velocity = [0.0, 0.0, 0.0]
        self.radius = 0.3
        self.height = 1.5
        self.onGround = False
        self.walkSpeed = 0.05

        self.path = []
    
    def tick(self, world):
        if len(self.path) > 0:
            x = self.path[0][0] - self.pos[0]
            z = self.path[0][2] - self.pos[2]

            mag = math.sqrt(x**2 + z**2)

            if mag < 0.5:
                self.path.pop(0)
            else:
                x /= mag
                z /= mag

                x *= self.walkSpeed
                z *= self.walkSpeed

                self.velocity[0] = x
                self.velocity[2] = z
        else:
            self.velocity[0] = 0.0
            self.velocity[2] = 0.0
    
    def updatePath(self, world, target: BlockPos):
        start = BlockPos(round(self.pos[0]), round(self.pos[1] + 0.01), round(self.pos[2]))

        if self.path == []:
            print(f"Finding path from {start} to {target}")

            path = findPath(start, target, world)
            if path is not None:
                self.path = path
                print(self.path)


def makePathFromChain(prevDirs, end: BlockPos) -> List[BlockPos]:
    result = []

    cur = end
    while cur in prevDirs:
        result.append(cur)
        cur = prevDirs[cur]

    result.append(cur)
    
    result.reverse()

    return result

def findPath(start: BlockPos, end: BlockPos, world) -> Optional[List[BlockPos]]:
    def heuristic(start: BlockPos, end: BlockPos):
        # Chebyshev distance

        xDist = abs(start.x - end.x)
        yDist = abs(start.y - end.y)
        zDist = abs(start.z - end.z)

        return (max(xDist, zDist) + yDist)

    # https://en.wikipedia.org/wiki/A*_search_algorithm

    prevDirs = dict()

    realCosts = { start: 0 }

    costs = { start: heuristic(start, end) }

    openSet = { start }

    MAX_RANGE = 10
    
    while len(openSet) != 0:
        minCost = None
        current = None
        for pos in openSet:
            if minCost is None or costs[pos] < minCost:
                current = pos
                minCost = costs[pos]
        assert(current is not None)

        print(f"Trying {current}")

        openSet.remove(current)
        
        if current == end:
            return makePathFromChain(prevDirs, end)

        for nextPos in destinations(current, world):
            if heuristic(nextPos, end) > MAX_RANGE:
                continue

            newRealCost = realCosts[current] + 1
            if nextPos not in realCosts or realCosts[nextPos] > newRealCost:
                prevDirs[nextPos] = current
                realCosts[nextPos] = newRealCost
                costs[nextPos] = realCosts[current] + heuristic(nextPos, end)
                if nextPos not in openSet:
                    openSet.add(nextPos)
    
    return None

def destinations(start: BlockPos, world):
    for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        pos = BlockPos(start.x + dx, start.y, start.z + dz)
        if world.coordsOccupied(pos): continue

        yield pos
    
    for (dx, dz) in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        pos = BlockPos(start.x + dx, start.y, start.z + dz)
        if world.coordsOccupied(pos): continue

        corner1 = BlockPos(start.x + dx, start.y, start.z)
        corner2 = BlockPos(start.x, start.y, start.z + dz)

        if world.coordsOccupied(corner1): continue
        if world.coordsOccupied(corner2): continue

        yield pos
    