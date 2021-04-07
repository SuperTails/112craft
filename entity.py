from typing import List, Tuple
import json
import numpy as np
from dataclasses import dataclass
from OpenGL.GL import * #type:ignore

EntityKind = str

CUBE_MESH_VERTICES = np.array([
    # Left face
    -0.5,  0.5,  0.5,  1/4, 1/2, # top-right
    -0.5,  0.5, -0.5,  0/4, 1/2, # top-left
    -0.5, -0.5, -0.5,  0/4, 0/2, # bottom-left
    -0.5, -0.5, -0.5,  0/4, 0/2, # bottom-left
    -0.5, -0.5,  0.5,  1/4, 0/2, # bottom-right
    -0.5,  0.5,  0.5,  1/4, 1/2, # top-right
    # Right face
     0.5,  0.5,  0.5,  2/4, 1/2, # top-left
     0.5, -0.5, -0.5,  3/4, 0/2, # bottom-right
     0.5,  0.5, -0.5,  3/4, 1/2, # top-right         
     0.5, -0.5, -0.5,  3/4, 0/2, # bottom-right
     0.5,  0.5,  0.5,  2/4, 1/2, # top-left
     0.5, -0.5,  0.5,  2/4, 0/2, # bottom-left     
    # Back face
    -0.5, -0.5, -0.5,  3/4, 0/2, # Bottom-left
     0.5,  0.5, -0.5,  4/4, 1/2, # top-right
     0.5, -0.5, -0.5,  4/4, 0/2, # bottom-right         
     0.5,  0.5, -0.5,  4/4, 1/2, # top-right
    -0.5, -0.5, -0.5,  3/4, 0/2, # bottom-left
    -0.5,  0.5, -0.5,  3/4, 1/2, # top-left
    # Front face
    -0.5, -0.5,  0.5,  1/4, 0/2, # bottom-left
     0.5, -0.5,  0.5,  2/4, 0/2, # bottom-right
     0.5,  0.5,  0.5,  2/4, 1/2, # top-right
     0.5,  0.5,  0.5,  2/4, 1/2, # top-right
    -0.5,  0.5,  0.5,  1/4, 1/2, # top-left
    -0.5, -0.5,  0.5,  1/4, 0/2, # bottom-left
    # Bottom face
    -0.5, -0.5, -0.5,  3/4, 2/2, # top-right
     0.5, -0.5, -0.5,  2/4, 2/2, # top-left
     0.5, -0.5,  0.5,  2/4, 1/2, # bottom-left
     0.5, -0.5,  0.5,  2/4, 1/2, # bottom-left
    -0.5, -0.5,  0.5,  3/4, 1/2, # bottom-right
    -0.5, -0.5, -0.5,  3/4, 2/2, # top-right
    # Top face
    -0.5,  0.5, -0.5,  1/4, 2/2, # top-left
     0.5,  0.5,  0.5,  2/4, 1/2, # bottom-right
     0.5,  0.5, -0.5,  2/4, 2/2, # top-right     
     0.5,  0.5,  0.5,  2/4, 1/2, # bottom-right
    -0.5,  0.5, -0.5,  1/4, 2/2, # top-left
    -0.5,  0.5,  0.5,  1/4, 1/2, # bottom-left
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
            u = result[row, 3]
            if u == 0/4:
                u = 0
            elif u == 1/4:
                u = self.size[2]
            elif u == 2/4:
                u = self.size[0] + self.size[2]
            elif u == 3/4:
                u = 2*self.size[0] + self.size[2]
            else:
                u = 2*(self.size[0] + self.size[2])
            
            v = result[row, 4]
            if v == 0/2:
                v = 0
            elif v == 1/2:
                v = self.size[1]
            else:
                v = self.size[1] + self.size[0]

            result[row, 3] = u + self.uv[0]
            result[row, 4] = v + (-(self.size[1] + self.size[0]) + (32.0 - self.uv[1]))
        
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

class Entity:
    pos: List[float]
    velocity: List[float]
    kind: EntityKind
    radius: float
    onGround: bool

    def __init__(self, kind: EntityKind, x: float, y: float, z: float):
        self.kind = kind
        self.pos = [x, y, z]
        self.velocity = [0.0, 0.0, 0.0]
        self.radius = 0.3
        self.height = 1.5
        self.onGround = False