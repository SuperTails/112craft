"""This module is used to manage most of the game's assets.

Textures, block attributes, mob kinds, etc. are all loaded into the `app`.
This also creates the texture atlas used for rendering chunks.
"""

import numpy as np
import render
import config
import copy
import entity
import os
from client import CLIENT_DATA
from util import Color, BlockId, ItemId
import util
from sound import Sound
from shader import ShaderProgram
from PIL import Image
import typing
from typing import List, Optional, Tuple
from player import Stack
from OpenGL.GL import * #type:ignore
import json
import requests
import random
from nbt import nbt
from quarry.types.registry import LookupRegistry
from quarry.types import nbt as quarrynbt
from dimregistry import DimensionCodec


class Recipe:
    inputs: List[List[Optional[ItemId]]]
    outputs: Stack

    def __init__(self, grid: List[str], outputs: Stack, maps: dict[str, ItemId]):
        self.inputs = []
        for row in grid:
            newRow = []
            for col in row:
                if col == '-':
                    newRow.append(None)
                else:
                    newRow.append(maps[col])
            self.inputs.append(newRow)
        self.outputs = outputs

    def isCraftedBy(self, ingredients: List[List[Optional[ItemId]]]):
        dim = len(ingredients)

        rowOffset = 0

        for r in range(dim):
            if any(map(lambda c: c is not None, ingredients[r])):
                rowOffset = r
                break

        colOffset = 0

        for c in range(dim):
            if any(map(lambda r: r[c] is not None, ingredients)):
                colOffset = c
                break

        for rowIdx in range(dim):
            for colIdx in range(dim):
                if rowIdx + rowOffset >= dim or colIdx + colOffset >= dim:
                    ingr = None
                else:
                    ingr = ingredients[rowIdx + rowOffset][colIdx + colOffset]

                if self.inputs[rowIdx][colIdx] != ingr:
                    return False

        return True

def loadEntityRenderData(app):
    CLIENT_DATA.entityRenderData = {}

    for path, _, files in os.walk('assets/Vanilla_Resource_Pack_1.16.220/entity'):
        for file in files:
            data = entity.openRenderData(path + '/' + file)
            CLIENT_DATA.entityRenderData[data.identifier] = data

def loadEntityModels(app):
    CLIENT_DATA.entityModels = dict()

    for path, _, files in os.walk('assets/Vanilla_Resource_Pack_1.16.220/models'):
        for file in files:
            CLIENT_DATA.entityModels.update(entity.openModels(path + '/' + file, app))
    
    CLIENT_DATA.entityModels.update(entity.openModels('assets/block.geo.json', app))

    toMerge = []

    for name in CLIENT_DATA.entityModels:
        if ':' in name:
            toMerge.append(name)
    
    while len(toMerge) > 0:
        nameIdx = 0
        while nameIdx < len(toMerge):
            name = toMerge[nameIdx]
            name1, name2 = name.split(':')

            canMerge = True
            for i in range(len(toMerge)):
                if i != nameIdx and name2 == toMerge[i].split(':')[0]:
                    canMerge = False
                    break
                    
            if canMerge:
                override = CLIENT_DATA.entityModels.pop(name)
                CLIENT_DATA.entityModels[name1] = entity.merge(app, override, CLIENT_DATA.entityModels[name2])
                toMerge.pop(nameIdx)
            else:
                nameIdx += 1


    
def loadEntityAnimations(app):
    CLIENT_DATA.entityAnimations = dict()

    for path, _, files in os.walk('assets/Vanilla_Resource_Pack_1.16.220/animations'):
        for file in files:
            CLIENT_DATA.entityAnimations.update(entity.openAnimations(path + '/' + file))

def loadEntityAnimControllers(app):
    CLIENT_DATA.entityAnimControllers = {}

    for path, _, files in os.walk('assets/Vanilla_Resource_Pack_1.16.220/animation_controllers'):
        for file in files:
            CLIENT_DATA.entityAnimControllers.update(entity.openAnimControllers(path + '/' + file))

def loadEntityTextures(app):
    CLIENT_DATA.entityTextures = {}
    CLIENT_DATA.entityTextures['creeper'] = loadTexture('assets/creeper.png')
    CLIENT_DATA.entityTextures['fox'] = loadTexture('assets/fox.png')
    CLIENT_DATA.entityTextures['zombie'] = loadTexture('assets/Vanilla_Resource_Pack_1.16.220/textures/entity/zombie/zombie.png')
    CLIENT_DATA.entityTextures['skeleton'] = loadTexture('assets/Vanilla_Resource_Pack_1.16.220/textures/entity/skeleton/skeleton.png')
    CLIENT_DATA.entityTextures['player'] = loadTexture('assets/Vanilla_Resource_Pack_1.16.220/textures/entity/steve.png')

def imageToTexture(image: Image.Image, flip=True) -> int:
    if flip:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    texture = glGenTextures(1) #type:ignore
    glBindTexture(GL_TEXTURE_2D, texture)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    arr = np.asarray(image, dtype=np.uint8)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, arr) #type:ignore
    glGenerateMipmap(GL_TEXTURE_2D)

    glBindTexture(GL_TEXTURE_2D, 0)

    return texture

def loadTexture(path) -> int:
    im = Image.open(path).convert(mode='RGBA')
    return imageToTexture(im)

def loadSkyboxVao():
    # https://learnopengl.com/code_viewer_gh.php?code=src/4.advanced_opengl/6.1.cubemaps_skybox/cubemaps_skybox.cpp
    vertices = np.array([
        -1.0,  1.0, -1.0,
        -1.0, -1.0, -1.0,
         1.0, -1.0, -1.0,
         1.0, -1.0, -1.0,
         1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,

        -1.0, -1.0,  1.0,
        -1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0,  1.0,  1.0,
        -1.0, -1.0,  1.0,

         1.0, -1.0, -1.0,
         1.0, -1.0,  1.0,
         1.0,  1.0,  1.0,
         1.0,  1.0,  1.0,
         1.0,  1.0, -1.0,
         1.0, -1.0, -1.0,

        -1.0, -1.0,  1.0,
        -1.0,  1.0,  1.0,
         1.0,  1.0,  1.0,
         1.0,  1.0,  1.0,
         1.0, -1.0,  1.0,
        -1.0, -1.0,  1.0,

        -1.0,  1.0, -1.0,
         1.0,  1.0, -1.0,
         1.0,  1.0,  1.0,
         1.0,  1.0,  1.0,
        -1.0,  1.0,  1.0,
        -1.0,  1.0, -1.0,

        -1.0, -1.0, -1.0,
        -1.0, -1.0,  1.0,
         1.0, -1.0, -1.0,
         1.0, -1.0, -1.0,
        -1.0, -1.0,  1.0,
         1.0, -1.0,  1.0,
    ], dtype='float32')

    vao: int = glGenVertexArrays(1) #type:ignore
    vbo: int = glGenBuffers(1) #type:ignore

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    return vao

def loadSkyVao():
    vertices = np.array([
    # Back face
    -0.5, -0.5, -0.5,  0, 0, # Bottom-left
     0.5,  0.5, -0.5,  0, 0, # top-right
     0.5, -0.5, -0.5,  0, 0, # bottom-right         
     0.5,  0.5, -0.5,  0, 0, # top-right
    -0.5, -0.5, -0.5,  0, 0, # bottom-left
    -0.5,  0.5, -0.5,  0, 0, # top-left
    # Front face
    -0.5, -0.5,  0.5,  0, 0, # bottom-left
     0.5, -0.5,  0.5,  0, 0, # bottom-right
     0.5,  0.5,  0.5,  0, 0, # top-right
     0.5,  0.5,  0.5,  0, 0, # top-right
    -0.5,  0.5,  0.5,  0, 0, # top-left
    -0.5, -0.5,  0.5,  0, 0, # bottom-left
    # Left face
    -0.5,  0.5,  0.5,  1, 1/4, # top-right
    -0.5,  0.5, -0.5,  0, 1/4, # top-left
    -0.5, -0.5, -0.5,  0, 0/4, # bottom-left
    -0.5, -0.5, -0.5,  0, 0/4, # bottom-left
    -0.5, -0.5,  0.5,  1, 0/4, # bottom-right
    -0.5,  0.5,  0.5,  1, 1/4, # top-right
    # Right face
     0.5,  0.5,  0.5,  0, 2/4, # top-left
     0.5, -0.5, -0.5,  1, 3/4, # bottom-right
     0.5,  0.5, -0.5,  1, 2/4, # top-right         
     0.5, -0.5, -0.5,  1, 3/4, # bottom-right
     0.5,  0.5,  0.5,  0, 2/4, # top-left
     0.5, -0.5,  0.5,  0, 3/4, # bottom-left     
    # Bottom face
    -0.5, -0.5, -0.5,  1, 3/4, # top-right
     0.5, -0.5, -0.5,  0, 3/4, # top-left
     0.5, -0.5,  0.5,  0, 4/4, # bottom-left
     0.5, -0.5,  0.5,  0, 4/4, # bottom-left
    -0.5, -0.5,  0.5,  1, 4/4, # bottom-right
    -0.5, -0.5, -0.5,  1, 3/4, # top-right
    # Top face
    -0.5,  0.5, -0.5,  1, 1/4, # top-left
     0.5,  0.5,  0.5,  0, 2/4, # bottom-right
     0.5,  0.5, -0.5,  1, 2/4, # top-right     
     0.5,  0.5,  0.5,  0, 2/4, # bottom-right
    -0.5,  0.5, -0.5,  1, 1/4, # top-left
    -0.5,  0.5,  0.5,  0, 1/4, # bottom-left
    ], dtype='float32')

    vao: int = glGenVertexArrays(1) #type:ignore
    vbo: int = glGenBuffers(1) #type:ignore

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    #glBindBuffer(GL_ARRAY_BUFFER, buffer)

    #glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 1 * (4 * 4), ctypes.c_void_p(0 * (4 * 4)))
    #glEnableVertexAttribArray(2)
    #glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * (4 * 4), ctypes.c_void_p(1 * (4 * 4)))
    #glEnableVertexAttribArray(2)
    #glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * (4 * 4), ctypes.c_void_p(2 * (4 * 4)))
    #glEnableVertexAttribArray(2)
    #glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 4 * (4 * 4), ctypes.c_void_p(3 * (4 * 4)))
    #glEnableVertexAttribArray(2)

    #glVertexAttribDivisor(2, 1)
    #glVertexAttribDivisor(3, 1)
    #glVertexAttribDivisor(4, 1)
    #glVertexAttribDivisor(5, 1)

    #glBindBuffer(GL_ARRAY_BUFFER, 0)

    glBindVertexArray(0)

    return vao
 

def loadCubeVao():
    vertices = np.array([
    # Back face
    -0.5, -0.5, -0.5,  3/4, 1/3, # Bottom-left
     0.5,  0.5, -0.5,  4/4, 2/3, # top-right
     0.5, -0.5, -0.5,  4/4, 1/3, # bottom-right         
     0.5,  0.5, -0.5,  4/4, 2/3, # top-right
    -0.5, -0.5, -0.5,  3/4, 1/3, # bottom-left
    -0.5,  0.5, -0.5,  3/4, 2/3, # top-left
    # Front face
    -0.5, -0.5,  0.5,  3/4, 1/3, # bottom-left
     0.5, -0.5,  0.5,  4/4, 1/3, # bottom-right
     0.5,  0.5,  0.5,  4/4, 2/3, # top-right
     0.5,  0.5,  0.5,  4/4, 2/3, # top-right
    -0.5,  0.5,  0.5,  3/4, 2/3, # top-left
    -0.5, -0.5,  0.5,  3/4, 1/3, # bottom-left
    # Left face
    -0.5,  0.5,  0.5,  1/4, 2/3, # top-right
    -0.5,  0.5, -0.5,  0/4, 2/3, # top-left
    -0.5, -0.5, -0.5,  0/4, 1/3, # bottom-left
    -0.5, -0.5, -0.5,  0/4, 1/3, # bottom-left
    -0.5, -0.5,  0.5,  1/4, 1/3, # bottom-right
    -0.5,  0.5,  0.5,  1/4, 2/3, # top-right
    # Right face
     0.5,  0.5,  0.5,  2/4, 2/3, # top-left
     0.5, -0.5, -0.5,  3/4, 1/3, # bottom-right
     0.5,  0.5, -0.5,  3/4, 2/3, # top-right         
     0.5, -0.5, -0.5,  3/4, 1/3, # bottom-right
     0.5,  0.5,  0.5,  2/4, 2/3, # top-left
     0.5, -0.5,  0.5,  2/4, 1/3, # bottom-left     
    # Bottom face
    -0.5, -0.5, -0.5,  3/4, 1/3, # top-right
     0.5, -0.5, -0.5,  2/4, 1/3, # top-left
     0.5, -0.5,  0.5,  2/4, 0/3, # bottom-left
     0.5, -0.5,  0.5,  2/4, 0/3, # bottom-left
    -0.5, -0.5,  0.5,  3/4, 0/3, # bottom-right
    -0.5, -0.5, -0.5,  3/4, 1/3, # top-right
    # Top face
    -0.5,  0.5, -0.5,  2/4, 2/3, # top-left
     0.5,  0.5,  0.5,  1/4, 3/3, # bottom-right
     0.5,  0.5, -0.5,  2/4, 2/3, # top-right     
     0.5,  0.5,  0.5,  1/4, 3/3, # bottom-right
    -0.5,  0.5, -0.5,  2/4, 2/3, # top-left
    -0.5,  0.5,  0.5,  1/4, 3/3, # bottom-left
    ], dtype='float32')

    vao: int = glGenVertexArrays(1) #type:ignore
    vbo: int = glGenBuffers(1) #type:ignore

    buffer = glGenBuffers(1) #type:ignore

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    #glBindBuffer(GL_ARRAY_BUFFER, buffer)

    #glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 1 * (4 * 4), ctypes.c_void_p(0 * (4 * 4)))
    #glEnableVertexAttribArray(2)
    #glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * (4 * 4), ctypes.c_void_p(1 * (4 * 4)))
    #glEnableVertexAttribArray(2)
    #glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * (4 * 4), ctypes.c_void_p(2 * (4 * 4)))
    #glEnableVertexAttribArray(2)
    #glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 4 * (4 * 4), ctypes.c_void_p(3 * (4 * 4)))
    #glEnableVertexAttribArray(2)

    #glVertexAttribDivisor(2, 1)
    #glVertexAttribDivisor(3, 1)
    #glVertexAttribDivisor(4, 1)
    #glVertexAttribDivisor(5, 1)

    #glBindBuffer(GL_ARRAY_BUFFER, 0)

    glBindVertexArray(0)

    return vao, buffer
    
def loadTkTextures(app):
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

    app.tkTextures = {}

    for (blockId, sides) in app.texturePaths.items():
        tex = blockImageToTkTexture(loadBlockUVFromSides(app, **sides))
        app.tkTextures[blockId] = tex

    # Vertices in CCW order
    faces: List[render.Face] = [
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

    app.cube = render.Model(vertices, faces)

def getFaceColor(img: Image.Image, cornerX: int, cornerY: int, faceSide: bool) -> Color:
    avgR = 0
    avgG = 0
    avgB = 0

    amount = 0
    for cy in range(16):
        xRange = (0, 16 - cy) if faceSide else (cy, 16)
        for cx in range(xRange[0], xRange[1]):
            x = cornerX + cx
            y = cornerY + cy

            pixel = img.getpixel((x, y))[0:3] #type:ignore
            avgR += pixel[0]
            avgG += pixel[1]
            avgB += pixel[2]
            amount += 1
    
    avgR //= amount
    avgG //= amount
    avgB //= amount

    return f'#{avgR:02X}{avgG:02X}{avgB:02X}'

def blockImageToTkTexture(img: Image.Image):
    OFFSETS = (
        # Left
        (0, 16),
        # Right
        (32, 16),
        # Near
        (16, 16),
        # Far
        (48, 16),
        # Bottom
        (32, 16),
        # Top
        (16, 0),
    )

    tex = []

    for (cornerX, cornerY) in OFFSETS:
        for faceSide in (False, True):
            tex.append(getFaceColor(img, cornerX, cornerY, faceSide))

    return tex

def tint(im: Image.Image, color: Color) -> Image.Image:
    rTint = int(color[1:3], 16)
    gTint = int(color[3:5], 16)
    bTint = int(color[5:7], 16)

    def makeTint(c):
        def tint(p):
            return int(p * c / 0xFF)
        return tint
    
    r = im.getchannel('R').point(makeTint(rTint))
    g = im.getchannel('G').point(makeTint(gTint))
    b = im.getchannel('B').point(makeTint(bTint))
    a = im.getchannel('A')

    im = Image.merge('RGBA', (r, g, b, a))

    return im

class ResourcePack:
    root: str
    imageCache: dict[str, Image.Image]

    def __init__(self, root: str):
        self.root = root[:-1] if root.endswith('/') else root

        self.imageCache = {}
    
    def getImg(self, path: str) -> Image.Image:
        try:
            return self.imageCache[path]
        except KeyError:
            im = Image.open(self.root + '/' + path)
            if 'grass_top' in path or 'leaves' in path:
                #im = tint(im, '#79C05A')
                im = tint(im, '#7CBD6B')
            
            im = im.convert('RGBA')
            
            self.imageCache[path] = im
            return self.imageCache[path]
    
    def getItemTex(self, name: str) -> Image.Image:
        return self.getImg(f'textures/items/{name}.png')
    
    def getBlockTex(self, name: str) -> Image.Image:
        return self.getImg(f'textures/blocks/{name}.png')

FACE_NAMES = ['left', 'right', 'near', 'far', 'bottom', 'top']

def getFaceTextureList(**sides) -> List[str]:
    """
    Returns a list where each element is the texture that face will contain.
    """

    MAPPINGS = (
        ('all', ['left', 'right', 'near', 'far', 'bottom', 'top']),
        ('side', ['left', 'right', 'near', 'far']),
        ('south', ['near']),
        ('north', ['far']),
        ('west', ['left']),
        ('east', ['right']),
        ('up', ['top']),
        ('down', ['bottom']),
    )

    usages = [None] * 6

    for (side, faceNames) in MAPPINGS:
        if side in sides:
            texName = sides[side]
            sides.pop(side)

            for faceName in faceNames:
                usages[FACE_NAMES.index(faceName)] = texName

    if len(sides) != 0:
        raise Exception(f"Unknown sides {sides}")
    
    if not all(usages):
        raise Exception(f"Not all faces covered by {sides}")
    
    return typing.cast(List[str], usages)

def getFacesUsedForTexture(**sides) -> dict[str, set[str]]:
    """
    Returns a dict mapping texture names to the faces they are used on.
    This combines duplicates and removes sides overwritten by later textures.
    """
    
    usages = getFaceTextureList(**sides)

    result = {}

    for (faceName, usage) in zip(FACE_NAMES, usages):
        if usage not in result:
            result[usage] = set()
        
        result[usage].add(faceName)
    
    return result

def loadBlockUVFromSides(app, **sides) -> Image.Image:
    """
    Valid sides:
    * `all` - all sides of the block
    * `side` - All vertical sides of the block
    * `east` - ???
    * `west` - ???
    * `north` - ???
    * `south` - The front face
    * `up` - The top face
    * `down` - The bottom face
    """

    w = 16
    h = 16

    facesUsed = getFacesUsedForTexture(**sides)

    CORNERS = {
        'left': (0*w, 1*h),
        'right': (2*w, 1*h),
        'near': (1*w, 1*h),
        'far': (3*w, 1*h),
        'bottom': (2*w, 0*h),
        'top': (1*w, 0*h),
    }

    result = Image.new("RGBA", (w * 4, h * 2))

    for (texName, faceNames) in facesUsed.items():
        tex = app.rePack.getBlockTex(texName)
        for faceName in faceNames:
            result.paste(tex, CORNERS[faceName])
    
    return result

def loadBlockImage(path, tesselate=False):
    tex = Image.open(path)
    tex = tex.convert(mode='RGBA')
    if tesselate:
        newTex = Image.new("RGBA", (16 * 4, 16 * 3))
        for i in range(4):
            newTex.paste(tex, (i * 16, 16))
        newTex.paste(tex, (1 * 16, 0))
        newTex.paste(tex, (2 * 16, 2 * 16))
        tex = newTex
    return tex

def loadTextureAtlas(app):
    app.textureIdx = dict()
    app.textureIndices = dict()

    totalUnique = 0

    for (name, sides) in app.texturePaths.items():
        if name == 'redstone_wire':
            totalUnique += 16
        if name == 'redstone_torch':
            totalUnique += 2
        elif name == 'redstone_wall_torch':
            totalUnique += 2
        else:
            totalUnique += len(getFacesUsedForTexture(**sides))

    width = 16 * totalUnique

    atlas = Image.new("RGBA", (width, 16))

    idx = 0

    for (name, sides) in app.texturePaths.items():
        indices = getFaceTextureList(**sides)
        texNames = list(getFacesUsedForTexture(**sides).keys())

        app.textureIndices[name] = [idx + texNames.index(name) for name in indices]

        if name == 'redstone_wire':
            crossIm = app.rePack.getBlockTex('redstone_dust_cross')
            lineIm = app.rePack.getBlockTex('redstone_dust_line')
            for east in [False, True]:
                for north in [False, True]:
                    for south in [False, True]:
                        for west in [False, True]:

                            if east and west and not south and not north:
                                myIm: Image.Image = lineIm.copy()
                            elif north and south and not east and not west:
                                myIm: Image.Image = lineIm.copy()
                                myIm = myIm.rotate(90.0)
                            else:
                                myIm: Image.Image = crossIm.copy()

                                if not west:
                                    for x in range(0, 5):
                                        for y in range(0, 16):
                                            myIm.putpixel((x, y), 0)
                                if not east:
                                    for x in range(16 - 5, 16):
                                        for y in range(0, 16):
                                            myIm.putpixel((x, y), 0)
                                if not south:
                                    for x in range(0, 16):
                                        for y in range(0, 5):
                                            myIm.putpixel((x, y), 0)
                                if not north:
                                    for x in range(0, 16):
                                        for y in range(16 - 5, 16):
                                            myIm.putpixel((x, y), 0)

                            atlas.paste(myIm, (idx * 16, 0))
                            idx += 1
        elif name in ('redstone_torch', 'redstone_wall_torch'):
            for name in ('redstone_torch_on', 'redstone_torch_off'):
                im = app.rePack.getBlockTex(name)
                atlas.paste(im, (idx * 16, 0))
                idx += 1
        else:
            for name in texNames:
                im = app.rePack.getBlockTex(name)
                atlas.paste(im, (idx * 16, 0))
                idx += 1
    
    CLIENT_DATA.atlasWidth = width
        
    return imageToTexture(atlas)

def loadGlTextures(app):
    app.cubeVao, app.cubeBuffer = loadCubeVao()
    CLIENT_DATA.skyboxVao = loadSkyVao()

    CLIENT_DATA.glTextures = {}

    for (name, sides) in app.texturePaths.items():
        CLIENT_DATA.glTextures[name] = imageToTexture(loadBlockUVFromSides(app, **sides))

    CLIENT_DATA.breakTextures = []
    for i in range(10):
        CLIENT_DATA.breakTextures.append(loadTexture(f'assets/destroy_stage_{i}.png'))

    CLIENT_DATA.blockProgram = ShaderProgram('shaders/blockShader.vert', 'shaders/blockShader.frag')
    CLIENT_DATA.chunkProgram = ShaderProgram('shaders/chunkShader.vert', 'shaders/chunkShader.frag')
    CLIENT_DATA.guiProgram = ShaderProgram('shaders/guiShader.vert', 'shaders/guiShader.frag')
    CLIENT_DATA.entityProgram = ShaderProgram('shaders/entityShader.vert', 'shaders/entityShader.frag')
    CLIENT_DATA.skyProgram = ShaderProgram('shaders/skyShader.vert', 'shaders/skyShader.frag')
    CLIENT_DATA.transProgram = ShaderProgram('shaders/transShader.vert', 'shaders/transShader.frag')

    vertices = np.array([
        1.0,  1.0, 1.0, 0.0, # top right
        1.0, -1.0, 1.0, 1.0, # bottom right
        -1.0, -1.0, 0.0, 1.0, # bottom left
        -1.0,  1.0, 0.0, 0.0, # top left 
    ], dtype='float32')

    indices = np.array([
        0, 1, 3,
        1, 2, 3,
    ], dtype='uint32')

    vao: int = glGenVertexArrays(1) #type:ignore
    vbo: int = glGenBuffers(1) #type:ignore
    ebo: int = glGenBuffers(1) #type:ignore

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)

    glBindVertexArray(0)

    CLIENT_DATA.fullscreenVao = vao


#def createTk

#def createTkFace(im: Image.Image, offsetX: int, offsetY: int) -> List[int]:

def registerBlock(
    app,
    blockId: str,
    texturePath: Tuple[str, bool],
    itemTexturePath: Optional[str],
    hardness: Tuple[str, float],
    drops: dict[str, Optional[str]],
    opaque: bool = True,
    collides: bool = True,
    fuelValue: int = 0): 

    # TODO: INTEGRATE WITH `loadTkTextures`

    app.commonItemTextures[blockId] = itemTexturePath

    # TODO: Everything else
    
def getAttackDamage(app, item: str):
    if item == 'wooden_sword':
        return 4.5
    elif item == 'stone_sword':
        return 6.0
    elif item == 'iron_sword':
        return 6.5
    else:
        return 0.0
    
def getVersionJson():
    # From https://minecraft.fandom.com/wiki/Minecraft_Wiki

    URL = 'https://launchermeta.mojang.com/v1/packages/436877ffaef948954053e1a78a366b8b7c204a91/1.16.5.json'

    if not os.path.exists('assets/1.16.5.json'):
        data = requests.get(URL).text
        with open('assets/1.16.5.json', 'w') as f:
            f.write(data)
    
    with open('assets/1.16.5.json', 'r') as f:
        return json.load(f)

def getRegistry() -> LookupRegistry:
    if not os.path.exists('assets/jar/1.16.5.jar'):
        os.makedirs('assets/jar/', exist_ok=True)
        url = getVersionJson()['downloads']['server']['url']
        data = requests.get(url).content
        with open('assets/jar/1.16.5.jar', 'wb') as f:
            f.write(data)
        
    return LookupRegistry.from_jar('assets/jar/1.16.5.jar')

def getAssetIndex():
    versionJson = getVersionJson()
    url = versionJson['assetIndex']['url']

    if not os.path.exists('assets/1.16.assets.json'):
        data = requests.get(url).text
        with open('assets/1.16.assets.json', 'w') as f:
            f.write(data)
    
    with open('assets/1.16.assets.json', 'r') as f:
        return json.load(f)

def downloadSound(assetIndex, name: str):
    path = 'assets/sounds/' + name
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        hashStr = assetIndex['objects'][name]['hash']
        url = f'http://resources.download.minecraft.net/{hashStr[:2]}/{hashStr}'

        print(f"Downloading {name}")
    
        data = requests.get(url)
        with open(path, 'wb') as f:
            f.write(data.content)

def getSoundKind(app, blockId: BlockId):
    try:
        return app.soundKinds[blockId]
    except:
        print(f"Using fallback sound kind for {blockId}")
        return 'grass'
        
def getStepSound(app, blockId: BlockId):
    kind = getSoundKind(app, blockId)
    return random.choice(app.stepSounds[kind])

def getDigSound(app, blockId: BlockId):
    kind = getSoundKind(app, blockId)
    return random.choice(app.digSounds[kind])

def loadSoundArray(app, index, path: str) -> List[Sound]:
    sounds = []

    i = 1
    try:
        while i < 10:
            downloadSound(index, path.format(i))

            sounds.append(Sound('assets/sounds/' + path.format(i)))

            i += 1
    except KeyError:
        if i == 1:
            raise Exception(f"Likely invalid sound type {path}")

    return sounds

def loadSoundArray2(app, index, category: str) -> dict[str, List[Sound]]:
    result = {}

    for blockKind in ['grass', 'gravel', 'stone', 'wood']:
        result[blockKind] = loadSoundArray(app, index, f'minecraft/sounds/{category}/{blockKind}{{}}.ogg')
    
    return result


def loadSounds(app):
    index = getAssetIndex()

    app.stepSounds = {}
    app.digSounds = {}

    app.stepSounds = loadSoundArray2(app, index, 'step')
    app.digSounds = loadSoundArray2(app, index, 'dig')

    app.hurtSounds = {}
    
    app.hurtSounds['player'] = loadSoundArray(app, index, 'minecraft/sounds/damage/hit{}.ogg')
    app.hurtSounds['zombie'] = loadSoundArray(app, index, 'minecraft/sounds/mob/zombie/hurt{}.ogg')
    app.hurtSounds['creeper'] = loadSoundArray(app, index, 'minecraft/sounds/mob/creeper/say{}.ogg')
    app.hurtSounds['skeleton'] = loadSoundArray(app, index, 'minecraft/sounds/mob/skeleton/hurt{}.ogg')
    app.hurtSounds['fox'] = loadSoundArray(app, index, 'minecraft/sounds/mob/fox/hurt{}.ogg')

    app.soundKinds = {
        'grass': 'grass',
        'tnt': 'grass',
        'dirt': 'gravel',
        'cobblestone': 'stone',
        'stone': 'stone',
        'furnace': 'stone',
        'coal_ore': 'stone',
        'iron_ore': 'stone',
        'redstone_ore': 'stone',
        'diamond_ore': 'stone',
        'oak_log': 'wood',
        'oak_planks': 'wood',
        'torch': 'wood',
        'wall_torch': 'wood',
        'redstone_torch': 'wood',
        'redstone_wall_torch': 'wood',
        'redstone_wire': 'stone',
        'crafting_table': 'wood',
    }

def loadResources(app):
    app.rePack = ResourcePack('assets/Vanilla_Resource_Pack_1.16.220')

    loadSounds(app)

    app.furnaceRecipes = {
        'iron_ore': 'iron_ingot'
    }

    util.REGISTRY = getRegistry()

    # Default dimension codec from https://wiki.vg/
    nbtfile = quarrynbt.NBTFile.load('assets/default_codec.nbt')
    assert(nbtfile.root_tag is not None)
    util.DIMENSION_CODEC = DimensionCodec.fromNbt(nbtfile.root_tag)

    app.hardnesses = {}

    app.iconSheet = app.rePack.getImg('textures/gui/icons.png')

    CLIENT_DATA.guiTextures = {
        'heart_outline': app.iconSheet.crop((16, 0, 16 + 9, 9)),
        'heart_outline_hurt': app.iconSheet.crop((16 + 9, 0, 16 + 9*2, 9)),
        'heart': app.iconSheet.crop((16 + 9 * 4, 0, 16 + 9 * 5, 9)),
        'heart_half': app.iconSheet.crop((16 + 9 * 5, 0, 16 + 9 * 6, 9)),
    }

    app.texturePaths = {
        'grass': { 'down': 'dirt', 'up': 'grass_top', 'side': 'grass_side_carried' },
        'dirt': { 'all': 'dirt' },
        'stone': { 'all': 'stone' },
        'coal_ore': { 'all': 'coal_ore' },
        'iron_ore': { 'all': 'iron_ore' },
        'redstone_ore': { 'all': 'redstone_ore' },
        'diamond_ore': { 'all': 'diamond_ore' },
        'cobblestone': { 'all': 'cobblestone' },
        'oak_leaves': { 'all': 'leaves_oak_opaque' },
        'oak_log': { 'side': 'log_oak', 'up': 'log_oak_top', 'down': 'log_oak_top' },
        'bedrock': { 'all': 'bedrock' },
        'oak_planks': { 'all': 'planks_oak' },
        'crafting_table': {
            "down": 'planks_oak',
            "east": 'crafting_table_side',
            "north": 'crafting_table_front',
            "south": 'crafting_table_front',
            "up": 'crafting_table_top',
            "west": 'crafting_table_side',
        },
        'furnace': {
            'down' : 'furnace_top',
            'east' : 'furnace_side',
            'north' : 'furnace_side',
            'south' : 'furnace_front_off',
            'up' : 'furnace_top',
            'west' : 'furnace_side',
        },
        'obsidian': { 'all': 'obsidian' },
        'nether_portal': { 'all': 'portal_placeholder' },
        'glowstone': { 'all': 'glowstone' },
        'tnt': { 'side': 'tnt_side', 'down': 'tnt_bottom', 'up': 'tnt_top' },
        'torch': { 'all': 'torch_on' },
        'wall_torch': { 'all': 'torch_on' },
        'netherrack': { 'all': 'netherrack' },
        'water': { 'all': 'water_placeholder' },
        'flowing_water': { 'all': 'water_placeholder' },
        'lava': { 'all': 'lava_placeholder' },
        'flowing_lava': { 'all': 'lava_placeholder' },
        'redstone_wire': { 'all': 'redstone_dust_cross' },
        'redstone_torch': { 'all': 'redstone_torch_on' },
        'redstone_wall_torch': { 'all': 'redstone_torch_on' },
        'sand': { 'all': 'sand' },
        'sandstone': { 'up': 'sandstone_top', 'side': 'sandstone_normal', 'down': 'sandstone_bottom', },
    }

    app.hardnesses = {
        'grass': ('shovel', 1.0),
        'dirt': ('shovel', 0.8),
        'sand': ('shovel', 0.8),
        'stone': ('pickaxe', 5.0),
        'sandstone': ('pickaxe', 3.0),
        'furnace': ('pickaxe', 5.0),
        'glowstone': ('pickaxe', 2.0),
        'coal_ore': ('pickaxe', 6.0),
        'iron_ore': ('pickaxe', 6.0),
        'redstone_ore': ('pickaxe', 6.0),
        'diamond_ore': ('pickaxe', 6.0),
        'cobblestone': ('pickaxe', 6.0),
        'obsidian': ('pickaxe', 20.0),
        'netherrack': ('pickaxe', 0.5),
        'oak_leaves': (None, 0.5),
        'oak_log': ('axe', 2.0),
        'oak_planks': ('axe', 2.0),
        'crafting_table': ('axe', 2.0),
        'bedrock': (None, float('inf')),
        'nether_portal': (None, float('inf')),
        'tnt': (None, 0.1),
        'torch': (None, 0.1),
        'redstone_torch': (None, 0.1),
        'wall_torch': (None, 0.1),
        'redstone_wall_torch': (None, 0.1),
        'redstone_wire': (None, 0.1),
    }

    global HARDNESSES

    HARDNESSES = app.hardnesses

    app.blockDrops = {
        'grass': { '': 'dirt' },
        'dirt': { '': 'dirt' }, 
        'stone': { '': None, 'pickaxe': 'cobblestone' },
        'coal_ore': { '': None, 'pickaxe': 'coal' },
        'iron_ore': { '': None, 'pickaxe': 'iron_ore' },
        'redstone_ore': { '': None, 'pickaxe': 'redstone' },
        'diamond_ore': { '': None, 'pickaxe': 'diamond' },
        'cobblestone': { '': None, 'pickaxe': 'cobblestone' },
        'netherrack': { '': None, 'pickaxe': 'netherrack' },
        'obsidian': { '': None, 'pickaxe': 'obsidian' },
        'furnace': { 'pickaxe': 'furnace' },
        'oak_leaves': { '': None },
        'oak_log': { '': 'oak_log' },
        'oak_planks': { '': 'oak_planks' },
        'crafting_table': { '': 'crafting_table' },
        'bedrock': { '': None },
        'nether_portal': { '': None },
        'netherrack': { '': 'netherrack' },
        'glowstone': { '': 'glowstone' },
        'tnt': { '', 'tnt' },
        'torch': { '': 'torch' },
        'wall_torch': { '': 'torch' },
        'redstone_torch': { '': 'redstone_torch' },
        'redstone_wall_torch': { '': 'redstone_torch' },
        'redstone_wire': { '': 'redstone' },
        'sand': { '': 'sand' },
        'sandstone': { 'pickaxe': 'sandstone' },
    }

    loadTkTextures(app)
    if config.USE_OPENGL_BACKEND:
        loadGlTextures(app)
        app.textures = CLIENT_DATA.glTextures
        CLIENT_DATA.textureAtlas = loadTextureAtlas(app)
        loadEntityTextures(app)
        loadEntityAnimations(app)

        # https://learnopengl.com/Advanced-OpenGL/Framebuffers
        CLIENT_DATA.translucentFb = glGenFramebuffers(1) #type:ignore
        glBindFramebuffer(GL_FRAMEBUFFER, CLIENT_DATA.translucentFb)

        CLIENT_DATA.transColorTex = glGenTextures(1) #type:ignore
        glBindTexture(GL_TEXTURE_2D, CLIENT_DATA.transColorTex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, app.width, app.height, 0, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0)) #type:ignore
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, CLIENT_DATA.transColorTex, 0)

        CLIENT_DATA.transDepthTex = glGenTextures(1) #type:ignore
        glBindTexture(GL_TEXTURE_2D, CLIENT_DATA.transDepthTex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, app.width, app.height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, ctypes.c_void_p(0)) #type:ignore
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, CLIENT_DATA.transDepthTex, 0) #type:ignore

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        sunPath = 'assets/Vanilla_Resource_Pack_1.16.220/textures/environment/sun.png' 
        moonPath = 'assets/Vanilla_Resource_Pack_1.16.220/textures/environment/moon_phases.png'

        moonTex = Image.open(moonPath)

        tex = Image.open(sunPath)
        moonTex = moonTex.crop((0, 0, tex.width, tex.height))

        tex = tex.convert(mode='RGBA')
        newTex = Image.new("RGBA", (tex.width, tex.height * 4))
        newTex.paste(tex, (0, 1 * tex.height))
        newTex.paste(moonTex, (0, 3 * tex.height))

        CLIENT_DATA.sunTex = imageToTexture(newTex)

        #app.sunTex = loadTexture('assets/furnace.png')

        #app.sunTex = loadTexture(sunPath, True)

        im = loadBlockImage(sunPath, False)
        im = im.transpose(Image.FLIP_TOP_BOTTOM)

        app.sunCubeTex = glGenTextures(1) #type:ignore
        glBindTexture(GL_TEXTURE_CUBE_MAP, app.sunCubeTex)

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        arr = np.asarray(im, dtype=np.uint8)

        for i in range(6):
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA, im.width, im.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, arr) #type:ignore

    else:
        app.textures = app.tkTextures
        app.textureIndices = None
    
    loadEntityModels(app)
    loadEntityRenderData(app)
    loadEntityAnimControllers(app)
    
    entity.registerEntityKinds(app)

    app.recipes = [
        Recipe(
            [
                'l--',
                '---',
                '---'
            ],
            Stack('oak_planks', 4),
            { 'l': 'oak_log' }
        ),
        Recipe(
            [
                'p--',
                'p--',
                '---'
            ],
            Stack('stick', 4),
            { 'p': 'oak_planks' }
        ),
        Recipe(
            [
                'pp-',
                'pp-',
                '---'
            ],
            Stack('crafting_table', 1),
            { 'p': 'oak_planks' }
        ),
        Recipe(
            [
                'ppp',
                '-s-',
                '-s-',
            ],
            Stack('wooden_pickaxe', 1),
            { 'p': 'oak_planks', 's': 'stick' }
        ),
        Recipe(
            [
                'ccc',
                '-s-',
                '-s-',
            ],
            Stack('stone_pickaxe', 1),
            { 'c': 'cobblestone', 's': 'stick' }
        ),
        Recipe(
            [
                'pp-',
                'ps-',
                '-s-',
            ],
            Stack('wooden_axe', 1),
            { 'p': 'oak_planks', 's': 'stick' }
        ),
        Recipe(
            [
                'p--',
                's--',
                's--',
            ],
            Stack('wooden_shovel', 1),
            { 'p': 'oak_planks', 's': 'stick' }
        ),
        Recipe(
            [
                'ccc',
                'c-c',
                'ccc',
            ],
            Stack('furnace', 1),
            { 'c': 'cobblestone' }
        ),
        Recipe(
            [
                'c--',
                's--',
                '---',
            ],
            Stack('torch', 4),
            { 'c': 'coal', 's': 'stick' }
        ),
        Recipe(
            [
                'p--',
                'p--',
                's--',
            ],
            Stack('wooden_sword', 1),
            { 'p': 'oak_planks', 's': 'stick' }
        ),
        Recipe(
            [
                'p--',
                'p--',
                's--',
            ],
            Stack('stone_sword', 1),
            { 'p': 'cobblestone', 's': 'stick' }
        ),
        Recipe(
            [
                'p--',
                'p--',
                's--',
            ],
            Stack('iron_sword', 1),
            { 'p': 'iron_ingot', 's': 'stick' }
        ),
        Recipe(
            [
                'r--',
                's--',
                's--',
            ],
            Stack('redstone_torch', 1),
            { 'r': 'redstone', 's': 'stick' }
        )
    ]

    commonItemTextures = {
        'air': Image.open('assets/AirItem.png'),
        'stick': app.rePack.getItemTex('stick'),
        'coal': app.rePack.getItemTex('coal'),
        'iron_ingot': app.rePack.getItemTex('iron_ingot'),
        'diamond': app.rePack.getItemTex('diamond'),
        'redstone': app.rePack.getItemTex('redstone_dust'),
        'wooden_sword': app.rePack.getItemTex('wood_sword'),
        'wooden_pickaxe': app.rePack.getItemTex('wood_pickaxe'),
        'wooden_axe': app.rePack.getItemTex('wood_axe'),
        'wooden_shovel': app.rePack.getItemTex('wood_shovel'),
        'stone_sword': app.rePack.getItemTex('stone_sword'),
        'stone_pickaxe': app.rePack.getItemTex('stone_pickaxe'),
        'stone_shovel': app.rePack.getItemTex('stone_shovel'),
        'iron_sword': app.rePack.getItemTex('iron_sword'),
        'iron_pickaxe': app.rePack.getItemTex('iron_pickaxe'),
        'iron_shovel': app.rePack.getItemTex('iron_shovel'),
        'torch': Image.open('assets/Vanilla_Resource_Pack_1.16.220/textures/blocks/torch_on.png'),
        'bucket': app.rePack.getItemTex('bucket_empty'),
        'lava_bucket': app.rePack.getItemTex('bucket_lava'),
        'water_bucket': app.rePack.getItemTex('bucket_water'),
        'flint_and_steel': app.rePack.getItemTex('flint_and_steel'),
    }

    app.tkItemTextures = copy.copy(commonItemTextures)
    app.glItemTextures = commonItemTextures

    for (name, _) in app.textures.items():
        if name in commonItemTextures:
            continue

        if config.USE_OPENGL_BACKEND:
            sides = app.texturePaths[name]
            newGlTex = render.drawItemFromBlock2(25, loadBlockUVFromSides(app, **sides))
            app.glItemTextures[name] = newGlTex
        else:
            newTkTex = render.drawItemFromBlock(25, app.tkTextures[name])
            app.tkItemTextures[name] = newTkTex
    
    if config.USE_OPENGL_BACKEND:
        CLIENT_DATA.itemTextures = app.glItemTextures
    else:
        CLIENT_DATA.itemTextures = app.tkItemTextures
    
def getBlockDrop(app, block: BlockId, tool: ItemId) -> ItemId:
    drops = app.blockDrops[block]

    pickaxes = { 'wooden_pickaxe': 0.5, 'stone_pickaxe': 0.25 }
    axes = { 'wooden_axe': 0.5 }
    shovels = { 'wooden_shovel': 0.5 }

    if tool in pickaxes:
        toolKind = 'pickaxe'
    elif tool in axes:
        toolKind = 'axe'
    elif tool in shovels:
        toolKind = 'shovel'
    else:
        toolKind = ''
    
    if toolKind in drops:
        return drops[toolKind]
    else:
        return drops['']

HARDNESSES = {}

def getHardnessAgainst(block: BlockId, tool: ItemId) -> float:
    (goodTool, base) = HARDNESSES[block]

    pickaxes = { 'wooden_pickaxe': 0.5, 'stone_pickaxe': 0.25 }
    axes = { 'wooden_axe': 0.5 }
    shovels = { 'wooden_shovel': 0.5 }

    if goodTool == 'pickaxe' and tool in pickaxes:
        return base * pickaxes[tool]
    elif goodTool == 'axe' and tool in axes:
        return base * axes[tool]
    elif goodTool == 'shovel' and tool in shovels:
        return base * shovels[tool]
    else:
        return base
