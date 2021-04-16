"""This module is used to manage most of the game's assets.

Textures, block attributes, mob kinds, etc. are all loaded into the `app`.
This also creates the texture atlas used for rendering chunks.
"""

import numpy as np
import world
import render
import config
import copy
import entity
import os
from client import CLIENT_DATA
from util import Color, BlockId
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
from quarry.types.registry import LookupRegistry

class Recipe:
    inputs: List[List[Optional[world.ItemId]]]
    outputs: Stack

    def __init__(self, grid: List[str], outputs: Stack, maps: dict[str, world.ItemId]):
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

    def isCraftedBy(self, ingredients: List[List[Optional[world.ItemId]]]):
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

def loadEntityModels(app):
    CLIENT_DATA.entityModels = dict()

    for path, _, files in os.walk('assets/Vanilla_Resource_Pack_1.16.220/models'):
        for file in files:
            CLIENT_DATA.entityModels.update(entity.openModels(path + '/' + file, app))
    
    CLIENT_DATA.entityModels.update(entity.openModels('assets/block.geo.json', app))
    
def loadEntityAnimations(app):
    CLIENT_DATA.entityAnimations = dict()

    for path, _, files in os.walk('assets/Vanilla_Resource_Pack_1.16.220/animations'):
        for file in files:
            if 'bee' in file:
                # Who in their right mind puts COMMENTS in a JSON file????
                continue
            CLIENT_DATA.entityAnimations.update(entity.openAnimations(path + '/' + file))

def loadEntityTextures(app):
    CLIENT_DATA.entityTextures = {}
    CLIENT_DATA.entityTextures['creeper'] = loadTexture('assets/creeper.png')
    CLIENT_DATA.entityTextures['fox'] = loadTexture('assets/fox.png')
    CLIENT_DATA.entityTextures['zombie'] = loadTexture('assets/Vanilla_Resource_Pack_1.16.220/textures/entity/zombie/zombie.png')
    CLIENT_DATA.entityTextures['skeleton'] = loadTexture('assets/Vanilla_Resource_Pack_1.16.220/textures/entity/skeleton/skeleton.png')

    # TODO:
    #CLIENT_DATA.entityTextures['player'] = loadTexture('assets/Vanilla_Resource_Pack_1.16.220/textures/entity/steve.png')
    CLIENT_DATA.entityTextures['player'] = loadTexture('assets/Vanilla_Resource_Pack_1.16.220/textures/entity/zombie/zombie.png')

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
                im = tint(im, '#79C05A')

            self.imageCache[path] = im
            return self.imageCache[path]
    
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

    for (_, sides) in app.texturePaths.items():
        totalUnique += len(getFacesUsedForTexture(**sides))

    width = 16 * totalUnique

    atlas = Image.new("RGBA", (width, 16))

    idx = 0

    for (name, sides) in app.texturePaths.items():
        indices = getFaceTextureList(**sides)
        texNames = list(getFacesUsedForTexture(**sides).keys())

        app.textureIndices[name] = [idx + texNames.index(name) for name in indices]

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
        'dirt': 'gravel',
        'cobblestone': 'stone',
        'stone': 'stone',
        'furnace': 's tone',
        'coal_ore': 'stone',
        'iron_ore': 'stone',
        'log': 'wood',
        'planks': 'wood',
        'torch': 'wood',
        'crafting_table': 'wood',
    }

def loadResources(app):
    app.rePack = ResourcePack('assets/Vanilla_Resource_Pack_1.16.220')
    
    loadSounds(app)

    app.furnaceRecipes = {
        'iron_ore': 'iron_ingot'
    }

    app.hardnesses = {}

    app.texturePaths = {
        'grass': { 'down': 'dirt', 'up': 'grass_top', 'side': 'grass_side_carried' },
        'dirt': { 'all': 'dirt' },
        'stone': { 'all': 'stone' },
        'coal_ore': { 'all': 'coal_ore' },
        'iron_ore': { 'all': 'iron_ore' },
        'cobblestone': { 'all': 'cobblestone' },
        'leaves': { 'all': 'leaves_oak_opaque' },
        'log': { 'side': 'log_oak', 'up': 'log_oak_top', 'down': 'log_oak_top' },
        'bedrock': { 'all': 'bedrock' },
        'planks': { 'all': 'planks_oak' },
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
        'glowstone': { 'all': 'glowstone' },
        'torch':     { 'all': 'torch_on' },
    }

    app.hardnesses = {
        'grass': ('shovel', 1.0),
        'dirt': ('shovel', 0.8),
        'stone': ('pickaxe', 5.0),
        'furnace': ('pickaxe', 5.0),
        'glowstone': ('pickaxe', 2.0),
        'coal_ore': ('pickaxe', 6.0),
        'iron_ore': ('pickaxe', 6.0),
        'cobblestone': ('pickaxe', 6.0),
        'leaves': (None, 0.5),
        'log': ('axe', 2.0),
        'planks': ('axe', 2.0),
        'crafting_table': ('axe', 2.0),
        'bedrock': (None, float('inf')),
        'torch': (None, 0.1),
    }

    global HARDNESSES

    HARDNESSES = app.hardnesses

    app.blockDrops = {
        'grass': { '': 'dirt' },
        'dirt': { '': 'dirt' }, 
        'stone': { '': None, 'pickaxe': 'cobblestone' },
        'coal_ore': { '': None, 'pickaxe': 'coal' },
        'iron_ore': { '': None, 'pickaxe': 'iron_ore' },
        'cobblestone': { '': 'cobblestone' },
        'furnace': { 'pickaxe': 'furnace' },
        'leaves': { '': None },
        'log': { '': 'log' },
        'planks': { '': 'planks' },
        'crafting_table': { '': 'crafting_table' },
        'bedrock': { '': None },
        'glowstone': { '': 'glowstone' },
        'torch': { '': 'torch' },
    }

    loadTkTextures(app)
    if config.USE_OPENGL_BACKEND:
        loadGlTextures(app)
        app.textures = CLIENT_DATA.glTextures
        CLIENT_DATA.textureAtlas = loadTextureAtlas(app)
        loadEntityTextures(app)
        loadEntityAnimations(app)

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
    
    entity.registerEntityKinds(app)

    app.recipes = [
        Recipe(
            [
                'l--',
                '---',
                '---'
            ],
            Stack('planks', 4),
            { 'l': 'log' }
        ),
        Recipe(
            [
                'p--',
                'p--',
                '---'
            ],
            Stack('stick', 4),
            { 'p': 'planks' }
        ),
        Recipe(
            [
                'pp-',
                'pp-',
                '---'
            ],
            Stack('crafting_table', 1),
            { 'p': 'planks' }
        ),
        Recipe(
            [
                'ppp',
                '-s-',
                '-s-',
            ],
            Stack('wooden_pickaxe', 1),
            { 'p': 'planks', 's': 'stick' }
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
            { 'p': 'planks', 's': 'stick' }
        ),
        Recipe(
            [
                'p--',
                's--',
                's--',
            ],
            Stack('wooden_shovel', 1),
            { 'p': 'planks', 's': 'stick' }
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
            { 'p': 'planks', 's': 'stick' }
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
    ]

    commonItemTextures = {
        'air': Image.open('assets/AirItem.png'),
        'stick': Image.open('assets/Stick.png'),
        'coal': Image.open('assets/coal.png'),
        'wooden_pickaxe': Image.open('assets/WoodenPickaxe.png'),
        'stone_pickaxe': Image.open('assets/StonePickaxe.png'),
        'wooden_axe': Image.open('assets/WoodenAxe.png'),
        'wooden_shovel': Image.open('assets/WoodenShovel.png'),
        'iron_sword': Image.open('assets/Vanilla_Resource_Pack_1.16.220/textures/items/iron_sword.png'),
        'stone_sword': Image.open('assets/Vanilla_Resource_Pack_1.16.220/textures/items/stone_sword.png'),
        'wooden_sword': Image.open('assets/Vanilla_Resource_Pack_1.16.220/textures/items/wood_sword.png'),
        'iron_ingot': Image.open('assets/Vanilla_Resource_Pack_1.16.220/textures/items/iron_ingot.png'),
        'torch': Image.open('assets/Vanilla_Resource_Pack_1.16.220/textures/blocks/torch_on.png'),
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
    
def getBlockDrop(app, block: world.BlockId, tool: world.ItemId) -> world.ItemId:
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

def getHardnessAgainst(block: world.BlockId, tool: world.ItemId) -> float:
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
