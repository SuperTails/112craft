import numpy as np
import world
import render
import config
import copy
from shader import ShaderProgram
from PIL import Image
from typing import List, Optional
from player import Slot
from OpenGL.GL import * #type:ignore

class Recipe:
    inputs: List[List[Optional[world.ItemId]]]
    outputs: Slot

    def __init__(self, grid: List[str], outputs: Slot, maps: dict[str, world.ItemId]):
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

def loadTexture(path: str, tesselate=False) -> int:
    texture = glGenTextures(1) #type:ignore
    glBindTexture(GL_TEXTURE_2D, texture)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    grassTex = Image.open(path)
    grassTex = grassTex.convert(mode='RGBA')
    if tesselate:
        newTex = Image.new("RGBA", (16 * 4, 16 * 3))
        for i in range(4):
            newTex.paste(grassTex, (i * 16, 16))
        newTex.paste(grassTex, (1 * 16, 0))
        newTex.paste(grassTex, (2 * 16, 2 * 16))
        grassTex = newTex
    grassTex = grassTex.transpose(Image.FLIP_TOP_BOTTOM)
    grassArr = np.asarray(grassTex, dtype=np.uint8)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, grassTex.width, grassTex.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, grassArr) #type:ignore
    glGenerateMipmap(GL_TEXTURE_2D)

    glBindTexture(GL_TEXTURE_2D, 0)

    return texture

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
    -0.5,  0.5, -0.5,  1/4, 3/3, # top-left
     0.5,  0.5,  0.5,  2/4, 2/3, # bottom-right
     0.5,  0.5, -0.5,  2/4, 3/3, # top-right     
     0.5,  0.5,  0.5,  2/4, 2/3, # bottom-right
    -0.5,  0.5, -0.5,  1/4, 3/3, # top-left
    -0.5,  0.5,  0.5,  1/4, 2/3, # bottom-left
    ], dtype='float32')

    vao: int = glGenVertexArrays(1) #type:ignore
    vbo: int = glGenBuffers(1) #type:ignore

    buffer = glGenBuffers(1) #type:ignore

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, buffer)

    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 1 * (4 * 4), ctypes.c_void_p(0 * (4 * 4)))
    glEnableVertexAttribArray(2)
    #glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * (4 * 4), ctypes.c_void_p(1 * (4 * 4)))
    #glEnableVertexAttribArray(2)
    #glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * (4 * 4), ctypes.c_void_p(2 * (4 * 4)))
    #glEnableVertexAttribArray(2)
    #glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 4 * (4 * 4), ctypes.c_void_p(3 * (4 * 4)))
    #glEnableVertexAttribArray(2)

    glVertexAttribDivisor(2, 1)
    #glVertexAttribDivisor(3, 1)
    #glVertexAttribDivisor(4, 1)
    #glVertexAttribDivisor(5, 1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)

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

    leavesTexture = [
        '#206000', '#256505',
        '#257000', '#256505',
        '#206010', '#206505',
        '#206505', '#256005',
        '#306005', '#256500',
        '#206500', '#306505',
    ]

    logTexture = [
        '#705020', '#655020',
        '#705520', '#655025',
        '#705025', '#705020',
        '#755020', '#705A2A',
        '#755520', '#7A4A20',
        '#705525', '#70502A',
    ]

    bedrockTexture = [
        '#0A0A10', '#0E0A10',
        '#0A1010', '#0A0A10',
        '#0A0A18', '#0E1010',
        '#100A10', '#080A10',
        '#0A0810', '#0A0A18',
        '#0A0A1E', '#100A10',
    ]

    planksTexture = [
        '#BE9A60', '#B4915D',
        '#AC8C53', '#9C814B',
        '#937240', '#7B6036',
        '#7B6036', '#654E2B', 
        '#9C814B', '#BE9A60',
        '#B4915D', '#AC8C53'
    ]

    craftingTableTexture = [
        '#A36F45', '#443C34',
        '#715836', '#727274',
        '#482E18', '#888173',
        '#534423', '#B7B5B2',
        '#AB673C', '#71381B',
        '#B4915D', '#AC8C53'
    ]

    app.tkTextures = {
        'grass': grassTexture,
        'stone': stoneTexture,
        'leaves': leavesTexture,
        'log': logTexture,
        'bedrock': bedrockTexture,
        'planks': planksTexture,
        'crafting_table': craftingTableTexture,
    }

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


def loadGlTextures(app):
    app.cubeVao, app.cubeBuffer = loadCubeVao()

    app.glTextures = {
        'grass': loadTexture('assets/grass.png'),
        'stone': loadTexture('assets/missing.png'),
        'leaves': loadTexture('assets/leaves.png'),
        'log': loadTexture('assets/log.png'),
        'bedrock': loadTexture('assets/missing.png'),
        'planks': loadTexture('assets/missing.png'),
        'crafting_table': loadTexture('assets/missing.png'),
    }
    
    app.breakTextures = []
    for i in range(10):
        app.breakTextures.append(loadTexture(f'assets/destroy_stage_{i}.png', tesselate=True))

    app.blockProgram = ShaderProgram('shaders/blockShader.vert', 'shaders/blockShader.frag')
    
    app.guiProgram = ShaderProgram('shaders/guiShader.vert', 'shaders/guiShader.frag')

def loadResources(app):
    loadTkTextures(app)
    if config.USE_OPENGL_BACKEND:
        loadGlTextures(app)
        app.textures = app.glTextures
    else:
        app.textures = app.tkTextures

    app.hardnesses = {
        'grass': ('shovel', 1.0),
        'stone': ('pickaxe', 5.0),
        'leaves': (None, 0.5),
        'log': ('axe', 2.0),
        'planks': ('axe', 2.0),
        'crafting_table': ('axe', 2.0),
        'bedrock': (None, float('inf')),
    }

    app.recipes = [
        Recipe(
            [
                'l--',
                '---',
                '---'
            ],
            Slot('planks', 4),
            { 'l': 'log' }
        ),
        Recipe(
            [
                'p--',
                'p--',
                '---'
            ],
            Slot('stick', 4),
            { 'p': 'planks' }
        ),
        Recipe(
            [
                'pp-',
                'pp-',
                '---'
            ],
            Slot('crafting_table', 1),
            { 'p': 'planks' }
        ),
        Recipe(
            [
                'ppp',
                '-s-',
                '-s-',
            ],
            Slot('wooden_pickaxe', 1),
            { 'p': 'planks', 's': 'stick' }
        ),
        Recipe(
            [
                'ccc',
                '-s-',
                '-s-',
            ],
            Slot('stone_pickaxe', 1),
            { 'c': 'stone', 's': 'stick' }
        ),
        Recipe(
            [
                'pp-',
                'ps-',
                '-s-',
            ],
            Slot('wooden_axe', 1),
            { 'p': 'planks', 's': 'stick' }
        ),
        Recipe(
            [
                'p--',
                's--',
                's--',
            ],
            Slot('wooden_shovel', 1),
            { 'p': 'planks', 's': 'stick' }
        )
    ]

    commonItemTextures = {
        'air': Image.open('assets/AirItem.png'),
        'stick': Image.open('assets/Stick.png'),
        'wooden_pickaxe': Image.open('assets/WoodenPickaxe.png'),
        'stone_pickaxe': Image.open('assets/StonePickaxe.png'),
        'wooden_axe': Image.open('assets/WoodenAxe.png'),
        'wooden_shovel': Image.open('assets/WoodenShovel.png'),
    }

    app.tkItemTextures = copy.copy(commonItemTextures)
    app.glItemTextures = commonItemTextures

    textureNames = {
        'grass': ('assets/grass.png'),
        'stone': ('assets/missing.png'),
        'leaves': ('assets/leaves.png'),
        'log': ('assets/log.png'),
        'bedrock': ('assets/missing.png'),
        'planks': ('assets/missing.png'),
        'crafting_table': ('assets/missing.png'),
    }

    for (name, _) in app.textures.items():
        if config.USE_OPENGL_BACKEND:
            newGlTex = render.drawItemFromBlock2(25, Image.open(textureNames[name]))
            app.glItemTextures[name] = newGlTex
        else:
            newTkTex = render.drawItemFromBlock(25, app.tkTextures[name])
            app.tkItemTextures[name] = newTkTex
    
    if config.USE_OPENGL_BACKEND:
        app.itemTextures = app.glItemTextures
    else:
        app.itemTextures = app.tkItemTextures

def getHardnessAgainst(app, block: world.BlockId, tool: world.ItemId) -> float:
    (goodTool, base) = app.hardnesses[block]

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