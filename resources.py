import numpy as np
import world
import render
import config
import copy
import entity
from sound import Sound
from shader import ShaderProgram
from PIL import Image
from typing import List, Optional, Tuple
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

def loadEntityModels(app):
    app.entityModels = dict()
    app.entityModels.update(entity.openModels('assets/creeper.geo.json'))
    app.entityModels.update(entity.openModels('assets/fox.geo.json'))

def loadEntityAnimations(app):
    app.entityAnimations = dict()
    app.entityAnimations.update(entity.openAnimations('assets/creeper.animation.json'))
    app.entityAnimations.update(entity.openAnimations('assets/fox.animation.json'))

def loadEntityTextures(app):
    app.entityTextures = {}
    app.entityTextures['creeper'] = loadTexture('assets/creeper.png')
    app.entityTextures['fox'] = loadTexture('assets/fox.png')

def imageToTexture(image: Image.Image) -> int:
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

def loadTexture(path: str, tesselate=False) -> int:
    im = loadBlockImage(path, tesselate)
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    return imageToTexture(im)

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
    
    dirtTexture = [
        '#FF0000', '#FF0000',
        '#A52A2A', '#A52A2A',
        '#A52A2A', '#A52A2A',
        '#A52A2A', '#A52A2A',
        '#A52A2A', '#A52A2A',
        '#a52A2A', '#A52A2A']
    
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
        'dirt': dirtTexture,
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


    totalAmt = 0
    for (_, (_, tess)) in app.texturePaths.items():
        if tess:
            totalAmt += 1
        else:
            totalAmt += 2

    width = 16 * totalAmt

    atlas = Image.new("RGBA", (width, 16))

    idx = 0

    for (name, (path, tess)) in app.texturePaths.items():
        app.textureIndices[name] = [0] * 6

        im = Image.open(path)

        if tess:
            atlas.paste(im, (idx * 16, 0))
            app.textureIndices[name] = [idx] * 6
            idx += 1
        else:
            top = im.copy()
            top = top.crop((16, 0, 32, 16))

            atlas.paste(top, (idx * 16, 0))
            # FIXME:
            app.textureIndices[name][4] = idx
            app.textureIndices[name][5] = idx
            idx += 1
            
            side = im.copy()
            side = side.crop((0, 16, 16, 32))

            atlas.paste(side, (idx * 16, 0))
            app.textureIndices[name][0] = idx
            app.textureIndices[name][1] = idx
            app.textureIndices[name][2] = idx
            app.textureIndices[name][3] = idx
            idx += 1
    
    atlas = atlas.transpose(Image.FLIP_TOP_BOTTOM)
    
    app.atlasWidth = width
        
    return imageToTexture(atlas)

def loadGlTextures(app):
    app.cubeVao, app.cubeBuffer = loadCubeVao()

    app.glTextures = dict()

    for (name, (path, tess)) in app.texturePaths.items():
        app.glTextures[name] = loadTexture(path, tesselate=tess)

    app.breakTextures = []
    for i in range(10):
        app.breakTextures.append(loadTexture(f'assets/destroy_stage_{i}.png', tesselate=True))

    app.blockProgram = ShaderProgram('shaders/blockShader.vert', 'shaders/blockShader.frag')

    app.chunkProgram = ShaderProgram('shaders/chunkShader.vert', 'shaders/chunkShader.frag')
    
    app.guiProgram = ShaderProgram('shaders/guiShader.vert', 'shaders/guiShader.frag')

    app.entityProgram = ShaderProgram('shaders/entityShader.vert', 'shaders/entityShader.frag')

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

    app.texturePaths[blockId] = texturePath
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

def loadResources(app):
    app.sounds = {
        'grass': Sound('assets/grass.ogg'),
        'destroy_grass': Sound('assets/destroy_grass.ogg')
    }

    app.furnaceRecipes = {
        'iron_ore': 'iron_ingot'
    }

    app.texturePaths = {}
    app.hardnesses = {}

    app.texturePaths = {
        'grass': ('assets/grass.png', False),
        'dirt': ('assets/dirt.png', True),
        'stone': ('assets/stone.png', True),
        'coal_ore': ('assets/coal_ore.png', True),
        'iron_ore': ('assets/iron_ore.png', True),
        'cobblestone': ('assets/cobblestone.png', True),
        'leaves': ('assets/leaves.png', False),
        'log': ('assets/log.png', False),
        'bedrock': ('assets/bedrock.png', True),
        'planks': ('assets/oak_planks.png', True),
        'crafting_table': ('assets/crafting_table.png', False),
        'furnace': ('assets/furnace.png', False),
        'glowstone': ('assets/Vanilla_Resource_Pack_1.16.220/textures/blocks/glowstone.png', True),
        'torch':     ('assets/Vanilla_Resource_Pack_1.16.220/textures/blocks/torch_on.png', True),
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
        app.textures = app.glTextures
        app.textureAtlas = loadTextureAtlas(app)
        loadEntityModels(app)
        loadEntityTextures(app)
        loadEntityAnimations(app)
        entity.registerEntityKinds(app)
    else:
        app.textures = app.tkTextures
        app.textureIndices = None

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
            { 'c': 'cobblestone', 's': 'stick' }
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
        ),
        Recipe(
            [
                'ccc',
                'c-c',
                'ccc',
            ],
            Slot('furnace', 1),
            { 'c': 'cobblestone' }
        ),
        Recipe(
            [
                'c--',
                's--',
                '---',
            ],
            Slot('torch', 4),
            { 'c': 'coal', 's': 'stick' }
        ),
        Recipe(
            [
                'p--',
                'p--',
                's--',
            ],
            Slot('wooden_sword', 1),
            { 'p': 'planks', 's': 'stick' }
        ),
        Recipe(
            [
                'p--',
                'p--',
                's--',
            ],
            Slot('stone_sword', 1),
            { 'p': 'cobblestone', 's': 'stick' }
        ),
        Recipe(
            [
                'p--',
                'p--',
                's--',
            ],
            Slot('iron_sword', 1),
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
            (path, tess) = app.texturePaths[name]
            newGlTex = render.drawItemFromBlock2(25, loadBlockImage(path, tesselate=tess))
            app.glItemTextures[name] = newGlTex
        else:
            newTkTex = render.drawItemFromBlock(25, app.tkTextures[name])
            app.tkItemTextures[name] = newTkTex
    
    if config.USE_OPENGL_BACKEND:
        app.itemTextures = app.glItemTextures
    else:
        app.itemTextures = app.tkItemTextures

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
