import numpy as np
import world
import render
from typing import List, Optional
from player import Slot

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


def loadResources(app):
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

    app.textures = {
        'grass': grassTexture,
        'stone': stoneTexture,
        'leaves': leavesTexture,
        'log': logTexture,
        'bedrock': bedrockTexture,
        'planks': planksTexture,
        'crafting_table': craftingTableTexture,
    }

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

    app.itemTextures = {
        'air': app.loadImage('assets/AirItem.png'),
        'stick': app.loadImage('assets/Stick.png'),
        'wooden_pickaxe': app.loadImage('assets/WoodenPickaxe.png'),
        'stone_pickaxe': app.loadImage('assets/StonePickaxe.png'),
        'wooden_axe': app.loadImage('assets/WoodenAxe.png'),
        'wooden_shovel': app.loadImage('assets/WoodenShovel.png'),
    }

    for (name, tex) in app.textures.items():
        newTex = render.drawItemFromBlock(25, tex)
        app.itemTextures[name] = newTex

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
