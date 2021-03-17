from cmu_112_graphics import Image, runApp
from PIL import ImageDraw
from PIL.ImageDraw import Draw
import numpy as np
import math
import render
import world
from button import Button, ButtonManager, createSizedBackground
from world import Chunk, ChunkPos
from typing import List, Optional
from render import getCachedImage
from enum import Enum
from player import Player, Slot

# =========================================================================== #
# ----------------------------- THE APP ------------------------------------- #
# =========================================================================== #

# Author: Carson Swoveland (cswovela)
# Part of a term project for 15112

# I've incorporated Minecraft into every year of my education so far,
# and I don't plan to stop any time soon.

class Mode:
    """Represents some state of the game that can accept events.

    For example, the title screen is a mode, a loading screen is a mode,
    the actual gameplay is a mode, etc.
    """

    def __init__(self): pass

    def mousePressed(self, app, event): pass
    def mouseReleased(self, app, event): pass
    def timerFired(self, app): pass
    def sizeChanged(self, app): pass
    def redrawAll(self, app, canvas): pass
    def keyPressed(self, app, event): pass
    def keyReleased(self, app, event): pass

class StartupMode(Mode):
    loadStage: int = 0

    def __init__(self, app):
        loadResources(app)

        app.timerDelay = 10

        # TODO: Fix
        app.worldSeed = 40

        app.chunks = {
            ChunkPos(0, 0, 0): Chunk(ChunkPos(0, 0, 0))
        }

        app.chunks[ChunkPos(0, 0, 0)].generate(app, app.worldSeed)
    
    def timerFired(self, app):
        if self.loadStage < 20:
            world.loadUnloadChunks(app, [0.0, 0.0, 0.0])
        elif self.loadStage < 30:
            world.tickChunks(app)
        else:
            app.mode = TitleMode(app)
            
        self.loadStage += 1
    
    def redrawAll(self, app, canvas):
        leftX = app.width * 0.25
        rightX = app.width * 0.75

        height = 20

        canvas.create_rectangle(leftX, app.height / 2 - height, rightX, app.height / 2 + height)

        progress = self.loadStage / 30.0

        midX = leftX + (rightX - leftX) * progress

        canvas.create_rectangle(leftX, app.height / 2 - height, midX, app.height / 2 + height, fill='red')

class TitleMode(Mode):
    buttons: ButtonManager
    titleText: Image

    def __init__(self, app):
        self.buttons = ButtonManager()

        self.titleText = app.loadImage('assets/TitleText.png')
        self.titleText = app.scaleImage(self.titleText, 3)

        survivalButton = Button(app, 0.5, 0.4, app.btnBg, "Play Survival")
        creativeButton = Button(app, 0.5, 0.55, app.btnBg, "Play Creative")

        self.buttons.addButton('playSurvival', survivalButton) # type: ignore
        self.buttons.addButton('playCreative', creativeButton) # type: ignore

    def timerFired(self, app):
        app.cameraYaw += 0.01

    def mousePressed(self, app, event):
        self.buttons.onPress(event.x, event.y)
    
    def mouseReleased(self, app, event):
        btn = self.buttons.onRelease(event.x, event.y)
        if btn is not None:
            print(f"Pressed {btn}")
            if btn == 'playSurvival':
                app.mode = PlayingMode(app, False)
            elif btn == 'playCreative':
                app.mode = PlayingMode(app, True)

    def redrawAll(self, app, canvas):
        render.redrawAll(app, canvas, doDrawHud=False)

        canvas.create_image(app.width / 2, 50, image=getCachedImage(self.titleText))
        
        self.buttons.draw(app, canvas)
    
    def sizeChanged(self, app):
        self.buttons.canvasSizeChanged(app)
    
def setMouseCapture(app, value: bool) -> None:
    """If True, locks the mouse to the center of the window and hides it.
    
    This is True when playing the game normally, and False
    in menus and GUIs.
    """

    app.captureMouse = value

class PlayingMode(Mode):
    lookedAtBlock = None
    mouseHeld: bool = False

    player: Player

    def __init__(self, app, creative: bool):
        app.timerDelay = 30
        setMouseCapture(app, True)

        self.player = Player(app, creative)

    def redrawAll(self, app, canvas):
        render.redrawAll(app, canvas)
    
    def timerFired(self, app):
        self.lookedAtBlock = world.lookedAtBlock(app)

        updateBlockBreaking(app, self)

        world.tick(app)

    def mousePressed(self, app, event):
        self.mouseHeld = True

        block = world.lookedAtBlock(app)
        if block is not None:
            (pos, face) = block
            if self.player.hotbarIdx != 0:
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
                pos = world.BlockPos(x, y, z)

                if not world.coordsInBounds(app, pos): return

                slot = self.player.inventory[self.player.hotbarIdx]
                if slot.amount == 0: return
                
                if slot.amount > 0:
                    slot.amount -= 1

                world.addBlock(app, pos, slot.item)
    
    def mouseReleased(self, app, event):
        self.mouseHeld = False
    
    def keyPressed(self, app, event):
        if len(event.key) == 1 and event.key.isdigit():
            keyNum = int(event.key)
            if keyNum != 0:
                self.player.hotbarIdx = keyNum - 1
        elif event.key == 'w':
            app.w = True
        elif event.key == 's':
            app.s = True
        elif event.key == 'a':
            app.a = True
        elif event.key == 'd':
            app.d = True
        elif event.key == 'e':
            app.mode = InventoryMode(app, app.mode)
        elif event.key == 'Space':
            if self.player.onGround:
                app.mode.player.velocity[1] = 0.35
        elif event.key == 'Escape':
            setMouseCapture(app, not app.captureMouse)

    def keyReleased(self, app, event):
        if event.key == 'w':
            app.w = False
        elif event.key == 's':
            app.s = False 
        elif event.key == 'a':
            app.a = False
        elif event.key == 'd':
            app.d = False

class InventoryMode(Mode):
    submode: Mode
    heldItem: Slot = Slot('', 0)

    def __init__(self, app, submode: Mode):
        setMouseCapture(app, False)
        self.submode = submode

    def redrawAll(self, app, canvas):
        self.submode.redrawAll(app, canvas)

        render.drawMainInventory(app, canvas)

        if app.mousePos is not None:
            render.drawSlot(app, canvas, app.mousePos[0], app.mousePos[1],
                self.heldItem, drawBackground=False)

    
    def mousePressed(self, app, event):
        clickedSlot = None
        for i in range(36):
            (x, y, w) = render.getSlotCenterAndSize(app, i)
            x0, y0 = x - w/2, y - w/2
            x1, y1 = x + w/2, y + w/2
            if x0 < event.x and event.x < x1 and y0 < event.y and event.y < y1:
                clickedSlot = i
        
        if clickedSlot is not None and clickedSlot != 0:
            player = self.submode.player
            self.heldItem, player.inventory[clickedSlot] = player.inventory[clickedSlot], self.heldItem
            
    
    def keyPressed(self, app, event):
        if event.key == 'e':
            app.mode = self.submode
            setMouseCapture(app, True)

# Initializes all the data needed to run 112craft
def appStarted(app):
    app.mode = StartupMode(app)

    app.btnBg = createSizedBackground(app, 200, 40)

    app.tickTimes = [0.0] * 10
    app.tickTimeIdx = 0

    # ----------------
    # Player variables
    # ----------------
    app.breakingBlock = 1.0
    app.breakingBlockPos = world.BlockPos(0, 0, 0)

    app.gravity = 0.10

    app.cameraYaw = 0
    app.cameraPitch = 0
    app.cameraPos = [2.0, 9.5, 4.0]

    # -------------------
    # Rendering Variables
    # -------------------
    app.vpDist = 0.25
    app.vpWidth = 3.0 / 4.0
    app.vpHeight = app.vpWidth * app.height / app.width 
    app.wireframe = False
    app.renderDistanceSq = 10**2

    app.horizFov = math.atan(app.vpWidth / app.vpDist)
    app.vertFov = math.atan(app.vpHeight / app.vpDist)

    print(f"Horizontal FOV: {app.horizFov} ({math.degrees(app.horizFov)}°)")
    print(f"Vertical FOV: {app.vertFov} ({math.degrees(app.vertFov)}°)")

    app.csToCanvasMat = render.csToCanvasMat(app.vpDist, app.vpWidth,
                        app.vpHeight, app.width, app.height)

    # ---------------
    # Input Variables
    # ---------------
    app.mouseMovedDelay = 10

    app.w = False
    app.s = False
    app.a = False
    app.d = False

    app.prevMouse = None

    setMouseCapture(app, False)

def updateBlockBreaking(app, mode: PlayingMode):
    if mode.mouseHeld and mode.player.hotbarIdx == 0 and mode.lookedAtBlock is not None:
        pos = mode.lookedAtBlock[0]
        if mode.player.creative:
            app.breakingBlockPos = pos
            app.breakingBlock = 1000.0
        else:
            if app.breakingBlockPos == pos: 
                app.breakingBlock += 0.1
            else:
                app.breakingBlockPos = pos
                app.breakingBlock = 0.0
        
        blockId = world.getBlock(app, pos)
        hardness = app.hardnesses[blockId]       

        if app.breakingBlock >= hardness:
            brokenName = world.getBlock(app, pos)
            world.removeBlock(app, pos)
            mode.player.pickUpItem(app, brokenName)
    else:
        app.breakingBlock = 0.0


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

    app.textures = {
        'grass': grassTexture,
        'stone': stoneTexture,
        'leaves': leavesTexture,
        'log': logTexture,
        'bedrock': bedrockTexture,
        'planks': planksTexture,
    }

    app.hardnesses = {
        'grass': 1.0,
        'stone': 5.0,
        'leaves': 0.5,
        'log': 2.0,
        'planks': 2.0,
        'bedrock': float('inf'),
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

    app.itemTextures = {
        'air': app.loadImage('assets/AirItem.png')
    }

    for (name, tex) in app.textures.items():
        newTex = render.drawItemFromBlock(25, tex)
        app.itemTextures[name] = newTex


def keyPressed(app, event):
    app.mode.keyPressed(app, event)

def keyReleased(app, event):
    app.mode.keyReleased(app, event)

def mousePressed(app, event):
    app.mode.mousePressed(app, event)

def mouseReleased(app, event):
    app.mode.mouseReleased(app, event)

def timerFired(app):
    app.mode.timerFired(app)

def sizeChanged(app):
    app.csToCanvasMat = render.csToCanvasMat(app.vpDist, app.vpWidth, app.vpHeight,
                        app.width, app.height)
    
    app.mode.sizeChanged(app)

def mouseDragged(app, event):
    mouseMovedOrDragged(app, event)

def mouseMoved(app, event):
    mouseMovedOrDragged(app, event)

def mouseMovedOrDragged(app, event):
    if not app.captureMouse:
        app.prevMouse = None
        app.mousePos = (event.x, event.y)
    else:
        app.mousePos = None

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


def redrawAll(app, canvas):
    if app.captureMouse:
        canvas.configure(cursor='none')
    else:
        canvas.configure(cursor='arrow')

    app.mode.redrawAll(app, canvas)


def main():
    runApp(width=600, height=400, mvcCheck=True)

if __name__ == '__main__':
    main()