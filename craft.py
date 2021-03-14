from cmu_112_graphics import *
from PIL import ImageDraw
from PIL.ImageDraw import Draw
import numpy as np
import math
import render
import world
from button import Button, ButtonManager
from world import Chunk, ChunkPos
from typing import List
from render import getCachedImage
from enum import Enum
from player import Player

# =========================================================================== #
# ----------------------------- THE APP ------------------------------------- #
# =========================================================================== #

# Author: Carson Swoveland (cswovela)
# Part of a term project for 15112

# I've incorporated Minecraft into every year of my education so far,
# and I don't plan to stop any time soon.

def createSizedBackground(app, width: int, height: int):
    cobble = app.loadImage('assets/CobbleBackground.png')
    cobble = app.scaleImage(cobble, 2)
    cWidth, cHeight = cobble.size

    newCobble = Image.new(cobble.mode, (width, height))
    for xIdx in range(math.ceil(width / cWidth)):
        for yIdx in range(math.ceil(height / cHeight)):
            xOffset = xIdx * cWidth
            yOffset = yIdx * cHeight

            newCobble.paste(cobble, (xOffset, yOffset))

    return newCobble

class Mode:
    finished: bool = False

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

        # FIXME: This does not work with resizing!
        self.buttons.addButton('playSurvival', Button(app.width / 2, app.height * 0.4, background=app.btnBg, text="Play Survival")) # type: ignore
        self.buttons.addButton('playCreative', Button(app.width / 2, app.height * 0.55, background=app.btnBg, text="Play Creative")) # type: ignore

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
    
def setMouseCapture(app, value: bool) -> None:
    """If True, locks the mouse to the center of the window and hides it.
    
    This is True when playing the game normally, and False
    in menus and GUIs.
    """

    app.captureMouse = value


class PlayingMode(Mode):
    lookedAtBlock = world.BlockPos(0, 0, 0)
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
        
        if self.mouseHeld and self.player.hotbarIdx == 0 and self.lookedAtBlock is not None:
            pos = self.lookedAtBlock[0]
            if self.player.creative:
                app.breakingBlockPos = pos
                app.breakingBlock = 1.0
            else:
                if app.breakingBlockPos == pos: 
                    app.breakingBlock += 0.1
                else:
                    app.breakingBlockPos = pos
                    app.breakingBlock = 0.0
            
            if app.breakingBlock > 1.0:
                app.breakingBlock = 1.0
            
            if app.breakingBlock == 1.0:
                brokenName = world.getBlock(app, pos)
                world.removeBlock(app, pos)
                self.player.pickUpItem(app, brokenName)
        else:
            app.breakingBlock = 0.0

        world.tick(app)

    def mousePressed(self, app, event):
        self.mouseHeld = True

        block = world.lookedAtBlock(app)
        if block is not None:
            (pos, face) = block
            if self.player.hotbarIdx != 0:
                slot = self.player.inventory[self.player.hotbarIdx]
                if slot.amount == 0: return
                
                if slot.amount > 0:
                    slot.amount -= 1

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

                world.addBlock(app, world.BlockPos(x, y, z), slot.item)
    
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
        elif event.key == 'Space':
            if hasattr(app.mode, 'player') and app.mode.player.onGround:
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
    app.renderDistanceSq = 11**2

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

    app.textures = {
        'grass': grassTexture,
        'stone': stoneTexture,
        'leaves': leavesTexture,
        'log': logTexture,
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

def mouseDragged(app, event):
    mouseMovedOrDragged(app, event)

def mouseMoved(app, event):
    mouseMovedOrDragged(app, event)

def mouseMovedOrDragged(app, event):
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