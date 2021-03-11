from cmu_112_graphics import *
import numpy as np
import math
import render
import world
from world import Chunk, ChunkPos
from typing import List

# =========================================================================== #
# ----------------------------- THE APP ------------------------------------- #
# =========================================================================== #

# Author: Carson Swoveland (cswovela)
# Part of a term project for 15112

# I've incorporated Minecraft into every year of my education so far,
# and I don't plan to stop any time soon.


# Initializes all the data needed to run 112craft
def appStarted(app):
    loadResources(app)

    # ---------------
    # World variables
    # ---------------
    app.chunks = {
        ChunkPos(0, 0, 0): Chunk(ChunkPos(0, 0, 0))
    }

    app.chunks[ChunkPos(0, 0, 0)].generate(app)

    app.timerDelay = 30

    app.tickTimes = [0.0] * 10
    app.tickTimeIdx = 0

    # ----------------
    # Player variables
    # ----------------
    app.playerHeight = 1.5
    app.playerWidth = 0.6
    app.playerRadius = app.playerWidth / 2
    app.playerOnGround = False
    app.playerVel = [0.0, 0.0, 0.0]
    app.playerWalkSpeed = 0.2
    app.playerReach = 4.0
    app.selectedBlock = 'air'
    app.gravity = 0.10

    app.cameraYaw = 0
    app.cameraPitch = 0
    app.cameraPos = [4.0, 14.0 + app.playerHeight, 4.0]

    # -------------------
    # Rendering Variables
    # -------------------
    app.vpDist = 0.25
    app.vpWidth = 3.0 / 4.0
    app.vpHeight = app.vpWidth * app.height / app.width 
    app.wireframe = False
    app.renderDistanceSq = 6**2

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

    app.captureMouse = False


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

def sizeChanged(app):
    app.csToCanvasMat = render.csToCanvasMat(app.vpDist, app.vpWidth, app.vpHeight,
                        app.width, app.height)

def mousePressed(app, event):
    block = world.lookedAtBlock(app)
    if block is not None:
        (pos, face) = block
        if app.selectedBlock == 'air':
            world.removeBlock(app, pos)
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
            
            world.addBlock(app, world.BlockPos(x, y, z), app.selectedBlock)

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

def timerFired(app):
    world.tick(app)

def keyPressed(app, event):
    if event.key == '1':
        app.selectedBlock = 'air'
    elif event.key == '2':
        app.selectedBlock = 'grass'
    elif event.key == '3':
        app.selectedBlock = 'stone'
    elif event.key == '4':
        app.selectedBlock = 'leaves'
    elif event.key == '5':
        app.selectedBlock = 'log'
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

def keyReleased(app, event):
    if event.key == 'w':
        app.w = False
    elif event.key == 's':
        app.s = False 
    elif event.key == 'a':
        app.a = False
    elif event.key == 'd':
        app.d = False

def redrawAll(app, canvas):
    render.redrawAll(app, canvas)

# =========================================================================== #
# ------------------------- IDK WHAT TO NAME THIS --------------------------- #
# =========================================================================== #

# P'A = |PB| * |OA| / (|OB|) 

def main():
    runApp(width=600, height=400, mvcCheck=False)

if __name__ == '__main__':
    main()