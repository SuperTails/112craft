from openglapp import runApp
from PIL import Image
from PIL import ImageDraw
from PIL.ImageDraw import Draw
import numpy as np
import math
import render
import world
import copy
import glfw
from button import Button, ButtonManager, createSizedBackground
from world import Chunk, ChunkPos
from typing import List, Optional, Tuple
from enum import Enum
from player import Player, Slot
from resources import loadResources, getHardnessAgainst

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
    def rightMousePressed(self, app, event): pass
    def mouseReleased(self, app, event): pass
    def rightMouseReleased(self, app, event): pass
    def timerFired(self, app): pass
    def sizeChanged(self, app): pass
    def redrawAll(self, app, window, canvas): pass
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
    
    def redrawAll(self, app, window, canvas):
        # TODO:
        return

        leftX = app.width * 0.25
        rightX = app.width * 0.75

        height = 20

        canvas.create_rectangle(leftX, app.height / 2 - height, rightX, app.height / 2 + height)

        progress = self.loadStage / 30.0

        midX = leftX + (rightX - leftX) * progress

        canvas.create_rectangle(leftX, app.height / 2 - height, midX, app.height / 2 + height, fill='red')

class TitleMode(Mode):
    buttons: ButtonManager
    titleText: Image.Image

    def __init__(self, app):
        self.buttons = ButtonManager()

        self.titleText = Image.open('assets/TitleText.png')
        self.titleText = self.titleText.resize((self.titleText.width * 3, self.titleText.height * 3), Image.NEAREST)

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

    def redrawAll(self, app, window, canvas):
        render.redrawAll(app, canvas, doDrawHud=False)

        canvas.create_image(app.width / 2, 50, image=self.titleText)
        
        self.buttons.draw(app, canvas)
    
    def sizeChanged(self, app):
        self.buttons.canvasSizeChanged(app)
    
def setMouseCapture(app, value: bool) -> None:
    """If True, locks the mouse to the center of the window and hides it.
    
    This is True when playing the game normally, and False
    in menus and GUIs.
    """

    app.captureMouse = value

    if app.captureMouse:
        glfw.set_input_mode(app.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    else:
        glfw.set_input_mode(app.window, glfw.CURSOR, glfw.CURSOR_NORMAL)

class PlayingMode(Mode):
    lookedAtBlock = None
    mouseHeld: bool = False

    player: Player

    def __init__(self, app, creative: bool):
        app.timerDelay = 30
        setMouseCapture(app, True)

        self.player = Player(app, creative)

    def redrawAll(self, app, window, canvas):
        render.redrawAll(app, canvas)
    
    def timerFired(self, app):
        self.lookedAtBlock = world.lookedAtBlock(app)

        updateBlockBreaking(app, self)

        world.tick(app)

    def mousePressed(self, app, event):
        self.mouseHeld = True

    def rightMousePressed(self, app, event):
        block = world.lookedAtBlock(app)
        if block is not None:
            (pos, face) = block
            faceIdx = ['left', 'right', 'back', 'front', 'bottom', 'top'].index(face) * 2
            pos2 = world.adjacentBlockPos(pos, faceIdx)

            if not world.coordsInBounds(app, pos2): return

            if world.getBlock(app, pos) == 'crafting_table':
                app.mode = InventoryMode(app, self, name='crafting_table')
            else:
                slot = self.player.inventory[self.player.hotbarIdx]
                if slot.amount == 0: return
                
                if slot.item not in app.textures: return

                if slot.amount > 0:
                    slot.amount -= 1

                world.addBlock(app, pos2, slot.item)
    
    def mouseReleased(self, app, event):
        self.mouseHeld = False
    
    def keyPressed(self, app, event):
        if len(event.key) == 1 and event.key.isdigit():
            keyNum = int(event.key)
            if keyNum != 0:
                self.player.hotbarIdx = keyNum - 1
        elif event.key == 'W':
            app.w = True
        elif event.key == 'S':
            app.s = True
        elif event.key == 'A':
            app.a = True
        elif event.key == 'D':
            app.d = True
        elif event.key == 'E':
            app.mode = InventoryMode(app, self, name='inventory')
            app.w = app.s = app.a = app.d = False
        elif event.key == 'Space':
            if self.player.onGround:
                app.mode.player.velocity[1] = 0.35
        elif event.key == 'Escape':
            setMouseCapture(app, not app.captureMouse)

    def keyReleased(self, app, event):
        if event.key == 'W':
            app.w = False
        elif event.key == 'S':
            app.s = False 
        elif event.key == 'A':
            app.a = False
        elif event.key == 'D':
            app.d = False
        
class CraftingGui:
    craftInputs: List[List[Slot]]
    craftOutput: Slot

    def __init__(self, size: int):
        self.craftInputs = [[Slot('', 0) for _ in range(size)] for _ in range(size)]
        self.craftOutput = Slot('', 0)

    def onClick(self, app, isRight, mx, my):
        clickedSlot = self.clickedCraftInputIdx(app, mx, my)
        print(f"clicked craft slot {clickedSlot}")
        if clickedSlot is not None:
            (rowIdx, colIdx) = clickedSlot
            if isRight:
                app.mode.onRightClickIntoNormalSlot(app, self.craftInputs[rowIdx][colIdx])
            else:
                app.mode.onLeftClickIntoNormalSlot(app, self.craftInputs[rowIdx][colIdx])
        
        if self.clickedCraftOutput(app, mx, my):
            merged = app.mode.heldItem.tryMergeWith(self.craftOutput)
            if merged is not None:
                for r in range(len(self.craftInputs)):
                    for c in range(len(self.craftInputs)):
                        if self.craftInputs[r][c].amount > 0:
                            self.craftInputs[r][c].amount -= 1

                app.mode.heldItem = merged
                print(f"set held item to {merged}")
        

        def toid(s): return None if s.isEmpty() else s.item

        c = [[toid(i) for i in row] for row in self.craftInputs]

        self.craftOutput = Slot('', 0)

        for r in app.recipes:
            if r.isCraftedBy(c):
                self.craftOutput = copy.copy(r.outputs)
                break
        
        print(f"output is {self.craftOutput}")

    def clickedCraftOutput(self, a, b, c) -> bool:
        raise Exception("Attempt to check output in abstract class")

    def clickedCraftInputIdx(self, app, mx, my) -> Optional[Tuple[int, int]]:
        raise Exception("Attempt to check input in abstract class")


class InventoryCraftingGui(CraftingGui):
    def __init__(self):
        super().__init__(2)

    def redrawAll(self, app, canvas):
        (_, _, w) = render.getSlotCenterAndSize(app, 0)
        for (rowIdx, row) in enumerate(self.craftInputs):
            for (colIdx, col) in enumerate(row):
                x = colIdx * w + 350
                y = rowIdx * w + 100
                render.drawSlot(app, canvas, x, y, col)

        render.drawSlot(app, canvas, 460, 100 + w / 2, self.craftOutput)
    
    def clickedCraftInputIdx(self, app, mx, my) -> Optional[Tuple[int, int]]:
        (_, _, w) = render.getSlotCenterAndSize(app, 0)
        for rowIdx in range(2):
            for colIdx in range(2):
                x = colIdx * w + 350
                y = rowIdx * w + 100
                x0, y0 = x - w/2, y - w/2
                x1, y1 = x + w/2, y + w/2
                if x0 < mx and mx < x1 and y0 < my and my < y1:
                    return (rowIdx, colIdx)
        return None
    
    def clickedCraftOutput(self, app, mx, my) -> bool:
        (_, _, w) = render.getSlotCenterAndSize(app, 0)
        
        x = 470
        y = 100 + w / 2
        x0, y0 = x - w/2, y - w/2
        x1, y1 = x + w/2, y + w/2

        return x0 < mx and mx < x1 and y0 < my and my < y1
    

class CraftingTableGui(CraftingGui):
    def __init__(self):
        super().__init__(3)

    def clickedCraftInputIdx(self, app, mx, my) -> Optional[Tuple[int, int]]:
        (_, _, w) = render.getSlotCenterAndSize(app, 0)
        for rowIdx in range(3):
            for colIdx in range(3):
                (x, y) = self.craftSlotCenter(app, rowIdx, colIdx)
                x0, y0 = x - w/2, y - w/2
                x1, y1 = x + w/2, y + w/2
                if x0 < mx and mx < x1 and y0 < my and my < y1:
                    return (rowIdx, colIdx)
        return None
    
    def craftOutputCenter(self, app) -> Tuple[int, int]:
        (_, _, w) = render.getSlotCenterAndSize(app, 0)
        
        return (app.width // 2 + w * 2, 70 + w)
    
    def craftSlotCenter(self, app, rowIdx, colIdx) -> Tuple[int, int]:
        (_, _, w) = render.getSlotCenterAndSize(app, 0)

        x = app.width / 2 + (colIdx - 3) * w
        y = 70 + rowIdx * w

        return (x, y)
    
    def clickedCraftOutput(self, app, mx, my) -> bool:
        (_, _, w) = render.getSlotCenterAndSize(app, 0)
        
        (x, y) = self.craftOutputCenter(app)
        x0, y0 = x - w/2, y - w/2
        x1, y1 = x + w/2, y + w/2

        return x0 < mx and mx < x1 and y0 < my and my < y1
    
    def redrawAll(self, app, canvas):
        (_, _, w) = render.getSlotCenterAndSize(app, 0)
        for (rowIdx, row) in enumerate(self.craftInputs):
            for (colIdx, col) in enumerate(row):
                (x, y) = self.craftSlotCenter(app, rowIdx, colIdx)
                render.drawSlot(app, canvas, x, y, col)
        
        (x, y) = self.craftOutputCenter(app)

        render.drawSlot(app, canvas, x, y, self.craftOutput)

class InventoryMode(Mode):
    submode: PlayingMode
    heldItem: Slot = Slot('', 0)
    player: Player

    def __init__(self, app, submode: PlayingMode, name: str):
        setMouseCapture(app, False)
        self.submode = submode
        self.heldItem = Slot('', 0)
        self.craftOutput = Slot('', 0)

        self.player = submode.player

        if name == 'inventory':
            self.gui = InventoryCraftingGui()
        elif name == 'crafting_table':
            self.gui = CraftingTableGui()
        else:
            1 / 0

    def redrawAll(self, app, window, canvas):
        self.submode.redrawAll(app, window, canvas)

        render.drawMainInventory(app, canvas)

        self.gui.redrawAll(app, canvas)
        
        if app.mousePos is not None:
            render.drawSlot(app, canvas, app.mousePos[0], app.mousePos[1],
                self.heldItem, drawBackground=False)
    
    def mousePressed(self, app, event):
        self.someMousePressed(app, event, False)
    
    def rightMousePressed(self, app, event):
        self.someMousePressed(app, event, True)
    
    def clickedInventorySlotIdx(self, app, mx, my) -> Optional[int]:
        for i in range(36):
            (x, y, w) = render.getSlotCenterAndSize(app, i)
            x0, y0 = x - w/2, y - w/2
            x1, y1 = x + w/2, y + w/2
            if x0 < mx and mx < x1 and y0 < my and my < y1:
                return i
        
        return None
    
    def onRightClickIntoNormalSlot(self, app, normalSlot):
        if self.heldItem.isEmpty():
            # Picks up half of the slot
            if normalSlot.isInfinite():
                amountTaken = 1
            else:
                amountTaken = math.ceil(normalSlot.amount / 2)
                normalSlot.amount -= amountTaken
            self.heldItem = Slot(normalSlot.item, amountTaken)
        else:
            newStack = normalSlot.tryMergeWith(Slot(self.heldItem.item, 1))
            if newStack is not None:
                if not self.heldItem.isInfinite():
                    self.heldItem.amount -= 1
                normalSlot.item = newStack.item
                normalSlot.amount = newStack.amount
    
    def onLeftClickIntoNormalSlot(self, app, normalSlot):
        newStack = self.heldItem.tryMergeWith(normalSlot)
        if newStack is None or self.heldItem.isEmpty():
            tempItem = self.heldItem.item
            tempAmount = self.heldItem.amount
            self.heldItem.item = normalSlot.item
            self.heldItem.amount = normalSlot.amount
            normalSlot.item = tempItem
            normalSlot.amount = tempAmount
        else:
            self.heldItem = Slot('', 0)
            normalSlot.item = newStack.item
            normalSlot.amount = newStack.amount

    def someMousePressed(self, app, event, isRight: bool):
        (_, _, w) = render.getSlotCenterAndSize(app, 0)

        clickedSlot = self.clickedInventorySlotIdx(app, event.x, event.y) 
        print(f"clicked inv slot {clickedSlot}")
        if clickedSlot is not None:
            player = self.submode.player
            if isRight:
                self.onRightClickIntoNormalSlot(app, player.inventory[clickedSlot])
            else:
                self.onLeftClickIntoNormalSlot(app, player.inventory[clickedSlot])

        
        self.gui.onClick(app, isRight, event.x, event.y)
 
    def keyPressed(self, app, event):
        if event.key == 'E':
            dim = len(self.gui.craftInputs)
            for r in range(dim):
                for c in range(dim):
                    self.submode.player.pickUpItem(app, self.gui.craftInputs[r][c])
            self.submode.player.pickUpItem(app, self.heldItem)
            self.heldItem = Slot('', 0)
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
    if mode.mouseHeld and mode.lookedAtBlock is not None:
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

        toolSlot = mode.player.inventory[mode.player.hotbarIdx]
        if toolSlot.isEmpty():
            tool = ''
        else:
            tool = toolSlot.item

        hardness = getHardnessAgainst(app, blockId, tool)

        if app.breakingBlock >= hardness:
            brokenName = world.getBlock(app, pos)
            world.removeBlock(app, pos)
            mode.player.pickUpItem(app, Slot(brokenName, 1))
    else:
        app.breakingBlock = 0.0


def keyPressed(app, event):
    app.mode.keyPressed(app, event)

def keyReleased(app, event):
    app.mode.keyReleased(app, event)

def rightMousePressed(app, event):
    app.mode.rightMousePressed(app, event)

def mousePressed(app, event):
    app.mode.mousePressed(app, event)

def rightMouseReleased(app, event):
    app.mode.rightMouseReleased(app, event)

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
        app.prevMouse = (event.x, event.y)

def redrawAll(app, window, canvas):
    app.mode.redrawAll(app, window, canvas)

def main():
    runApp(width=600, height=400)

if __name__ == '__main__':
    main()