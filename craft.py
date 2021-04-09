import openglapp
from PIL import Image
from PIL import ImageDraw
from PIL.ImageDraw import Draw
from pathlib import Path
import os
import numpy as np
import math
import render
import world
import copy
import glfw
import config
import random
import cmu_112_graphics
import tkinter
import entity
import tick
from util import ChunkPos
from button import Button, ButtonManager, createSizedBackground
from world import Chunk, World
from typing import List, Optional, Tuple
from enum import Enum
from player import Player, Slot
from resources import loadResources, getHardnessAgainst, getBlockDrop, getAttackDamage

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

def worldToFolderName(name: str) -> str:
    result = ''
    for c in name:
        if c.isalnum():
            result += c
        else:
            result += '_'
    return result

class WorldLoadMode(Mode):
    loadStage: int = 0

    def __init__(self, app, worldName, nextMode, seed=40, importPath=''):
        self.nextMode = nextMode

        app.timerDelay = 10

        #app.world = World(worldName, seed, anvilpath='C:/Users/Carson/AppData/Roaming/.minecraft/saves/TheTempleofNotch/region/')
        app.world = World(worldName, seed, importPath=importPath)

        app.world.loadChunk((app.textures, app.cube, app.textureIndices), ChunkPos(0, 0, 0))
    
    def timerFired(self, app):
        if self.loadStage < 40:
            world.loadUnloadChunks(app, [0.0, 0.0, 0.0])
        elif self.loadStage < 60:
            world.tickChunks(app, maxTime=5.0)
        else:
            app.mode = self.nextMode(app)
            
        self.loadStage += 1
    
    def redrawAll(self, app, window, canvas):
        leftX = app.width * 0.25
        rightX = app.width * 0.75

        height = 20

        canvas.create_rectangle(leftX, app.height / 2 - height, rightX, app.height / 2 + height)

        progress = self.loadStage / 60.0

        midX = leftX + (rightX - leftX) * progress

        canvas.create_rectangle(leftX, app.height / 2 - height, midX, app.height / 2 + height, fill='red')

def getWorldNames() -> List[str]:
    return list(os.listdir('saves/'))

class WorldListMode(Mode):
    buttons: ButtonManager
    selectedWorld: Optional[int]
    worlds: List[str]

    def __init__(self, app):
        self.buttons = ButtonManager()
        self.selectedWorld = None

        self.worlds = getWorldNames()

        playButton = Button(app, 0.2, 0.7, 200, 40, "Play")
        createButton = Button(app, 0.8, 0.7, 200, 40, "Create New")

        self.buttons.addButton('play', playButton)
        self.buttons.addButton('create', createButton)
    
    def redrawAll(self, app, window, canvas):
        self.buttons.draw(app, canvas)

        if self.selectedWorld is not None:
            cx = app.width * 0.5
            cy = app.height * 0.20 + 30 * self.selectedWorld
            canvas.create_rectangle(cx - 100, cy - 15, cx + 100, cy + 15)

        for (i, worldName) in enumerate(self.worlds):
            canvas.create_text(app.width * 0.5, app.height * 0.20 + 30 * i, text=worldName)

    def mousePressed(self, app, event):
        self.buttons.onPress(app, event.x, event.y)

        # FIXME: Make these into buttons!
        if app.width * 0.5 - 100 < event.x and event.x < app.width * 0.5 + 100:
            yIdx = round((event.y - (app.height * 0.20)) / 30)
            if 0 <= yIdx and yIdx < len(self.worlds):
                self.selectedWorld = yIdx
    
    def mouseReleased(self, app, event):
        btn = self.buttons.onRelease(app, event.x, event.y)
        if btn is not None:
            print(f"Pressed {btn}")
            if btn == 'create':
                app.mode = CreateWorldMode(app)
            elif btn == 'play' and self.selectedWorld is not None:
                # FIXME: Gamemodes, seed
                def makePlayingMode(app): return PlayingMode(app, False)
                app.mode = WorldLoadMode(app, self.worlds[self.selectedWorld], makePlayingMode)


def posInBox(x, y, bounds) -> bool:
    (x0, y0, x1, y1) = bounds
    return x0 <= x and x <= x1 and y0 <= y and y <= y1

class CreateWorldMode(Mode):
    buttons: ButtonManager
    worldName: str

    worldSource: str
    importPath: str
    seed: Optional[int]

    def __init__(self, app):
        self.worldName = ''
        self.buttons = ButtonManager()

        self.seed = None

        self.importPath = ''

        survivalButton = Button(app, 0.2, 0.90, 200, 40, "Play Survival")
        self.buttons.addButton('playSurvival', survivalButton)

        creativeButton = Button(app, 0.8, 0.90, 200, 40, "Play Creative")
        self.buttons.addButton('playCreative', creativeButton)

        worldTypeButton = Button(app, 0.5, 0.4, 300, 40, "")
        self.buttons.addButton('worldSource', worldTypeButton)

        self.setWorldSource('generated')

    def setWorldSource(self, ty: str):
        self.worldSource = ty
        if self.worldSource == 'generated':
            self.buttons.buttons['worldSource'].text = 'World Source:  Generated'
        elif self.worldSource == 'imported':
            self.buttons.buttons['worldSource'].text = 'World Source:  Imported'
        else:
            1 / 0

    def worldNameBoxBounds(self, app):
        return (app.width * 0.5 - 100, app.height * 0.25 - 15,
                app.width * 0.5 + 100, app.height * 0.25 + 15)
    
    def worldPathBoxBounds(self, app):
        return (app.width * 0.5 - 100, app.height * 0.60 - 15,
                app.width * 0.5 + 100, app.height * 0.60 + 15)
        
    def redrawAll(self, app, window, canvas):
        self.buttons.draw(app, canvas)

        canvas.create_text(app.width * 0.5, app.height * 0.15, text="World Name:")

        (x0, y0, x1, y1) = self.worldNameBoxBounds(app)
        canvas.create_rectangle(x0, y0, x1, y1)

        if self.worldSource == 'imported':
            (x0, y0, x1, y1) = self.worldPathBoxBounds(app)
            canvas.create_rectangle(x0, y0, x1, y1)

            if len(self.importPath) > 18:
                importText = '...' + self.importPath[-15:]
            else:
                importText = self.importPath
            
            x = (x1 + x0) / 2
            y = (y1 + y0) / 2

            canvas.create_text(x, y0 - 5, text='World Folder:', anchor='s')

            canvas.create_text(x0 + 5, y, text=importText, anchor='w')

        canvas.create_text(app.width * 0.5, app.height * 0.25, text=self.worldName)
    
    def keyPressed(self, app, event):
        key = event.key.lower()
        # FIXME: CHECK IF BACKSPACE IS CORRECT FOR TKINTER TOO
        if key == 'backspace' and len(self.worldName) > 0:
            self.worldName = self.worldName[:-1]
        elif len(key) == 1 and len(self.worldName) < 15:
            self.worldName += key

    def mousePressed(self, app, event):
        self.buttons.onPress(app, event.x, event.y)

        if posInBox(event.x, event.y, self.worldPathBoxBounds(app)):
            # https://stackoverflow.com/questions/30678508/how-to-use-tkinter-filedialog-without-a-window

            from cmu_112_graphics import Tk
            from cmu_112_graphics import filedialog

            Tk().withdraw()
            self.importPath = filedialog.askdirectory()

    def mouseReleased(self, app, event):
        btn = self.buttons.onRelease(app, event.x, event.y)
        if btn is not None:
            print(f"Pressed {btn}")
            if (btn == 'playSurvival' or btn == 'playCreative') and self.worldName != '':
                # FIXME: CHECK FOR DUPLICATE WORLD NAMES
                isCreative = btn == 'playCreative'
                makePlayingMode = lambda app: PlayingMode(app, isCreative)

                if self.worldSource == 'imported':
                    importPath = self.importPath
                else:
                    importPath = ''

                app.mode = WorldLoadMode(app, self.worldName, makePlayingMode, seed=random.random(), importPath=importPath)
            elif btn == 'worldSource':
                if self.worldSource == 'generated':
                    self.setWorldSource('imported')
                else:
                    self.setWorldSource('generated')

class TitleMode(Mode):
    buttons: ButtonManager
    titleText: Image.Image

    def __init__(self, app):
        self.buttons = ButtonManager()

        if config.USE_OPENGL_BACKEND:
            self.titleText = Image.open('assets/TitleText.png')
            self.titleText = self.titleText.resize((self.titleText.width * 3, self.titleText.height * 3), Image.NEAREST)
        else:
            self.titleText = app.loadImage('assets/TitleText.png')
            self.titleText = app.scaleImage(self.titleText, 3)

        playButton = Button(app, 0.5, 0.4, 200, 40, "Play")
        self.buttons.addButton('play', playButton)

    def timerFired(self, app):
        app.cameraYaw += 0.01

    def mousePressed(self, app, event):
        self.buttons.onPress(app, event.x, event.y)
    
    def mouseReleased(self, app, event):
        btn = self.buttons.onRelease(app, event.x, event.y)
        if btn is not None:
            print(f"Pressed {btn}")
            worldNames = getWorldNames()
            if worldNames == []:
                app.mode = CreateWorldMode(app)
            else:
                app.mode = WorldListMode(app)

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

    if config.USE_OPENGL_BACKEND:
        if app.captureMouse:
            glfw.set_input_mode(app.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        else:
            glfw.set_input_mode(app.window, glfw.CURSOR, glfw.CURSOR_NORMAL)

def submitChat(app, text: str):
    if text.startswith('/'):
        text = text.removeprefix('/')

        parts = text.split()

        print(f"COMMAND {text}")

        if parts[0] == 'pathfind':
            player: Player = app.mode.player
            target = player.getBlockPos()
            for entity in app.entities:
                entity.updatePath(app.world, target)
        elif parts[0] == 'give':
            itemId = parts[1]
            if len(parts) == 3:
                amount = int(parts[2])
            else:
                amount = 1
            app.mode.player.pickUpItem(app, Slot(itemId, amount))
        elif parts[0] == 'hide':
            app.doDrawHud = False
        elif parts[0] == 'show':
            app.doDrawHud = True
        elif parts[0] == 'cinematic':
            app.cinematic = not app.cinematic
        elif parts[0] == 'time':
            if parts[1] == 'set':
                if parts[2] == 'day':
                    app.time = 1000
                elif parts[2] == 'night':
                    app.time = 13000
                elif parts[2] == 'midnight':
                    app.time = 18000
                else:
                    app.time = int(parts[2])
            elif parts[1] == 'add':
                app.time += int(parts[2])
        elif parts[0] == 'gamemode':
            if parts[1] == 'creative':
                app.mode.player.creative = True
            elif parts[1] == 'survival':
                app.mode.player.creative = False

    else:
        print(f"CHAT: {text}")

class ChatMode(Mode):
    text: str

    def __init__(self, app, submode, text):
        self.submode = submode
        self.player = self.submode.player
        self.text = text
    
    def keyPressed(self, app, event):
        key = event.key.upper()
        if key == 'ESCAPE':
            app.mode = self.submode
        elif key == 'ENTER':
            app.mode = self.submode
            submitChat(app, self.text)
        elif key == 'BACKSPACE':
            if self.text != '':
                self.text = self.text[:-1]
        elif len(key) == 1:
            self.text += key.lower()
        
    def redrawAll(self, app, window, canvas):
        canvas.create_rectangle(0, app.height * 2 / 3 - 10, app.width, app.height * 2 / 3 + 10, fill='#333333')

        canvas.create_text(0, app.height * 2 / 3, text=self.text, anchor='w')

        self.submode.redrawAll(app, window, canvas)
    
    def timerFired(self, app):
        self.submode.timerFired(app)


class PlayingMode(Mode):
    lookedAtBlock = None
    mouseHeld: bool = False

    player: Player

    def __init__(self, app, creative: bool):
        app.timerDelay = 30
        setMouseCapture(app, True)

        self.player = Player(app, creative)

    def redrawAll(self, app, window, canvas):
        render.redrawAll(app, canvas, doDrawHud=app.doDrawHud)
    
    def timerFired(self, app):
        self.lookedAtBlock = world.lookedAtBlock(app)

        if app.cinematic:
            app.cameraPitch += app.pitchSpeed * 0.05
            app.cameraYaw += app.yawSpeed * 0.05

            app.pitchSpeed *= 0.95
            app.yawSpeed *= 0.95
        
        if self.player.flying:
            if app.space:
                self.player.velocity[1] = 0.2
            elif app.shift:
                self.player.velocity[1] = -0.2
            else:
                self.player.velocity[1] = 0.0

        updateBlockBreaking(app, self)

        tick.tick(app)

    def mousePressed(self, app, event):
        self.mouseHeld = True

        idx = tick.lookedAtEntity(app)
        if idx is not None:
            entity = app.entities[idx]

            knockback = [entity.pos[0] - self.player.pos[0], entity.pos[2] - self.player.pos[2]]
            mag = math.sqrt(knockback[0]**2 + knockback[1]**2)
            knockback[0] /= mag
            knockback[1] /= mag

            slot = self.player.inventory[self.player.hotbarIdx]
            
            if slot.isEmpty():
                dmg = 1.0
            else:
                dmg = 1.0 + getAttackDamage(app, slot.item)

            entity.hit(dmg, knockback)

    def rightMousePressed(self, app, event):
        block = world.lookedAtBlock(app)
        if block is not None:
            (pos, face) = block
            faceIdx = ['left', 'right', 'back', 'front', 'bottom', 'top'].index(face) * 2
            pos2 = world.adjacentBlockPos(pos, faceIdx)

            if not app.world.coordsInBounds(pos2): return

            if app.world.getBlock(pos) == 'crafting_table':
                app.mode = InventoryMode(app, self, name='crafting_table')
            elif app.world.getBlock(pos) == 'furnace':
                (ckPos, ckLocal) = world.toChunkLocal(pos)
                furnace = app.world.chunks[ckPos].tileEntities[ckLocal]
                app.mode = InventoryMode(app, self, name='furnace', extra=furnace)
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
        key = event.key.upper()
        if len(key) == 1 and key.isdigit():
            keyNum = int(key)
            if keyNum != 0:
                self.player.hotbarIdx = keyNum - 1
        elif key == 'W':
            app.w = True
        elif key == 'S':
            app.s = True
        elif key == 'A':
            app.a = True
        elif key == 'D':
            app.d = True
        elif key == 'E':
            app.mode = InventoryMode(app, self, name='inventory')
            app.w = app.s = app.a = app.d = False
        elif key == 'SPACE' or key == ' ':
            app.space = True
            if self.player.onGround:
                app.mode.player.velocity[1] = 0.35
            elif self.player.creative and not self.player.onGround:
                self.player.flying = True
        elif key == 'SHIFT':
            app.shift = True
        elif key == 'ESCAPE':
            setMouseCapture(app, not app.captureMouse)
        elif key == 'T':
            app.mode = ChatMode(app, self, '')
        elif key == '/':
            app.mode = ChatMode(app, self, '/')
        elif self.player.flying:
            self.player.velocity[1] = 0.0

    def keyReleased(self, app, event):
        key = event.key.upper()
        if key == 'W':
            app.w = False
        elif key == 'S':
            app.s = False 
        elif key == 'A':
            app.a = False
        elif key == 'D':
            app.d = False
        elif key == 'SHIFT':
            app.shift = False
        elif key == 'SPACE' or key == ' ':
            app.space = False

class FurnaceGui:
    furnace: world.Furnace

    def __init__(self, furnace: world.Furnace):
        self.furnace = furnace

    def inputPos(self, app):
        (_, _, w) = render.getSlotCenterAndSize(app, 0)
        return (app.width / 2 - 50, app.height / 4 - 30)
    
    def fuelPos(self, app):
        (_, _, w) = render.getSlotCenterAndSize(app, 0)
        return (app.width / 2 - 50, app.height / 4 + 30)
    
    def outputPos(self, app):
        (_, _, w) = render.getSlotCenterAndSize(app, 0)
        return (app.width / 2 + 50, app.height / 4)
    
    def onClick(self, app, isRight, mx, my):
        (_, _, w) = render.getSlotCenterAndSize(app, 0)

        slot = None

        (x, y) = self.inputPos(app)
        if x-w/2 <= mx and mx <= x+w/2 and y-w/2 <= my and my <= y+w/2:
            slot = self.furnace.inputSlot

        (x, y) = self.outputPos(app)
        if x-w/2 <= mx and mx <= x+w/2 and y-w/2 <= my and my <= y+w/2:
            slot = self.furnace.outputSlot

        (x, y) = self.fuelPos(app)
        if x-w/2 <= mx and mx <= x+w/2 and y-w/2 <= my and my <= y+w/2:
            slot = self.furnace.fuelSlot
        
        if slot is not None:
            if isRight:
                app.mode.onRightClickIntoNormalSlot(app, slot)
            else:
                app.mode.onLeftClickIntoNormalSlot(app, slot)
    
    def redrawAll(self, app, canvas):
        (x, y) = self.inputPos(app)
        render.drawSlot(app, canvas, x, y, self.furnace.inputSlot)

        (x, y) = self.outputPos(app)
        render.drawSlot(app, canvas, x, y, self.furnace.outputSlot)

        (x, y) = self.fuelPos(app)
        render.drawSlot(app, canvas, x, y, self.furnace.fuelSlot)
        
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

class PauseMode(Mode):
    submode: PlayingMode
    buttons: ButtonManager

    def __init__(self, app, submode: PlayingMode):
        setMouseCapture(app, False)
        self.submode = submode

    def redrawAll(self, app, window, canvas):
        self.submode.redrawAll(app, window, canvas)



class InventoryMode(Mode):
    submode: PlayingMode
    heldItem: Slot = Slot('', 0)
    player: Player

    def __init__(self, app, submode: PlayingMode, name: str, extra=None):
        setMouseCapture(app, False)
        self.submode = submode
        self.heldItem = Slot('', 0)
        self.craftOutput = Slot('', 0)

        self.player = submode.player

        if name == 'inventory':
            self.gui = InventoryCraftingGui()
        elif name == 'crafting_table':
            self.gui = CraftingTableGui()
        elif name == 'furnace':
            self.gui = FurnaceGui(extra)
        else:
            raise Exception(f"unknown gui {name}")
        
    def timerFired(self, app):
        tick.tick(app)

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
        key = event.key.upper()
        if key == 'E':
            if hasattr(self.gui, 'craftInputs'):
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
    loadResources(app)

    #app.mode = WorldLoadMode(app, 'world', TitleMode)
    def makePlayingMode(app): return PlayingMode(app, False)
    app.mode = WorldLoadMode(app, 'demoworld', makePlayingMode, seed=random.random())
    #app.mode = CreateWorldMode(app)

    app.entities = [entity.Entity(app, 'creeper', 0.0, 71.0, 1.0), entity.Entity(app, 'fox', 5.0, 72.0, 3.0)]

    app.btnBg = createSizedBackground(app, 200, 40)

    app.doDrawHud = True
    app.cinematic = False

    app.time = 0

    app.yawSpeed = 0.0
    app.pitchSpeed = 0.0

    app.tickTimes = [0.0] * 10
    app.tickTimeIdx = 0

    # ----------------
    # Player variables
    # ----------------
    app.breakingBlock = 0.0
    app.breakingBlockPos = world.BlockPos(0, 0, 0)

    app.gravity = 0.10

    app.cameraYaw = 0
    app.cameraPitch = 0

    if world.CHUNK_HEIGHT == 256: #type:ignore
        app.cameraPos = [2.0, 72, 4.0]
    else:
        app.cameraPos = [2.0, 10.0, 4.0]

    # -------------------
    # Rendering Variables
    # -------------------
    app.vpDist = 0.25
    app.vpWidth = 3.0 / 4.0
    app.vpHeight = app.vpWidth * app.height / app.width 
    app.wireframe = False
    app.renderDistanceSq = 9**2

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
    app.space = False
    app.shift = False

    app.prevMouse = None

    setMouseCapture(app, False)

def appStopped(app):
    # FIXME:
    if hasattr(app, 'world'):
        app.world.save()

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

        #app.sounds['grass'].play()
        
        blockId = app.world.getBlock(pos)

        toolSlot = mode.player.inventory[mode.player.hotbarIdx]
        if toolSlot.isEmpty():
            tool = ''
        else:
            tool = toolSlot.item

        hardness = getHardnessAgainst(app, blockId, tool)

        if app.breakingBlock >= hardness:
            droppedItem = getBlockDrop(app, app.world.getBlock(pos), tool)
            world.removeBlock(app, pos)
            app.sounds['destroy_grass'].play()
            if droppedItem is not None:
                mode.player.pickUpItem(app, Slot(droppedItem, 1))
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

        app.yawSpeed += 0.01 * (xChange - app.yawSpeed)
        app.pitchSpeed += 0.01 * (yChange - app.pitchSpeed)

        if not app.cinematic:
            app.cameraPitch += yChange * 0.01
            app.cameraYaw += xChange * 0.01
        
        if app.cameraPitch < -math.pi / 2 * 0.95:
            app.cameraPitch = -math.pi / 2 * 0.95
        elif app.cameraPitch > math.pi / 2 * 0.95:
            app.cameraPitch = math.pi / 2 * 0.95

    if app.captureMouse:
        if config.USE_OPENGL_BACKEND:
            app.prevMouse = (event.x, event.y)
        else:
            x = app.width / 2
            y = app.height / 2
            app._theRoot.event_generate('<Motion>', warp=True, x=x, y=y)
            app.prevMouse = (x, y)


class CachedImageCanvas(cmu_112_graphics.WrappedCanvas):
    def __init__(self, c):
        self._canvas = c
    
    def create_rectangle(self, *args, **kwargs):
        self._canvas.create_rectangle(*args, **kwargs)
    
    def create_polygon(self, *args, **kwargs):
        self._canvas.create_polygon(*args, **kwargs)
    
    def create_image(self, x, y, image, **kwargs):
        self._canvas.create_image(x, y, image=render.getCachedImage(image), **kwargs)
    
    def create_text(self, *args, **kwargs):
        self._canvas.create_text(*args, **kwargs)
    
    def create_oval(self, *args, **kwargs):
        self._canvas.create_oval(*args, **kwargs)

def redrawAll(app, *args):
    if config.USE_OPENGL_BACKEND:
        window, canvas = args
    else:
        window = ()
        [canvas] = args

        if app.captureMouse:
            canvas.configure(cursor='none')
        else:
            canvas.configure(cursor='arrow')

        canvas = CachedImageCanvas(canvas)

    app.mode.redrawAll(app, window, canvas)

def main():
    if config.USE_OPENGL_BACKEND:
        openglapp.runApp(width=600, height=400)
    else:
        cmu_112_graphics.runApp(width=600, height=400)

if __name__ == '__main__':
    main()