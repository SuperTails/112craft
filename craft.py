"""The main entry point of the app.

All of the different parts of the game are divided into "Modes".
Any events that occur are delegated to the current mode to be handled.
The modes act like an FSM. For example, pressing 'E' in `PlayingMode` will
result in a transition to the `InventoryMode` state.

# WorldLoadMode #
Displays a loading screen while a World is generated or loaded from disk.

# WorldListMode #
Displays a list of the currently saved worlds and allows the player to 
select one or create a new world.

# CreateWorldMode #
Displays various options the player can set to create a new world. 

# TitleMode # 
Used at the title screen. Not very exciting.

# ChatMode #
Used when the chat box is opened during gameplay.

# PlayingMode #
The 'normal' state of gameplay. The player can walk, look around, etc.

# InventoryMode #
Used when the player has an inventory window opened. This can include
when the player right clicks on, for example, a furnace or a crafting table.
This mode keeps a reference to what kind of GUI is opened.
"""

import config
import OpenGL
OpenGL.ERROR_CHECKING = config.OPENGL_ERROR_CHECKING

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
import random
import time
import cmu_112_graphics
import tkinter
import entity
from tick import *
import tick
import server
from queue import SimpleQueue
from client import ClientState, lookedAtEntity, getLookVector
from util import ChunkPos, BlockPos
import util
from button import Button, ButtonManager, createSizedBackground
from world import Chunk, World
from typing import List, Optional, Tuple, Any
from enum import Enum
from player import Player, Slot, Stack
import inventory
import resources
from resources import loadResources, getHardnessAgainst, getBlockDrop, getAttackDamage
from nbt import nbt
from dataclasses import dataclass
import network

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

    def __init__(self, app, worldName, local: bool, nextMode, seed: Optional[Any] = None, importPath=''):
        self.nextMode = nextMode

        app.timerDelay = 10

        if seed is None:
            seed = random.random()

        app.client.local = local

        if local:
            app.server = ServerState.open(worldName, seed, importPath, app)
            self.centerPos = [16.0 * app.server.preloadPos.x, 0.0, 16.0 * app.server.preloadPos.z]

            app.client.world = app.server.getLocalDimension().world
        else:
            if hasattr(app, 'server'):
                delattr(app, 'server')

            app.client.world = World('saves/temp', world.NullGen(), 0)
            app.client.world.local = False

            self.centerPos = [0.0, 0.0, 0.0]

            network.c2sQueue.put((worldName, 25565))
        
    def timerFired(self, app):
        loader = app.server.getLocalDimension() if app.client.local else app.client

        if self.loadStage < 10:
            loader.world.loadUnloadChunks(self.centerPos, (app.textures, app.cube, app.textureIndices))
        elif self.loadStage < 20:
            loader.world.addChunkDetails((app.textures, app.cube, app.textureIndices), maxTime=5.0)
        else:
            app.mode = self.nextMode(app, Player(app))
            
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
                def makePlayingMode(app, player): return PlayingMode(app, player)
                app.mode = WorldLoadMode(app, self.worlds[self.selectedWorld], True, makePlayingMode)


def posInBox(x, y, bounds) -> bool:
    (x0, y0, x1, y1) = bounds
    return x0 <= x and x <= x1 and y0 <= y and y <= y1

class DirectConnectMode(Mode):
    buttons: ButtonManager

    ip: str

    def __init__(self, app):
        self.ip = ''
        self.buttons = ButtonManager()

        connectButton = Button(app, 0.5, 0.6, 200, 40, 'Connect')
        self.buttons.addButton('connect', connectButton)
    
    def ipBoxBounds(self, app):
        return (app.width * 0.5 - 100, app.height * 0.25 - 15,
                app.width * 0.5 + 100, app.height * 0.25 + 15)
    
    def redrawAll(self, app, window, canvas):
        self.buttons.draw(app, canvas)

        (x0, y0, x1, y1) = self.ipBoxBounds(app)
        canvas.create_rectangle(x0, y0, x1, y1)

        x = (x1 + x0) / 2
        canvas.create_text(x, y0 - 5, text='Server Address:', anchor='s')

        canvas.create_text(app.width * 0.5, app.height * 0.25, text=self.ip)
    
    def keyPressed(self, app, event):
        key = event.key.lower()
        # FIXME: CHECK IF BACKSPACE IS CORRECT FOR TKINTER TOO
        if key == 'backspace' and len(self.ip) > 0:
            self.ip = self.ip[:-1]
        elif len(key) == 1 and len(self.ip) < 30:
            self.ip += key
    
    def mousePressed(self, app, event):
        self.buttons.onPress(app, event.x, event.y)

    def mouseReleased(self, app, event):
        btn = self.buttons.onRelease(app, event.x, event.y)
        if btn == 'connect':
            def makePlayingMode(app, player): return PlayingMode(app, player)
            app.mode = WorldLoadMode(app, self.ip, False, makePlayingMode)


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

                def makePlayingMode(app, player): return PlayingMode(app, player)

                if self.worldSource == 'imported':
                    importPath = self.importPath
                else:
                    importPath = ''

                app.mode = WorldLoadMode(app, self.worldName, True, makePlayingMode, seed=random.random(), importPath=importPath)
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

        singleButton = Button(app, 0.5, 0.4, 200, 40, "Singleplayer")
        multiButton = Button(app, 0.5, 0.6, 200, 40, "Multiplayer")

        self.buttons.addButton('singleplayer', singleButton)
        self.buttons.addButton('multiplayer', multiButton)

    def timerFired(self, app):
        app.client.cameraYaw += 0.01

    def mousePressed(self, app, event):
        self.buttons.onPress(app, event.x, event.y)
    
    def mouseReleased(self, app, event):
        btn = self.buttons.onRelease(app, event.x, event.y)
        if btn == 'singleplayer':
            print(f"Pressed {btn}")
            worldNames = getWorldNames()
            if worldNames == []:
                app.mode = CreateWorldMode(app)
            else:
                app.mode = WorldListMode(app)
        elif btn == 'multiplayer':
            app.mode = DirectConnectMode(app)

    def redrawAll(self, app, window, canvas):
        render.redrawAll(app.client, canvas, doDrawHud=False)

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

def drawChatHistory(app, client: ClientState, canvas, useAge=True):
    for i, (sendTime, msg) in enumerate(client.chat[::-1]):
        if (useAge and time.time() - sendTime > 5.0) or i > 10:
            return
        else:
            y = app.height * 2/3 - 20 * (i+1)

            canvas.create_rectangle(0, y - 10, app.width, y + 10, fill='#333333')

            msgStr: str = msg.to_string()

            if msgStr.startswith('chat.type.announcement'):
                msgStr = msgStr.removeprefix('chat.type.announcement')[1:-1]
                name, text = msgStr.split(', ', 1)
                msgStr = f'[{name}] {text}'
            elif msgStr.startswith('chat.type.text'):
                msgStr = msgStr.removeprefix('chat.type.text')[1:-1]
                name, text = msgStr.split(', ', 1)
                msgStr = f'<{name}> {text}'

            canvas.create_text(0, y, text=msgStr, anchor='w')

class ChatMode(Mode):
    text: str

    def __init__(self, app, submode, text):
        self.submode = submode
        self.text = text
    
    def keyPressed(self, app, event):
        key = event.key.upper()
        if key == 'ESCAPE':
            app.mode = self.submode
        elif key == 'ENTER':
            app.mode = self.submode
            if not self.text.isspace():
                sendChatMessage(app, self.text)
        elif key == 'BACKSPACE':
            if self.text != '':
                self.text = self.text[:-1]
        elif key == 'SPACE':
            self.text += ' '
        elif len(key) == 1:
            self.text += key.lower()
        
    def redrawAll(self, app, window, canvas):
        self.submode.redrawAll(app, window, canvas)
        
        drawChatHistory(app, app.client, canvas, useAge=False)

        canvas.create_rectangle(0, app.height * 2 / 3 - 10, app.width, app.height * 2 / 3 + 10, fill='#333333')

        canvas.create_text(0, app.height * 2 / 3, text=self.text, anchor='w')
    
    def timerFired(self, app):
        self.submode.timerFired(app)

class GameOverMode(Mode):
    buttons: ButtonManager

    def __init__(self, app):
        setMouseCapture(app, False)

        respawnButton = Button(app, 0.5, 0.5, 200, 40, "Respawn")

        self.buttons = ButtonManager()
        self.buttons.addButton("respawn", respawnButton)

    def redrawAll(self, app, window, canvas):
        render.redrawAll(app.client, canvas, doDrawHud=False)
        canvas.create_text(app.width / 2, app.height / 3, text="Game Over!")
        self.buttons.draw(app, canvas)
    
    def mousePressed(self, app, event):
        self.buttons.onPress(app, event.x, event.y)
    
    def mouseReleased(self, app, event):
        btn = self.buttons.onRelease(app, event.x, event.y)
        if btn == 'respawn':
            # FIXME:
            player = Player(app, False)
            player.pos = [0.0, 75.0, 0.0]
            app.cameraPos = [0.0, 75.0, 0.0]
            app.mode = PlayingMode(app, player)

class PlayingMode(Mode):
    lookedAtBlock = None
    mouseHeld: bool = False

    overlay: Optional['InventoryMode']

    def __init__(self, app, player: Player):
        app.timerDelay = 50
        setMouseCapture(app, True)

        app.client.player = player

        self.overlay = None

        sendClientStatus(app, 0)

    def redrawAll(self, app, window, canvas):
        doDrawHud = app.doDrawHud and self.overlay is None

        render.redrawAll(app.client, canvas, doDrawHud)

        drawChatHistory(app, app.client, canvas)

        if self.overlay is not None:
            self.overlay.redrawAll(app, window, canvas)
    
    def timerFired(self, app):
        self.lookedAtBlock = app.client.lookedAtBlock()

        player = app.client.getPlayer()

        if app.client.cinematic:
            # TODO: Use framerate instead
            app.client.cameraPitch += app.pitchSpeed * 0.05
            app.client.cameraYaw += app.yawSpeed * 0.05

            app.pitchSpeed *= 0.95
            app.yawSpeed *= 0.95
        
        if player.flying:
            if app.client.space:
                player.velocity[1] = 0.2
            elif app.client.shift:
                player.velocity[1] = -0.2
            else:
                player.velocity[1] = 0.0

        updateBlockBreaking(app, self)

        handleS2CPackets(self, app, app.client)
        
        client: ClientState = app.client
        sendPlayerLook(app, client.cameraYaw, client.cameraPitch, client.getPlayer().onGround)
        sendPlayerPosition(app, player.pos[0], player.pos[1], player.pos[2], player.onGround)
        sendPlayerMovement(app, player.onGround)

        tick.clientTick(app.client, (app.textures, app.cube, app.textureIndices))

        if hasattr(app, 'server'):
            tick.serverTick(app, app.server)

        if player.health <= 0.0:
            app.mode = GameOverMode(app)

    def mousePressed(self, app, event):
        if self.overlay is not None:
            self.overlay.mousePressed(app, event)
            return

        self.mouseHeld = True

        player: Player = app.client.getPlayer()

        idx = lookedAtEntity(app.client)
        if idx is not None:
            ent = app.client.entities[idx]

            # TODO: Sneaking

            sendInteractEntity(app, ent.entityId, network.InteractKind.ATTACK,
                sneaking=False)

    def rightMousePressed(self, app, event):
        if self.overlay is not None:
            self.overlay.rightMousePressed(app, event)
            return

        player: Player = app.client.getPlayer()

        stack = player.inventory[player.hotbarIdx].stack

        useFluids = stack.item == 'bucket' and not stack.isEmpty()

        block = app.client.lookedAtBlock(useFluids)
        if block is not None:
            (pos, face) = block
            faceIdx = ['left', 'right', 'back', 'front', 'bottom', 'top'].index(face) * 2
            pos2 = world.adjacentBlockPos(pos, faceIdx)

            mcFace = { 'bottom': 0, 'top': 1, 'back': 2, 'front': 3, 'left': 4, 'right': 5 }[face]

            if not app.client.world.coordsInBounds(pos2): return

            blockId = app.client.world.getBlock(pos)

            if blockId in ['crafting_table', 'furnace']:
                mcFace = 2*(mcFace // 2) + (1-(mcFace%2))

                sendPlayerPlacement(app, 0, pos, mcFace, 0.5, 0.5, 0.5, False)
            else:
                if stack.amount == 0: return

                if stack.item in app.textures or stack.item == 'redstone':
                    if stack.amount > 0:
                        stack.amount -= 1
                    
                    if stack.item == 'torch':
                        if face in ('top', 'bottom'):
                            placedId = 'torch'
                            placedState = {}
                        else:
                            placedId = 'wall_torch'
                            if face == 'left':
                                placedState = { 'facing': 'west' }
                            elif face == 'right':
                                placedState = { 'facing': 'east' }
                            elif face == 'front':
                                placedState = { 'facing': 'north' }
                            else:
                                placedState = { 'facing': 'south' }
                    elif stack.item == 'redstone_torch':
                        if face in ('top', 'bottom'):
                            placedId = 'redstone_torch'
                            placedState = {}
                        else:
                            placedId = 'redstone_wall_torch'
                            if face == 'left':
                                placedState = { 'facing': 'west' }
                            elif face == 'right':
                                placedState = { 'facing': 'east' }
                            elif face == 'front':
                                placedState = { 'facing': 'north' }
                            else:
                                placedState = { 'facing': 'south' }
                        
                        placedState['lit'] = 'true'
                    elif stack.item == 'redstone':
                        placedId = 'redstone_wire'
                        placedState = {
                            'east': 'none',
                            'north': 'none',
                            'south': 'none',
                            'west': 'none',
                            'power': '0',
                        }
                    else:
                        placedId = stack.item
                        placedState = {}
                    
                    # TODO: Cursor position, inside block??
                    sendPlayerPlacement(app, 0, pos2, mcFace, 0.5, 0.5, 0.5, False)
                    
                    if config.UGLY_HACK:
                        app.client.world.setBlock((app.textures, app.cube, app.textureIndices), pos2, placedId, placedState)

                    resources.getDigSound(app, app.client.world.getBlock(pos2)).play()
                else:
                    sendUseItem(app, 0)
    
    def mouseReleased(self, app, event):
        if self.overlay is not None:
            self.overlay.mouseReleased(app, event)
            return

        self.mouseHeld = False
    
    def keyPressed(self, app, event):
        if self.overlay is not None:
            doExit = self.overlay.keyPressed(app, event)
            sendCloseWindow(app, self.overlay.windowId)
            if doExit:
                self.overlay = None
            return

        client: ClientState = app.client
        player = client.getPlayer()
        assert(isinstance(player, Player))

        key = event.key.upper()
        if len(key) == 1 and key.isdigit():
            keyNum = int(key)
            if keyNum != 0:
                player.hotbarIdx = keyNum - 1
                sendHeldItemChange(app, keyNum - 1)
        elif key == 'W':
            client.w = True
        elif key == 'S':
            client.s = True
        elif key == 'A':
            client.a = True
        elif key == 'D':
            client.d = True
        elif key == 'E':
            self.overlay = InventoryMode(app, 0, name='inventory')
            client.w = client.s = client.a = client.d = False
        elif key == 'Q':
            stack = player.inventory[player.hotbarIdx].stack
            if not stack.isEmpty():
                sendPlayerDigging(app, network.DiggingAction.DROP_ITEM, BlockPos(0, 0, 0), 0)

                '''
                ent = entity.Entity(app, 'item', player.pos[0], player.pos[1] + player.height - 0.5, player.pos[2])
                ent.extra.stack = Stack(stack.item, 1)

                look = getLookVector(app.client)
                ent.velocity[0] = look[0] * 0.5
                ent.velocity[1] = look[1] * 0.3 + 0.2
                ent.velocity[2] = look[2] * 0.5

                if not stack.isInfinite():
                    stack.amount -= 1
                
                app.entities.append(ent)
                '''

        elif key == 'SPACE' or key == ' ':
            client.space = True
            if player.onGround:
                player.velocity[1] = 0.35
            elif player.creative and not player.onGround:
                player.flying = True
        elif key == 'SHIFT':
            client.shift = True
        elif key == 'ESCAPE':
            setMouseCapture(app, not app.captureMouse)
        elif key == 'H':
            app.doDrawHud = not app.doDrawHud
        elif key == 'J':
            app.client.cinematic = not app.client.cinematic
        elif key == 'T':
            app.mode = ChatMode(app, self, '')
        elif key == '/':
            app.mode = ChatMode(app, self, '/')
        elif player.flying:
            player.velocity[1] = 0.0

    def keyReleased(self, app, event):
        if self.overlay is not None:
            self.overlay.keyReleased(app, event)
            return

        client: ClientState = app.client

        key = event.key.upper()
        if key == 'W':
            client.w = False
        elif key == 'S':
            client.s = False 
        elif key == 'A':
            client.a = False
        elif key == 'D':
            client.d = False
        elif key == 'SHIFT':
            client.shift = False
        elif key == 'SPACE' or key == ' ':
            client.space = False

def handleS2CPackets(mode, app, client: ClientState):
    player = client.getPlayer()
    entities = client.entities

    while not network.s2cQueue.empty():
        packet = network.s2cQueue.get()
        if isinstance(packet, network.PlayerPositionAndLookS2C):
            player.pos[0] = packet.x + (player.pos[0] if packet.xRel else 0.0)
            player.pos[1] = packet.y + (player.pos[1] if packet.yRel else 0.0)
            player.pos[2] = packet.z + (player.pos[2] if packet.zRel else 0.0)
            # TODO:
            # yaw, pitch

            sendTeleportConfirm(app, packet.teleportId)
        elif isinstance(packet, network.ChunkDataS2C):
            chunkPos = ChunkPos(packet.x, 0, packet.z)
            
            if chunkPos in client.world.chunks:
                del client.world.chunks[chunkPos]

            client.world.serverChunks[chunkPos] = packet
        elif isinstance(packet, network.TimeUpdateS2C):
            # TODO: World age

            client.time = packet.dayTime
        elif isinstance(packet, network.AckPlayerDiggingS2C):
            print(packet)
        elif isinstance(packet, network.SpawnEntityS2C):
            kind = util.REGISTRY.decode('minecraft:entity_type', packet.kind).removeprefix('minecraft:')

            if kind not in ['item', 'tnt']:
                print(f'Ignoring entity kind {packet.kind}')
            else:
                print(f'Adding entity {kind} with ID {packet.entityId}')

                # TODO: remove `app`, UUID
                ent = Entity(app, packet.entityId, kind, packet.x, packet.y, packet.z)
                ent.velocity[0] = packet.xVel / 8000
                ent.velocity[1] = packet.yVel / 8000
                ent.velocity[2] = packet.zVel / 8000
                ent.headYaw = packet.yaw
                ent.headPitch = packet.pitch

                entities.append(ent)
        elif isinstance(packet, network.SpawnMobS2C):
            kind = util.REGISTRY.decode('minecraft:entity_type', packet.kind).removeprefix('minecraft:')
            if kind in ['zombie', 'creeper', 'fox', 'skeleton']:
                ent = Entity(app, packet.entityId, kind, packet.x, packet.y, packet.z)
                ent.velocity[0] = packet.xVel / 8000
                ent.velocity[1] = packet.yVel / 8000
                ent.velocity[2] = packet.zVel / 8000

                ent.headYaw = packet.yaw
                ent.headPitch = packet.pitch

                entities.append(ent)
            else:
                print(f'Ignoring unknown mob kind {kind}')
        elif isinstance(packet, network.SpawnPlayerS2C):
            ent = Player(app)
            ent.entityId = packet.entityId
            ent.headYaw = packet.yaw
            ent.headPitch = packet.pitch
            ent.pos = [packet.x, packet.y, packet.z]

            entities.append(ent)
        elif isinstance(packet, network.EntityLookS2C):
            for ent in entities:
                if ent.entityId == packet.entityId:
                    ent.bodyAngle = packet.bodyYaw
                    ent.headPitch = packet.headPitch
                    break
        elif isinstance(packet, network.EntityHeadLookS2C):
            for ent in entities:
                if ent.entityId == packet.entityId:
                    ent.headYaw = packet.headYaw
                    break
        elif isinstance(packet, network.EntityVelocityS2C):
            for ent in entities:
                if ent.entityId == packet.entityId:
                    ent.velocity[0] = packet.xVel / 8000
                    ent.velocity[1] = packet.yVel / 8000
                    ent.velocity[2] = packet.zVel / 8000
                    break
        elif isinstance(packet, network.EntityLookRelMoveS2C):
            for ent in entities:
                if ent.entityId == packet.entityId:
                    ent.pos[0] += packet.dx / (128*32)
                    ent.pos[1] += packet.dy / (128*32)
                    ent.pos[2] += packet.dz / (128*32)
                    # TODO: Is this body or head yaw?
                    ent.headYaw = packet.yaw
                    ent.headPitch = packet.pitch
                    ent.onGround = packet.onGround
                    break
        elif isinstance(packet, network.EntityRelMoveS2C):
            for ent in entities:
                if ent.entityId == packet.entityId:
                    ent.pos[0] += packet.dx / (128*32)
                    ent.pos[1] += packet.dy / (128*32)
                    ent.pos[2] += packet.dz / (128*32)
                    ent.onGround = packet.onGround
                    break
        elif isinstance(packet, network.EntityTeleportS2C):
            for ent in entities:
                if ent.entityId == packet.entityId:
                    ent.pos = [packet.x, packet.y, packet.z]
                    # TODO: Is this body or head yaw?
                    ent.headYaw = packet.yaw
                    ent.headPitch = packet.pitch
                    ent.onGround = packet.onGround
                    break
        elif isinstance(packet, network.EntityMetadataS2C):
            for ent in entities:
                if ent.entityId == packet.entityId:
                    print(packet)
                    for (ty, idx), value in packet.metadata.items():
                        if idx == 7 and ent.kind.name == 'item':
                            if value['item'] is None:
                                # TODO: ????
                                pass
                            else:
                                itemId = util.REGISTRY.decode('minecraft:item', value['item']).removeprefix('minecraft:')
                                print(itemId)
                                ent.extra.stack = Stack(itemId, value['count'])
                        elif idx == 7 and ent.kind.name == 'tnt':
                            ent.extra.fuse = value
                        else:
                            # TODO:
                            pass
                    break
        elif isinstance(packet, network.WindowItemsS2C):
            print(packet)
        elif isinstance(packet, network.SetSlotS2C):
            if packet.itemId is None:
                stack = Stack('', 0)
            else:
                stack = Stack(
                    util.REGISTRY.decode('minecraft:item', packet.itemId).removeprefix('minecraft:'),
                    packet.count)

            if packet.windowId == 0:
                if 9 <= packet.slotIdx < 45:
                    print(f'Setting player inventory at {packet.slotIdx} to {stack}')
                    player.inventory[packet.slotIdx % 36].stack = stack
                else:
                    # TODO:
                    print(f'Other slot: {packet.slotIdx}')
            elif hasattr(mode.overlay, 'heldItem') and packet.windowId == -1 and packet.slotIdx == -1:
                mode.overlay.heldItem = stack
            elif mode.overlay is not None and hasattr(mode.overlay.gui, 'slots') and packet.windowId == mode.overlay.windowId:
                mode.overlay.gui.slots[packet.slotIdx][2].stack = stack
            else:
                # TODO:
                print(f'window ID: {packet.windowId}, stack: {stack}')
        elif isinstance(packet, network.WindowPropertyS2C):
            if mode.overlay is not None:
                if packet.windowId == mode.overlay.windowId:
                    if hasattr(mode.overlay.gui, 'fuelLeft'):
                        if packet.property == 0:
                            mode.overlay.gui.fuelLeft = packet.value
                        elif packet.property == 1:
                            mode.overlay.gui.maxFuelLeft = packet.value
                        elif packet.property == 2:
                            mode.overlay.gui.progress = packet.value
                        elif packet.property == 3:
                            mode.overlay.gui.maxProgress = packet.value
                        else:
                            raise Exception(packet)
                    else:
                        print(f'TODO: {packet}')
                else:
                    # TODO:
                    print(f'window ID: {packet.windowId}')
        elif isinstance(packet, network.DestroyEntitiesS2C):
            entIdx = 0
            while entIdx < len(entities):
                if entities[entIdx].entityId in packet.entityIds:
                    entities.pop(entIdx)
                else:
                    entIdx += 1
        elif isinstance(packet, network.UpdateLightS2C):
            #print(f'Light update at {packet.chunkX} {packet.chunkZ}, trust edges: {packet.trustEdges}')

            # TODO:

            if ChunkPos(packet.chunkX, 0, packet.chunkZ) not in client.world.chunks:
                # FIXME:
                continue

            chunk = client.world.chunks[ChunkPos(packet.chunkX, 0, packet.chunkZ)]

            for idx in range(1, 17):
                skyLight = packet.skyLights[idx]
                if skyLight is not None:
                    pass
                elif packet.emptySkyLights[idx]:
                    chunk.lightLevels[:, (idx-1)*16:idx*16, :] = 0
                
                blockLight = packet.blockLights[idx]
                if blockLight is not None:
                    pass
                elif packet.emptyBlockLights[idx]:
                    chunk.blockLightLevels[:, (idx-1)*16:idx*16, :] = 0

        elif isinstance(packet, network.MultiBlockChangeS2C):
            if ChunkPos(packet.chunkX, 0, packet.chunkZ) not in client.world.chunks:
                continue
        
            chunk = client.world.chunks[ChunkPos(packet.chunkX, 0, packet.chunkZ)]

            for blockStateId, pos in packet.blocks:
                blockStateId = util.REGISTRY.decode_block(blockStateId)
                blockId = blockStateId.pop('name').removeprefix('minecraft:')
                blockId = world.convertBlock(blockId, (app.textures, app.cube, app.textureIndices))

                pos = BlockPos(pos.x, pos.y + packet.chunkSectionY * 16, pos.z)

                try:
                    chunk.setBlock(client.world, (app.textures, app.cube, app.textureIndices), pos, blockId, blockStateId, doBlockUpdates=False)
                except Exception as e:
                    print(f'Ignoring exception in MultiBlockChange: {e}')

        elif isinstance(packet, network.BlockChangeS2C):
            blockStateId = util.REGISTRY.decode_block(packet.blockId)
            blockId = blockStateId.pop('name').removeprefix('minecraft:')
            blockId = world.convertBlock(blockId, (app.textures, app.cube, app.textureIndices))

            try:
                if not client.local:
                    client.world.setBlock((app.textures, app.cube, app.textureIndices), packet.location, blockId, blockStateId, doBlockUpdates=False)
            except KeyError as e:
                print(f'Ignoring exception when handling BlockChange packet: {e}')
        elif isinstance(packet, network.WindowConfirmationS2C):
            print(packet)
        elif isinstance(packet, network.OpenWindowS2C):
            windowName = util.REGISTRY.decode('minecraft:menu', packet.kind).removeprefix('minecraft:')

            print(f'Opening window {windowName} with ID {packet.windowId}')

            app.mode.overlay = InventoryMode(app, packet.windowId, windowName)
        elif isinstance(packet, network.ChatMessageS2C):
            app.client.chat.append((time.time(), packet.data))
        elif isinstance(packet, network.JoinGameS2C):
            app.client.player.entityId = packet.entityId

            # TODO:
            print(packet.dimension)

            util.DIMENSION_CODEC = packet.dimensionCodec
        elif isinstance(packet, network.RespawnS2C):
            print(f'RESPAWN: {packet.worldName}, {packet.dimension}')
            if hasattr(app, 'server'):
                app.client.world = app.server.getLocalDimension().world
            else:
                app.client.world.chunks = {}
                app.client.world.serverChunks = {}
        elif packet is None:
            raise Exception("Disconnected")


class ContainerGui:
    slots: List[Tuple[int, int, Slot]]
    actionNum: int

    def __init__(self, slots):
        self.slots = slots
        self.actionNum = 1
    
    def getMcSlot(self, idx: int) -> int:
        return idx
    
    def onClick(self, app, isRight, mx, my, windowId):
        (_, _, w) = render.getSlotCenterAndSize(app, 0)

        for (i, (x, y, slot)) in enumerate(self.slots):
            x0, x1 = x - w/2, x + w/2
            y0, y1 = y - w/2, y + w/2

            if x0 <= mx <= x1 and y0 <= my <= y1:
                if isRight:
                    button = 1
                    mode = 0
                else:
                    button = 0
                    mode = 0
                
                stack = self.slots[i][2].stack
                if stack.isEmpty():
                    item = None
                    count = 0
                else:
                    item = util.REGISTRY.encode('minecraft:item', 'minecraft:' + stack.item)
                    count = stack.amount

                sendClickWindow(app, windowId, self.getMcSlot(i), button, self.actionNum, mode, item, count)

                self.actionNum += 1

                inventory.onSlotClicked(app.mode.overlay.heldItem, app, isRight, slot)
                self.postClick(app, i)
    
    def postClick(self, app, slotIdx):
        pass
        
    def redrawAll(self, app, canvas):
        for (x, y, slot) in self.slots:
            render.drawSlot(app, canvas, x, y, slot)

class FurnaceGui(ContainerGui):
    fuelLeft: int
    maxFuelLeft: int
    progress: int
    maxProgress: int

    def __init__(self, app):
        self.fuelLeft = 0
        self.maxFuelLeft = 1
        self.progress = 0
        self.maxProgress = 200

        slots = [
            (app.width / 2 - 50, app.height / 4 - 30, Slot()),
            (app.width / 2 - 50, app.height / 4 + 30, Slot()),
            (app.width / 2 + 50, app.height / 4, Slot(canInput=False)),
        ]

        super().__init__(slots)
    
    def redrawAll(self, app, canvas):
        super().redrawAll(app, canvas)

        fuelText = str(round(self.fuelLeft / self.maxFuelLeft * 100)) + '%'
        canvas.create_text(app.width / 2 - 50, app.height / 4, text=fuelText)

        progressText = str(round(self.progress / self.maxProgress * 100)) + '%'
        canvas.create_text(app.width / 2, app.height / 4, text=progressText)

def craftingGuiPostClick(gui, app, slotIdx):
    if isinstance(gui, CraftingTableGui):
       totalCraftSlots = 9+1
    else:
        totalCraftSlots = 4+1

    if slotIdx == 0 and gui.prevOutput != gui.slots[0][2].stack:
        # Something was crafted
        for (_, _, slot) in gui.slots[1:totalCraftSlots]:
            if slot.stack.amount > 0:
                slot.stack.amount -= 1

    def toid(s): return None if s.isEmpty() else s.item

    rowLen = round(math.sqrt(totalCraftSlots - 1))

    c = []

    for rowIdx in range(rowLen):
        row = []
        for colIdx in range(rowLen):
            row.append(toid(gui.slots[1 + rowIdx * rowLen + colIdx][2].stack))
        c.append(row)
    
    gui.slots[0][2].stack = Stack('', 0)

    for r in app.recipes:
        if r.isCraftedBy(c):
            gui.slots[0][2].stack = copy.copy(r.outputs)
            break
    
    gui.prevOutput = copy.copy(gui.slots[0][2].stack)

class InventoryCraftingGui(ContainerGui):
    prevOutput: Stack

    def __init__(self, app):
        self.prevOutput = Stack('', 0)

        (_, _, w) = render.getSlotCenterAndSize(app, 0)

        slots = []

        slots.append((460, 100 + w / 2, Slot(canInput=False)))

        for rowIdx in range(2):
            for colIdx in range(2):
                x = colIdx * w + 350
                y = rowIdx * w + 100
                slots.append((x, y, Slot(persistent=False)))

        super().__init__(slots)
    
    def postClick(self, app, slotIdx):
        craftingGuiPostClick(self, app, slotIdx)

class CraftingTableGui(ContainerGui):
    prevOutput: Stack

    def __init__(self, app):
        self.prevOutput = Stack('', 0)

        slots = []
        
        (_, _, w) = render.getSlotCenterAndSize(app, 0)

        slots.append((app.width // 2 + w * 2, 70 + w, Slot(canInput=False)))

        for rowIdx in range(3):
            for colIdx in range(3):
                x = app.width / 2 + (colIdx - 3) * w
                y = 70 + rowIdx * w

                slots.append((x, y, Slot(persistent=False)))
        

        super().__init__(slots)
    
    def postClick(self, app, slotIdx):
        craftingGuiPostClick(self, app, slotIdx)
    
class PauseMode(Mode):
    submode: PlayingMode
    buttons: ButtonManager

    def __init__(self, app, submode: PlayingMode):
        setMouseCapture(app, False)
        self.submode = submode

    def redrawAll(self, app, window, canvas):
        self.submode.redrawAll(app, window, canvas)

def getInventorySlots(app, player) -> List[Any]:
    result = []

    for i in range(27):
        (x, y, _) = render.getSlotCenterAndSize(app, i + 9)
        slot = player.inventory[i + 9]
        result.append((x, y, slot))

    for i in range(9):
        (x, y, _) = render.getSlotCenterAndSize(app, i)
        slot = player.inventory[i]
        result.append((x, y, slot))
    
    return result

class InventoryMode(Mode):
    heldItem: Stack
    player: Player
    windowId: int

    def __init__(self, app, windowId: int, name: str, extra=None):
        setMouseCapture(app, False)
        self.heldItem = Stack('', 0)
        self.craftOutput = Stack('', 0)
        self.windowId = windowId

        self.player = app.client.getPlayer()

        if name == 'inventory':
            self.gui = InventoryCraftingGui(app)
            self.gui.slots += [(0, 0, Slot())] * 4 # Armor slots
        elif name == 'crafting':
            self.gui = CraftingTableGui(app)
        elif name == 'furnace':
            self.gui = FurnaceGui(app)
        else:
            raise Exception(f"unknown gui {name}")
        
        self.gui.slots += getInventorySlots(app, self.player)
        
    def redrawAll(self, app, window, canvas):
        render.drawMainInventory(app.client, canvas)

        self.gui.redrawAll(app, canvas)
        
        if app.mousePos is not None:
            render.drawStack(app.client, canvas, app.mousePos[0], app.mousePos[1],
                self.heldItem)
    
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
    
    def someMousePressed(self, app, event, isRight: bool):
        (_, _, w) = render.getSlotCenterAndSize(app, 0)

        self.gui.onClick(app, isRight, event.x, event.y, self.windowId)
 
    def keyPressed(self, app, event):
        key = event.key.upper()
        if key == 'E':
            for (_, _, slot) in self.gui.slots:
                if not slot.persistent:
                    self.player.pickUpItem(app, slot.stack)

            self.player.pickUpItem(app, self.heldItem)
            self.heldItem = Stack('', 0)
            setMouseCapture(app, True)
            return True
        return False

# Initializes all the data needed to run 112craft
def appStarted(app):
    loadResources(app)

    app.btnBg = createSizedBackground(app, 200, 40)

    app.doDrawHud = True

    app.time = 0

    app.yawSpeed = 0.0
    app.pitchSpeed = 0.0

    # ----------------
    # Player variables
    # ----------------
    app.breakingBlock = 0.0
    app.breakingBlockPos = world.BlockPos(0, 0, 0)
    app.newBreakingBlockPos = world.BlockPos(0, 0, 0)

    app.gravity = 0.10

    if world.CHUNK_HEIGHT == 256: #type:ignore
        app.cameraPos = [2.0, 72, 4.0]
    else:
        app.cameraPos = [2.0, 10.0, 4.0]

    # -------------------
    # Rendering Variables
    # -------------------

    client = ClientState()

    client.height = app.height
    client.width = app.width

    client.cinematic = False

    client.tickTimes = [0.0] * 10
    client.tickTimeIdx = 0

    client.lastTickTime = time.time()

    client.gravity = app.gravity

    client.chat = []

    client.vpDist = 0.25
    client.vpWidth = 3.0 / 4.0
    client.vpHeight = client.vpWidth * app.height / app.width 
    client.wireframe = False
    client.renderDistanceSq = 9**2

    client.breakingBlock = 0.0
    client.breakingBlockPos = BlockPos(0, 0, 0)
    client.lastDigSound = time.time()

    client.cameraPos = [0.0, 0.0, 0.0]
    client.cameraYaw = 0.0
    client.cameraPitch = 0.0
    
    client.time = 0

    client.entities = []

    client.horizFov = math.atan(client.vpWidth / client.vpDist)
    client.vertFov = math.atan(client.vpHeight / client.vpDist)

    print(f"Horizontal FOV: {client.horizFov} ({math.degrees(client.horizFov)}°)")
    print(f"Vertical FOV: {client.vertFov} ({math.degrees(client.vertFov)}°)")

    client.csToCanvasMat = render.csToCanvasMat(client.vpDist, client.vpWidth,
                        client.vpHeight, client.width, client.height)

    app.client = client

    #def makeTitleMode(app, _player): return TitleMode(app)
    #app.mode = WorldLoadMode(app, 'world', True, makeTitleMode)
    def makePlayingMode(app, player): return PlayingMode(app, player)
    app.mode = WorldLoadMode(app, 'world', True, makePlayingMode, seed=random.randint(0, 2**31))
    #app.mode = CreateWorldMode(app)

    # ---------------
    # Input Variables
    # ---------------
    app.mouseMovedDelay = 10

    client.w = False
    client.s = False
    client.a = False
    client.d = False
    client.space = False
    client.shift = False

    app.prevMouse = None

    setMouseCapture(app, False)

def appStopped(app):
    if hasattr(app, 'server'):
        app.server.save()

def updateBlockBreaking(app, mode: PlayingMode):
    client: ClientState = app.client

    if mode.mouseHeld and mode.lookedAtBlock is not None:
        pos, face = mode.lookedAtBlock

        face = { 'bottom': 0, 'top': 1, 'back': 2, 'front': 3, 'left': 4, 'right': 5 }[face]

        if client.breakingBlock == 0.0:
            sendPlayerDigging(app, network.DiggingAction.START_DIGGING, pos, face)

        if client.getPlayer().creative:
            client.breakingBlockPos = pos
            client.breakingBlock = 1000.0
        else:
            if client.breakingBlockPos == pos: 
                client.breakingBlock += 0.1
            else:
                client.breakingBlockPos = pos
                client.breakingBlock = 0.0

        blockId = client.world.getBlock(pos)

        # TODO: This should be a packet too
        if time.time() - client.lastDigSound > 0.2:
            resources.getStepSound(app, blockId).play(halfPitch=True, volume=0.3)
            client.lastDigSound = time.time()

        toolStack = client.player.inventory[client.player.hotbarIdx].stack
        if toolStack.isEmpty():
            tool = ''
        else:
            tool = toolStack.item

        hardness = getHardnessAgainst(blockId, tool)

        if client.breakingBlock >= hardness:
            sendPlayerDigging(app, network.DiggingAction.FINISH_DIGGING, pos, face)

            resources.getDigSound(app, blockId).play()

            if not app.client.local:
                app.client.world.setBlock((app.textures, app.cube, app.textureIndices), pos, 'air', doBlockUpdates=False)
    else:
        if app.breakingBlock > 0.0:
            # FIXME: Face
            sendPlayerDigging(app, network.DiggingAction.CANCEL_DIGGING, app.breakingBlockPos, 0)

        client.breakingBlock = 0.0


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
    
        client: ClientState = app.client

        if not client.cinematic:
            client.cameraPitch += yChange * 0.01
            client.cameraYaw += xChange * 0.01
        
        if client.cameraPitch < -math.pi / 2 * 0.95:
            client.cameraPitch = -math.pi / 2 * 0.95
        elif client.cameraPitch > math.pi / 2 * 0.95:
            client.cameraPitch = math.pi / 2 * 0.95

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
    try:
        if config.USE_OPENGL_BACKEND:
            openglapp.runApp(width=600, height=400)
        else:
            cmu_112_graphics.runApp(width=600, height=400)
    except Exception as e:
        import traceback

        # https://stackoverflow.com/questions/35498555/formatting-exceptions-as-python-does
        traceback.print_exception(type(e), e, e.__traceback__)
    
    network.c2sQueue.put(None)

import threading

from time import sleep

if __name__ == '__main__':
    gameThread = threading.Thread(target=main)
    gameThread.start()

    network.go()
    print("Waiting for game thread to close...")
    gameThread.join()