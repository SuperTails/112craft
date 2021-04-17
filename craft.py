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
import time
import cmu_112_graphics
import tkinter
import entity
from tick import *
import tick
import server
from queue import SimpleQueue
from client import ClientState
from util import ChunkPos, BlockPos
from button import Button, ButtonManager, createSizedBackground
from world import Chunk, World
from typing import List, Optional, Tuple, Any
from enum import Enum
from player import Player, Slot, Stack
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

    def __init__(self, app, worldName, local: bool, nextMode, seed=None, importPath=''):
        self.nextMode = nextMode

        app.timerDelay = 10

        if seed is None:
            seed = random.random()

        app.world = World(worldName, seed, importPath=importPath)
        app.world.local = local

        if local:
            try:
                path = app.world.saveFolderPath() + '/entities.dat'

                nbtfile = nbt.NBTFile(path)

                self.player = Player(app, tag=nbtfile["Entities"][0])

                app.entities = [entity.Entity(app, nbt=tag) for tag in nbtfile["Entities"][1:]]
            except FileNotFoundError:
                self.player = Player(app)
                self.player.pos[1] = 75.0
                app.entities = [entity.Entity(app, 'skeleton', 0.0, 71.0, 1.0), entity.Entity(app, 'fox', 5.0, 72.0, 3.0)]

            cx = math.floor(self.player.pos[0] / 16)
            cy = math.floor(self.player.pos[1] / world.CHUNK_HEIGHT)
            cz = math.floor(self.player.pos[2] / 16)

            app.world.loadChunk((app.textures, app.cube, app.textureIndices), ChunkPos(cx, cy, cz))

        else:
            self.player = Player(app)
            app.entities = []
        
            network.host = worldName
        
    def timerFired(self, app):
        if self.loadStage < 10:
            world.loadUnloadChunks(app, self.player.pos)
        elif self.loadStage < 20:
            world.tickChunks(app, maxTime=5.0)
        else:
            tick.syncClient(app)
            app.mode = self.nextMode(app, self.player)
            
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
                makePlayingMode = lambda app: PlayingMode(app, Player(app, isCreative))

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

def submitChat(app, text: str):
    network.c2sQueue.put(network.ChatMessageC2S(text))

    if text.startswith('/'):
        text = text.removeprefix('/')

        parts = text.split()

        print(f"COMMAND {text}")

        if parts[0] == 'pathfind':
            player: Player = app.mode.player
            target = player.getBlockPos()
            for ent in app.entities:
                ent.updatePath(app.world, target)
        elif parts[0] == 'give':
            itemId = parts[1]
            if len(parts) == 3:
                amount = int(parts[2])
            else:
                amount = 1
            app.mode.player.pickUpItem(app, Stack(itemId, amount))
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
        elif parts[0] == 'summon':
            player = app.mode.player
            ent = entity.Entity(app, parts[1],
                player.pos[0]+0.5, player.pos[1]+0.5, player.pos[2]+0.5)
            app.entities.append(ent)
        elif parts[0] == 'tp':
            player = app.mode.player
            player.pos[0] = float(parts[1])
            player.pos[1] = float(parts[2])
            player.pos[2] = float(parts[3])

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
        elif key == 'SPACE':
            self.text += ' '
        elif len(key) == 1:
            self.text += key.lower()
        
    def redrawAll(self, app, window, canvas):
        self.submode.redrawAll(app, window, canvas)

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

    player: Player

    def __init__(self, app, player: Player):
        app.world.local = False
        app.world.registry = resources.getRegistry()

        app.timerDelay = 100
        setMouseCapture(app, True)

        self.player = player

        sendClientStatus(app, 0)

    def redrawAll(self, app, window, canvas):
        render.redrawAll(app.client, canvas, doDrawHud=app.doDrawHud)
    
    def timerFired(self, app):
        self.lookedAtBlock = world.lookedAtBlock(app)

        if app.cinematic:
            # TODO: Use framerate instead
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

        while not network.s2cQueue.empty():
            packet = network.s2cQueue.get()
            if isinstance(packet, network.PlayerPositionAndLookS2C):
                player = app.client.getPlayer()
                player.pos[0] = packet.x + (player.pos[0] if packet.xRel else 0.0)
                player.pos[1] = packet.y + (player.pos[1] if packet.yRel else 0.0)
                player.pos[2] = packet.z + (player.pos[2] if packet.zRel else 0.0)
                # TODO:
                # yaw, pitch

                sendTeleportConfirm(app, packet.teleportId)
            elif isinstance(packet, network.ChunkDataS2C):
                chunkPos = ChunkPos(packet.x, 0, packet.z)
                
                if chunkPos in app.world.chunks:
                    del app.world.chunks[chunkPos]

                app.world.serverChunks[chunkPos] = packet
            elif isinstance(packet, network.TimeUpdateS2C):
                # TODO: World age

                app.time = packet.dayTime
            elif isinstance(packet, network.AckPlayerDiggingS2C):
                print(packet)
            elif isinstance(packet, network.SpawnEntityS2C):
                if packet.kind == 37:
                    kind = 'item'
                else:
                    kind = None
                
                if kind is None:
                    print(f'Ignoring entity kind {packet.kind}')
                else:
                    # TODO: UUID
                    ent = entity.Entity(app, kind, packet.x, packet.y, packet.z)
                    ent.velocity[0] = packet.xVel / 8000
                    ent.velocity[1] = packet.yVel / 8000
                    ent.velocity[2] = packet.zVel / 8000
                    ent.headYaw = packet.yaw
                    ent.headPitch = packet.pitch
                    ent.entityId = packet.entityId

                    app.entities.append(ent)
            elif isinstance(packet, network.SpawnPlayerS2C):
                ent = Player(app)
                ent.entityId = packet.entityId
                ent.headYaw = packet.yaw
                ent.headPitch = packet.pitch
                ent.pos = [packet.x, packet.y, packet.z]

                app.entities.append(ent)
            elif isinstance(packet, network.EntityLookS2C):
                for ent in app.entities:
                    if ent.entityId == packet.entityId:
                        ent.bodyAngle = packet.bodyYaw
                        ent.headPitch = packet.headPitch
                        break
            elif isinstance(packet, network.EntityHeadLookS2C):
                for ent in app.entities:
                    if ent.entityId == packet.entityId:
                        ent.headYaw = packet.headYaw
                        break
            elif isinstance(packet, network.EntityVelocityS2C):
                for ent in app.entities:
                    if ent.entityId == packet.entityId:
                        ent.velocity[0] = packet.xVel / 8000
                        ent.velocity[1] = packet.yVel / 8000
                        ent.velocity[2] = packet.zVel / 8000
                        break
            elif isinstance(packet, network.EntityLookRelMoveS2C):
                for ent in app.entities:
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
                for ent in app.entities:
                    if ent.entityId == packet.entityId:
                        ent.pos[0] += packet.dx / (128*32)
                        ent.pos[1] += packet.dy / (128*32)
                        ent.pos[2] += packet.dz / (128*32)
                        ent.onGround = packet.onGround
                        break
            elif isinstance(packet, network.EntityTeleportS2C):
                for ent in app.entities:
                    if ent.entityId == packet.entityId:
                        ent.pos = [packet.x, packet.y, packet.z]
                        # TODO: Is this body or head yaw?
                        ent.headYaw = packet.yaw
                        ent.headPitch = packet.pitch
                        ent.onGround = packet.onGround
                        break
            elif isinstance(packet, network.EntityMetadataS2C):
                for ent in app.entities:
                    if ent.entityId == packet.entityId:
                        print(packet.metadata)
                        for (ty, idx), value in packet.metadata.items():
                            if idx == 7 and ent.kind.name == 'item':
                                if value['item'] is None:
                                    # TODO: ????
                                    pass
                                else:
                                    itemId = app.world.registry.decode('minecraft:item', value['item']).removeprefix('minecraft:')
                                    print(itemId)
                                    ent.extra.stack = Stack(itemId, value['count'])
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
                        app.world.registry.decode('minecraft:item', packet.itemId).removeprefix('minecraft:'),
                        packet.count)

                if packet.windowId == 0:
                    if 9 <= packet.slotIdx < 45:
                        print(f'Setting player inventory at {packet.slotIdx} to {stack}')
                        app.mode.player.inventory[packet.slotIdx % 36].stack = stack
                    else:
                        # TODO:
                        print(f'Other slot: {packet.slotIdx}')
                else:
                    # TODO:
                    print(f'window ID: {packet.windowId}')
            elif isinstance(packet, network.DestroyEntitiesS2C):
                entIdx = 0
                while entIdx < len(app.entities):
                    if app.entities[entIdx].entityId in packet.entityIds:
                        app.entities.pop(entIdx)
                    else:
                        entIdx += 1
            elif isinstance(packet, network.BlockChangeS2C):
                blockId = app.world.registry.decode_block(packet.blockId)
                blockId = blockId['name'].removeprefix('minecraft:')
                blockId = world.convertBlock(blockId, (app.textures, app.cube, app.textureIndices))

                try:
                    world.setBlock(app, packet.location, blockId)
                except KeyError:
                    pass
            elif isinstance(packet, network.WindowConfirmationS2C):
                print(packet)
            elif packet is None:
                raise Exception("Disconnected")
        
        player = app.client.getPlayer()

        client: ClientState = app.client
        sendPlayerLook(app, client.cameraYaw, client.cameraPitch, client.getPlayer().onGround)
        sendPlayerPosition(app, player.pos[0], player.pos[1], player.pos[2], player.onGround)
        sendPlayerMovement(app, player.onGround)

        tick.tick(app)

        if self.player.health <= 0.0:
            app.mode = GameOverMode(app)

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
                dmg = 1.0 + getAttackDamage(app, slot.stack.item)

            entity.hit(app, dmg, knockback)

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
                stack = self.player.inventory[self.player.hotbarIdx].stack
                if stack.amount == 0: return
                
                if stack.item not in app.textures: return

                if stack.amount > 0:
                    stack.amount -= 1
                
                mcFace = { 'bottom': 0, 'top': 1, 'back': 2, 'front': 3, 'left': 4, 'right': 5 }[face]
                
                # TODO: Cursor position, inside block??
                sendPlayerPlacement(app, 0, pos2, mcFace, 0.5, 0.5, 0.5, False)
                
                world.addBlock(app, pos2, stack.item)

                resources.getDigSound(app, app.world.getBlock(pos2)).play()
    
    def mouseReleased(self, app, event):
        self.mouseHeld = False
    
    def keyPressed(self, app, event):
        key = event.key.upper()
        if len(key) == 1 and key.isdigit():
            keyNum = int(key)
            if keyNum != 0:
                sendHeldItemChange(app, keyNum - 1)
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
        elif key == 'Q':
            stack = self.player.inventory[self.player.hotbarIdx].stack
            if not stack.isEmpty():
                ent = entity.Entity(app, 'item', self.player.pos[0], self.player.pos[1] + self.player.height - 0.5, self.player.pos[2])
                ent.extra.stack = Stack(stack.item, 1)

                look = world.getLookVector(app)
                ent.velocity[0] = look[0] * 0.5
                ent.velocity[1] = look[1] * 0.3 + 0.2
                ent.velocity[2] = look[2] * 0.5

                if not stack.isInfinite():
                    stack.amount -= 1
                
                app.entities.append(ent)

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

class ContainerGui:
    slots: List[Tuple[int, int, Slot]]
    windowId: int
    actionNum: int

    def __init__(self, slots, windowId: int):
        self.slots = slots
        self.windowId = windowId
        self.actionNum = 1
    
    def getMcSlot(self, idx: int) -> int:
        return idx
    
    def onClick(self, app, isRight, mx, my):
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
                    item = app.world.registry.encode('minecraft:item', 'minecraft:' + stack.item)
                    count = stack.amount

                sendClickWindow(app, self.windowId, self.getMcSlot(i), button, self.actionNum, mode, item, count)

                self.actionNum += 1

                app.mode.onSlotClicked(app, isRight, slot)
                self.postClick(app, i)
    
    def postClick(self, app, slotIdx):
        pass
        
    def redrawAll(self, app, canvas):
        for (x, y, slot) in self.slots:
            render.drawSlot(app, canvas, x, y, slot)

class FurnaceGui(ContainerGui):
    furnace: world.Furnace

    def __init__(self, app, furnace: world.Furnace):
        self.furnace = furnace

        slots = [
            (app.width / 2 - 50, app.height / 4 - 30, self.furnace.inputSlot),
            (app.width / 2 - 50, app.height / 4 + 30, self.furnace.fuelSlot),
            (app.width / 2 + 50, app.height / 4, self.furnace.outputSlot),
        ]

        super().__init__(slots, app.world.registry.encode('minecraft:menu', 'minecraft:furnace'))

def craftingGuiPostClick(gui, app, slotIdx):
    if slotIdx == 0 and gui.prevOutput != gui.slots[0][2].stack:
        # Something was crafted
        for (_, _, slot) in gui.slots:
            if slot.stack.amount > 0:
                slot.stack.amount -= 1

    def toid(s): return None if s.isEmpty() else s.item

    rowLen = round(math.sqrt((len(gui.slots) - 1)))

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

        super().__init__(slots, 0)
    
    def postClick(self, app, slotIdx):
        craftingGuiPostClick(self, app, slotIdx)

class CraftingTableGui(ContainerGui):
    prevOutput: Stack

    def __init__(self, app):
        self.prevOutput = Stack('', 0)

        slots = []
        
        (_, _, w) = render.getSlotCenterAndSize(app, 0)

        for rowIdx in range(3):
            for colIdx in range(3):
                x = app.width / 2 + (colIdx - 3) * w
                y = 70 + rowIdx * w

                slots.append((x, y, Slot(persistent=False)))
        
        slots.append((app.width // 2 + w * 2, 70 + w, Slot(canInput=False)))

        super().__init__(slots, app.world.registry.encode('minecraft:menu', 'minecraft:crafting'))
    
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

class InventoryGui(ContainerGui):
    def __init__(self, app, player):
        slots = []

        for i in range(36):
            (x, y, _) = render.getSlotCenterAndSize(app, i)
            slot = player.inventory[i]
            slots.append((x, y, slot))
        
        super().__init__(slots, 0)
    
    def getMcSlot(self, idx: int) -> int:
        if idx <= 9:
            return idx + 36
        else:
            return idx

class InventoryMode(Mode):
    submode: PlayingMode
    heldItem: Stack
    player: Player

    def __init__(self, app, submode: PlayingMode, name: str, extra=None):
        setMouseCapture(app, False)
        self.submode = submode
        self.heldItem = Stack('', 0)
        self.craftOutput = Stack('', 0)

        self.player = submode.player

        self.guis: List[Any] = [InventoryGui(app, self.player)]

        if name == 'inventory':
            self.guis.append(InventoryCraftingGui(app))
        elif name == 'crafting_table':
            self.guis.append(CraftingTableGui(app))
        elif name == 'furnace':
            self.guis.append(FurnaceGui(app, extra))
        else:
            raise Exception(f"unknown gui {name}")
        
    def timerFired(self, app):
        self.submode.timerFired(app)

    def redrawAll(self, app, window, canvas):
        self.submode.redrawAll(app, window, canvas)

        render.drawMainInventory(app.client, canvas)

        for gui in self.guis:
            gui.redrawAll(app, canvas)
        
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
    
    def onSlotClicked(self, app, isRight: bool, slot: Slot):
        if slot.canInput and slot.canOutput:
            if isRight:
                self.onRightClickIntoNormalSlot(app, slot)
            else:
                self.onLeftClickIntoNormalSlot(app, slot)
        elif not slot.canInput and slot.canOutput:
            print(f"before: {self.heldItem}, {slot.stack}")
            merged = self.heldItem.tryMergeWith(slot.stack)
            print(f"merged: {merged}")
            if merged is not None:
                self.heldItem = merged
                slot.stack = Stack('', 0)
        else:
            raise Exception("TODO")
    
    def onRightClickIntoNormalSlot(self, app, normalSlot):
        normalSlot = normalSlot.stack
        if self.heldItem.isEmpty():
            # Picks up half of the slot
            if normalSlot.isInfinite():
                amountTaken = 1
            else:
                amountTaken = math.ceil(normalSlot.amount / 2)
                normalSlot.amount -= amountTaken
            self.heldItem = Stack(normalSlot.item, amountTaken)
        else:
            newStack = normalSlot.tryMergeWith(Stack(self.heldItem.item, 1))
            if newStack is not None:
                if not self.heldItem.isInfinite():
                    self.heldItem.amount -= 1
                normalSlot.item = newStack.item
                normalSlot.amount = newStack.amount
    
    def onLeftClickIntoNormalSlot(self, app, normalSlot):
        normalSlot = normalSlot.stack
        newStack = self.heldItem.tryMergeWith(normalSlot)
        if newStack is None or self.heldItem.isEmpty():
            tempItem = self.heldItem.item
            tempAmount = self.heldItem.amount
            self.heldItem.item = normalSlot.item
            self.heldItem.amount = normalSlot.amount
            normalSlot.item = tempItem
            normalSlot.amount = tempAmount
        else:
            self.heldItem = Stack('', 0)
            normalSlot.item = newStack.item
            normalSlot.amount = newStack.amount

    def someMousePressed(self, app, event, isRight: bool):
        (_, _, w) = render.getSlotCenterAndSize(app, 0)

        for gui in self.guis:
            gui.onClick(app, isRight, event.x, event.y)
 
    def keyPressed(self, app, event):
        key = event.key.upper()
        if key == 'E':
            for gui in self.guis:
                for (_, _, slot) in gui.slots:
                    if not slot.persistent:
                        self.submode.player.pickUpItem(app, slot.stack)

            self.submode.player.pickUpItem(app, self.heldItem)
            self.heldItem = Stack('', 0)
            app.mode = self.submode
            setMouseCapture(app, True)

# Initializes all the data needed to run 112craft
def appStarted(app):
    loadResources(app)

    #app.mode = WorldLoadMode(app, 'world', TitleMode)
    def makePlayingMode(app, player): return PlayingMode(app, player)
    app.mode = WorldLoadMode(app, 'localhost', False, makePlayingMode, seed=random.randint(0, 2**31))
    #app.mode = CreateWorldMode(app)

    #app.entities = [entity.Entity(app, 'skeleton', 0.0, 71.0, 1.0), entity.Entity(app, 'fox', 5.0, 72.0, 3.0)]

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
    app.newBreakingBlockPos = world.BlockPos(0, 0, 0)

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

    client = ClientState()

    client.height = app.height
    client.width = app.width

    client.vpDist = 0.25
    client.vpWidth = 3.0 / 4.0
    client.vpHeight = client.vpWidth * app.height / app.width 
    client.wireframe = False
    client.renderDistanceSq = 9**2

    client.breakingBlock = 0.0
    client.breakingBlockPos = BlockPos(0, 0, 0)
    client.lastDigSound = time.time()

    client.horizFov = math.atan(client.vpWidth / client.vpDist)
    client.vertFov = math.atan(client.vpHeight / client.vpDist)

    print(f"Horizontal FOV: {client.horizFov} ({math.degrees(client.horizFov)}°)")
    print(f"Vertical FOV: {client.vertFov} ({math.degrees(client.vertFov)}°)")

    client.csToCanvasMat = render.csToCanvasMat(client.vpDist, client.vpWidth,
                        client.vpHeight, client.width, client.height)

    app.client = client

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
    if hasattr(app, 'world') and app.world.local:
        app.world.save()

        path = app.world.saveFolderPath() + '/entities.dat'

        nbtfile = nbt.NBTFile()
        nbtfile.name = "Entities"
        nbtfile.tags.append(entity.toNbt([app.mode.player] + app.entities))
        nbtfile.write_file(path)

def updateBlockBreaking(app, mode: PlayingMode):
    client: ClientState = app.client

    if mode.mouseHeld and mode.lookedAtBlock is not None:
        pos, face = mode.lookedAtBlock

        face = { 'bottom': 0, 'top': 1, 'back': 2, 'front': 3, 'left': 4, 'right': 5 }[face]

        if client.breakingBlock == 0.0:
            sendPlayerDigging(app, network.DiggingAction.START_DIGGING, pos, face)

        if mode.player.creative:
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

        if client.breakingBlock >= hardness * 2.0:
            sendPlayerDigging(app, network.DiggingAction.FINISH_DIGGING, pos, face)

            resources.getDigSound(app, blockId).play()

            # HACK:
            app.world.setBlock((app.textures, app.cube, app.textureIndices), pos, 'air')
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

        if not app.cinematic:
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
    if config.USE_OPENGL_BACKEND:
        openglapp.runApp(width=600, height=400)
    else:
        cmu_112_graphics.runApp(width=600, height=400)
    
    network.c2sQueue.put(None)

import threading

from time import sleep

if __name__ == '__main__':
    gameThread = threading.Thread(target=main)
    gameThread.start()

    while network.host is None:
        sleep(0.1)

    network.go()
    print("Waiting for game thread to close...")
    gameThread.join()