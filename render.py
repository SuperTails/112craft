"""This module does all of the drawing for both backends.

In particualar, it has:
- View and camera matrix calculation
- The entire process of drawing models
- Hidden surface/view frustrum/back face culling
- GUI drawing helper functions

This module does NOT handle mesh/buffer creation. Those are created in
the modules where they are needed. This one is only for drawing them.

The `redrawAll` function is the most interesting place to start.
"""

import math
import time
from tkinter.constants import X
import world
import config
import numpy as np
import copy
from model import *
from resources import getHardnessAgainst
from math import sin, cos
from numpy import ndarray
from typing import List, Tuple, Optional, Any
from world import BlockPos, adjacentBlockPos
from inventory import Slot, Stack
from cmu_112_graphics import ImageTk # type: ignore
from PIL import Image, ImageDraw
from OpenGL.GL import * #type:ignore

# Author: Carson Swoveland (cswovela)
# Part of a term project for 15112

# I shall name my custom rendering engine "Campfire", because:
#   1. It's a minecraft block
#   2. It sounds clever and unique
#   3. You can warm your hands by the heat of your CPU 

# =========================================================================== #
# ---------------------- COORDINATE CONVERSIONS ----------------------------- #
# =========================================================================== #

def faceNormal(v0, v1, v2):
    """Returns the normal vector for the face represented by v0, v1, v2"""

    v0 = toCartesian(v0)
    v1 = toCartesian(v1)
    v2 = toCartesian(v2)

    a = v1 - v0
    b = v2 - v0
    cross = np.cross(a, b)
    return cross

def isBackFace(v0, v1, v2) -> bool:
    """
    If v0, v1, v2 are vertices in camera space, returns True if the
    face they represent is facing away from the camera.
    """

    # From https://en.wikipedia.org/wiki/Back-face_culling

    normal = faceNormal(v0, v1, v2)
    v0 = toCartesian(v0)

    return -np.dot(v0, normal) >= 0

def blockFaceLight(app, blockPos: BlockPos, faceIdx: int) -> int:
    """Returns the light level of the given face of the block"""

    pos = adjacentBlockPos(blockPos, faceIdx)
    if app.world.coordsInBounds(pos):
        (chunk, (x, y, z)) = app.world.getChunk(pos)
        return chunk.lightLevels[x, y, z]
    else:
        return 7

def isBackBlockFace(app, blockPos: BlockPos, faceIdx: int) -> bool:
    """Returns True if the given face of the block is facing away from the camera"""

    faceIdx //= 2
    (x, y, z) = world.blockToWorld(blockPos)
    xDiff = app.cameraPos[0] - x
    yDiff = app.cameraPos[1] - y
    zDiff = app.cameraPos[2] - z
    # Left, right, near, far, bottom, top
    if faceIdx == 0:
        # Left
        return xDiff > -0.5
    elif faceIdx == 1:
        # Right
        return xDiff < 0.5
    elif faceIdx == 2:
        # Near
        return zDiff > -0.5
    elif faceIdx == 3:
        # Far
        return zDiff < 0.5
    elif faceIdx == 4:
        # Bottom
        return yDiff > -0.5
    else:
        # Top
        return yDiff < 0.5
    
# FIXME: This does NOT preserve the CCW vertex ordering!
# And also adds stuff to `vertices`
def clip(app, vertices: List[Any], face: Face) -> List[Face]:
    def outOfView(idx): return vertices[idx][2] < app.vpDist

    numVisible = (not outOfView(face[0])) + (
        (not outOfView(face[1])) + (not outOfView(face[2])))

    if numVisible == 0:
        return []
    elif numVisible == 3:
        return [face]

    [v0, v1, v2] = sorted(face, key=outOfView)
    
    [[x0], [y0], [z0], _] = vertices[v0]
    [[x1], [y1], [z1], _] = vertices[v1]
    [[x2], [y2], [z2], _] = vertices[v2]

    if numVisible == 2:
        xd = (x2 - x0) * (app.vpDist - z0) / (z2 - z0) + x0
        yd = (y2 - y0) * (app.vpDist - z0) / (z2 - z0) + y0

        xc = (x2 - x1) * (app.vpDist - z1) / (z2 - z1) + x1
        yc = (y2 - y1) * (app.vpDist - z1) / (z2 - z1) + y1

        dIdx = len(vertices)
        vertices.append(np.array([[xd], [yd], [app.vpDist], [1.0]]))
        cIdx = len(vertices)
        vertices.append(np.array([[xc], [yc], [app.vpDist], [1.0]]))

        face0: Face = (v0, v1, dIdx)
        face1: Face = (v0, v1, cIdx)

        return [face0, face1]
    else:
        xa = (x1 - x0) * (app.vpDist - z0) / (z1 - z0) + x0
        ya = (y1 - y0) * (app.vpDist - z0) / (z1 - z0) + y0

        xb = (x2 - x0) * (app.vpDist - z0) / (z2 - z0) + x0
        yb = (y2 - y0) * (app.vpDist - z0) / (z2 - z0) + y0

        aIdx = len(vertices)
        vertices.append(np.array([[xa], [ya], [app.vpDist], [1.0]]))
        bIdx = len(vertices)
        vertices.append(np.array([[xb], [yb], [app.vpDist], [1.0]]))

        clippedFace: Face = (v0, aIdx, bIdx)

        return [clippedFace]

def cullInstance(app, toCamMat: ndarray, inst: Instance, blockPos: Optional[BlockPos]) -> List[Tuple[Any, Face, Color]]:
    """
    This converts the instance's vertices to points in camera space, and then:

    For all blocks, the following happens:
        - Faces pointing away from the camera are removed
        - Faces that are hidden 'underground' are removed
        - The color of each face is adjusted based on lighting
        - ~~A "fog" is applied~~ NOT IMPLEMENTED! TODO:

    For anything else:
        - Normal back face culling is applied
    
    Then the faces are clipped, which may remove, modify, or split faces.
    Then a list of faces, their vertices, and their colors are returned
    """

    vertices = [toCamMat @ v for v in inst.worldSpaceVertices()]

    faces = []

    skipNext = False

    for (faceIdx, (face, color)) in enumerate(zip(inst.model.faces, inst.texture)):
        if skipNext:
            skipNext = False
            continue 

        if blockPos is not None:
            if not inst.visibleFaces[faceIdx]:
                continue
            
            if isBackBlockFace(app, blockPos, faceIdx):
                skipNext = True
                continue

            light = blockFaceLight(app, blockPos, faceIdx)
            r = int(color[1:3], base=16)
            g = int(color[3:5], base=16)
            b = int(color[5:7], base=16)

            brightness = (light + 1) / 8
            r *= brightness
            g *= brightness
            b *= brightness

            if blockPos == app.breakingBlockPos and app.breakingBlock != 0.0:
                avg = (r + g + b) / 3.0

                toolSlot = app.mode.player.inventory[app.mode.player.hotbarIdx]
                if toolSlot.isEmpty():
                    tool = ''
                else:
                    tool = toolSlot.item

                hardness = getHardnessAgainst(app, app.world.getBlock(blockPos), tool)

                desaturation = app.breakingBlock / hardness
                r += (avg - r) * desaturation
                g += (avg - g) * desaturation
                b += (avg - b) * desaturation

            r = max(0.0, min(255.0, r))
            g = max(0.0, min(255.0, g))
            b = max(0.0, min(255.0, b))

            color = '#{:02X}{:02X}{:02X}'.format(int(r), int(g), int(b))
        else:
            # Backface culling (surprisingly expensive)
            '''
            backFace = isBackFace(
                vertices[face[0]], 
                vertices[face[1]],
                vertices[face[2]]
            )
            if backFace:
                continue
            '''

        for clippedFace in clip(app, vertices, face):
            faces.append([vertices, clippedFace, color])

    return faces

def blockPosIsVisible2(camX, camY, camZ, lookX, lookY, lookZ, pos: BlockPos) -> bool:
    [blockX, blockY, blockZ] = world.blockToWorld(pos)

    dot = lookX * (blockX - camX) + lookY * (blockY - camY) + lookZ * (blockZ - camZ)

    return dot >= 0.0

def blockPosIsVisible(app, pos: BlockPos) -> bool:
    pitch = app.cameraPitch
    yaw = app.cameraYaw 

    lookX = cos(pitch)*sin(-yaw)
    lookY = sin(pitch)
    lookZ = cos(pitch)*cos(-yaw)

    [camX, camY, camZ] = app.cameraPos

    # This is only a conservative estimate, so we move the camera "back"
    # to make sure we don't miss blocks behind us
    camX -= lookX
    camY -= lookY
    camZ -= lookZ

    [blockX, blockY, blockZ] = world.blockToWorld(pos)

    dot = lookX * (blockX - camX) + lookY * (blockY - camY) + lookZ * (blockZ - camZ)

    return dot >= 0

def renderInstancesTk(app, canvas):
    faces = drawToFaces(app)

    toCamMat = worldToCameraMat(app.cameraPos, app.cameraYaw, app.cameraPitch)

    for entity in app.entities:
        model = app.entityModels[entity.kind.model]

        for mesh in model.models:
            offset = np.array([[entity.pos[0]], [entity.pos[1]], [entity.pos[2]]])

            texture = [
                '#000000', '#111111',
                '#222222', '#333333',
                '#444444', '#555555',
                '#666666', '#777777',
                '#888888', '#999999',
                '#AAAAAA', '#BBBBBB',
            ]

            inst = Instance(mesh, offset, texture)
        
            faces += cullInstance(app, toCamMat, inst, None)

        #texture = app.entityTextures[entity.kind.name]

        '''
        modelMat = np.array([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            entity.pos[0], entity.pos[1], entity.pos[2], 1.0
        ], dtype='float32')
        '''

        '''
        for (vao, num) in model.vaos:
            if vao == 0: 
                i += 1
                continue

            (x, y, z) = entity.getRotation(app, i)

            glUniform1f(app.entityProgram.getUniformLocation("rotX"), x)
            glUniform1f(app.entityProgram.getUniformLocation("rotY"), y)
            glUniform1f(app.entityProgram.getUniformLocation("rotZ"), z)

            immunity = 1.0 if entity.immunity > 0 else 0.0
            glUniform1f(app.entityProgram.getUniformLocation("immunity"), immunity)

            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, num * 5)

            i += 1
        '''


    def zCoord(d): return -(d[0][d[1][0]][2] + d[0][d[1][1]][2] + d[0][d[1][2]][2])
    
    faces.sort(key=zCoord)

    drawToCanvas(app, canvas, faces)

def renderInstancesGl(app, canvas):
    view = glViewMat(app.cameraPos, app.cameraYaw, app.cameraPitch)

    th = math.tan(0.5 * math.radians(70.0));
    zf = 100.0;
    zn = 0.1;

    projection = np.array([
        [(app.height / app.width) / th, 0.0, 0.0, 0.0],
        [0.0, 1.0 / th, 0.0, 0.0],
        [0.0, 0.0, zf / (zf - zn), 1.0],
        [0.0, 0.0, -(zf * zn) / (zf - zn), 0.0],
    ], dtype='float32')

    #glBindVertexArray(app.cubeVao)

    chunkVaos = []

    for (pos, chunk) in app.world.chunks.items():
        if not chunk.isVisible: continue 

        [cx, cy, cz] = chunk.pos
        cx *= 16
        cy *= world.CHUNK_HEIGHT
        cz *= 16

        if hasattr(chunk, 'meshVaos'):
            for i in range(len(chunk.meshVaos)):
                chunkVaos.append((chunk.meshVertexCounts[i], chunk.meshVaos[i], pos, i))
    
    breakingBlockAmount = 0.0

    if app.breakingBlock != 0.0 and hasattr(app.mode, 'player'):
        toolStack = app.mode.player.inventory[app.mode.player.hotbarIdx].stack
        if toolStack .isEmpty():
            tool = ''
        else:
            tool = toolStack.item

        blockId = app.world.getBlock(app.breakingBlockPos)

        if blockId != 'air':
            hardness = getHardnessAgainst(app, blockId, tool)

            breakingBlockAmount = app.breakingBlock / hardness

    b = math.floor(breakingBlockAmount * 10.0)

    app.chunkProgram.useProgram()
    glUniformMatrix4fv(app.chunkProgram.getUniformLocation("view"), 1, GL_FALSE, view) #type:ignore
    glUniformMatrix4fv(app.chunkProgram.getUniformLocation("projection"), 1, GL_FALSE, projection) #type:ignore
    glUniform1f(app.chunkProgram.getUniformLocation("atlasWidth"), app.atlasWidth)
    glUniform1i(app.chunkProgram.getUniformLocation("gameTime"), app.time)

    glUniform1i(app.chunkProgram.getUniformLocation("blockTexture"), 0)
    glUniform1i(app.blockProgram.getUniformLocation("breakTexture"), 1)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, app.textureAtlas)

    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, app.breakTextures[b])

    (cp, lp) = world.toChunkLocal(app.breakingBlockPos)
    breakBlockIdx = lp.x * 16 * 16 + (lp.y % world.MESH_HEIGHT) * 16 + lp.z
    breakBlockLoc = app.chunkProgram.getUniformLocation("breakBlockIdx")

    #print("drawing a chunk vao")
    for amt, chunkVao, pos, i in chunkVaos:
        if breakingBlockAmount > 0.0 and cp == pos and (lp.y // world.MESH_HEIGHT) == i:
            glUniform1i(breakBlockLoc, breakBlockIdx)
        else:
            glUniform1i(breakBlockLoc, -1)


        glBindVertexArray(chunkVao)

        glDrawArrays(GL_TRIANGLES, 0, amt * 7)
    
    drawEntities(app, view, projection)

    # https://learnopengl.com/Advanced-OpenGL/Cubemaps
    glDisable(GL_CULL_FACE)
    glDepthFunc(GL_LEQUAL)
    app.skyProgram.useProgram()
    view[3, 0:3] = 0.0
    glUniformMatrix4fv(app.skyProgram.getUniformLocation("view"), 1, GL_FALSE, view) #type:ignore
    glUniformMatrix4fv(app.skyProgram.getUniformLocation("projection"), 1, GL_FALSE, projection) #type:ignore
    glUniform1i(app.skyProgram.getUniformLocation("gameTime"), app.time)
    glUniform1i(app.skyProgram.getUniformLocation("sunTex"), 0)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, app.sunTex)

    glBindVertexArray(app.skyboxVao)
    glDrawArrays(GL_TRIANGLES, 0, 36)
    glBindVertexArray(0)

    glEnable(GL_CULL_FACE)
    glDepthFunc(GL_LESS)


def drawEntities(app, view, projection):
    app.entityProgram.useProgram()
    glUniform1i(app.blockProgram.getUniformLocation("skin"), 0)
    glUniformMatrix4fv(app.entityProgram.getUniformLocation("view"), 1, GL_FALSE, view) #type:ignore
    glUniformMatrix4fv(app.entityProgram.getUniformLocation("projection"), 1, GL_FALSE, projection) #type:ignore

    modelPos = app.entityProgram.getUniformLocation("model")
    rotPos = app.entityProgram.getUniformLocation("rot")

    glActiveTexture(GL_TEXTURE0)

    for entity in app.entities:
        model = app.entityModels[entity.kind.model]
        texture = app.entityTextures[entity.kind.name]

        glBindTexture(GL_TEXTURE_2D, texture)

        modelMat = np.array([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            entity.pos[0], entity.pos[1], entity.pos[2], 1.0
        ], dtype='float32')

        glUniformMatrix4fv(modelPos, 1, GL_FALSE, modelMat) #type:ignore
        glUniform1f(rotPos, entity.bodyAngle)

        i = 0

        for (vao, num) in model.vaos:
            if vao == 0: 
                i += 1
                continue

            (x, y, z) = entity.getRotation(app, i)

            glUniform1f(app.entityProgram.getUniformLocation("rotX"), x)
            glUniform1f(app.entityProgram.getUniformLocation("rotY"), y)
            glUniform1f(app.entityProgram.getUniformLocation("rotZ"), z)

            immunity = 1.0 if entity.immunity > 0 else 0.0
            glUniform1f(app.entityProgram.getUniformLocation("immunity"), immunity)

            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, num * 5)

            i += 1




def doTheDraw(app, modelUniformLoc, amt):
    #bId = chunk.blocks[i // 256, (i // 16) % 16, i % 16]
    
    #glDrawArrays(GL_TRIANGLES, 0, 36)
    glDrawArraysInstanced(GL_TRIANGLES, 0, 36, amt)


# Straight down: 40ms
# Forward: 35ms

def makeFrustrumCullCheck(app, pitch, yaw):
    lookX = cos(pitch)*sin(-yaw)
    lookY = sin(pitch)
    lookZ = cos(pitch)*cos(-yaw)

    [camX, camY, camZ] = app.cameraPos

    camX -= lookX
    camY -= lookY
    camZ -= lookZ

    def wrapper(blockPos):
        return blockPosIsVisible2(camX, camY, camZ, lookX, lookY, lookZ, blockPos)
    
    return wrapper

def drawToFaces(app):
    # These perform view frustrum culling. The functions are precalculated at
    # the beginning of the loop because that causes *insanely* good speedups.
    check1 = makeFrustrumCullCheck(app, app.cameraPitch, app.cameraYaw)
    check2 = makeFrustrumCullCheck(app, app.cameraPitch, app.cameraYaw + (app.horizFov / 2))
    check3 = makeFrustrumCullCheck(app, app.cameraPitch, app.cameraYaw - (app.horizFov / 2))
    check4 = makeFrustrumCullCheck(app, app.cameraPitch - (app.vertFov / 2), app.cameraYaw)
    check5 = makeFrustrumCullCheck(app, app.cameraPitch + (app.vertFov / 2), app.cameraYaw)

    [camX, camY, camZ] = app.cameraPos

    renderDist = math.sqrt(app.renderDistanceSq) + 1

    toCamMat = worldToCameraMat(app.cameraPos, app.cameraYaw, app.cameraPitch)
    faces = []
    for chunk in app.world.chunks.values():
        if not chunk.isVisible: continue 

        [cx, cy, cz] = chunk.pos
        cx *= 16
        cy *= world.CHUNK_HEIGHT
        cz *= 16

        for (i, inst) in enumerate(chunk.instances):
            if inst is not None:
                (inst, unburied) = inst
                if unburied:
                    #wx = chunk.pos[0] * 16 + (i // 256)
                    #wy = chunk.pos[1] * 16 + (i // 16) % 16
                    #wz = chunk.pos[2] * 16 + (i % 16)

                    #print(f"Rendering {cx} {cy} {cz} {i} with inst {inst._worldSpaceVertices[0]}")

                    wy = cy + (i // 256)
                    wx = cx + (i // 16) % 16
                    wz = cz + (i % 16)

                    if abs(wx - camX) >= renderDist: continue
                    if abs(wz - camZ) >= renderDist: continue

                    blockPos = BlockPos(wx, wy, wz)

                    # View frustrum culling
                    #if not check1(blockPos): continue
                    if not check2(blockPos): continue
                    if not check3(blockPos): continue
                    if not check4(blockPos): continue
                    if not check5(blockPos): continue

                    wx -= camX
                    wy -= camY
                    wz -= camZ
                    if wx**2 + wz**2 <= app.renderDistanceSq:
                        faces += cullInstance(app, toCamMat, inst, blockPos)
    return faces

def drawToCanvas(app, canvas, faces):
    wv = app.csToCanvasMat[0, 0]
    hv = app.csToCanvasMat[1, 1]
    x = app.csToCanvasMat[0, 2]
    y = app.csToCanvasMat[1, 2]

    def csToCanvas(v):
        a = v[0, 0]
        b = v[1, 0]
        c = v[2, 0]

        x1 = a * wv / c + x 
        y1 = b * hv / c + y
        return (x1, y1)

    for i in range(len(faces)):
        if type(faces[i][0]) != type((0, 0)):
            verts = [csToCanvas(v) for v in faces[i][0]]
            faces[i][0] = (verts, True)

        ((vertices, _), face, color) = faces[i]

        (x0, y0) = vertices[face[0]]
        (x1, y1) = vertices[face[1]]
        (x2, y2) = vertices[face[2]]

        #if app.wireframe:
        #    edges = [(v0, v1), (v0, v2), (v1, v2)]

        #    for (v0, v1) in edges:            
        #        canvas.create_line(v0[0], v0[1], v1[0], v1[1], fill=color)
        #else:

        canvas.create_polygon(x0, y0, x1, y1, x2, y2, fill=color)

frameTimes = [0.0] * 10
frameTimeIdx = 0

def drawTextOutlined(canvas, x, y, **kwargs):
    """
    This is simply a shim function that calls `create_text` twice,
    with the text colored black, then white.
    This makes it more easily legible on both dark and light backgrounds.
    """

    canvas.create_text(x + 2, y + 2, fill='#444', **kwargs)
    canvas.create_text(x, y, fill='white', **kwargs)

def getSlotCenterAndSize(app, slotIdx) -> Tuple[int, int, int]:
    slotWidth = app.itemTextures['air'].width + 7
    if slotIdx < 9:
        margin = 10
        x = (slotIdx - 4) * slotWidth + app.width / 2
        y = app.height - margin - slotWidth / 2
        return (x, y, slotWidth)
    else:
        rowNum = (36 // 9) - 1
        rowIdx = (slotIdx // 9) - 1
        x = ((slotIdx % 9) - 4) * slotWidth + app.width / 2
        y = app.height / 2 - (rowIdx - (rowNum - 1) / 2) * slotWidth 
        return (x, y, slotWidth)


def drawMainInventory(app, canvas):
    # FIXME: 
    if hasattr(app.mode, 'player'):
        player = app.mode.player
    else:
        player = app.mode.submode.player

    slotWidth = app.itemTextures['air'].width + 7

    for i in range(9, 36):
        slot = player.inventory[i]

        (x, y, _) = getSlotCenterAndSize(app, i)

        drawSlot(app, canvas, x, y, slot)


def drawHotbar(app, canvas):
    # FIXME: 
    if hasattr(app.mode, 'player'):
        player = app.mode.player
    else:
        player = app.mode.submode.player

    slotWidth = app.itemTextures['air'].width + 7

    margin = 10

    for (i, slot) in enumerate(player.inventory[:9]):
        x = (i - 4) * slotWidth + app.width / 2

        y = app.height - margin - slotWidth / 2

        drawSlot(app, canvas, x, y, slot)
    
    x = (player.hotbarIdx - 4) * slotWidth + app.width / 2
    y = app.height - margin - slotWidth
    canvas.create_rectangle(x - slotWidth / 2, y,
        x + slotWidth / 2,
        y + slotWidth,
        outline='white')

def drawStack(app, canvas, x, y, stack: Stack):
    slotWidth = app.itemTextures['air'].width + 6

    if not stack.isEmpty():
        tex = app.itemTextures[stack.item]
        image = tex
        canvas.create_image(x, y, image=image)

        # Slots that are infinite or with a single item just don't have a number displayed
        if not stack.isInfinite() and stack.amount != 1:
            cornerX = x + 0.3 * slotWidth
            cornerY = y + 0.2 * slotWidth

            qty = stack.amount

            drawTextOutlined(canvas, cornerX, cornerY, text=str(qty), font='Arial 12 bold')

def drawSlot(app, canvas, x, y, slot: Slot):
    """x and y are the *center* of the slot"""

    slotWidth = app.itemTextures['air'].width + 6

    canvas.create_rectangle(x - slotWidth / 2, y - slotWidth / 2,
        x + slotWidth / 2,
        y + slotWidth / 2,
        fill='#8b8b8b', outline='#373737')

    drawStack(app, canvas, x, y, slot.stack)

def drawHud(app, canvas, startTime):
    # Indicates the center of the screen
    canvas.create_oval(app.width / 2 - 1, app.height / 2 - 1, 
        app.width / 2 + 1, app.height / 2 + 1)

    drawHotbar(app, canvas)

    tickTime = sum(app.tickTimes) / len(app.tickTimes) * 1000.0

    drawTextOutlined(canvas, 10, 30, text=f'Tick Time: {tickTime:.2f}ms', anchor='nw')
    
    global frameTimes
    global frameTimeIdx

    endTime = time.time()
    frameTimes[frameTimeIdx] = (endTime - startTime)
    frameTimeIdx += 1
    frameTimeIdx %= len(frameTimes)
    frameTime = sum(frameTimes) / len(frameTimes) * 1000.0

    drawTextOutlined(canvas, 10, 10, text=f'Frame Time: {frameTime:.2f}ms', anchor='nw')

    drawTextOutlined(canvas, 10, 50, text=f"Eyes: {app.cameraPos[0]:.2f}, {app.cameraPos[1]:.2f}, {app.cameraPos[2]:.2f}", anchor='nw')
    
    chunkX = math.floor(app.cameraPos[0] / 16)
    chunkY = math.floor(app.cameraPos[1] / world.CHUNK_HEIGHT)
    chunkZ = math.floor(app.cameraPos[2] / 16)

    drawTextOutlined(canvas, 10, 140, text=f'Chunk coords: {chunkX}, {chunkY}, {chunkZ}', anchor='nw')

    # FIXME:
    if hasattr(app.mode, 'player'):
        player = app.mode.player
        drawTextOutlined(canvas, 10, 90, text=f"Feet: {player.pos[0]:.2f}, {player.pos[1]:.2f}, {player.pos[2]:.2f}", anchor='nw')

        feetPos = (app.cameraPos[0], app.cameraPos[1] - app.mode.player.height + 0.1, app.cameraPos[2])
        if feetPos[1] < world.CHUNK_HEIGHT - 1:
            lightLevel = app.world.getLightLevel(world.nearestBlockPos(feetPos[0], feetPos[1], feetPos[2]))
            blockLightLevel = app.world.getBlockLightLevel(world.nearestBlockPos(feetPos[0], feetPos[1], feetPos[2]))
            drawTextOutlined(canvas, 10, 190, text=f'Sky {lightLevel}, Block {blockLightLevel}', anchor='nw')

def redrawAll(app, canvas, doDrawHud=True):
    startTime = time.time()
    
    # The sky
    # TODO:
    if not config.USE_OPENGL_BACKEND:
        canvas.create_rectangle(0.0, 0.0, app.width, app.height, fill='#0080FF')

    # The world
    if config.USE_OPENGL_BACKEND:
        renderInstancesGl(app, canvas)
    else:
        renderInstancesTk(app, canvas)

    #origin = worldToCanvas(app, np.array([[0.0], [0.0], [0.0]]))
    #xAxis = worldToCanvas(app, np.array([[1.0], [0.0], [0.0]]))
    #yAxis = worldToCanvas(app, np.array([[0.0], [1.0], [0.0]]))
    #zAxis = worldToCanvas(app, np.array([[0.0], [0.0], [1.0]]))

    #xpoint = worldToCameraMat(app.cameraPos, app.cameraYaw, app.cameraPitch) @ toHomogenous(np.array([[1.0], [0.0], [0.0]]))
    #xpoint = toCartesian(xpoint)
    # print(f"x point: {xpoint}")

    #canvas.create_line(origin[0], origin[1], xAxis[0], xAxis[1], fill='red')
    #canvas.create_line(origin[0], origin[1], yAxis[0], yAxis[1], fill='green')
    #canvas.create_line(origin[0], origin[1], zAxis[0], zAxis[1], fill='blue')

    if doDrawHud: drawHud(app, canvas, startTime)

def drawItemFromBlock2(sz: int, base: Image.Image) -> Image.Image:
    from typing import cast

    midX = int((sz - 1) / 2)

    partH = sz / 5

    bottomLeft = (0, int(partH * 4))
    bottomMiddle = (midX, sz - 1)
    bottomRight = (sz - 1, int(partH * 4))

    centerLeft = (0, int(partH))
    centerMiddle = (midX, int(partH * 2))
    centerRight = (sz - 1, int(partH))

    im = Image.new('RGBA', (sz, sz))
    draw = cast(ImageDraw.ImageDraw, ImageDraw.Draw(im))

    for x in range(0, 16):
        srcX = 16 + x
        dstX = int(x / 16.0 * midX)

        for y in range(0, 16):
            srcY = 16 + y
            dstY = centerLeft[1] + y + int(partH * x / 16.0)

            pixel = base.getpixel((srcX, srcY))

            im.putpixel((dstX, dstY), pixel)
    
    for x in range(0, 16):
        srcX = 32 + x
        dstX = int((x / 16.0 + 1.0) * midX)

        for y in range(0, 16):
            srcY = 16 + y
            dstY = int(centerMiddle[1] + (y / 16.0) * partH * 3 - partH * (x / 16.0))

            pixel = base.getpixel((srcX, srcY))


            im.putpixel((dstX, dstY), pixel)

    for x in range(0, 16):
        srcX = 16 + x

        for y in range(0, 16):
            srcY = y

            dstX = int((x - y + 16) / 32.0 * sz)
            dstY = int((y + x) / 32.0 * partH * 2)

            pixel = base.getpixel((srcX, srcY))

            im.putpixel((dstX, dstY), pixel)

    return im

def drawItemFromBlock(size: int, textures: List[Color]) -> Image.Image:
    from typing import cast

    sz = size

    im = Image.new('RGBA', (sz, sz))

    # This cast does nothing but suppress a boatload of 
    # false-positive type errors from Pylance. 
    draw = cast(ImageDraw.ImageDraw, ImageDraw.Draw(im))

    midX = int((sz - 1) / 2)

    partH = sz / 5

    fill = '#000'

    bottomLeft = (0, int(partH * 4))
    bottomMiddle = (midX, sz - 1)
    bottomRight = (sz - 1, int(partH * 4))

    centerLeft = (0, int(partH))
    centerMiddle = (midX, int(partH * 2))
    centerRight = (sz - 1, int(partH))

    top = (midX, 0)

    # 2, 6, 10

    color = textures[2]
    draw.polygon([bottomMiddle, bottomLeft, centerLeft], fill=color)
    color = textures[3]
    draw.polygon([bottomMiddle, centerLeft, centerMiddle], fill=color)

    color = textures[6]
    draw.polygon([bottomMiddle, bottomRight, centerRight], fill=color)
    color = textures[7]
    draw.polygon([bottomMiddle, centerRight, centerMiddle], fill=color)

    color = textures[10]
    draw.polygon([centerMiddle, centerLeft, centerRight], fill=color)
    color = textures[11]
    draw.polygon([centerLeft, centerRight, top], fill=color)

    draw.line([bottomMiddle, bottomLeft], fill=fill) # Bottom left
    draw.line([bottomMiddle, bottomRight], fill=fill) # Bottom right
    draw.line([bottomMiddle, centerMiddle], fill=fill) # Center vertical
    draw.line([bottomLeft, centerLeft], fill=fill) # Left vertical
    draw.line([bottomRight, centerRight], fill=fill) # Right vertical
    draw.line([centerLeft, top], fill=fill) # Very top left
    draw.line([centerRight, top], fill=fill) # Very top right
    draw.line([centerMiddle, centerLeft], fill=fill) # Top left
    draw.line([centerMiddle, centerRight], fill=fill) # Top right

    return im

def getCachedImage(image):
# From:
# https://www.kosbie.net/cmu/fall-19/15-112/notes/notes-animations-part2.html
    if ('cachedPhotoImage' not in image.__dict__):
        image.cachedPhotoImage = ImageTk.PhotoImage(image)
    return image.cachedPhotoImage

