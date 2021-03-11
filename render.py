import math
import time
import world
import numpy as np
from math import sin, cos
from numpy import ndarray
from typing import List, Tuple, Optional, Any
from world import BlockPos, adjacentBlockPos

# =========================================================================== #
# ---------------------------- RENDERING ------------------------------------ #
# =========================================================================== #

# Author: Carson Swoveland (cswovela)
# Part of a term project for 15112

# I shall name my custom rendering engine "Campfire", because:
#   1. It's a minecraft block
#   2. It sounds clever and unique
#   3. You can warm your hands by the heat of your CPU 

Color = str

# Always holds indices into the model's list of vertices
Face = Tuple[int, int, int]

class Model:
    vertices: List[ndarray]
    faces: List[Face]

    def __init__(self, vertices: List[ndarray], faces: List[Face]):
        self.vertices = vertices

        self.faces = []
        for face in faces:
            if len(face) == 4:
                # FIXME:
                1 / 0
            elif len(face) == 3:
                self.faces.append(face)
            else:
                raise Exception("Invalid number of vertices for face")

class Instance:
    model: Model
    trans: ndarray
    texture: List[Color]
    visibleFaces: List[bool]

    _worldSpaceVertices: List[ndarray]

    def __init__(self, model: Model, trans: ndarray, texture: List[Color]):
        self.model = model
        self.trans = trans
        self.texture = texture

        self._worldSpaceVertices = list(map(toHomogenous, self.worldSpaceVerticesUncached()))
        self.visibleFaces = [True] * len(model.faces)

    def worldSpaceVertices(self) -> List[ndarray]:
        return self._worldSpaceVertices
    
    def worldSpaceVerticesUncached(self) -> List[ndarray]:
        result = []
        for vertex in self.model.vertices:
            result.append(vertex + self.trans)
        return result
    
def toHomogenous(cartesian: ndarray) -> ndarray:
    #assert(cartesian.shape[1] == 1)

    #return np.vstack((cartesian, np.array([[1]])))

    # This one line change makes the world load *twice as fast*
    return np.array([[cartesian[0, 0]], [cartesian[1, 0]], [cartesian[2, 0]], [1.0]])

def toCartesian(cartesian: ndarray) -> ndarray:
    assert(cartesian.shape[1] == 1)

    cart = cartesian.ravel()

    return cart[:-1] / cart[-1]

def rotateX(ang):
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, math.cos(ang), -math.sin(ang), 0.0],
        [0.0, math.sin(ang), math.cos(ang), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotateY(ang):
    return np.array([
        [math.cos(ang), 0.0, math.sin(ang), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-math.sin(ang), 0.0, math.cos(ang), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotateZ(ang):
    return np.array([
        [math.cos(ang), -math.sin(ang), 0.0, 0.0],
        [math.sin(ang), math.cos(ang), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def translationMat(x, y, z):
    return np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0]])

def wsToCanvasMat(camPos, yaw, pitch, vpDist, vpWidth, vpHeight,
    canvWidth, canvHeight):
    vpToCanv = vpToCanvasMat(vpWidth, vpHeight, canvWidth, canvHeight)
    return vpToCanv @ camToVpMat(vpDist) @ wsToCamMat(camPos, yaw, pitch)

def csToCanvasMat(vpDist, vpWidth, vpHeight, canvWidth, canvHeight):
    vpToCanv = vpToCanvasMat(vpWidth, vpHeight, canvWidth, canvHeight)
    return vpToCanv @ camToVpMat(vpDist)

def wsToCamMat(camPos, yaw, pitch):
    # Original technique from
    # https://gamedev.stackexchange.com/questions/168542/camera-view-matrix-from-position-yaw-pitch-worldup
    # (My axes are oriented differently, so the matrix is different)

    # I think I made a mistake in the calculations but this fixes it lol
    yaw = -yaw

    # Here is the camera matrix:
    #
    # x = camPos[0]
    # y = camPos[1]
    # z = camPos[2]
    #
    # cam = [
    #     [cos(yaw),  -sin(pitch)*sin(yaw), cos(pitch)*sin(yaw), x],
    #     [0.0,       cos(pitch),           sin(pitch),          y],
    #     [-sin(yaw), -sin(pitch)*cos(yaw), cos(pitch)*cos(yaw), z],
    #     [0.0,       0.0,                  0.0,                 1.0]
    #]
    #
    # cam = np.linalg.inv(cam)

    y = yaw
    p = pitch

    a = camPos[0]
    b = camPos[1]
    c = camPos[2]

    # This is the manually-calculated inverse of the matrix shown above
    cam = np.array([
        [cos(y), 0.0, -sin(y), c*sin(y) - a*cos(y)],
        [-sin(p)*sin(y), cos(p), -sin(p)*cos(y), c*sin(p)*cos(y) + a*sin(p)*sin(y) - b*cos(p)],
        [cos(p)*sin(y), sin(p), cos(p)*cos(y), -b*sin(p) - a*sin(y)*cos(p) - c*cos(y)*cos(p)],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return cam

def vpToCanvasMat(vpWidth, vpHeight, canvWidth, canvHeight):
    w = canvWidth / vpWidth
    h = -canvHeight / vpHeight

    x = canvWidth * 0.5
    y = canvHeight * 0.5

    return np.array([
        [w, 0.0, x],
        [0.0, h, y],
        [0.0, 0.0, 1.0]])

def wsToCam(point, camPos):
    x = point[0] - camPos[0]
    y = point[1] - camPos[1]
    z = point[2] - camPos[2]

    return [x, y, z]

def camToVpMat(vpDist):
    vpd = vpDist

    return np.array([
        [vpd, 0.0, 0.0, 0.0],
        [0.0, vpd, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])

def camToVp(point, vpDist):
    vpX = point[0] * vpDist / point[2]
    vpY = point[1] * vpDist / point[2]

    return [vpX, vpY]

def vpToCanvas(point, vpWidth, vpHeight, canvWidth, canvHeight):
    canvX = (point[0] / vpWidth + 0.5) * canvWidth
    canvY = (-point[1] / vpHeight + 0.5) * canvHeight

    return [canvX, canvY]

def wsToCanvas(app, point):
    point = toHomogenous(point)
    mat = wsToCanvasMat(app.cameraPos, app.cameraYaw, app.cameraPitch,
        app.vpDist, app.vpWidth, app.vpHeight, app.width, app.height)

    point = mat @ point

    point = toCartesian(point)

    # point = wsToCam(point, app.cameraPos)
    # point = camToVp(point, app.vpDist)
    #point = vpToCanvas(point, app.vpWidth, app.vpHeight, app.width, app.height)
    return point

def faceNormal(v0, v1, v2):
    v0 = toCartesian(v0)
    v1 = toCartesian(v1)
    v2 = toCartesian(v2)

    a = v1 - v0
    b = v2 - v0
    cross = np.cross(a, b)
    return cross

# Vertices must be in camera space
def isBackFace(v0, v1, v2) -> bool:
    # From https://en.wikipedia.org/wiki/Back-face_culling

    normal = faceNormal(v0, v1, v2)
    v0 = toCartesian(v0)

    return -np.dot(v0, normal) >= 0

def blockFaceVisible(app, blockPos: BlockPos, faceIdx: int) -> bool:
    (x, y, z) = adjacentBlockPos(blockPos, faceIdx)

    if world.coordsOccupied(app, BlockPos(x, y, z)):
        return False

    return True

def blockFaceLight(app, blockPos: BlockPos, faceIdx: int) -> int:
    pos = adjacentBlockPos(blockPos, faceIdx)
    (chunk, (x, y, z)) = world.getChunk(app, pos)
    return chunk.lightLevels[x, y, z]

def isBackBlockFace(app, blockPos: BlockPos, faceIdx: int) -> bool:
    # Left 
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
    outOfView = lambda idx: vertices[idx][2] < app.vpDist

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

# This converts the instance's vertices to points in camera space, and then:
# For all blocks, the following happens:
#       - Faces pointing away from the camera are removed
#       - Faces that are hidden 'underground' are removed
#       - The color of each face is adjusted based on lighting
#       - ~~A "fog" is applied~~ NOT IMPLEMENTED!
# For anything else:
#       - Normal back face culling is applied
# 
# Then, the faces are clipped, which may remove, modify, or split faces
# Then a list of faces, their vertices, and their colors are returned
def cullInstance(app, toCamMat: ndarray, inst: Instance, blockPos: Optional[BlockPos]) -> List[Tuple[Any, Face, Color]]:
    vertices = list(map(lambda v: toCamMat @ v, inst.worldSpaceVertices()))

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

            if not blockFaceVisible(app, blockPos, faceIdx):
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

            '''
            avg = (r + g + b) / 3.0
            desaturation = min(vertices[face[0]][2] / 8.0, 1.0)**2
            r += (0.0 - r) * desaturation
            g += (128.0 - g) * desaturation
            b += (255.0 - b) * desaturation
            '''

            r = max(0.0, min(255.0, r))
            g = max(0.0, min(255.0, g))
            b = max(0.0, min(255.0, b))

            color = '#{:02X}{:02X}{:02X}'.format(int(r), int(g), int(b))
        else:
            # Backface culling (surprisingly expensive)
            backFace = isBackFace(
                vertices[face[0]], 
                vertices[face[1]],
                vertices[face[2]]
            )
            if backFace:
                continue

        for clippedFace in clip(app, vertices, face):
            faces.append([vertices, clippedFace, color])

    return faces

def blockPosIsVisible(app, pos: BlockPos) -> bool:
    pitch = app.cameraPitch
    yaw = app.cameraYaw 

    lookX = cos(pitch)*sin(-yaw)
    lookY = sin(pitch)
    lookZ = cos(pitch)*cos(-yaw)

    [camX, camY, camZ] = app.cameraPos
    [blockX, blockY, blockZ] = world.blockToWorld(pos)

    # This is only a conservative estimate, so we move the camera "back"
    # to make sure we don't miss blocks behind us
    camX -= lookX
    camY -= lookY
    camZ -= lookZ

    dot = lookX * (blockX - camX) + lookY * (blockY - camY) + lookZ * (blockZ - camZ)

    return dot >= 0

def renderInstances(app, canvas):
    faces = drawToFaces(app)

    zCoord = lambda d: -(d[0][d[1][0]][2] + d[0][d[1][1]][2] + d[0][d[1][2]][2])
    
    faces.sort(key=zCoord)

    drawToCanvas(app, canvas, faces)

# Straight down: 40ms
# Forward: 35ms

def drawToFaces(app):
    toCamMat = wsToCamMat(app.cameraPos, app.cameraYaw, app.cameraPitch)
    faces = []
    for chunkPos in app.chunks:
        chunk = app.chunks[chunkPos]
        if chunk.isVisible:
            for (i, inst) in enumerate(chunk.instances):
                if inst is not None:
                    (inst, unburied) = inst
                    if unburied:
                        wx = chunk.pos[0] * 16 + (i // 256)
                        wy = chunk.pos[1] * 16 + (i // 16) % 16
                        wz = chunk.pos[2] * 16 + (i % 16)
                        blockPos = BlockPos(wx, wy, wz)
                        if blockPosIsVisible(app, blockPos):
                            (x, y, z) = blockPos
                            x -= app.cameraPos[0]
                            y -= app.cameraPos[1]
                            z -= app.cameraPos[2]
                            if x**2 + y**2 + z**2 <= app.renderDistanceSq:
                                faces += cullInstance(app, toCamMat, inst, blockPos)
    return faces

def drawToCanvas(app, canvas, faces):
    mat = app.csToCanvasMat

    for i in range(len(faces)):
        if type(faces[i][0]) != type((0, 0)):
            verts = list(map(lambda v: toCartesian(mat @ v), faces[i][0]))
            faces[i][0] = (verts, True)

        ((vertices, _), face, color) = faces[i]

        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        if app.wireframe:
            edges = [(v0, v1), (v0, v2), (v1, v2)]

            for (v0, v1) in edges:            
                canvas.create_line(v0[0], v0[1], v1[0], v1[1], fill=color)
        else:
            canvas.create_polygon(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], fill=color)
            '''
            vs = [v0, v1, v2]

            if vs[1][1] <= vs[0][1] and vs[1][1] <= vs[2][1]:
                vs[0], vs[1] = vs[1], vs[0]
            elif vs[2][1] <= vs[0][1] and vs[2][1] <= vs[1][1]:
                vs[0], vs[2] = vs[2], vs[0]
            
            if vs[2][1] <= vs[1][1]:
                vs[1], vs[2] = vs[2], vs[1]

            yMin = int(vs[0][1])
            yMid = int(vs[1][1])
            yMax = int(vs[2][1])

            print(yMin, yMid, yMax)

            if yMin == yMid or yMid == yMax:
                continue

            tallSlope = (vs[2][0] - vs[0][0]) / (yMax - yMin)
            shortSlope1 = (vs[1][0] - vs[0][0]) / (yMid - yMin)
            shortSlope2 = (vs[2][0] - vs[1][0]) / (yMax - yMid)

            for y in range(yMin, yMax + 1):
                tallX = vs[0][0] + tallSlope * (y - yMin)

                if y < yMid:
                    shortX = vs[0][0] + shortSlope1 * (y - yMin)
                else:
                    shortX = vs[1][0] + shortSlope2 * (y - yMin)
                
                minX = int(min(tallX, shortX))
                maxX = int(max(tallX, shortX))

                #for x in range(minX, maxX + 1):
                canvas.create_rectangle(minX, y, maxX, y, outline=color)
            '''

frameTimes = [0.0] * 10
frameTimeIdx = 0

# This is simply a shim function that calls `create_text` twice, with
# the text colored black, then white.
# This makes it more easily legible on both dark and light backgrounds.
def drawTextOutlined(canvas, x, y, **kwargs):
    canvas.create_text(x + 1, y + 1, fill='black', **kwargs)
    canvas.create_text(x, y, fill='white', **kwargs)


def redrawAll(app, canvas):
    startTime = time.time()
    
    canvas.create_rectangle(0.0, 0.0, app.width, app.height, fill='#0080FF')

    renderInstances(app, canvas)

    origin = wsToCanvas(app, np.array([[0.0], [0.0], [0.0]]))
    xAxis = wsToCanvas(app, np.array([[1.0], [0.0], [0.0]]))
    yAxis = wsToCanvas(app, np.array([[0.0], [1.0], [0.0]]))
    zAxis = wsToCanvas(app, np.array([[0.0], [0.0], [1.0]]))

    xpoint = wsToCamMat(app.cameraPos, app.cameraYaw, app.cameraPitch) @ toHomogenous(np.array([[1.0], [0.0], [0.0]]))
    xpoint = toCartesian(xpoint)
    # print(f"x point: {xpoint}")

    canvas.create_line(origin[0], origin[1], xAxis[0], xAxis[1], fill='red')
    canvas.create_line(origin[0], origin[1], yAxis[0], yAxis[1], fill='green')
    canvas.create_line(origin[0], origin[1], zAxis[0], zAxis[1], fill='blue')

    canvas.create_oval(app.width / 2 - 1, app.height / 2 - 1, 
        app.width / 2 + 1, app.height / 2 + 1)

    tickTime = sum(app.tickTimes) / len(app.tickTimes) * 1000.0

    drawTextOutlined(canvas, 10, 25, text=f'Tick Time: {tickTime:.2f}ms', anchor='nw')
    
    global frameTimes
    global frameTimeIdx

    endTime = time.time()
    frameTimes[frameTimeIdx] = (endTime - startTime)
    frameTimeIdx += 1
    frameTimeIdx %= len(frameTimes)
    frameTime = sum(frameTimes) / len(frameTimes) * 1000.0

    drawTextOutlined(canvas, 10, 10, text=f'Frame Time: {frameTime:.2f}ms', anchor='nw')

    chunkX = int(app.cameraPos[0] / 16)
    chunkY = int(app.cameraPos[1] / 16)
    chunkZ = int(app.cameraPos[2] / 16)

    drawTextOutlined(canvas, 10, 55, text=f'Chunk coords: {chunkX}, {chunkY}, {chunkZ}', anchor='nw')