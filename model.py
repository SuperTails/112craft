import config
import copy
from math import sin, cos
from numpy import ndarray
import numpy as np
from typing import List, Tuple
from util import Color

# Always holds indices into the model's list of vertices
Face = Tuple[int, int, int]

class Model:
    vertices: List[ndarray]
    faces: List[Face]

    def __init__(self, vertices: List[ndarray], faces: List[Face]):
        self.vertices = vertices
        self.faces = faces
    
    def transformed(self, f) -> 'Model':
        vertices = [f(v) for v in self.vertices]
        return Model(vertices, self.faces)
    
    def scaled(self, x: float, y: float, z: float) -> 'Model':
        vertices = [v * [x, y, z] for v in self.vertices]
        return Model(vertices, self.faces)

    def fuse(self, other: 'Model') -> 'Model':
        vertices = self.vertices + other.vertices
        faces = copy.copy(self.faces)
        offset = len(faces)
        faces += [(f[0] + offset, f[1] + offset, f[2] + offset) for f in other.faces]
        return Model(vertices, faces)

'''
class CubeInstance:
    trans: ndarray

    texture: int

    def __init__(self, model: Model, trans: ndarray, texture: int):
        self.trans = trans
        self.texture = texture
    
        self._worldSpaceVertices = [toHomogenous(v) for v in self.worldSpaceVerticesUncached()]
        self.visibleFaces = [True] * len(model.faces)

    def worldSpaceVertices(self) -> List[ndarray]:
        return self._worldSpaceVertices
    
    def worldSpaceVerticesUncached(self) -> List[ndarray]:
        return [vertex + self.trans for vertex in self.model.vertices]
'''
    
class Instance:
    """An actual occurrence of a Model in the world.

    An Instance is essentially a Model that has been given a texture
    and a position in the world, so it can actually be displayed.
    """

    model: Model

    # The model's translation (i.e. position)
    trans: ndarray

    texture: List[Color]

    # A cache of what faces are visible, to speed up rendering.
    # This is only used for block models.
    visibleFaces: List[bool]

    _worldSpaceVertices: List[ndarray]

    def __init__(self, model: Model, trans: ndarray, texture: List[Color]):
        self.model = model
        self.trans = trans
        self.texture = texture

        if not config.USE_OPENGL_BACKEND:
            self._worldSpaceVertices = [toHomogenous(v) for v in self.worldSpaceVerticesUncached()]
        self.visibleFaces = [True] * len(model.faces)

    def worldSpaceVertices(self) -> List[ndarray]:
        return self._worldSpaceVertices
    
    def worldSpaceVerticesUncached(self) -> List[ndarray]:
        return [vertex + self.trans for vertex in self.model.vertices]

def toHomogenous(cartesian: ndarray) -> ndarray:
    #assert(cartesian.shape[1] == 1)

    #return np.vstack((cartesian, np.array([[1]])))

    # This one line change makes the world load *twice as fast*
    return np.array([[cartesian[0, 0]], [cartesian[1, 0]], [cartesian[2, 0]], [1.0]])

def toCartesian(c: ndarray) -> ndarray:
    f = c[2][0]

    return np.array([ c[0][0] / f, c[1][0] / f ])

def toCartesianList(c: ndarray):
    f = c[2][0]

    return ( c[0][0] / f, c[1][0] / f )

def worldToCanvasMat(camPos, yaw, pitch, vpDist, vpWidth, vpHeight,
    canvWidth, canvHeight):
    vpToCanv = viewToCanvasMat(vpWidth, vpHeight, canvWidth, canvHeight)
    return vpToCanv @ cameraToViewMat(vpDist) @ worldToCameraMat(camPos, yaw, pitch)

def csToCanvasMat(vpDist, vpWidth, vpHeight, canvWidth, canvHeight):
    vpToCanv = viewToCanvasMat(vpWidth, vpHeight, canvWidth, canvHeight)
    return vpToCanv @ cameraToViewMat(vpDist)

def worldToCameraMat(camPos, yaw, pitch):
    """Calculates a matrix that converts from world space to camera space"""

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
        [        cos(y),    0.0,        -sin(y),                         c*sin(y)-a*cos(y)],
        [-sin(p)*sin(y), cos(p), -sin(p)*cos(y),  c*sin(p)*cos(y)+a*sin(p)*sin(y)-b*cos(p)],
        [ cos(p)*sin(y), sin(p),  cos(p)*cos(y), -b*sin(p)-a*sin(y)*cos(p)-c*cos(y)*cos(p)],
        [           0.0,    0.0,            0.0,                                       1.0]
    ], dtype='float32')

    return cam

def glViewMat(camPos, yaw, pitch):
    return worldToCameraMat(camPos, yaw, pitch).transpose()

def viewToCanvasMat(vpWidth, vpHeight, canvWidth, canvHeight):
    """Calculates a matrix that converts from the view plane to the canvas"""

    w = canvWidth / vpWidth
    h = -canvHeight / vpHeight

    x = canvWidth * 0.5
    y = canvHeight * 0.5

    return np.array([
        [w, 0.0, x],
        [0.0, h, y],
        [0.0, 0.0, 1.0]])

def cameraToViewMat(vpDist):
    """Calculates a matrix that converts from camera space to the view plane"""

    vpd = vpDist

    return np.array([
        [vpd, 0.0, 0.0, 0.0],
        [0.0, vpd, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])

def worldToCanvas(app, point):
    point = toHomogenous(point)
    mat = worldToCanvasMat(app.cameraPos, app.cameraYaw, app.cameraPitch,
        app.vpDist, app.vpWidth, app.vpHeight, app.width, app.height)

    point = mat @ point

    point = toCartesian(point)

    return point

