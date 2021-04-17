from typing import List, Any, Optional, Tuple
import typing
from shader import ShaderProgram
from world import World, nearestBlockCoord
from entity import Entity
from player import Player
from util import BlockPos, rayAABBIntersect
import math
from math import sin, cos

class ClientData:
    chunkProgram: ShaderProgram
    blockProgram: ShaderProgram
    skyProgram: ShaderProgram
    entityProgram: ShaderProgram
    guiProgram: ShaderProgram

    textureAtlas: Any
    atlasWidth: int

    breakTextures: List[Any]

    sunTex: Any
    skyboxVao: Any

    entityModels: dict[Any, Any]
    entityTextures: Any
    entityAnimations: Any

    glTextures: dict[Any, Any]

    itemTextures: Any

CLIENT_DATA = ClientData()

class ClientState:
    world: World
    entities: List[Entity]
    time: int
    player: Player

    breakingBlock: float
    breakingBlockPos: BlockPos
    lastDigSound: float

    cameraPos: List[float]
    cameraPitch: float
    cameraYaw: float

    height: int
    width: int

    horizFov: float
    vertFov: float

    vpDist: float
    vpWidth: float
    vpHeight: float

    wireframe: bool

    renderDistanceSq: int

    chat: List[Tuple[float, Any]]

    tickTimes: List[float]
    tickTimeIdx: int

    csToCanvasMat: Any

    gravity: float

    w: bool
    a: bool
    s: bool
    d: bool
    shift: bool
    space: bool

    def getPlayer(self) -> Optional[Player]:
        return self.player

def lookedAtBlock(client: ClientState) -> Optional[Tuple[BlockPos, str]]:
    lookX = cos(client.cameraPitch)*sin(-client.cameraYaw)
    lookY = sin(client.cameraPitch)
    lookZ = cos(client.cameraPitch)*cos(-client.cameraYaw)

    if lookX == 0.0:
        lookX = 1e-6
    if lookY == 0.0:
        lookY = 1e-6
    if lookZ == 0.0:
        lookZ = 1e-6

    mag = math.sqrt(lookX**2 + lookY**2 + lookZ**2)
    lookX /= mag
    lookY /= mag
    lookZ /= mag

    # From the algorithm/code shown here:
    # http://www.cse.yorku.ca/~amana/research/grid.pdf

    x = nearestBlockCoord(client.cameraPos[0])
    y = nearestBlockCoord(client.cameraPos[1])
    z = nearestBlockCoord(client.cameraPos[2])

    stepX = 1 if lookX > 0.0 else -1
    stepY = 1 if lookY > 0.0 else -1
    stepZ = 1 if lookZ > 0.0 else -1

    tDeltaX = 1.0 / abs(lookX)
    tDeltaY = 1.0 / abs(lookY)
    tDeltaZ = 1.0 / abs(lookZ)

    nextXWall = x + 0.5 if stepX == 1 else x - 0.5
    nextYWall = y + 0.5 if stepY == 1 else y - 0.5
    nextZWall = z + 0.5 if stepZ == 1 else z - 0.5

    tMaxX = (nextXWall - client.cameraPos[0]) / lookX
    tMaxY = (nextYWall - client.cameraPos[1]) / lookY
    tMaxZ = (nextZWall - client.cameraPos[2]) / lookZ

    blockPos = None
    lastMaxVal = 0.0
    
    reach = client.getPlayer().reach

    while 1:
        if client.world.coordsOccupied(BlockPos(x, y, z)):
            blockPos = BlockPos(x, y, z)
            break

        minVal = min(tMaxX, tMaxY, tMaxZ)

        if minVal == tMaxX:
            x += stepX
            # FIXME: if outside...
            lastMaxVal = tMaxX
            tMaxX += tDeltaX
        elif minVal == tMaxY:
            y += stepY
            lastMaxVal = tMaxY
            tMaxY += tDeltaY
        else:
            z += stepZ
            lastMaxVal = tMaxZ
            tMaxZ += tDeltaZ
        
        if lastMaxVal > reach:
            break
    
    if blockPos is None:
        return None
    else:
        pointX = client.cameraPos[0] + lastMaxVal * lookX
        pointY = client.cameraPos[1] + lastMaxVal * lookY
        pointZ = client.cameraPos[2] + lastMaxVal * lookZ

        pointX -= blockPos.x
        pointY -= blockPos.y
        pointZ -= blockPos.z

        if abs(pointX) > abs(pointY) and abs(pointX) > abs(pointZ):
            face = 'right' if pointX > 0.0 else 'left'
        elif abs(pointY) > abs(pointX) and abs(pointY) > abs(pointZ):
            face = 'top' if pointY > 0.0 else 'bottom'
        else:
            face = 'front' if pointZ > 0.0 else 'back'
        
        return (blockPos, face)

def lookedAtEntity(client: ClientState) -> Optional[int]:
    lookX = cos(client.cameraPitch)*sin(-client.cameraYaw)
    lookY = sin(client.cameraPitch)
    lookZ = cos(client.cameraPitch)*cos(-client.cameraYaw)

    if lookX == 0.0:
        lookX = 1e-6
    if lookY == 0.0:
        lookY = 1e-6
    if lookZ == 0.0:
        lookZ = 1e-6

    mag = math.sqrt(lookX**2 + lookY**2 + lookZ**2)
    lookX /= mag
    lookY /= mag
    lookZ /= mag

    rayOrigin = typing.cast(Tuple[float, float, float], tuple(client.cameraPos))

    rayDir = (lookX, lookY, lookZ)

    inters = []

    reach = client.getPlayer().reach

    for idx, entity in enumerate(client.entities):
        if abs(entity.pos[0] - client.cameraPos[0]) + abs(entity.pos[2] - client.cameraPos[2]) > 2 * reach:
            continue

        (aabb0, aabb1) = entity.getAABB()
        inter = rayAABBIntersect(rayOrigin, rayDir, aabb0, aabb1)
        if inter is not None:
            inters.append((idx, inter))
        
    def dist(inter):
        (_, i) = inter
        dx = i[0] - rayOrigin[0]
        dy = i[1] - rayOrigin[1]
        dz = i[2] - rayOrigin[2]
        return math.sqrt(dx**2 + dy**2 + dz**2)
    
    inters.sort(key=dist)

    if inters == []:
        return None
    else:
        inter = inters[0]
        if dist(inter) > reach:
            print(f"dist: {dist(inter)}")
            return None
        else:
            return inter[0]
