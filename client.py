from typing import List, Any, Optional, Tuple
import typing
from shader import ShaderProgram
import world
from world import World, nearestBlockCoord
from entity import Entity, EntityModel, Animation, EntityRenderData, AnimController
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

    entityRenderData: dict[str, EntityRenderData]
    entityModels: dict[str, EntityModel]
    entityTextures: dict[str, int]
    entityAnimations: dict[str, Animation]
    entityAnimControllers: dict[str, AnimController]

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
    
    local: bool

    renderDistanceSq: int

    lastTickTime: float

    chat: List[Tuple[float, Any]]

    tickTimes: List[float]
    tickTimeIdx: int

    csToCanvasMat: Any

    gravity: float

    cinematic: bool

    w: bool
    a: bool
    s: bool
    d: bool
    shift: bool
    space: bool

    def getPlayer(self) -> Optional[Player]:
        return self.player

    def lookedAtBlock(self, useFluids: bool = False) -> Optional[Tuple[BlockPos, str]]:
        player = self.getPlayer()
        assert(player is not None)
        cameraPos = typing.cast(Tuple[float, float, float], tuple(self.cameraPos))
        return self.world.lookedAtBlock(player.reach, cameraPos, self.cameraPitch, self.cameraYaw, useFluids)

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

def getLookVector(client: ClientState) -> Tuple[float, float, float]:
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

    return (lookX, lookY, lookZ)

