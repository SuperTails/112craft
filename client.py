from typing import List, Any, Optional
from shader import ShaderProgram
from world import World
from entity import Entity
from player import Player
from util import BlockPos

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

    tickTimes: List[float]

    csToCanvasMat: Any

    def getPlayer(self) -> Optional[Player]:
        return self.player