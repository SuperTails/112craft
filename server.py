from typing import List, Any
from world import World
from entity import Entity
from player import Player
from util import BlockPos

class ServerState:
    world: World
    entities: List[Entity]
    players: List[Player]
    time: int

    renderDistanceSq: int

    chat: List[Any]

    tickTimes: List[float]
    tickTimeIdx: int

    gravity: float