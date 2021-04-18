from typing import List, Any
from world import World
from entity import Entity
from player import Player
from util import BlockPos

class ServerState:
    world: World
    entities: List[Entity]
    players: List[Player]

    breakingBlock: float
    breakingBlockPos: BlockPos

    localPlayer: int

    teleportId: int

    time: int

    renderDistanceSq: int

    chat: List[Any]

    tickTimes: List[float]
    tickTimeIdx: int

    gravity: float

    def __init__(self):
        self.teleportId = 1
        self.breakingBlock = 0.0
        self.breakingBlockPos = BlockPos(0, 0, 0)

        self.tickTimes = [0.0] * 10
        self.tickTimeIdx = 0

        self.gravity = 0.10

    def getLocalPlayer(self) -> Player:
        for player in self.players:
            if player.entityId == self.localPlayer:
                return player
        raise Exception("No local player")