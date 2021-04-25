from typing import List, Any
from world import World
import entity
from entity import Entity
from player import Player
from util import BlockPos
from nbt import nbt
from inventory import Stack, Slot
from dataclasses import dataclass

@dataclass
class Window:
    playerId: int
    pos: BlockPos
    kind: str

class ServerState:
    world: World
    entities: List[Entity]
    players: List[Player]

    heldItems: dict[int, Stack]
    craftSlots: dict[int, List[Slot]]

    breakingBlock: float
    breakingBlockPos: BlockPos

    localPlayer: int

    teleportId: int
    nextEntityId: int

    nextWindowId: int
    openWindows: dict[int, Window]

    time: int

    renderDistanceSq: int

    chat: List[Any]

    tickTimes: List[float]
    tickTimeIdx: int

    gravity: float

    def __init__(self):
        self.teleportId = 1
        self.nextEntityId = 2
        self.nextWindowId = 1
        self.breakingBlock = 0.0
        self.breakingBlockPos = BlockPos(0, 0, 0)

        self.openWindows = {}
        self.heldItems = {}
        self.craftSlots = {}

        self.tickTimes = [0.0] * 10
        self.tickTimeIdx = 0

        self.time = 0

        self.gravity = 0.10
    
    def getWindowId(self):
        self.nextWindowId += 1
        return self.nextWindowId
    
    def getEntityId(self):
        self.nextEntityId += 1
        return self.nextEntityId

    def getLocalPlayer(self) -> Player:
        for player in self.players:
            if player.entityId == self.localPlayer:
                return player
        raise Exception("No local player")
    
    def save(self):
        self.world.save()

        path = self.world.saveFolderPath() + '/entities.dat'

        nbtfile = nbt.NBTFile()
        nbtfile.name = "Entities"
        # FIXME:
        nbtfile.tags.append(entity.toNbt([self.getLocalPlayer()] + self.entities)) #type:ignore
        nbtfile.write_file(path)
    
    @classmethod
    def open(cls, folderPath, worldName, seed, importPath):
        server = cls()

        server.world = World(worldName, seed, importPath=importPath)