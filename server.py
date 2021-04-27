from typing import List, Any
import world
from world import World
import entity
import math
import os
from entity import Entity
from player import Player
from util import BlockPos, ChunkPos
from nbt import nbt
from inventory import Stack, Slot
from dimension import Dimension
from dataclasses import dataclass

@dataclass
class Window:
    playerId: int
    pos: BlockPos
    kind: str

class ServerState:
    dimensions: List[Dimension]
    players: List[Player]

    preloadPos: ChunkPos

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

    preloadProgress: int

    saveName: str

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

        self.preloadProgress = 0

        self.players = []
    
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
    
    def getLocalDimension(self) -> Dimension:
        player = self.getLocalPlayer()
        return self.getDimensionOf(player)
    
    def getDimensionOf(self, player: Player) -> Dimension:
        return self.getDimension(player.dimension)
    
    def getDimension(self, name: str) -> Dimension:
        if name == 'overworld':
            return self.dimensions[0]
        elif name == 'nether':
            return self.dimensions[1]
        else:
            # TODO:
            raise Exception()
    
    def save(self):
        self.saveWorld()
        self.saveEntities()
        self.savePlayers()
    
    def saveWorld(self):
        for dim in self.dimensions:
            dim.world.save()
    
    def saveEntities(self):
        for dim in self.dimensions:
            # FIXME:
            path = self.saveFolderPath() + '/entities.dat'

            nbtfile = nbt.NBTFile()
            nbtfile.name = 'Entities'
            # FIXME:
            nbtfile.tags.append(entity.toNbt(dim.entities))
            nbtfile.write_file(path)

    def savePlayers(self):
        os.makedirs(self.saveFolderPath() + '/playerdata', exist_ok=True)

        for player in self.players:
            # FIXME:
            path = self.saveFolderPath() + '/playerdata/player.dat'

            nbtfile = nbt.NBTFile()
            nbtfile.name = 'Player'
            nbtfile.tags = player.toNbt().tags
            nbtfile.write_file(path)
        
    def addPlayer(self, app):
        path = self.saveFolderPath() + '/playerdata/player.dat'

        try:
            nbtfile = nbt.NBTFile(filename=path)
            player = Player(app, tag=nbtfile)
        except FileNotFoundError:
            player = Player(app)
            player.pos[1] = 75.0
            player.entityId = self.getEntityId()

        # TODO: Send `Join Game` packet
        self.players.append(player)
        self.localPlayer = player.entityId
    
    def saveFolderPath(self):
        return f'saves/{self.saveName}'
    
    @classmethod
    def open(cls, worldName, seed, importPath, app):
        server = cls()

        server.saveName = worldName

        overworld = Dimension()
        overworld.world = World(server.saveFolderPath() + '/region', world.OverworldGen(), seed, importPath=importPath)

        try:
            path = server.saveFolderPath() + '/entities.dat'

            nbtfile = nbt.NBTFile(path)

            overworld.entities = [entity.Entity(app, server.getEntityId(), nbt=tag) for tag in nbtfile["Entities"][1:]]
        except FileNotFoundError:
            overworld.entities = [entity.Entity(app, server.getEntityId(), 'fox', 5.0, 75.0, 3.0)]
        
        nether = Dimension()
        nether.world = World(server.saveFolderPath() + '/DIM-1/region', world.NetherGen(), seed, importPath=importPath)

        try:
            path = server.saveFolderPath() + '/DIM-1/entities.dat'

            nbtfile = nbt.NBTFile(path)

            nether.entities = [entity.Entity(app, server.getEntityId(), nbt=tag) for tag in nbtfile["Entities"][1:]]
        except FileNotFoundError:
            nether.entities = []
        
        server.dimensions = [overworld, nether]

        server.addPlayer(app)
        preloadPos = server.players[0].pos

        cx = math.floor(preloadPos[0] / 16)
        cy = math.floor(preloadPos[1] / world.CHUNK_HEIGHT)
        cz = math.floor(preloadPos[2] / 16)

        server.preloadPos = ChunkPos(cx, cy, cz)

        server.getLocalDimension().world.loadChunk((app.textures, app.cube, app.textureIndices), server.preloadPos)

        return server