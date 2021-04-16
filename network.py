from twisted.internet import defer, reactor
from quarry.net.auth import Profile
from quarry.net.client import ClientFactory, ClientProtocol
from quarry.net.ticker import Ticker
from quarry.types.chunk import BlockArray
from queue import SimpleQueue
from dataclasses import dataclass
from typing import List, Any, Tuple, Optional
import math

c2sQueue = SimpleQueue()
s2cQueue = SimpleQueue()

@dataclass
class PlayerMovementC2S:
    onGround: bool

    def send(self, pro):
        pro.send_packet('player', pro.buff_type.pack('?', self.onGround))
    
@dataclass
class ClientStatusC2S:
    status: int

    def send(self, pro):
        pro.send_packet('client_status', pro.buff_type.pack_varint(self.status))

@dataclass
class ChatMessageC2S:
    message: str

    def send(self, pro):
        pro.send_packet('chat_message', pro.buff_type.pack_string(self.message))

@dataclass
class PlayerLookC2S:
    yaw: float
    pitch: float
    onGround: bool

    def send(self, pro):
        pro.send_packet('player_look',
            pro.buff_type.pack('ff?', 180-math.degrees(self.yaw), math.degrees(-self.pitch), self.onGround))

@dataclass
class PlayerPositionAndLookC2S:
    x: float
    y: float
    z: float

    yaw: float
    pitch: float

    onGround: bool

    def send(self, pro):
        pro.send_packet('player_position_and_look', 
            pro.buff_type.pack('dddff?', self.x, self.y+0.5, -self.z, 180-math.degrees(self.yaw), math.degrees(-self.pitch), self.onGround))

@dataclass
class PlayerPositionC2S:
    x: float
    y: float
    z: float
    onGround: bool

    def send(self, pro):
        pro.send_packet('player_position',
            pro.buff_type.pack('ddd?', self.x, self.y+0.5, -self.z, self.onGround))

@dataclass
class TeleportConfirmC2S:
    teleportId: int

    def send(self, pro):
        pro.send_packet('teleport_confirm', pro.buff_type.pack_varint(self.teleportId))

@dataclass
class PlayerPositionAndLookS2C:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float

    xRel: bool
    yRel: bool
    zRel: bool
    yawRel: bool
    pitchRel: bool

    teleportId: int

    @classmethod
    def fromBuf(cls, buf):
        x, y, z, yaw, pitch, flags = buf.unpack('dddffb')

        xRel = flags & 0x1
        yRel = flags & 0x2
        zRel = flags & 0x4
        pitchRel = flags & 0x8
        yawRel = flags & 0x10

        teleportId = buf.unpack_varint()

        return cls(x, y-0.5, -z, 180-yaw, -pitch, xRel, yRel, zRel, yawRel, pitchRel, teleportId)

@dataclass
class ChunkDataS2C:
    x: int
    z: int
    full: bool

    bitmask: int

    heightmap: Any

    biomes: Optional[List[int]]

    sections: List[Tuple[BlockArray, Any, Any]]

    blockEntities: List[Any]

    @classmethod
    def fromBuf(cls, buf):
        x, z, full = buf.unpack('ii?')
        bitmask = buf.unpack_varint()
        heightmap = buf.unpack_nbt()
        biomes = [buf.unpack_varint() for _ in range(buf.unpack_varint())]
        sections_length = buf.unpack_varint()
        sections = buf.unpack_chunk(bitmask)
        blockEntities = [buf.unpack_nbt() for _ in range(buf.unpack_varint())]

        return cls(x, -z-1, full, bitmask, heightmap, biomes, sections, blockEntities)


class MinecraftProtocol(ClientProtocol):
    def player_joined(self):
        print("hewwo")
        self.ticker.interval = 0.1

        pro = self

        def doTick():
            while not c2sQueue.empty():
                packet = c2sQueue.get()
                packet.send(pro)
        
        self.mainLoop = self.ticker.add_loop(1, doTick)
        self.ticker.start()
    
    def packet_block_change(self, buf):
        buf.discard()
    
    def packet_entity_status(self, buf):
        buf.discard()

    def packet_chat_message(self, buf):
        p_text = buf.unpack_chat().to_string()
        p_position = 0
        p_sender = None

        if self.protocol_version >= 47:
            p_position = buf.unpack('B')

        if self.protocol_version >= 736:
            p_sender = buf.unpack_uuid()
        
        if p_position in (0, 1) and p_text.strip():
            print(p_text)

    def packet_entity_velocity(self, buf):
        buf.discard()

    def packet_entity_look_and_relative_move(self, buf):
        buf.discard()

    def packet_entity_head_look(self, buf):
        buf.discard()

    def packet_sound_effect(self, buf):
        buf.discard()

    def packet_entity_relative_move(self, buf):
        buf.discard()

    def packet_entity_teleport(self, buf):
        buf.discard()
    
    def packet_update_light(self, buf):
        buf.discard()
    
    def packet_entity_properties(self, buf):
        buf.discard()
    
    def packet_entity_metadata(self, buf):
        buf.discard()
    
    def packet_spawn_mob(self, buf):
        buf.discard()
    
    def packet_player_position_and_look(self, buf):
        s2cQueue.put(PlayerPositionAndLookS2C.fromBuf(buf))

        buf.discard()
    
    def packet_chunk_data(self, buf):
        # https://quarry.readthedocs.io/en/latest/data_types/chunks.html
        try:
            s2cQueue.put(ChunkDataS2C.fromBuf(buf))
        except:
            pass

        buf.discard()
    
    def packet_unhandled(self, buf, name):
        print(f"Unhandled packet {name}")
        buf.discard()
    
    def packet_keep_alive(self, buf):
        self.send_packet('keep_alive', buf.read())

class MinecraftFactory(ClientFactory):
    protocol = MinecraftProtocol

@defer.inlineCallbacks
def main():
    with open('creds.txt', 'r') as f:
        [username, password] = f.readlines()
    
    username = username.strip()
    password = password.strip()

    profile = yield Profile.from_credentials(username, password)

    factory = MinecraftFactory(profile)

    factory.connect("localhost", 25565)

def go():
    main()
    reactor.run() #type:ignore