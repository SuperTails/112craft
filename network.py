from sys import float_repr_style
from twisted.internet import defer, reactor
from quarry.net.auth import Profile
from quarry.net.client import ClientFactory, ClientProtocol
from quarry.net.ticker import Ticker
from quarry.types.chunk import BlockArray
from queue import SimpleQueue
from dataclasses import dataclass
from typing import List, Any, Tuple, Optional
from enum import Enum
from util import BlockPos
from quarry.types.buffer import BufferUnderrun
from quarry.types.chat import Message
import math

c2sQueue = SimpleQueue()
s2cQueue = SimpleQueue()

class DiggingAction(Enum):
    START_DIGGING = 0,
    CANCEL_DIGGING = 1,
    FINISH_DIGGING = 2,
    DROP_ITEM_STACK = 3,
    DROP_ITEM = 4,
    UPDATE_ITEM = 5,

@dataclass
class PlayerDiggingC2S:
    action: DiggingAction
    location: BlockPos
    face: int

    def send(self, pro):
        if self.face == 2:
            face = 3
        elif self.face == 3:
            face = 2
        else:
            face = self.face
        
        pro.send_packet('player_digging',
            pro.buff_type.pack_varint(self.action.value[0]) +
            pro.buff_type.pack_position(self.location.x, self.location.y, -(self.location.z+1)) +
            pro.buff_type.pack('b', face))

@dataclass
class PlayerPlacementC2S:
    hand: int
    location: BlockPos
    face: int
    cx: float
    cy: float
    cz: float
    insideBlock: bool

    def send(self, pro):
        if self.face == 2:
            face = 3
        elif self.face == 3:
            face = 2
        else:
            face = self.face
        
        pro.send_packet('player_block_placement',
            pro.buff_type.pack_varint(self.hand) +
            pro.buff_type.pack_position(self.location.x, self.location.y, -(self.location.z+1)) +
            pro.buff_type.pack_varint(self.face) +
            pro.buff_type.pack('fff?', self.cx, self.cy, self.cz, self.insideBlock)
        )

@dataclass
class PlayerMovementC2S:
    onGround: bool

    def send(self, pro):
        pro.send_packet('player', pro.buff_type.pack('?', self.onGround))

@dataclass
class HeldItemChangeC2S:
    newSlot: int

    def send(self, pro):
        pro.send_packet('held_item_change', pro.buff_type.pack('h', self.newSlot))
    
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
            pro.buff_type.pack('dddff?', self.x+0.5, self.y+0.5, -(self.z+0.5), 180-math.degrees(self.yaw), math.degrees(-self.pitch), self.onGround))

@dataclass
class PlayerPositionC2S:
    x: float
    y: float
    z: float
    onGround: bool

    def send(self, pro):
        pro.send_packet('player_position',
            pro.buff_type.pack('ddd?', self.x+0.5, self.y+0.5, -(self.z+0.5), self.onGround))

@dataclass
class TeleportConfirmC2S:
    teleportId: int

    def send(self, pro):
        pro.send_packet('teleport_confirm', pro.buff_type.pack_varint(self.teleportId))
    
@dataclass
class ClickWindowC2S:
    windowId: int
    slotIdx: int
    button: int
    actionNum: int
    mode: int
    item: Any
    count: int

    def send(self, pro):
        pro.send_packet('click_window',
            pro.buff_type.pack('Bhbh', self.windowId, self.slotIdx, self.button, self.actionNum) +
            pro.buff_type.pack_varint(self.mode) +
            pro.buff_type.pack_slot(self.item, self.count))

@dataclass
class CloseWindowC2S:
    windowId: int

    def send(self, pro):
        pro.send_packet('close_window', pro.buff_type.pack_varint(self.windowId))

@dataclass
class UseItemC2S:
    hand: int

    def send(self, pro):
        pro.send_packet('use_item', pro.buff_type.pack_varint(self.hand))

@dataclass
class ChatMessageS2C:
    data: Any
    position: int
    sender: Message

    @classmethod
    def fromBuf(cls, buf):
        data = buf.unpack_chat()
        position = buf.unpack('b')
        sender = buf.unpack_uuid()

        return cls(data, position, sender)

@dataclass
class OpenWindowS2C:
    windowId: int
    kind: int
    title: Message

    @classmethod
    def fromBuf(cls, buf):
        windowId = buf.unpack_varint()
        kind = buf.unpack_varint()
        title = buf.unpack_chat()

        return cls(windowId, kind, title)

@dataclass
class AckPlayerDiggingS2C:
    location: BlockPos
    block: int
    status: DiggingAction
    successful: bool

    @classmethod
    def fromBuf(cls, buf):
        x, y, z = buf.unpack_position()
        location = BlockPos(x, y, -(z+1))

        block = buf.unpack_varint()

        status = DiggingAction((buf.unpack_varint(),))

        successful = buf.unpack('?')

        return cls(location, block, status, successful)

@dataclass
class TimeUpdateS2C:
    worldAge: int
    dayTime: int

    @classmethod
    def fromBuf(cls, buf):
        worldAge, dayTime = buf.unpack('ll')

        return cls(worldAge, dayTime)

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

        return cls(x-0.5, y-0.5, -(z+0.5), 180-yaw, -pitch, xRel, yRel, zRel, yawRel, pitchRel, teleportId)

@dataclass
class SpawnEntityS2C:
    entityId: int
    objectUUID: Any

    kind: int

    x: float
    y: float
    z: float

    pitch: float
    yaw: float

    data: int

    xVel: int
    yVel: int
    zVel: int

    @classmethod
    def fromBuf(cls, buf):
        entityId = buf.unpack_varint()
        objectUUID = buf.unpack_uuid()
        kind = buf.unpack_varint()
        x, y, z, pitch, yaw = buf.unpack('dddbb')
        data = buf.unpack('I')
        xVel, yVel, zVel = buf.unpack('hhh')

        return cls(entityId, objectUUID, kind, x-0.5, y-0.5, -(z+0.5), math.pi*2*-pitch/256, math.pi*2*-yaw/256, data, xVel, yVel, -zVel)

@dataclass
class SpawnMobS2C:
    entityId: int
    uuid: Any
    kind: int

    x: float
    y: float
    z: float

    yaw: float
    pitch: float
    headPitch: float

    xVel: int
    yVel: int
    zVel: int

    @classmethod
    def fromBuf(cls, buf):
        entityId = buf.unpack_varint()
        uuid = buf.unpack_uuid()
        kind = buf.unpack_varint()

        x, y, z, yaw, pitch, headPitch = buf.unpack('dddbbb')

        yaw = 2*math.pi*(0.5 + yaw/256)
        pitch = 2*math.pi*(0.5 + pitch/256)
        headPitch = 2*math.pi*(0.5 + headPitch/256)

        xVel, yVel, zVel = buf.unpack('hhh')

        return cls(entityId, uuid, kind, x-0.5, y-0.5, -(z+0.5), yaw, pitch, headPitch, xVel, yVel, -zVel)


@dataclass
class BlockChangeS2C:
    location: BlockPos
    blockId: int

    @classmethod
    def fromBuf(cls, buf):
        x, y, z = buf.unpack_position()
        location = BlockPos(x, y, -(z+1))
        blockId = buf.unpack_varint()
        
        return cls(location, blockId)


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

        return cls(x, -(z+1), full, bitmask, heightmap, biomes, sections, blockEntities)

@dataclass
class EntityMetadataS2C:
    entityId: int
    metadata: dict[Tuple[int, int], Any]

    @classmethod
    def fromBuf(cls, buf):
        entityId = buf.unpack_varint()
        metadata = buf.unpack_entity_metadata()

        return cls(entityId, metadata)

@dataclass
class EntityHeadLookS2C:
    entityId: int
    headYaw: float

    @classmethod
    def fromBuf(cls, buf):
        entityId = buf.unpack_varint()
        headYaw = buf.unpack('b')

        headYaw = 2*math.pi*(0.5 + headYaw/256)

        return cls(entityId, headYaw)

@dataclass
class EntityLookS2C:
    entityId: int
    bodyYaw: float
    headPitch: float

    @classmethod
    def fromBuf(cls, buf):
        entityId = buf.unpack_varint()
        bodyYaw, headPitch = buf.unpack('bb')

        bodyYaw = 2*math.pi*(0.5 + bodyYaw/256)
        headPitch = 2*math.pi*headPitch/256

        return cls(entityId, bodyYaw, headPitch)

@dataclass
class EntityVelocityS2C:
    entityId: int
    xVel: int
    yVel: int
    zVel: int

    @classmethod
    def fromBuf(cls, buf):
        entityId = buf.unpack_varint()
        xVel, yVel, zVel = buf.unpack('hhh')

        return cls(entityId, xVel, yVel, -zVel)

@dataclass
class EntityRelMoveS2C:
    entityId: int
    dx: int
    dy: int
    dz: int
    onGround: bool
    
    @classmethod
    def fromBuf(cls, buf):
        entityId = buf.unpack_varint()
        dx, dy, dz, onGround = buf.unpack('hhh?')

        return cls(entityId, dx, dy, -dz, onGround)

@dataclass
class EntityLookRelMoveS2C:
    entityId: int
    dx: int
    dy: int
    dz: int

    yaw: float
    pitch: float

    onGround: bool

    @classmethod
    def fromBuf(cls, buf):
        entityId = buf.unpack_varint()
        dx, dy, dz, yaw, pitch = buf.unpack('hhhbb')

        yaw = 2*math.pi*(0.5 + yaw/256)
        pitch = 2*math.pi*pitch/256

        onGround = buf.unpack('?')

        return cls(entityId, dx, dy, -dz, yaw, pitch, onGround)

@dataclass
class EntityTeleportS2C:
    entityId: int
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    onGround: bool

    @classmethod
    def fromBuf(cls, buf):
        entityId = buf.unpack_varint()
        x, y, z, yaw, pitch, onGround = buf.unpack('dddbb?')

        yaw = 2*math.pi*(0.5 + yaw/256)
        pitch = 2*math.pi*pitch/256

        return cls(entityId, x-0.5, y-0.5, -(z+0.5), yaw, pitch, onGround)
    
@dataclass
class DestroyEntitiesS2C:
    entityIds: List[int]

    @classmethod
    def fromBuf(cls, buf):
        count = buf.unpack_varint()
        entityIds = [buf.unpack_varint() for _ in range(count)]

        return cls(entityIds)

@dataclass
class WindowConfirmationS2C:
    windowId: int
    actionNum: int
    accepted: bool

    @classmethod
    def fromBuf(cls, buf):
        windowId, actionNum, accepted = buf.unpack('Bh?')

        return cls(windowId, actionNum, accepted)

@dataclass
class WindowItemsS2C:
    windowId: int
    stacks: List[Any]

    @classmethod
    def fromBuf(cls, buf):
        windowId = buf.unpack('B')

        stacks = [buf.unpack_slot() for _ in range(buf.unpack_varint())]

        return cls(windowId, stacks)

@dataclass
class SetSlotS2C:
    windowId: int
    slotIdx: int

    itemId: Optional[int]
    count: int

    @classmethod
    def fromBuf(cls, buf):
        windowId, slotIdx = buf.unpack('bh')
        stack = buf.unpack_slot()

        itemId = stack['item']
        if 'count' in stack:
            count = stack['count']
        else:
            count = 0

        return cls(windowId, slotIdx, itemId, count)

@dataclass
class SpawnPlayerS2C:
    entityId: int
    playerUUID: Any

    x: float
    y: float
    z: float

    yaw: float
    pitch: float

    @classmethod
    def fromBuf(cls, buf):
        entityId = buf.unpack_varint()
        playerUUID = buf.unpack_uuid()

        x, y, z, yaw, pitch = buf.unpack('dddbb')

        yaw = 2*math.pi*(0.5 + yaw/256)
        pitch = 2*math.pi*pitch/256

        return cls(entityId, playerUUID, x-0.5, y-0.5, -(z+0.5), yaw, pitch)

printI = 0

class MinecraftProtocol(ClientProtocol):
    stopped: bool = True

    def connection_lost(self, reason):
        print(f'Connection lost: {reason}')
        s2cQueue.put(None)
        
        if not self.stopped:
            reactor.stop() #type:ignore
            self.stopped = True

    def player_joined(self):
        self.stopped = False

        print("hewwo")
        self.ticker.interval = 0.1

        pro = self

        def doTick():
            while not c2sQueue.empty():
                packet = c2sQueue.get()
                global printI
                if printI < 20:
                    printI += 1
                    print(f'sending {packet}')
                if packet is None:
                    print('Stopping reactor!')
                    pro.ticker.stop()
                    reactor.stop() #type:ignore
                    self.stopped = True
                    break
                else:
                    packet.send(pro)
            
        self.mainLoop = self.ticker.add_loop(1, doTick)
        self.ticker.start()
    
    def packet_confirm_transaction(self, buf):
        s2cQueue.put(WindowConfirmationS2C.fromBuf(buf))
        buf.discard()
    
    def packet_spawn_object(self, buf):
        try:
            s2cQueue.put(SpawnEntityS2C.fromBuf(buf))
        except BufferUnderrun:
            # Sometimes the server's first packet is all zeros.
            # No idea why.
            pass
        buf.discard()
    
    def packet_spawn_player(self, buf):
        s2cQueue.put(SpawnPlayerS2C.fromBuf(buf))
        buf.discard()
    
    def packet_open_window(self, buf):
        s2cQueue.put(OpenWindowS2C.fromBuf(buf))
        buf.discard()
    
    def packet_window_items(self, buf):
        s2cQueue.put(WindowItemsS2C.fromBuf(buf))
        buf.discard()
    
    def packet_set_slot(self, buf):
        s2cQueue.put(SetSlotS2C.fromBuf(buf))
        buf.discard()
    
    def packet_acknowledge_player_digging(self, buf):
        s2cQueue.put(AckPlayerDiggingS2C.fromBuf(buf))
        buf.discard()
    
    def packet_time_update(self, buf):
        s2cQueue.put(TimeUpdateS2C.fromBuf(buf))
        buf.discard()
    
    def packet_destroy_entities(self, buf):
        s2cQueue.put(DestroyEntitiesS2C.fromBuf(buf))
        buf.discard()
    
    def packet_entity_look(self, buf):
        s2cQueue.put(EntityLookS2C.fromBuf(buf))
        buf.discard()
    
    def packet_entity_head_look(self, buf):
        s2cQueue.put(EntityHeadLookS2C.fromBuf(buf))
        buf.discard()

    def packet_entity_velocity(self, buf):
        s2cQueue.put(EntityVelocityS2C.fromBuf(buf))
        buf.discard()

    def packet_entity_look_and_relative_move(self, buf):
        s2cQueue.put(EntityLookRelMoveS2C.fromBuf(buf))
        buf.discard()

    def packet_entity_relative_move(self, buf):
        s2cQueue.put(EntityRelMoveS2C.fromBuf(buf))
        buf.discard()

    def packet_entity_teleport(self, buf):
        s2cQueue.put(EntityTeleportS2C.fromBuf(buf))
        buf.discard()
    
    def packet_entity_status(self, buf):
        buf.discard()

    def packet_entity_properties(self, buf):
        buf.discard()
    
    def packet_entity_metadata(self, buf):
        s2cQueue.put(EntityMetadataS2C.fromBuf(buf))
        buf.discard()
    
    def packet_block_change(self, buf):
        s2cQueue.put(BlockChangeS2C.fromBuf(buf))
        buf.discard()
    
    def packet_chat_message(self, buf):
        s2cQueue.put(ChatMessageS2C.fromBuf(buf))
        buf.discard()

    def packet_sound_effect(self, buf):
        buf.discard()

    def packet_update_light(self, buf):
        buf.discard()
    
    def packet_spawn_mob(self, buf):
        s2cQueue.put(SpawnMobS2C.fromBuf(buf))
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
def main(hostname, port):
    with open('creds.txt', 'r') as f:
        [username, password] = f.readlines()[:2]
    
    username = username.strip()
    password = password.strip()

    profile = yield Profile.from_credentials(username, password)

    factory = MinecraftFactory(profile)

    factory.connect(hostname, port)

def go():
    startPacket = c2sQueue.get(block=True)
    if startPacket is None:
        return
    (hostname, port) = startPacket

    main(hostname, port)
        
    reactor.run() #type:ignore