"""Rendering, representation, and behavior of entities is handled here.

Entity models are made up of bones. Each bone is individually posable. 
Each bone is made up of cubes, which stay in a fixed arrangement.

Every entity has a `kind`, which determines its attributes and behavior.

For behavior, an `Ai` is used. This simply stores a list of `Task`s.
Tasks run every tick until they are interrupted by a task with a higher
priority, or when they mark themselves as finished.
For example, most entities have a `WanderTask` with low priority, so when
they do not have any other goals, they will occasionally move around.
"""

from typing import List, Tuple, Optional, Any
import json
import numpy as np
import heapq
import math
import time
import random
import decimal
import copy
import molang
import model 
import functools
import config
from sound import Sound
from inventory import Stack
from util import BlockPos, roundHalfUp, ItemId
from dataclasses import dataclass
from OpenGL.GL import * #type:ignore
from nbt import nbt

CUBE_MESH_VERTICES = np.array([
    # Left face
    -0.5,  0.5,  0.5,  1.0, 1.0, # top-right
    -0.5,  0.5, -0.5,  0.0, 1.0, # top-left
    -0.5, -0.5, -0.5,  0.0, 0.0, # bottom-left
    -0.5, -0.5, -0.5,  0.0, 0.0, # bottom-left
    -0.5, -0.5,  0.5,  1.0, 0.0, # bottom-right
    -0.5,  0.5,  0.5,  1.0, 1.0, # top-right
    # Right face
    0.5,  0.5,  0.5,  0.0, 1.0, # top-left
    0.5, -0.5, -0.5,  1.0, 0.0, # bottom-right
    0.5,  0.5, -0.5,  1.0, 1.0, # top-right         
    0.5, -0.5, -0.5,  1.0, 0.0, # bottom-right
    0.5,  0.5,  0.5,  0.0, 1.0, # top-left
    0.5, -0.5,  0.5,  0.0, 0.0, # bottom-left     
    # Back face
    -0.5, -0.5, -0.5,  0.0, 0.0, # Bottom-left
    0.5,  0.5, -0.5,  1.0, 1.0, # top-right
    0.5, -0.5, -0.5,  1.0, 0.0, # bottom-right         
    0.5,  0.5, -0.5,  1.0, 1.0, # top-right
    -0.5, -0.5, -0.5,  0.0, 0.0, # bottom-left
    -0.5,  0.5, -0.5,  0.0, 1.0, # top-left
    # Front face
    -0.5, -0.5,  0.5,  0.0, 0.0, # bottom-left
    0.5, -0.5,  0.5,  1.0, 0.0, # bottom-right
    0.5,  0.5,  0.5,  1.0, 1.0, # top-right
    0.5,  0.5,  0.5,  1.0, 1.0, # top-right
    -0.5,  0.5,  0.5,  0.0, 1.0, # top-left
    -0.5, -0.5,  0.5,  0.0, 0.0, # bottom-left
    # Bottom face
    -0.5, -0.5, -0.5,  1.0, 1.0, # top-right
    0.5, -0.5, -0.5,  0.0, 1.0, # top-left
    0.5, -0.5,  0.5,  0.0, 0.0, # bottom-left
    0.5, -0.5,  0.5,  0.0, 0.0, # bottom-left
    -0.5, -0.5,  0.5,  1.0, 0.0, # bottom-right
    -0.5, -0.5, -0.5,  1.0, 1.0, # top-right
    # Top face
    -0.5,  0.5, -0.5,  0.0, 1.0, # top-left
    0.5,  0.5,  0.5,  1.0, 0.0, # bottom-right
    0.5,  0.5, -0.5,  1.0, 1.0, # top-right     
    0.5,  0.5,  0.5,  1.0, 0.0, # bottom-right
    -0.5,  0.5, -0.5,  0.0, 1.0, # top-left
    -0.5,  0.5,  0.5,  0.0, 0.0, # bottom-left
    ], dtype='float32')

@dataclass
class Cube:
    origin: Tuple[float, float, float]
    size: Tuple[int, int, int]
    uv: Tuple[int, int]

    def toModelTk(self, app) -> model.Model:
        factor = np.array([[self.size[0]], [self.size[1]], [self.size[2]]])
        offset = np.array([[self.origin[0]], [self.origin[1]], [self.origin[2]]])

        def trans(v):
            return (((v + 0.5) * factor) + offset) / 16

        return app.cube.transformed(trans)

    def toVerticesGl(self) -> np.ndarray:
        vertices = CUBE_MESH_VERTICES.reshape((-1, 5))

        factor = np.array([self.size[0], self.size[1], self.size[2], 1.0, 1.0])

        offset = np.array([self.origin[0], self.origin[1], self.origin[2], 0.0, 0.0])

        # https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array/8505658

        result = (vertices + np.array([0.5, 0.5, 0.5, 0.0, 0.0])) * factor + offset

        for row in range(vertices.shape[0]):
            face = ['left', 'right', 'front', 'back', 'bottom', 'top'][row // 6]

            if face == 'left':
                uOffset = 0
                uSize = self.size[2]

                vOffset = 0
                vSize = self.size[1]
            elif face == 'front':
                uOffset = self.size[2]
                uSize = self.size[0]

                vOffset = 0
                vSize = self.size[1]
            elif face == 'top':
                uOffset = self.size[2]
                uSize = self.size[0]

                vOffset = self.size[1]
                vSize = self.size[2]
            elif face == 'right':
                uOffset = self.size[2] + self.size[0]
                uSize = self.size[2]

                vOffset = 0
                vSize = self.size[1]
            elif face == 'back':
                uOffset = 2*self.size[2] + self.size[0]
                uSize = self.size[0]

                vOffset = 0
                vSize = self.size[1]
            else: # face == 'bottom'
                uOffset = self.size[2] + self.size[0]
                uSize = self.size[0]

                vOffset = self.size[1]
                vSize = self.size[2]

            uFrac = result[row, 3]
            vFrac = result[row, 4]

            result[row, 3] = self.uv[0] + uOffset + (1.0 - uFrac) * uSize
            result[row, 4] = (32.0 - (self.uv[1] + self.size[1] + self.size[2])) + vOffset + vFrac * vSize
        
        return result

@dataclass
class Bone:
    name: str
    pivot: Tuple[float, float, float]
    cubes: List[Cube]
    bind_pose_rotation: Tuple[float, float, float]
    neverRender: bool

    def toModelTk(self, app) -> model.Model:
        models = [c.toModelTk(app) for c in self.cubes]

        return functools.reduce(model.Model.fuse, models, model.Model([], []))

    def toVerticesGl(self) -> np.ndarray:
        self.innerVertices = []
        for cube in self.cubes:
            vertices = cube.toVerticesGl()

            pv = (self.pivot[0], self.pivot[1], self.pivot[2])

            withPivot = np.hstack((vertices, np.full(shape=(vertices.shape[0], 3), fill_value=pv)))

            self.innerVertices.append(withPivot)
        return np.vstack(self.innerVertices)

    def toVao(self) -> Tuple[int, int]:
        #if self.meshVaos[meshIdx] != 0:
        #    glDeleteVertexArrays(1, np.array([self.meshVaos[meshIdx]])) #type:ignore
        #    glDeleteBuffers(1, np.array([self.meshVbos[meshIdx]])) #type:ignore
        #
        #    self.meshVaos[meshIdx] = 0

        vertices = self.toVerticesGl().flatten().astype('float32')

        vao: int = glGenVertexArrays(1) #type:ignore
        vbo: int = glGenBuffers(1) #type:ignore

        glBindVertexArray(vao)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(5 * 4))
        glEnableVertexAttribArray(2)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(0)

        return (vao, len(vertices) // 5)

@dataclass
class BoneAnimation:
    rotation: List[Any]

@dataclass
class Animation:
    loop: bool
    bones: dict[str, BoneAnimation]

def parseBoneAnim(j) -> BoneAnimation:
    if 'rotation' in j:
        rotation = j['rotation']
    else:
        rotation = [0.0, 0.0, 0.0]

    return BoneAnimation(rotation)

def parseAnimation(j) -> Animation:
    loop = j['loop']
    bones = dict()
    for (name, bone) in j['bones'].items():
        bones[name] = parseBoneAnim(bone)
    return Animation(loop, bones)

def openAnimations(path) -> dict[str, Animation]:
    with open(path) as f:
        j = json.load(f)
    
    result = {}
    
    for (name, anim) in j['animations'].items():
        if name.startswith('animation.'):
            result[name] = parseAnimation(anim)
    
    return result


@dataclass
class EntityModel:
    bones: List[Bone]

    vaos: List[Tuple[int, int]]

    models: List[model.Model]

    def __init__(self, bones, app):
        self.bones = bones
        self.vaos = []
        self.models = []
        for bone in self.bones:
            if bone.cubes == [] or bone.neverRender or 'sleeping' in bone.name:
                if config.USE_OPENGL_BACKEND:
                    self.vaos.append((0, 0))
                else:
                    self.models.append(model.Model([], []))
            else:
                if config.USE_OPENGL_BACKEND:
                    self.vaos.append(bone.toVao())
                else:
                    self.models.append(bone.toModelTk(app))

def parseCube(j) -> Cube:
    origin = tuple(j['origin'])
    size = tuple(j['size'])
    if 'uv' in j:
        uv = tuple(j['uv'])
    else:
        # I do not understand why some of them just don't include any
        uv = (0, 0)
    return Cube(origin, size, uv) #type:ignore
    
def parseBone(j) -> Bone:
    name = j['name'].lower()
    if 'pivot' in j:
        pivot = tuple(j['pivot'])
    else:
        #FIXME: ???
        pivot = (0.0, 0.0, 0.0)
    if 'cubes' in j:
        cubes = list(map(parseCube, j['cubes']))
    else:
        cubes = []
    if 'bind_pose_rotation' in j:
        bind_pose_rotation = tuple(j['bind_pose_rotation'])
    else:
        bind_pose_rotation = (0.0, 0.0, 0.0)
    if 'neverRender' in j:
        neverRender = j['neverRender']
    else:
        neverRender = False
    return Bone(name, pivot, cubes, bind_pose_rotation, neverRender) #type:ignore

def parseModel(j, app) -> EntityModel:
    if 'bones' in j:
        bones = list(map(parseBone, j['bones']))
    else:
        bones = []
    return EntityModel(bones, app)

def openModels(path, app) -> dict[str, EntityModel]:
    with open(path) as f:
        j = json.load(f)
    
    result = {}
    
    for (name, model) in j.items():
        if name.startswith('geometry.'):
            result[name] = parseModel(model, app)
    
    return result

#print(models['fox'].bones[3].cubes[0].toVertices())

def getHurtSound(app, entity) -> Sound:
    if entity.kind.name in app.hurtSounds:
        name = entity.kind.name
    else:
        print(f"Using fallback hurt sound for entity {entity.kind.name}")
        name = 'player'
    
    return random.choice(app.hurtSounds[name])

class ItemData:
    stack: Stack
    age: int
    pickupDelay: int

    def __init__(self):
        self.stack = Stack('stone', 1)
        self.age = 6000
        # TODO: should increase if dropped by player or fox
        self.pickupDelay = 10
    
    def tick(self, entity: 'Entity'):
        self.age -= 1
        if self.age == 0:
            entity.health = 0

        if self.pickupDelay > 0: 
            self.pickupDelay -= 1


    def toNbt(self) -> nbt.TAG_Compound:
        tag = nbt.TAG_Compound()
        tag.tags.append(nbt.TAG_Short(self.age, 'Age'))
        tag.tags.append(nbt.TAG_Short(self.pickupDelay, 'PickupDelay'))

        item = self.stack.toNbt()
        assert(item is not None)
        item.name = 'Item'
        tag.tags.append(item)

        return tag

    def fromNbt(self, tag: nbt.TAG_Compound):
        self.age = tag['Age'].value
        self.pickupDelay = tag['PickupDelay'].value
        self.stack = Stack.fromNbt(tag['Item'])

@dataclass
class EntityKind:
    name: str
    model: str
    maxHealth: float
    walkSpeed: float
    radius: float
    height: float
    ai: 'Ai'
    extraData: Optional[Any]

    def __init__(self, name, model, maxHealth, walkSpeed, radius, height, ai, extraData=None):
        self.name = name
        self.model = model
        self.maxHealth = maxHealth
        self.walkSpeed = walkSpeed
        self.radius = radius
        self.height = height
        self.ai = ai
        self.extraData = extraData

def registerEntityKinds(app):
    app.entityKinds = {
        'creeper': EntityKind(
            name='creeper',
            model='geometry.creeper',
            maxHealth=20.0,
            walkSpeed=0.05,
            radius=0.3,
            height=1.7,
            ai=Ai([WanderTask()])
        ),
        'zombie': EntityKind(
            name='zombie',
            model='geometry.humanoid',
            maxHealth=20.0,
            walkSpeed=0.05,
            radius=0.3,
            height=1.9,
            ai=Ai([AttackTask(), FollowTask(), WanderTask()])
        ),
        'skeleton': EntityKind(
            name='skeleton',
            model='geometry.skeleton',
            maxHealth=20.0,
            walkSpeed=0.05,
            radius=0.3,
            height=1.9,
            ai=Ai([AttackTask(), FollowTask(), WanderTask()])
        ),
        'fox': EntityKind(
            name='fox',
            model='geometry.fox',
            maxHealth=20.0,
            walkSpeed=0.1,
            radius=0.35,
            height=0.6,
            ai=Ai([FollowTask(), WanderTask()])
        ),
        'player': EntityKind(
            name='player',
            model='geometry.humanoid',
            maxHealth=20.0,
            walkSpeed=0.2,
            radius=0.3,
            height=1.5,
            ai=Ai([NullTask()])
        ),
        'item': EntityKind(
            name='item',
            model='geometry.item',
            maxHealth=5.0,
            walkSpeed=0.0,
            radius=0.1,
            height=0.2,
            ai=Ai([NullTask()]),
            extraData=ItemData(),
        ),
    }

class Entity:
    pos: List[float]
    velocity: List[float]
    kind: EntityKind

    health: float

    radius: float
    height: float

    onGround: bool
    walkSpeed: float

    bodyAngle: float
    headYaw: float
    headPitch: float

    lifeTime: int
    distanceMoved: float

    path: List[BlockPos]

    variables: dict[str, float]

    ai: 'Ai'

    extra: Optional[Any]

    entityId: int

    def __init__(self, app, entityId: int, kind: str = '', x: float = 0.0, y: float = 0.0, z: float = 0.0, nbt: Optional[nbt.TAG_Compound] = None):
        if nbt is None:
            self.pos = [x, y, z]
            self.velocity = [0.0, 0.0, 0.0]
            self.onGround = False

            self.entityId = entityId

            self.bodyAngle = 0.0
            self.headPitch = 0.0
            self.headYaw = 0.0

            self.immunity = 0

            self.path = []

            self.kind = app.entityKinds[kind]

            self.variables = {}

            if self.kind.name == 'zombie':
                self.variables['gliding_speed_value'] = 1.0
            elif self.kind.name == 'creeper':
                self.variables['leg_rot'] = 0.0
            
            self.lifeTime = 0
            self.distanceMoved = 0.0

            self.radius = self.kind.radius
            self.height = self.kind.height
            self.walkSpeed = self.kind.walkSpeed
            self.health = self.kind.maxHealth
            self.ai = copy.deepcopy(self.kind.ai)
            self.extra = copy.deepcopy(self.kind.extraData)
        else:
            self.fromNbt(app, entityId, nbt)
    
    def fromNbt(self, entityId: int, app, data: nbt.TAG_Compound):
        kind = data["id"].value.removeprefix("minecraft:")

        Entity.__init__(self, entityId, app, kind=kind)
        
        self.pos = [tag.value for tag in data["Pos"].tags]
        self.velocity = [tag.value for tag in data["Motion"].tags]

        self.health = data["Health"].value
        self.onGround = data["OnGround"].value

        if self.extra is not None:
            self.extra.fromNbt(data)

    def toNbt(self) -> nbt.TAG_Compound:
        data = nbt.TAG_Compound()

        motion = nbt.TAG_List(name="Motion", type=nbt.TAG_Double)
        motion.append(nbt.TAG_Double(self.velocity[0]))
        motion.append(nbt.TAG_Double(self.velocity[1]))
        motion.append(nbt.TAG_Double(self.velocity[2]))
        data.tags.append(motion)

        pos = nbt.TAG_List(name="Pos", type=nbt.TAG_Double)
        pos.append(nbt.TAG_Double(self.pos[0]))
        pos.append(nbt.TAG_Double(self.pos[1]))
        pos.append(nbt.TAG_Double(self.pos[2]))
        data.tags.append(pos)
        
        data.tags.append(nbt.TAG_String(name="id", value=f"minecraft:{self.kind.name}"))

        data.tags.append(nbt.TAG_Float(name="Health", value=self.health))

        data.tags.append(nbt.TAG_Byte(name="OnGround", value=self.onGround))

        if self.extra is not None:
            extra = self.extra.toNbt()
            data.tags += extra.tags

        return data
    
    def hit(self, app, damage: float, knockback: Tuple[float, float]):
        if self.immunity == 0:
            getHurtSound(app, self).play()

            self.health -= damage

            self.velocity[0] += knockback[0] * 0.25
            self.velocity[1] += 0.2
            self.velocity[2] += knockback[1] * 0.25

            self.immunity = 10

    def getAABB(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        loX = self.pos[0] - self.radius
        loY = self.pos[1]
        loZ = self.pos[2] - self.radius

        hiX = self.pos[0] + self.radius
        hiY = self.pos[1] + self.height
        hiZ = self.pos[2] + self.radius

        return ((loX, loY, loZ), (hiX, hiY, hiZ))
        
    def getBlockPos(self) -> BlockPos:
        bx = roundHalfUp(self.pos[0])
        by = roundHalfUp(self.pos[1])
        bz = roundHalfUp(self.pos[2])
        return BlockPos(bx, by, bz)
    
    def getRotation(self, entityModels, entityAnimations, i):
        bone = entityModels[self.kind.model].bones[i]
        boneName = bone.name
        boneRot = bone.bind_pose_rotation

        if self.kind.name == 'creeper':
            anim = entityAnimations['animation.creeper.legs']
        elif self.kind.name == 'fox':
            anim = entityAnimations['animation.quadruped.walk']
        elif self.kind.name == 'zombie':
            anim = entityAnimations['animation.humanoid.move']
        elif self.kind.name == 'player':
            anim = entityAnimations['animation.humanoid.move']
        elif self.kind.name == 'skeleton':
            anim = entityAnimations['animation.humanoid.bow_and_arrow']
        elif self.kind.name == 'item':
            anim = None
        else:
            raise Exception(self.kind)
        
        '''
        if self.kind.name == 'zombie':
            self.variables['tcos0'] = molang.evalString("(Math.cos(query.modified_distance_moved * 38.17) * query.modified_move_speed / variable.gliding_speed_value) * 57.3", self)
        elif self.kind.name == 'creeper':
            self.variables['leg_rot'] = molang.evalString("Math.cos(query.modified_distance_moved * 38.17326) * 80.22 * query.modified_move_speed", self)
        '''
        
        #if anim is not None:
        #    print(boneName)

        if self.kind.name == 'item':
            rot = [0.0, self.lifeTime * 3.0, 0.0]
        elif boneName == 'head':
            rot = [math.degrees(self.headPitch), math.degrees(self.headYaw - self.bodyAngle), 0.0]
        elif anim is not None and boneName in anim.bones:
            (x, y, z) = anim.bones[boneName].rotation
            rot = [self.calc(x), self.calc(y), self.calc(z)]
        else:
            rot = [0.0, 0.0, 0.0]

        rot[0] += boneRot[0]
        rot[1] += boneRot[1]
        rot[2] += boneRot[2]

        rot[0] = math.radians(rot[0])
        rot[1] = math.radians(rot[1])
        rot[2] = math.radians(rot[2])

        return rot
    
    def getQuery(self, name):
        if name == 'target_x_rotation':
            # TODO:
            return 0.0
        elif name == 'target_y_rotation':
            # TODO:
            return 0.0
        elif name == 'modified_move_speed':
            return math.sqrt(self.velocity[0]**2 + self.velocity[2]**2)
        elif name == 'modified_distance_moved':
            return self.distanceMoved
        elif name == 'life_time':
            return self.lifeTime
        elif name == 'anim_time':
            # TODO:
            return self.lifeTime
        else:
            raise Exception(name)
    
    def calc(self, ex):
        mag = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if isinstance(ex, float):
            result = ex
        else:
            result = molang.evalString(ex, self)
        #elif '-variable.leg_rot' in ex:
        #    result = -30.0 * math.sin(time.time() * 3.0) * mag * 15.0
        #elif 'variable.leg_rot' in ex:
        #    result = 30.0 * math.sin(time.time() * 3.0) * mag * 15.0
        
        return result

    def tick(self, app, world, entities: List['Entity'], playerX, playerZ):
        #self.headYaw = math.atan2(playerX - self.pos[0], playerZ - self.pos[2])

        if math.sqrt(self.velocity[0]**2 + self.velocity[2]**2) > 0.01:
            goalAngle = math.atan2(self.velocity[0], self.velocity[2])

            # https://stackoverflow.com/questions/2708476/rotation-interpolation

            diff = ((goalAngle - self.bodyAngle + math.pi) % (2*math.pi)) - math.pi

            if diff > math.radians(10.0):
                change = math.radians(10.0)
            elif diff < math.radians(-10.0):
                change = math.radians(-10.0)
            else:
                change = diff

            self.bodyAngle += change
        
        self.distanceMoved += math.sqrt(self.velocity[0]**2 + self.velocity[2]**2)
        
        if self.immunity > 0:
            self.immunity -= 1
        
        self.lifeTime += 1
        
        self.ai.tick(app, self, world, entities)

        if len(self.path) > 0:
            x = self.path[0][0] - self.pos[0]
            z = self.path[0][2] - self.pos[2]

            mag = math.sqrt(x**2 + z**2)

            if mag < 0.5:
                self.path.pop(0)
                if self.path == []:
                    self.velocity[0] = 0.0
                    self.velocity[2] = 0.0
            else:
                x /= mag
                z /= mag

                x *= self.walkSpeed
                z *= self.walkSpeed

                self.velocity[0] = x
                self.velocity[2] = z
        elif self.onGround:
            self.velocity[0] = 0.0
            self.velocity[2] = 0.0
        
        if self.extra is not None:
            self.extra.tick(self)
    
    def updatePath(self, world, target: BlockPos):
        start = self.getBlockPos()

        if self.path == []:
            print(f"Finding path from {start} to {target}")

            path = findPath(start, target, world)
            if path is not None:
                self.path = path
                print(self.path)
        
#def fromNbt(app, data: nbt.TAG_List) -> List[Entity]:
#    return [Entity(app, nbt=tag) for tag in data.tags]

def toNbt(entities: List[Entity]) -> nbt.TAG_List:
    result = nbt.TAG_List(type=nbt.TAG_Compound, name="Entities")

    for entity in entities:
        result.append(entity.toNbt())
    
    return result

class Ai:
    taskIdx: int
    tasks: List[Any]

    def __init__(self, tasks):
        self.tasks = tasks
        self.taskIdx = 0
    
    def tick(self, app, entity: Entity, world, entities: List[Entity]):
        for highTask in range(self.taskIdx):
            if self.tasks[highTask].shouldStart(entity, world, entities):
                print(f"Switching to task {highTask} ({self.tasks[highTask]})")
                self.taskIdx = highTask
                break

        isDone = self.tasks[self.taskIdx].tick(app, entity, world, entities)

        if isDone:
            print(f"Stopping task {self.taskIdx} ({self.tasks[self.taskIdx]})")
            for lowTask in range(self.taskIdx+1, len(self.tasks)):
                if self.tasks[lowTask].shouldStart(entity, world, entities):
                    self.taskIdx = lowTask
                    break

class AttackTask:
    def shouldStart(self, entity: Entity, world, entities: List[Entity]):
        for other in entities:
            if other.kind.name == 'player':
                dx = entity.pos[0] - other.pos[0]
                dy = entity.pos[1] - other.pos[1]
                dz = entity.pos[2] - other.pos[2]
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                if dist < 1.0:
                    return True
    
    def tick(self, app, entity: Entity, world, entities: List[Entity]):
        for other in entities:
            if other.kind.name == 'player':
                dx = entity.pos[0] - other.pos[0]
                dy = entity.pos[1] - other.pos[1]
                dz = entity.pos[2] - other.pos[2]
                dist = math.sqrt(dx**2 + dy**2 + dz**2)

                if dist < 1.0:
                    knockX = other.pos[0] - entity.pos[0]
                    knockZ = other.pos[2] - entity.pos[2]
                    mag = math.sqrt(knockX**2 + knockZ**2)
                    knockX /= mag
                    knockZ /= mag

                    other.hit(app, 3.0, (knockX, knockZ))

                    return False
                else:
                    return True

class FollowTask:
    def shouldStart(self, entity: Entity, world, entities: List[Entity]):
        for other in entities:
            if other.kind.name == 'player' and not other.creative: #type:ignore
                dx = entity.pos[0] - other.pos[0]
                dy = entity.pos[1] - other.pos[1]
                dz = entity.pos[2] - other.pos[2]
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                if dist < 16.0:
                    # FIXME: CHECK FOR VALID PATH
                    return True
        return False

    def tick(self, app, entity: Entity, world, entities: List[Entity]):
        for other in entities:
            if other.kind.name == 'player' and not other.creative: #type:ignore
                dx = entity.pos[0] - other.pos[0]
                dy = entity.pos[1] - other.pos[1]
                dz = entity.pos[2] - other.pos[2]
                dist = math.sqrt(dx**2 + dy**2 + dz**2)

                if dist > 16.0:
                    return True
                
                if entity.path == [] and dist > 1.0:
                    startPos = entity.getBlockPos()
                    endPos = other.getBlockPos()
                    path = findPath(startPos, endPos, world)
                    if path is None:
                        return True
                    else:
                        entity.path = path
                        return False

class NullTask:
    def shouldStart(self, entity: Entity, world, entities):
        return True

    def tick(self, app, entity: Entity, world, entities):
        return False

class WanderTask:
    def shouldStart(self, entity, world, entities):
        return True

    def tick(self, app, entity: Entity, world, entities):
        wanderFreq = 0.01
        wanderDist = 3
        pos = entity.getBlockPos()
        if entity.path == [] and random.random() < wanderFreq:
            x = pos.x + random.randint(-wanderDist, wanderDist)
            y = pos.y + random.randint(-2, 2)
            z = pos.z + random.randint(-wanderDist, wanderDist)

            if (not world.coordsOccupied(BlockPos(x, y, z))
                and world.coordsOccupied(BlockPos(x, y - 1, z))):

                path = findPath(pos, BlockPos(x, y, z), world)
                if path is not None:
                    entity.path = path
        
        return False


def makePathFromChain(prevDirs, end: BlockPos) -> List[BlockPos]:
    result = []

    cur = end
    while cur in prevDirs:
        result.append(cur)
        cur = prevDirs[cur]

    result.append(cur)
    
    result.reverse()

    return result

def approxDistance(start: BlockPos, end: BlockPos):
    # Chebyshev distance

    xDist = abs(start.x - end.x)
    yDist = abs(start.y - end.y)
    zDist = abs(start.z - end.z)

    return (max(xDist, zDist) + yDist)


def findPath(start: BlockPos, end: BlockPos, world, maxDist=1) -> Optional[List[BlockPos]]:
    print(f"Start pos: {start} End pos: {end}")

    # https://en.wikipedia.org/wiki/A*_search_algorithm

    heuristic = approxDistance

    prevDirs = dict()

    realCosts = { start: 0 }

    costs = { start: heuristic(start, end) }

    openSet = { start }

    MAX_RANGE = 16
    
    while len(openSet) != 0:
        minCost = None
        current = None
        for pos in openSet:
            if minCost is None or costs[pos] < minCost:
                current = pos
                minCost = costs[pos]
        assert(current is not None)

        openSet.remove(current)
        
        if heuristic(current, end) <= maxDist:
            return makePathFromChain(prevDirs, end)

        for nextPos in destinations(current, world):
            if heuristic(nextPos, end) > MAX_RANGE:
                continue

            newRealCost = realCosts[current] + 1
            if nextPos not in realCosts or realCosts[nextPos] > newRealCost:
                prevDirs[nextPos] = current
                realCosts[nextPos] = newRealCost
                costs[nextPos] = realCosts[current] + heuristic(nextPos, end)
                if nextPos not in openSet:
                    openSet.add(nextPos)
    
    return None

def canStandAt(pos: BlockPos, world, needsFloor=True):
    # FIXME: Height

    if world.coordsOccupied(pos):
        return False
    elif not world.coordsOccupied(BlockPos(pos.x, pos.y - 1, pos.z)):
        if needsFloor:
            return False
        else:
            return True
    else:
        return True

def isValidDir(start: BlockPos, dx: int, dz: int, world) -> Optional[int]:
    # These are in order of most to least common, for efficiency
    for dy in (0, 1, -1, -2, -3):
        dest = BlockPos(start.x + dx, start.y + dy, start.z + dz)

        if not canStandAt(dest, world):
            continue

        if dy == 0:
            if dx != 0 and dz != 0:
                # This is a simple, flat, diagonal path

                corner1 = BlockPos(start.x + dx, start.y, start.z)
                corner2 = BlockPos(start.x, start.y, start.z + dz)

                if (not canStandAt(corner1, world, needsFloor=False)
                    or not canStandAt(corner2, world, needsFloor=False)):

                    continue
            else:
                # This is a straight path
                return dy
        elif dx != 0 and dz != 0:
            # TODO:
            continue
        elif dy < 0:
            corner = BlockPos(start.x + dx, start.y, start.z + dz)

            if not canStandAt(corner, world, needsFloor=False):
                continue
        
            return dy
        elif dy > 0:
            corner = BlockPos(start.x, start.y + 1, start.z)

            if not canStandAt(corner, world, needsFloor=False):
                continue
        
            return dy
    
    return None

def destinations(start: BlockPos, world):
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx == 0 and dz == 0: continue

            dy = isValidDir(start, dx, dz, world)

            if dy is not None:
                pos = BlockPos(start.x + dx, start.y + dy, start.z + dz)
                yield (pos)