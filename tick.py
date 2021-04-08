from entity import Entity
from util import BlockPos, roundHalfUp, rayAABBIntersect
import world
import time
import math
from math import cos, sin
import copy
import random
from typing import Optional

def lookedAtEntity(app) -> Optional[int]:
    lookX = cos(app.cameraPitch)*sin(-app.cameraYaw)
    lookY = sin(app.cameraPitch)
    lookZ = cos(app.cameraPitch)*cos(-app.cameraYaw)

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

    rayOrigin = app.cameraPos
    rayDir = (lookX, lookY, lookZ)

    inters = []

    for idx, entity in enumerate(app.entities):
        if abs(entity.pos[0] - app.cameraPos[0]) + abs(entity.pos[2] - app.cameraPos[2]) > 2 * app.mode.player.reach:
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
        if dist(inter) > app.mode.player.reach:
            return None
        else:
            return inter[0]

def tick(app):
    startTime = time.time()

    app.time += 1

    world.loadUnloadChunks(app, app.cameraPos)

    world.tickChunks(app)

    doMobSpawning(app)
    doMobDespawning(app)

    # Ticking is done in stages so that collision detection works as expected:
    # First we update the player's Y position and resolve Y collisions,
    # then we update the player's X position and resolve X collisions,
    # and finally update the player's Z position and resolve Z collisions.

    # W makes the player go forward, S makes them go backwards,
    # and pressing both makes them stop!
    z = float(app.w) - float(app.s)
    # Likewise for side to side movement
    x = float(app.d) - float(app.a)

    player = app.mode.player

    if x != 0.0 or z != 0.0:
        mag = math.sqrt(x*x + z*z)
        x /= mag
        z /= mag

        newX = math.cos(app.cameraYaw) * x - math.sin(app.cameraYaw) * z
        newZ = math.sin(app.cameraYaw) * x + math.cos(app.cameraYaw) * z

        x, z = newX, newZ

        x *= player.walkSpeed 
        z *= player.walkSpeed
    
    player.velocity[0] = x
    player.velocity[2] = z

    player.pos = copy.copy(app.cameraPos)
    player.pos[1] -= player.height
    collide(app, player)
    app.cameraPos = copy.copy(player.pos)
    app.cameraPos[1] += player.height

    entities = app.entities + [player]

    for entity in app.entities:
        if collide(app, entity) and entity.onGround:
            entity.velocity[1] = 0.40
        
        entity.tick(app.world, entities, player.pos[0], player.pos[2])

    endTime = time.time()
    app.tickTimes[app.tickTimeIdx] = (endTime - startTime)
    app.tickTimeIdx += 1
    app.tickTimeIdx %= len(app.tickTimes)

def doMobDespawning(app):
    player = app.mode.player

    idx = 0
    while idx < len(app.entities):
        [x, y, z] = app.entities[idx].pos
        dist = math.sqrt((x-player.pos[0])**2 + (y-player.pos[1])**2 + (z-player.pos[2])**2)

        maxDist = 128.0

        if dist > maxDist:
            app.entities.pop(idx)
        else:
            idx += 1

def doMobSpawning(app):
    mobCap = len(app.world.chunks) / 4

    random.seed(time.time())

    player = app.mode.player

    for (chunkPos, chunk) in app.world.chunks.items():
        chunk: world.Chunk
        if chunk.isTicking:
            if len(app.entities) > mobCap:
                return

            # FIXME: Random tick speed?
            x = random.randrange(0, 16) + chunkPos.x * 16
            y = random.randrange(0, world.CHUNK_HEIGHT) + chunkPos.y * world.CHUNK_HEIGHT
            z = random.randrange(0, 16) + chunkPos.z * 16

            dist = math.sqrt((x-player.pos[0])**2 + (y-player.pos[1])**2 + (z-player.pos[2])**2)

            minSpawnDist = 24.0
            
            if dist < minSpawnDist:
                continue

            if not isValidSpawnLocation(app, BlockPos(x, y, z)): continue

            packSize = 4
            for _ in range(packSize):
                x += random.randint(-2, 2)
                z += random.randint(-2, 2)
                if isValidSpawnLocation(app, BlockPos(x, y, z)):
                    app.entities.append(Entity(app, 'creeper', x, y, z))

def isValidSpawnLocation(app, pos: BlockPos):
    floor = BlockPos(pos.x, pos.y - 1, pos.z)
    feet = pos
    head = BlockPos(pos.x, pos.y + 1, pos.z)
    
    isOk = (app.world.coordsOccupied(floor)
        and not app.world.coordsOccupied(feet)
        and not app.world.coordsOccupied(head))
    
    return isOk

def collide(app, entity: Entity):
    entity.pos[1] += entity.velocity[1]

    if entity.onGround:
        if not world.hasBlockBeneath(app, entity):
            entity.onGround = False
    else:
        entity.velocity[1] -= app.gravity
        [_, yPos, _] = entity.pos
        #yPos -= entity.height
        yPos -= 0.1
        feetPos = round(yPos)
        if world.hasBlockBeneath(app, entity):
            entity.onGround = True
            entity.velocity[1] = 0.0
            #app.cameraPos[1] = (feetPos + 0.5) + entity.height
            entity.pos[1] = feetPos + 0.5

    for x in [entity.pos[0] - entity.radius * 0.99, entity.pos[0] + entity.radius * 0.99]:
        for z in [entity.pos[2] - entity.radius * 0.99, entity.pos[2] + entity.radius * 0.99]:
            hiYCoord = round(entity.pos[1] + entity.height)

            if app.world.coordsOccupied(BlockPos(round(x), hiYCoord, round(z))):
                yEdge = hiYCoord - 0.5
                entity.pos[1] = yEdge - entity.height
   
    hitWall = False

    minY = roundHalfUp((entity.pos[1]))
    maxY = roundHalfUp((entity.pos[1] + entity.height))

    entity.pos[0] += entity.velocity[0]

    for y in range(minY, maxY + 1):
        for z in [entity.pos[2] - entity.radius * 0.99, entity.pos[2] + entity.radius * 0.99]:
            x = entity.pos[0]

            hiXBlockCoord = round((x + entity.radius))
            loXBlockCoord = round((x - entity.radius))

            if app.world.coordsOccupied(BlockPos(hiXBlockCoord, y, round(z))):
                # Collision on the right, so move to the left
                xEdge = (hiXBlockCoord - 0.5)
                entity.pos[0] = xEdge - entity.radius
                hitWall = True
            elif app.world.coordsOccupied(BlockPos(loXBlockCoord, y, round(z))):
                # Collision on the left, so move to the right
                xEdge = (loXBlockCoord + 0.5)
                entity.pos[0] = xEdge + entity.radius
                hitWall = True
    
    entity.pos[2] += entity.velocity[2]

    for y in range(minY, maxY + 1):
        for x in [entity.pos[0] - entity.radius * 0.99, entity.pos[0] + entity.radius * 0.99]:
            z = entity.pos[2]

            hiZBlockCoord = round((z + entity.radius))
            loZBlockCoord = round((z - entity.radius))

            if app.world.coordsOccupied(BlockPos(round(x), y, hiZBlockCoord)):
                zEdge = (hiZBlockCoord - 0.5)
                entity.pos[2] = zEdge - entity.radius
                hitWall = True
            elif app.world.coordsOccupied(BlockPos(round(x), y, loZBlockCoord)):
                zEdge = (loZBlockCoord + 0.5)
                entity.pos[2] = zEdge + entity.radius
                hitWall = True
    
    return hitWall

