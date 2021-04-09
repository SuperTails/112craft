import decimal
from typing import NamedTuple, Tuple, Optional

def roundHalfUp(d):
    # Round to nearest with ties going away from zero.
    rounding = decimal.ROUND_HALF_UP
    # See other rounding options here:
    # https://docs.python.org/3/library/decimal.html#rounding-modes
    return int(decimal.Decimal(d).to_integral_value(rounding=rounding))

def rayAABBIntersect(
    rayOrigin: Tuple[float, float, float],
    rayDir: Tuple[float, float, float],
    corner0: Tuple[float, float, float],
    corner1: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:

    b = (corner0[0], corner1[1], corner1[2])
    i1 = rayAlignedPlaneIntersect(rayOrigin, rayDir, b, corner0)
    b = (corner1[0], corner0[1], corner1[2])
    i2 = rayAlignedPlaneIntersect(rayOrigin, rayDir, b, corner0)
    b = (corner1[0], corner1[1], corner0[2])
    i3 = rayAlignedPlaneIntersect(rayOrigin, rayDir, b, corner0)

    b = (corner1[0], corner0[1], corner0[2])
    i4 = rayAlignedPlaneIntersect(rayOrigin, rayDir, b, corner1)
    b = (corner0[0], corner1[1], corner0[2])
    i5 = rayAlignedPlaneIntersect(rayOrigin, rayDir, b, corner1)
    b = (corner0[0], corner0[1], corner1[2])
    i6 = rayAlignedPlaneIntersect(rayOrigin, rayDir, b, corner1)

    results = [i for i in [i1, i2, i3, i4, i5, i6] if i is not None]

    def dist(i):
        x = (i[0] - rayOrigin[0]) / rayDir[0]
        return x

    results.sort(key=dist)

    if results == []:
        return None
    else:
        return results[0]


def rayAlignedPlaneIntersect(
    rayOrigin: Tuple[float, float, float],
    rayDir: Tuple[float, float, float],
    plane0: Tuple[float, float, float],
    plane1: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:

    plane0 = (plane0[0] - rayOrigin[0], plane0[1] - rayOrigin[1], plane0[2] - rayOrigin[2])
    plane1 = (plane1[0] - rayOrigin[0], plane1[1] - rayOrigin[1], plane1[2] - rayOrigin[2])

    result = rayAlignedPlaneIntersect2(rayDir, plane0, plane1)

    if result is None:
        return None
    else:
        return (result[0] + rayOrigin[0], result[1] + rayOrigin[1], result[2] + rayOrigin[2])

def rayAlignedPlaneIntersect2(rayDir, plane0, plane1):
    planeMinX = min(plane0[0], plane1[0])
    planeMinY = min(plane0[1], plane1[1])
    planeMinZ = min(plane0[2], plane1[2])

    planeMaxX = max(plane0[0], plane1[0])
    planeMaxY = max(plane0[1], plane1[1])
    planeMaxZ = max(plane0[2], plane1[2])

    if planeMinX == planeMaxX:
        x = planeMinX
        t = x / rayDir[0]
        if t < 0:
            return None
        y = t * rayDir[1]
        z = t * rayDir[2]
        if planeMinY <= y and y <= planeMaxY and planeMinZ <= z and z <= planeMaxZ:
            return (x, y, z)
        else:
            return None
    elif planeMinY == planeMaxY:
        y = planeMinY
        t = y / rayDir[1]
        if t < 0:
            return None
        x = t * rayDir[0]
        z = t * rayDir[2]
        if planeMinX <= x and x <= planeMaxX and planeMinZ <= z and z <= planeMaxZ:
            return (x, y, z)
        else:
            return None
    elif planeMinZ == planeMaxZ:
        z = planeMinZ
        t = z / rayDir[2]
        if t < 0:
            return None
        x = t * rayDir[0]
        y = t * rayDir[1]
        if planeMinX <= x and x <= planeMaxX and planeMinY <= y and y <= planeMaxY:
            return (x, y, z)
        else:
            return None
    else:
        raise Exception("Plane is not axis-aligned")

class ChunkPos(NamedTuple):
    x: int
    y: int
    z: int

class BlockPos(NamedTuple):
    x: int
    y: int
    z: int

BlockId = str

ItemId = str
