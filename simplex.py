import math
from numba import jit

@jit(nopython=True) #type:ignore
def simplex(x, y, z, seed):
    def skew(x, y, z):
        f = (math.sqrt(3 + 1) - 1) / 3

        xp = x + (x + y + z) * f
        yp = y + (x + y + z) * f
        zp = z + (x + y + z) * f

        return (xp, yp, zp)

    def unskew(x, y, z):
        g = (1 - 1/math.sqrt(3 + 1)) / 3

        xs = x - (x + y + z) * g
        ys = y - (x + y + z) * g
        zs = z - (x + y + z) * g
        return (xs, ys, zs)

    # https://weber.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf
    # https://en.wikipedia.org/wiki/Simplex_noise

    xp, yp, zp = skew(x, y, z)

    xb = math.floor(xp)
    yb = math.floor(yp)
    zb = math.floor(zp)

    xs, ys, zs = unskew(xb, yb, zb)

    xi = x - xs
    yi = y - ys
    zi = z - zs

    if xi >= yi >= zi:
        vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
    elif xi >= zi >= yi:
        vertices = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)]
    elif yi >= xi >= zi:
        vertices = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)]
    elif yi >= zi >= xi:
        vertices = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)]
    elif zi >= xi >= yi:
        vertices = [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)]
    else: # zi > yi > xi
        vertices = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]
    
    vectors = [(1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),(1,0,1),(-1,0,1),(1,0,-1),
            (-1,0,-1),(0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1)]
    
    def combine(gx, gy, gz): return hash((gx, gy, gz, seed))

    def getGrad(gx, gy, gz): return vectors[combine(gx, gy, gz) % len(vectors)]

    total = 0

    for vert in vertices:
        vert = (vert[0], vert[1], vert[2])

        gx, gy, gz = getGrad(xb + vert[0], yb + vert[1], zb + vert[2])

        vert = unskew(vert[0] + xb, vert[1] + yb, vert[2] + zb)
        vert = (x - vert[0], y - vert[1], z - vert[2])

        #dx = x - vert[0]
        #dy = y - vert[1]
        #dz = z - vert[2]

        dx, dy, dz = vert

        d = math.sqrt(dx**2 + dy**2 + dz**2)

        #print(f'd**2: {d**2}')

        r_2 = 0.5

        total += max(0, r_2 - d**2)**4 * (dx * gx + dy * gy + dz * gz)

    return total

def getSimplexFractal(x, y, z, baseFreq: float, octaves: int, seed):
    """Returns the sum of multiple levels of perlin noise.

    A single iteration of perlin noise samples on a grid with a single
    size of cells, but this can look unnatural, so instead we add together the
    results from the grid, and a smaller grid, and a smaller grid... etc.
    
    Arguments:
     * `(x, y)`: The point sampled at
     * `baseFreq`: The lowest frequency of noise used
     * `octaves`: The number of different frequencies added together
     * `seed`: Seed used for random number generation
    """

    freq = baseFreq
    total = 0.0
    totalscale = 0.0
    persist = 0.5
    for i in range(octaves):
        freq *= 2
        amplitude = 0.5**i

        xs, ys, zs = x * freq, y * freq, z * freq

        total += amplitude * simplex(xs, ys, zs, seed)
        totalscale += amplitude
    
    return total / totalscale * 100