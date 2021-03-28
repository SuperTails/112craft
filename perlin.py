import random, math
from functools import lru_cache

@lru_cache
def getPerlinVectors(gridX, gridY, seed):
    # Vectors from https://mrl.cs.nyu.edu/~perlin/paper445.pdf 

    vectors = [(1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),(1,0,1),(-1,0,1),(1,0,-1),
               (-1,0,-1),(0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1)]
    
    def combine(gx, gy): return hash((gx, gy, seed))

    random.seed(combine(gridX, gridY))
    vtl = random.choice(vectors)

    random.seed(combine(gridX + 1, gridY))
    vtr = random.choice(vectors)

    random.seed(combine(gridX, gridY + 1))
    vbl = random.choice(vectors)

    random.seed(combine(gridX + 1, gridY + 1))
    vbr = random.choice(vectors)

    return (vtl, vtr, vbl, vbr)

def getPerlinValue(x, y, width, seed):
    """Returns a single sample of perlin noise.

    Arguments:
     * `(x, y)`: The point sampled at
     * `width`: The width of the grid being sampled from
     * `seed`: Seed used for random number generation
    """

    # Algorithm from https://adrianb.io/2014/08/09/perlinnoise.html
    # (it's all written in heavily-optimized C so no code was used)

    # First we pick four vectors, one for each corner of our current cell
    gridX = math.floor(x / width)
    gridY = math.floor(y / width)
    (vtl, vtr, vbl, vbr) = getPerlinVectors(gridX, gridY, seed)

    # Then we convert (x, y) to a fraction of our cell's width/height
    x -= gridX * width
    y -= gridY * width
    x /= width
    y /= width

    # And then we take the dot product of each corner vector with its
    # displacement to (x, y)
    dtl = vtl[0] * x + vtl[1] * y
    dtr = vtr[0] * (x-1.0) + vtr[1] * y
    dbl = vbl[0] * x + vbl[1] * (y-1.0)
    dbr = vbr[0] * (x-1.0) + vbr[1] * (y-1.0)

    # And then we interpolate between the values depending on how close we
    # are to each value. We use this `slope` function because linear
    # interpolation looks fairly jagged.
    def slope(t): return 6 * t**5 - 15 * t**4 + 10 * t**3
    def interp(a, b, t): return b * slope(t) + (1.0 - slope(t)) * a

    top = interp(dtl, dtr, x)
    bot = interp(dbl, dbr, x)

    return interp(top, bot, y)
    
def getPerlinFractal(x, y, baseFreq: float, octaves: int, seed):
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

        total += amplitude * getPerlinValue(x, y, 1.0 / freq, seed)
        totalscale += amplitude
    
    # 2D Perlin noise is always guaranteed to be in the range
    # [-sqrt(2) / 2, sqrt(2) / 2], but [-1, 1] is a more sensible range,
    # so I scale the output a little
    
    return total / totalscale * 1.41421