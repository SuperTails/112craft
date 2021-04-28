import random
import perlin
from functools import lru_cache

class BinarySeeder:
    def sample(self, cellX: int, cellY: int, seed: int):
        return hash((cellX, cellY, seed)) % 2 == 0

class UnitSeeder:
    def sample(self, cellX: int, cellY: int, seed: int):
        random.seed(hash((cellX, cellY, seed)))

        return random.random()

class VoronoiGen:
    cellSize: int

    def __init__(self, cellSize, seeder):
        self.cellSize = cellSize
        self.seeder = seeder
    
    @lru_cache
    def sample(self, x: int, y: int, seed):
        xOff = int(perlin.getPerlinFractal(x, y, 1 / self.cellSize, 2, seed) * self.cellSize / 3)
        yOff = int(perlin.getPerlinFractal(x, y, 1 / self.cellSize, 2, seed) * self.cellSize / 3)

        return self.getNearestSeed(x + xOff, y + yOff, seed)
    
    def getNearestSeed(self, x: int, y: int, seed):
        cellX = x // self.cellSize
        cellY = y // self.cellSize

        nearestDist = float('inf')
        nearestPoint = (-1, -1)
        nearestSeed = 0

        for otherCellX in (cellX - 1, cellX, cellX + 1):
            for otherCellY in (cellY - 1, cellY, cellY + 1):
                cellSeed, otherX, otherY = self.cellSeedAndPos(otherCellX, otherCellY, seed)

                dist = (otherX - x)**2 + (otherY - y)**2

                if nearestDist > dist:
                    nearestDist = dist
                    nearestPoint = (otherX, otherY)
                    nearestSeed = cellSeed
        
        return nearestSeed
    
    @lru_cache
    def cellSeedAndPos(self, cellX: int, cellY: int, seed):
        cellSeed = self.seeder.sample(cellX, cellY, seed)

        random.seed(hash((cellX, cellY, seed)))

        xOffset = random.randrange(0, self.cellSize)
        yOffset = random.randrange(0, self.cellSize)

        return (cellSeed, cellX * self.cellSize + xOffset, cellY * self.cellSize + yOffset)