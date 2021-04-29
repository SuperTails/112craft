import random
import perlin
import heapq
import math
from functools import lru_cache
from typing import Tuple, List, Any

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
    def sample2(self, x: int, y: int, seed) -> List[Tuple[Any, float]]:
        xOff = int(perlin.getPerlinFractal(x, y, 1 / self.cellSize, 2, seed) * self.cellSize / 3)
        yOff = int(perlin.getPerlinFractal(x, y, 1 / self.cellSize, 2, seed) * self.cellSize / 3)

        return self.getNearestSeeds(x + xOff, y + yOff, seed)
    
    def sample(self, x: int, y: int, seed) -> Tuple[Any, float]:
        return self.sample2(x, y, seed)[0]
    
    def getNearestSeeds(self, x: int, y: int, seed):
        cellX = x // self.cellSize
        cellY = y // self.cellSize

        points = []

        def dist(p):
            return p[1]

        for otherCellX in (cellX - 2, cellX - 1, cellX, cellX + 1, cellX + 2):
            for otherCellY in (cellY - 2, cellY - 1, cellY, cellY + 1, cellY + 2):
                cellSeed, otherX, otherY = self.cellSeedAndPos(otherCellX, otherCellY, seed)

                points.append((cellSeed, math.sqrt((otherX - x)**2 + (otherY - y)**2)))
            
        return heapq.nsmallest(2, points, key=dist)
    
    @lru_cache
    def cellSeedAndPos(self, cellX: int, cellY: int, seed):
        cellSeed = self.seeder.sample(cellX, cellY, seed)

        random.seed(hash((cellX, cellY, seed)))

        xOffset = random.randrange(0, self.cellSize)
        yOffset = random.randrange(0, self.cellSize)

        return (cellSeed, cellX * self.cellSize + xOffset, cellY * self.cellSize + yOffset)