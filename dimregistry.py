from quarry.types import nbt
from dataclasses import dataclass

@dataclass
class BiomeEntry:
    name: str
    biomeId: int

    @classmethod
    def fromNbt(cls, tag: nbt.TagCompound):
        name = tag.value['name'].value
        biomeId = tag.value['id'].value

        return BiomeEntry(name, biomeId)
        
@dataclass
class BiomeRegistry:
    biomes: list[BiomeEntry]

    @classmethod
    def fromNbt(cls, tag: nbt.TagCompound):
        assert(tag.value['type'].value == 'minecraft:worldgen/biome')

        biomes = [BiomeEntry.fromNbt(entry) for entry in tag.value['value'].value]

        return cls(biomes)
    
    def getBiome(self, biomeId: int) -> BiomeEntry:
        return self.biomes[biomeId]

@dataclass
class DimensionTypeRegistry:
    @classmethod
    def fromNbt(cls, tag: nbt.TagCompound):
        assert(tag.value['type'].value == 'minecraft:dimension_type')

        return cls()

@dataclass
class DimensionCodec:
    dimensionTypeRegistry: DimensionTypeRegistry
    biomeRegistry: BiomeRegistry

    @classmethod
    def fromNbt(cls, tag: nbt.TagCompound):
        tag = tag.value['']

        dtr = DimensionTypeRegistry.fromNbt(tag.value['minecraft:dimension_type'])
        br  = BiomeRegistry.fromNbt(tag.value['minecraft:worldgen/biome'])

        return cls(dtr, br)
