from quarry.types import nbt
from dataclasses import dataclass
from typing import Union

@dataclass
class DimensionType:
    ambientLight: float
    hasSkyLight: bool
    logicalHeight: int
    coordinateScale: float
    ultrawarm: bool
    hasCeiling: bool

    @classmethod
    def fromNbt(cls, tag: nbt.TagCompound):
        ambientLight = tag.value['ambient_light'].value
        hasSkyLight = bool(tag.value['has_skylight'].value)
        logicalHeight = tag.value['logical_height'].value
        coordinateScale = tag.value['coordinate_scale'].value
        ultrawarm = bool(tag.value['ultrawarm'].value)
        hasCeiling = bool(tag.value['has_ceiling'].value)

        return cls(ambientLight, hasSkyLight, logicalHeight, coordinateScale, ultrawarm, hasCeiling)

@dataclass
class DimensionEntry:
    name: str
    dimensionId: int
    ty: DimensionType

    @classmethod
    def fromNbt(cls, tag: nbt.TagCompound):
        name = tag.value['name'].value
        dimensionId = tag.value['id'].value

        ty = DimensionType.fromNbt(tag.value['element'])

        return cls(name, dimensionId, ty)


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
    
    def getBiome(self, biomeId: Union[int, str]) -> BiomeEntry:
        if isinstance(biomeId, int):
            for b in self.biomes:
                if b.biomeId == biomeId:
                    return b
            raise KeyError(biomeId)
        else:
            for b in self.biomes:
                if b.name == biomeId:
                    return b
            
            raise KeyError(biomeId)

@dataclass
class DimensionTypeRegistry:
    dimensions: list[DimensionEntry]

    @classmethod
    def fromNbt(cls, tag: nbt.TagCompound):
        assert(tag.value['type'].value == 'minecraft:dimension_type')

        dims = [DimensionEntry.fromNbt(entry) for entry in tag.value['value'].value]

        return cls(dims)
    
    def getDimension(self, dimId: Union[int, str]) -> DimensionEntry:
        if isinstance(dimId, int):
            return self.dimensions[dimId]
        else:
            for d in self.dimensions:
                if d.name == dimId:
                    return d
            
            raise KeyError(dimId)


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

    def getBiome(self, biomeId: Union[int, str]):
        return self.biomeRegistry.getBiome(biomeId)
    
    def getDimension(self, dimId: Union[int, str]):
        return self.dimensionTypeRegistry.getDimension(dimId)