from typing import List
from world import World
from entity import Entity

class ServerState:
    world: World
    entities: List[Entity]