from entity import Entity
from typing import List, Optional
from dataclasses import dataclass
from util import ItemId, BlockPos
import copy

@dataclass
class Slot:
    """Represents a single part of a player's inventory
    
    If the amount is zero the item is ignored and the slot is empty.
    If the amount is negative the item is considered to be infinite.
    """

    item: ItemId
    amount: int

    def isInfinite(self) -> bool:
        return self.amount < 0
    
    def isEmpty(self) -> bool:
        return self.amount == 0
    
    def tryMergeWith(self, other: 'Slot') -> Optional['Slot']:
        if self.isEmpty():
            return copy.copy(other)
        elif other.isEmpty():
            return copy.copy(self)
        elif self.item == other.item:
            # TODO: Stack sizes
            return Slot(self.item, self.amount + other.amount)
        else:
            return None

class Player(Entity):
    height: float
    radius: float

    walkSpeed: float
    reach: float

    hotbarIdx: int = 0

    creative: bool

    inventory: List[Slot]

    def __init__(self, app, creative: bool):
        self.kind = 'player'
        self.pos = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.radius = 0.3
        self.height = 1.5
        self.onGround = False
        self.walkSpeed = 0.2

        self.bodyAngle = 0.0
        self.headAngle = 0.0

        self.reach = 4.0

        self.creative = creative
        

        if self.creative:
            if len(app.itemTextures) > 36:
                # TODO:
                1 / 0

            self.inventory = [Slot(name, -1) for name in app.itemTextures]
            while len(self.inventory) < 36:
                self.inventory.append(Slot('', 0))
        else:
            self.inventory = [Slot('', 0) for _ in range(36)]
    
    def pickUpItem(self, app, newItem: Slot):
        """Adds an item to the player's inventory."""

        if newItem.isEmpty(): return

        # Prioritize existing stacks of the item first
        for (i, slot) in enumerate(self.inventory):
            if slot.isInfinite() and slot.item == newItem.item:
                # It just stacks into an infinite slot, so no change
                return
            elif newItem.isInfinite() and slot.item == newItem.item:
                # ditto
                return 
            elif slot.amount > 0 and slot.item == newItem.item:
                self.inventory[i].amount += newItem.amount
                return

        # If that fails, then just add the item to the next open space
        for (i, slot) in enumerate(self.inventory):
            if slot.isEmpty():
                self.inventory[i] = newItem
                return
        
        # TODO: Full inventory??
        1 / 0