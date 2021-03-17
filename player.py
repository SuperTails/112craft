from typing import List, Optional
from dataclasses import dataclass
from world import ItemId, BlockPos

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
            return other
        elif other.isEmpty():
            return self
        elif self.item == other.item:
            # TODO: Stack sizes
            return Slot(self.item, self.amount + other.amount)
        else:
            return None

class Player:
    height: float = 1.5
    radius: float = 0.3

    velocity: List[float] = [0.0, 0.0, 0.0]
    onGround: bool = False

    walkSpeed: float = 0.2
    reach: float = 4.0

    hotbarIdx: int = 0

    creative: bool

    inventory: List[Slot]

    def __init__(self, app, creative: bool):
        self.creative = creative

        if self.creative:
            if len(app.itemTextures) > 36:
                # TODO:
                1 / 0

            self.inventory = [Slot(name, -1) for name in app.itemTextures]
            while len(self.inventory) < 36:
                self.inventory.append(Slot('', 0))
        else:
            self.inventory = [Slot('', 0)] * 36
            # First slot is always reserved for breaking blocks
            self.inventory[0] = Slot('air', -1)
    
    def pickUpItem(self, app, newItem: Slot):
        """Adds an item to the player's inventory."""

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