from typing import List
from dataclasses import dataclass
from world import ItemId

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
            if len(app.itemTextures) > 9:
                # TODO:
                1 / 0

            self.inventory = [Slot(name, -1) for name in app.itemTextures]
            while len(self.inventory) < 9:
                self.inventory.append(Slot('', 0))
        else:
            self.inventory = [Slot('', 0)] * 9
            # First slot is always reserved for breaking blocks
            self.inventory[0] = Slot('air', -1)
    
    def pickUpItem(self, app, newItem: ItemId):
        """Adds an item to the player's inventory."""

        # Prioritize existing stacks of the item first
        for (i, slot) in enumerate(self.inventory):
            if slot.isInfinite() and slot.item == newItem:
                # It just stacks into an infinite slot, so no change
                return
            elif slot.amount > 0 and slot.item == newItem:
                self.inventory[i].amount += 1
                return

        # If that fails, then just add the item to the next open space
        for (i, slot) in enumerate(self.inventory):
            if slot.isEmpty():
                self.inventory[i] = Slot(newItem, 1)
                return
        
        # TODO: Full inventory??
        1 / 0
        
