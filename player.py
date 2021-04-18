"""Simply represents the player and their inventory.

Stack represents an item and an amount of that item.

Most of the player's behavior actually happens in `craft.py`,
which I may move in the future.

A player is simply an `Entity` with a few extra attributes, so there's
not that much too this module.
"""

from entity import Entity
from typing import List, Optional, Tuple, Literal, overload
from dataclasses import dataclass
from util import ItemId, BlockPos
from nbt import nbt
from inventory import Stack, Slot
import copy

class Player(Entity):
    reach: float

    hotbarIdx: int = 0

    creative: bool

    flying: bool

    inventory: List[Slot]

    def __init__(self, app, creative: bool = False, tag: Optional[nbt.TAG_Compound] = None):
        # FIXME: ID
        super().__init__(app, 1, 'player', 0.0, 0.0, 0.0, nbt=tag)

        self.reach = 4.0

        self.creative = creative
        self.flying = False
        
        if self.creative:
            if len(app.itemTextures) > 36:
                # TODO:
                1 / 0
            
            self.inventory = [Slot(stack=Stack(name, -1)) for name in app.itemTextures]
            
            while len(self.inventory) < 36:
                self.inventory.append(Slot(stack=Stack('', 0)))
        else:
            self.inventory = [Slot() for _ in range(36)]
        
        if tag is not None:
            for stackTag in tag["Inventory"]:
                (stack, slotIdx) = Stack.fromNbt(stackTag, getSlot=True)
                self.inventory[slotIdx].stack = stack
            
            gameMode = tag['playerGameType'].value
            if gameMode == 0:
                self.creative = False
            elif gameMode == 1:
                self.creative = True
            else:
                raise Exception(f'Invalid game mode {gameMode}')

            self.flying = tag['abilities']['flying'].value != 0
        
    def toNbt(self) -> nbt.TAG_Compound:
        tag = super().toNbt()

        inventory = nbt.TAG_List(type=nbt.TAG_Compound, name='Inventory')
        for (slotIdx, item) in enumerate(self.inventory):
            stackTag = item.stack.toNbt(slotIdx)
            if stackTag is not None:
                inventory.append(stackTag)

        tag.tags.append(inventory)

        gameMode = 1 if self.creative else 0
        tag.tags.append(nbt.TAG_Int(gameMode, 'playerGameType'))

        abilities = nbt.TAG_Compound()
        abilities.name = 'abilities'
        abilities.tags.append(nbt.TAG_Byte(int(self.flying), 'flying'))

        tag.tags.append(abilities)

        return tag
    
    def tick(self, app, world, entities, playerX, playerZ):
        if self.immunity > 0:
            self.immunity -= 1
    
    def pickUpItem(self, app, newItem: Stack):
        """Adds an item to the player's inventory."""

        if newItem.isEmpty(): return

        # Prioritize existing stacks of the item first
        for (i, slot) in enumerate(self.inventory):
            stack = slot.stack
            if stack.isInfinite() and stack.item == newItem.item:
                # It just stacks into an infinite slot, so no change
                return
            elif newItem.isInfinite() and stack.item == newItem.item:
                # ditto
                return 
            elif stack.amount > 0 and stack.item == newItem.item:
                self.inventory[i].stack.amount += newItem.amount
                return

        # If that fails, then just add the item to the next open space
        for (i, slot) in enumerate(self.inventory):
            if slot.isEmpty():
                self.inventory[i].stack = newItem
                return
        
        # TODO: Full inventory??
        1 / 0
    