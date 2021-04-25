from util import ItemId
import copy
import math
from nbt import nbt
from typing import Optional, Literal, Tuple, overload
from dataclasses import dataclass

@dataclass
class Stack:
    """Represents a single group of items
    
    If the amount is zero the item is ignored and the slot is empty.
    If the amount is negative the item is considered to be infinite.
    """

    item: ItemId
    amount: int

    def isInfinite(self) -> bool:
        return self.amount < 0
    
    def isEmpty(self) -> bool:
        return self.amount == 0
    
    def tryMergeWith(self, other: 'Stack') -> Optional['Stack']:
        if self.isEmpty():
            return copy.copy(other)
        elif other.isEmpty():
            return copy.copy(self)
        elif self.item == other.item:
            # TODO: Stack sizes
            return Stack(self.item, self.amount + other.amount)
        else:
            return None
    
    def toNbt(self, slotIdx = None) -> Optional[nbt.TAG_Compound]:
        if self.isEmpty():
            return None
        else:
            result = nbt.TAG_Compound()
            result.tags.append(nbt.TAG_String(f'minecraft:{self.item}', 'id'))
            result.tags.append(nbt.TAG_Byte(self.amount, 'Count'))
            if slotIdx is not None:
                result.tags.append(nbt.TAG_Byte(slotIdx, 'Slot'))
            return result
    
    @overload
    @classmethod
    def fromNbt(cls, tag: nbt.TAG_Compound) -> 'Stack':
        raise Exception()
    
    @overload
    @classmethod
    def fromNbt(cls, tag: nbt.TAG_Compound, getSlot: Literal[True]) -> Tuple['Stack', bool]:
        raise Exception()

    @classmethod
    def fromNbt(cls, tag: nbt.TAG_Compound, getSlot: bool = False):
        item = tag['id'].value.removeprefix('minecraft:')
        amount = tag['Count'].value

        stack = cls(item, amount)

        if getSlot:
            slotIdx = tag['Slot'].value
            return (stack, slotIdx)
        else:
            return stack

@dataclass
class Slot:
    stack: Stack
    canInput: bool
    canOutput: bool
    persistent: bool
    itemFilter: str

    def __init__(self, stack = None, canInput: bool = True, canOutput: bool = True, persistent: bool = True, itemFilter: str = ''):
        if stack is None:
            self.stack = Stack('', 0)
        else:
            self.stack = stack
        self.canInput = canInput
        self.canOutput = canOutput
        self.persistent = persistent
        self.itemFilter = itemFilter
    
    def isEmpty(self) -> bool:
        return self.stack.isEmpty()
    
    def isInfinite(self) -> bool:
        return self.stack.isInfinite()

def onSlotClicked(heldItem: Stack, app, isRight: bool, slot: Slot):
    if slot.canInput and slot.canOutput:
        if isRight:
            onRightClickIntoNormalSlot(heldItem, app, slot)
        else:
            onLeftClickIntoNormalSlot(heldItem, app, slot)
    elif not slot.canInput and slot.canOutput:
        print(f"before: {heldItem}, {slot.stack}")
        merged = heldItem.tryMergeWith(slot.stack)
        print(f"merged: {merged}")
        if merged is not None:
            heldItem.item, heldItem.amount = merged.item, merged.amount
            slot.stack = Stack('', 0)
    else:
        raise Exception("TODO")

def onRightClickIntoNormalSlot(heldItem: Stack, app, normalSlot: Slot):
    normalStack = normalSlot.stack
    if heldItem.isEmpty():
        # Picks up half of the slot
        if normalSlot.isInfinite():
            amountTaken = 1
        else:
            amountTaken = math.ceil(normalStack.amount / 2)
            normalStack.amount -= amountTaken
        heldItem.item = normalStack.item
        heldItem.amount = amountTaken
    else:
        newStack = normalStack.tryMergeWith(Stack(heldItem.item, 1))
        if newStack is not None:
            if not heldItem.isInfinite():
                heldItem.amount -= 1
            normalStack.item = newStack.item
            normalStack.amount = newStack.amount

def onLeftClickIntoNormalSlot(heldItem: Stack, app, normalSlot: Slot):
    normalStack = normalSlot.stack
    newStack = heldItem.tryMergeWith(normalStack)
    if newStack is None or heldItem.isEmpty():
        tempItem = heldItem.item
        tempAmount = heldItem.amount
        heldItem.item = normalStack.item
        heldItem.amount = normalStack.amount
        normalStack.item = tempItem
        normalStack.amount = tempAmount
    else:
        heldItem.amount = 0
        heldItem.item = ''
        normalStack.item = newStack.item
        normalStack.amount = newStack.amount

