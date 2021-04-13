from util import ItemId
import copy
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
