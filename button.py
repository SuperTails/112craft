from cmu_112_graphics import *
from typing import Callable, List, Optional
import math

def getCachedImage(image):
# From:
# https://www.kosbie.net/cmu/fall-19/15-112/notes/notes-animations-part2.html
    if ('cachedPhotoImage' not in image.__dict__):
        image.cachedPhotoImage = ImageTk.PhotoImage(image)
    return image.cachedPhotoImage

def createSizedBackground(app, width: int, height: int):
    cobble = app.loadImage('assets/CobbleBackground.png')
    cobble = app.scaleImage(cobble, 2)
    cWidth, cHeight = cobble.size

    newCobble = Image.new(cobble.mode, (width, height))
    for xIdx in range(math.ceil(width / cWidth)):
        for yIdx in range(math.ceil(height / cHeight)):
            xOffset = xIdx * cWidth
            yOffset = yIdx * cHeight

            newCobble.paste(cobble, (xOffset, yOffset))

    return newCobble

class Button:
    # Coords of top left
    x: int
    y: int

    width: int
    height: int

    background: Image
    text: str

    def __init__(self, x: int, y: int, background, text, anchor='c'):
        if anchor == 'c':
            (width, height) = background.size

            self.x = x - width // 2
            self.y = y - height // 2

            self.width = width
            self.height = height

            self.background = background
            self.text = text
        else:
            # TODO: Other anchors, if necessary
            1 / 0

    def isOver(self, x: int, y: int) -> bool:
        '''Returns True if the given position is inside this button'''

        return (self.x <= x and x <= self.x + self.width and 
            self.y <= y and y <= self.y + self.height)
    
    def draw(self, app, canvas):
        centerX = self.x + (self.width // 2)
        centerY = self.y + (self.height // 2)

        canvas.create_rectangle(self.x - 1, self.y - 1, self.x + self.width, self.y + self.height, outline='#555')
        canvas.create_image(centerX, centerY, image=getCachedImage(self.background))
        canvas.create_text(centerX, centerY, text=self.text, font='Arial 16 bold', fill='white')

class ButtonManager:
    buttons: dict[str, Button] = {}
    heldButtonName: Optional[str] = None

    def __init__(self): pass

    def onPress(self, x, y) -> None:
        for (buttonName, button) in self.buttons.items():
            if button.isOver(x, y):
                self.heldButtonName = buttonName
                break
    
    def addButton(self, name: str, button: Button) -> None:
        self.buttons[name] = button
    
    def onRelease(self, x, y) -> Optional[str]:
        '''Returns the name of the button that was just activated, if any'''

        if self.heldButtonName is not None:
            heldButton = self.buttons[self.heldButtonName]
            result = self.heldButtonName if heldButton.isOver(x, y) else None
            self.heldButtonName = None
            return result
        else:
            return None
                
    
    def draw(self, app, canvas) -> None:
        for button in self.buttons.values():
            button.draw(app, canvas)