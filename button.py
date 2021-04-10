"""Provides a `ButtonManager` for drawing and interacting with buttons.

A `Button` has an ID, a position, and some text or a background.
Multiple buttons can be added to a ButtonManager, and then they are all
checked at once when the mouse is pressed. The button manager then returns
the ID of the button that was pressed.
"""

from typing import Callable, List, Optional
from PIL import Image
import math

def createSizedBackground(app, width: int, height: int):
    cobble = Image.open('assets/CobbleBackground.png')
    cobble = cobble.resize((cobble.width * 2, cobble.height * 2), resample=Image.NEAREST)
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

    xFrac: float
    yFrac: float

    width: int
    height: int

    background: Image.Image
    text: str

    def __init__(self, app, x: float, y: float, width: int, height: int, text: str):
        """Creates a button.

        The x and y are proportions of the screen width and height, e.g.
        x=0.1 and y=0.9 would put the center of the button in the bottom-left
        corner of the canvas.

        `background` is the image to be displayed on the button, and 
        `text` will be displayed on top of that image
        """

        self.width = width
        self.height = height

        self.background = createSizedBackground(app, self.width, self.height)
        self.text = text

        self.xFrac = x
        self.yFrac = y

        self.canvasSizeChanged(app)

    def canvasSizeChanged(self, app): 
        """Recalculates where this button should be on the screen.

        This should be called when the canvas resizes.
        """

        self.x = int(self.xFrac * app.width) - self.width // 2
        self.y = int(self.yFrac * app.height) - self.height // 2

    def isOver(self, app, x: int, y: int) -> bool:
        '''Returns True if the given position is inside this button'''

        x0 = int(self.xFrac * app.width) - self.width // 2
        y0 = int(self.yFrac * app.height) - self.height // 2
        x1 = x0 + self.width
        y1 = y0 + self.height

        return (x0 <= x and x <= x1 and y0 <= y and y <= y1)
    
    def draw(self, app, canvas):
        centerX = self.x + (self.width // 2)
        centerY = self.y + (self.height // 2)

        canvas.create_rectangle(self.x - 1, self.y - 1, self.x + self.width, self.y + self.height, outline='#555')
        canvas.create_image(centerX, centerY, image=self.background)
        canvas.create_text(centerX, centerY, text=self.text, font='Arial 16 bold', fill='white')

class ButtonManager:
    buttons: dict[str, Button]
    heldButtonName: Optional[str]

    def __init__(self):
        self.buttons = {}
        self.heldButtonName = None

    def onPress(self, app, x, y) -> None:
        for (buttonName, button) in self.buttons.items():
            if button.isOver(app, x, y):
                self.heldButtonName = buttonName
                break
    
    def addButton(self, name: str, button: Button) -> None:
        self.buttons[name] = button
    
    def onRelease(self, app, x, y) -> Optional[str]:
        '''Returns the name of the button that was just activated, if any'''

        if self.heldButtonName is not None:
            heldButton = self.buttons[self.heldButtonName]
            result = self.heldButtonName if heldButton.isOver(app, x, y) else None
            self.heldButtonName = None
            return result
        else:
            return None
                
    def canvasSizeChanged(self, app) -> None:
        for button in self.buttons.values():
            button.canvasSizeChanged(app)
    
    def draw(self, app, canvas) -> None:
        for button in self.buttons.values():
            button.draw(app, canvas)