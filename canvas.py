"""Provides a wrapper struct that is meant to simulate a TKinter canvas.

This canvas allows for using the same GUI/text drawing code between both
the TKinter and OpenGL backends.
It draws on a PIL image and then renders to an OpenGL texture.

In theory any set of drawing calls to this canvas should appear identical
to the same set of drawing calls to a TKinter canvas...
...but it's not perfect.

Due to a limitation in PIL drawing functions, this canvas does not have
an alpha channel that works properly. Instead, any pixels that are
ALPHA_COLOR are removed from the image.

I miss the TKinter canvas.
"""

from shader import ShaderProgram
import typing
from PIL import Image, ImageDraw, ImageFont
from OpenGL.GL import * #type:ignore
import numpy

def convertAnchor(anchor):
    if anchor == 'center':
        return 'mm'

    if len(anchor) == 1:
        if anchor == 'n' or anchor == 's':
            horiz = 'center'
            vert = anchor
        else:
            horiz = anchor
            vert = 'center'
    else:
        vert = anchor[0]
        horiz = anchor[1]
    
    if horiz == 'center':
        horiz2 = 'm'
    elif horiz == 'w':
        horiz2 = 'l'
    else:
        horiz2 = 'r'
    
    if vert == 'center':
        vert2 = 'm'
    elif vert == 'n':
        vert2 = 'a'
    else:
        vert2 = 'd'

    return horiz2 + vert2

ALPHA_COLOR = (0xFE, 0xFE, 0xFE)

class Canvas:
    image: Image.Image
    draw: ImageDraw.ImageDraw

    vao: int
    program: ShaderProgram

    texture: int

    width: int
    height: int

    font: ImageFont.ImageFont

    def __init__(self, width, height):
        self.image = Image.new("RGB", (width, height), color=ALPHA_COLOR)
        self.draw = typing.cast(ImageDraw.ImageDraw, ImageDraw.Draw(self.image, "RGBA"))

        self._createGlSurface()
        self._createGlTexture()

        self.width = width
        self.height = height

        self.program = ShaderProgram('shaders/guiShader.vert', 'shaders/guiShader.frag')

        self.font = ImageFont.truetype('assets/minecraft_font.ttf', 16)
    
    def create_oval(self, x0, y0, x1, y1, **kwargs):
        if 'fill' in kwargs:
            assert(kwargs['fill'] not in ['#000', '#000000', 'black'])

        fill = kwargs.pop('fill', '#0000')
        outline = kwargs.pop('outline', '#111F')
        width = kwargs.pop('width', 1)
        assert(len(kwargs) == 0)
        self.draw.ellipse([x0, y0, x1, y1], fill=fill, outline=outline, width=width)
    
    def create_rectangle(self, x0, y0, x1, y1, **kwargs):
        if 'fill' in kwargs:
            assert(kwargs['fill'] not in ['#000', '#000000', 'black'])

        fill = kwargs.pop('fill', '#0000')
        outline = kwargs.pop('outline', '#111F')
        width = kwargs.pop('width', 1)

        assert(len(kwargs) == 0)

        self.draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=width)
    
    def create_text(self, x, y, **kwargs):
        text = kwargs.pop('text')
        fill = kwargs.pop('fill', '#FFF')
        anchor = convertAnchor(kwargs.pop('anchor', 'center'))
        font = kwargs.pop('font', ())

        if len(kwargs) != 0:
            print(kwargs)
            assert(False)

        self.draw.text((int(x), int(y)), text=text, fill=fill, font=self.font, anchor=anchor)
    
    def create_image(self, x, y, **kwargs):
        image = kwargs.pop('image')
        anchor = kwargs.pop('anchor', 'center')

        if len(kwargs) != 0:
            print(kwargs)
            assert(False)

        if anchor == 'center':
            x -= image.width // 2
            y -= image.height // 2
        else:
            print(anchor)
            assert(False)

        self.image.paste(image, box=(int(x), int(y)))

    def _createGlSurface(self):
        vertices = numpy.array([
             1.0,  1.0, 1.0, 0.0, # top right
             1.0, -1.0, 1.0, 1.0, # bottom right
            -1.0, -1.0, 0.0, 1.0, # bottom left
            -1.0,  1.0, 0.0, 0.0, # top left 
        ], dtype='float32')

        indices = numpy.array([
            0, 1, 3,
            1, 2, 3,
        ], dtype='uint32')

        vao: int = glGenVertexArrays(1) #type:ignore
        vbo: int = glGenBuffers(1) #type:ignore
        ebo: int = glGenBuffers(1) #type:ignore

        glBindVertexArray(vao)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(0)

        self.vao = vao
    
    def _createGlTexture(self):
        self.texture = glGenTextures(1) #type:ignore
        glBindTexture(GL_TEXTURE_2D, self.texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        #grassTex = grassTex.transpose(Image.FLIP_TOP_BOTTOM)
        arr = numpy.asarray(self.image, dtype=numpy.uint8)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.image.width, self.image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, arr) #type:ignore

        glBindTexture(GL_TEXTURE_2D, 0)
    
    def redraw(self):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        arr = numpy.asarray(self.image, dtype=numpy.uint8)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.image.width, self.image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, arr) #type:ignore
        #glBindTexture(GL_TEXTURE_2D, 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.program.useProgram()
        glUniform3f(self.program.getUniformLocation("alphaColor"), ALPHA_COLOR[0] / 255.0, ALPHA_COLOR[1] / 255.0, ALPHA_COLOR[2] / 255.0)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, ctypes.c_void_p(0)) #type:ignore

        glBlendFunc(GL_SRC_ALPHA, GL_ZERO)

        self.image = Image.new("RGB", (self.width, self.height), color=ALPHA_COLOR)
        self.draw = typing.cast(ImageDraw.ImageDraw, ImageDraw.Draw(self.image, "RGBA"))