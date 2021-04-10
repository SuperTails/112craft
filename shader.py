"""This module represents a vertex+fragment shader pair.

This is a simple abstraction to make loading them from disk and using
them easier.
"""

from OpenGL.GL import * #type:ignore

def readFile(path):
    with open(path, 'r') as f:
        return f.read()

class ShaderProgram:
    shaderProgram: int

    def __init__(self, vertexPath: str, fragmentPath: str):
        # From https://learnopengl.com 

        vertexShader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertexShader, readFile(vertexPath)) #type:ignore
        glCompileShader(vertexShader)

        fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragmentShader, readFile(fragmentPath)) #type:ignore
        glCompileShader(fragmentShader)

        self.shaderProgram = glCreateProgram() #type:ignore
        glAttachShader(self.shaderProgram, vertexShader)
        glAttachShader(self.shaderProgram, fragmentShader)
        glLinkProgram(self.shaderProgram)

        glDeleteShader(vertexShader)
        glDeleteShader(fragmentShader)
    
    def useProgram(self):
        glUseProgram(self.shaderProgram)
    
    def getUniformLocation(self, name: str) -> int:
        return glGetUniformLocation(self.shaderProgram, name)