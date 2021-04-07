#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 texCoord;

out vec2 TexCoord;

out vec3 bPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    bPos = aPos;
    gl_Position = projection * view * model * vec4(aPos / 16.0, 1.0) ;
    TexCoord = texCoord;
}