#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in ivec4 instanceMatrix; 

uniform mat4 view;
uniform mat4 projection;

uniform ivec4 breakingBlockPos;
uniform float breakingBlockAmount;

out vec3 ourColor;
out vec2 TexCoord;
out float breakColor;

void main() {
    gl_Position = projection * view * (instanceMatrix + vec4(aPos, 1.0));
    //ourColor = aColor;
    TexCoord = aTexCoord;

    if (instanceMatrix == breakingBlockPos) {
        breakColor = breakingBlockAmount;
    } else {
        breakColor = 0.0;
    }
}