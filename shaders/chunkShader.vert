#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform mat4 view;
uniform mat4 projection;

uniform float atlasWidth;

out vec2 TexCoord;
out float breakColor;

void main() {
    gl_Position = projection * view * (vec4(aPos, 1.0));
    //gl_Position = vec4(aPos * 0.1, 1.0);
    //ourColor = aColor;
    TexCoord = vec2(aTexCoord.x / atlasWidth, aTexCoord.y);

    breakColor = 0.0;
}