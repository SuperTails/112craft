#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform mat4 projection;
uniform mat4 view;

out vec3 TexCoords;
out vec2 SunTexCoords;

// https://learnopengl.com/Advanced-OpenGL/Cubemaps
void main() {
    TexCoords = aPos * 2.0;
    vec4 pos = projection * view * vec4(aPos * 2.0, 1.0);

    SunTexCoords = aTexCoord;

    gl_Position = pos.xyww;
}