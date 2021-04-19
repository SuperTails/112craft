#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 texCoord;
layout (location = 2) in vec3 pivot;

out vec2 TexCoord;

out vec3 bPos;

uniform float rot;

uniform float rotX;
uniform float rotY;
uniform float rotZ;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    bPos = aPos;

    float rot = rot + 3.14159;

    mat4 rotMat = mat4(
        cos(rot), 0.0, -sin(rot), 0.0,
             0.0, 1.0,       0.0, 0.0,
        sin(rot), 0.0,  cos(rot), 0.0,
             0.0, 0.0,       0.0, 1.0
    );

    mat3 rotMatZ = mat3(
        cos(rotZ), -sin(rotZ), 0.0,
        sin(rotZ), cos(rotZ),  0.0,
        0.0, 0.0, 1.0
    );

    mat3 rotMatX = mat3(
        1.0, 0.0, 0.0,
        0.0, cos(rotX), -sin(rotX),
        0.0, sin(rotX), cos(rotX)
    );

    mat3 rotMatY = mat3(
        cos(rotY), 0.0, -sin(rotY),
              0.0, 1.0,        0.0,
        sin(rotY), 0.0,  cos(rotY)
    );

    vec3 pivot2 = pivot / 16.0;

    vec3 pivoted = (rotMatZ * rotMatY * rotMatX * (aPos / 16.0 - pivot2)) + pivot2;

    gl_Position = projection * view * model * rotMat * vec4(pivoted, 1.0) ;
    TexCoord = texCoord;
}