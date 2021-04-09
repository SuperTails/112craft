#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in float aLight;
layout (location = 3) in float aBlockLight;
layout (location = 4) in float aBlockIdx;
layout (location = 5) in float aAtlasIdx;

uniform mat4 view;
uniform mat4 projection;

uniform int breakBlockIdx;

uniform float atlasWidth;

out vec2 AtlasTexCoord;
out vec2 FaceTexCoord;

out float breakColor;
out float light;
out float blockLight;
flat out int blockIdx;

void main() {
    gl_Position = projection * view * (vec4(aPos, 1.0));
    //gl_Position = vec4(aPos * 0.1, 1.0);
    //ourColor = aColor;

    FaceTexCoord = aTexCoord;
    AtlasTexCoord = vec2((aAtlasIdx + aTexCoord.x) * 16.0 / atlasWidth, aTexCoord.y);

    blockIdx = int(aBlockIdx);

    if (breakBlockIdx == blockIdx) {
        breakColor = 1.0;
    } else {
        breakColor = 0.0;
    }

    light = aLight;
    blockLight = aBlockLight;
}