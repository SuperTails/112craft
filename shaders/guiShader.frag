#version 330 core

out vec4 FragColor;

in vec2 TexCoord;

uniform vec3 alphaColor;

uniform sampler2D guiTexture;

void main() {
    FragColor = texture(guiTexture, TexCoord);
    if (FragColor == vec4(alphaColor, 1.0)) {
        discard;
    }
}