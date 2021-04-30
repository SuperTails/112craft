#version 330 core

in vec2 TexCoords;

out vec4 FragColor;

uniform sampler2D transTexture;
uniform sampler2D depthTexture;

void main() {
    FragColor = texture(transTexture, TexCoords);

    if (FragColor.a == 0.0) {
        discard;
    }

    FragColor.a = 0.7;

    gl_FragDepth = texture(depthTexture, TexCoords).r;
}