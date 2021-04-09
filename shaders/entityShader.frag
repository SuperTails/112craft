#version 330 core

in vec2 TexCoord;

in vec3 bPos;

out vec4 FragColor;

uniform sampler2D skin;

uniform float immunity;

void main() {
    //FragColor = vec4(TexCoord.x, TexCoord.y, 0.5, 0.5);

    vec2 newCoord = vec2(TexCoord.x / 64.0, TexCoord.y / 32.0);

    FragColor = texture(skin, newCoord);

    if (immunity > 0.0) {
        FragColor.r = 0.9 + 0.1 * FragColor.r;
    }

    //FragColor = vec4(abs(bPos) * 0.1, 1.0);
}