#version 330 core

out vec4 FragColor;

//in vec3 ourColor;
in vec2 TexCoord;

in float breakColor;

uniform sampler2D blockTexture;
uniform sampler2D breakTexture;

void main() {
    //FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
    //FragColor = vertexColor;
    //FragColor = vec4(ourColor, 1.0);
    vec3 c = texture(blockTexture, TexCoord).rgb;
    vec4 d = texture(breakTexture, TexCoord);

    FragColor = vec4(c - (d.rgb * d.a * breakColor), 1.0);
}