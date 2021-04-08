#version 330 core

out vec4 FragColor;

//in vec3 ourColor;
in vec2 TexCoord;

in float breakColor;
in float light;
in float blockLight;

uniform sampler2D blockTexture;
uniform sampler2D breakTexture;

uniform int gameTime;

void main() {
    //FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
    //FragColor = vertexColor;
    //FragColor = vec4(ourColor, 1.0);

    vec3 c = texture(blockTexture, TexCoord).rgb;
    vec4 d = texture(breakTexture, TexCoord);

    float dayFrac = ((gameTime % 24000) / 24000.0) - 0.75;

    float daylightFactor = sin(dayFrac * 2.0 * 3.14159 /2.0);
    daylightFactor *= daylightFactor;

    float daylight = light * daylightFactor;
    
    float totalLight = max(blockLight, daylight);

    vec3 baseColor = c - (d.rgb * d.a * breakColor);

    float light2 = 0.18 + pow(totalLight / 8.0, 1.5);

    FragColor = vec4(baseColor * light2, 1.0);

    //FragColor = vec4(0.5, 1.0, 1.0, 1.0);
}