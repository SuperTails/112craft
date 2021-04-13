#version 330 core

out vec4 FragColor;

in vec3 TexCoords;
in vec2 SunTexCoords;

uniform int gameTime;

uniform sampler2D sunTex;

void main() {
    float mag = sqrt(TexCoords.x*TexCoords.x + TexCoords.y*TexCoords.y + TexCoords.z*TexCoords.z);

    vec3 NormCoords = TexCoords / mag;

    float pitch = atan(TexCoords.y, sqrt(TexCoords.x*TexCoords.x + TexCoords.z*TexCoords.z)) / 3.14159;

    vec3 hiSkyBg = vec3(0x87 / 255.0, 0xA5 / 255.0, 0xFF / 255.0);
    vec3 loSkyBg = vec3(0x51 / 255.0, 0x5F / 255.0, 0xCC / 255.0);

    float factor = 1.0 / (1.0 + exp(-20.0 * pitch));
    
    vec3 skyBg = factor * hiSkyBg + (1 - factor) * loSkyBg;

    //skyBg = vec3(pitch, 0.0, 0.0);

    float dayFrac = ((gameTime % 24000) / 24000.0) - 0.75;
    float daylightFactor = sin(dayFrac * 2.0 * 3.14159 /2.0);
    daylightFactor *= daylightFactor;

    skyBg *= daylightFactor;

    float yaw = acos(TexCoords.z / mag);
    // 0.0 is the exact location of the sun
    float yawFrac = mod((yaw / (3.1415926)), 1.0);

    float pitchDist = (((gameTime % 24000) / 24000.0)) - pitch;

    float yawDist = yawFrac;
    if (yawDist > 0.5) {
        yawDist = 1.0 - yawFrac;
    }

    //vec2 AdjustCoord = SunTexCoords;
    vec2 AdjustCoord = SunTexCoords;
    AdjustCoord.y += (gameTime % 24000)  / 24000.0;

    vec4 sunColor = texture(sunTex, AdjustCoord);

    FragColor = vec4(skyBg + sunColor.rgb, 1.0);

    //FragColor = vec4(skyBg * (1 - sunColor.a) + sunColor.rgb * sunColor.a, 1.0);

    //FragColor = vec4(skyBg * (1.0 - sunColor.a) + sunColor.rgb * sunColor.a, 1.0);

    //FragColor = vec4(pitchFrac, 0.0, pitchDist, 1.0);
}