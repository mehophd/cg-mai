#version 450

struct PointLight {
    vec4 position; // xyz — позиция в мире
    vec4 color;    // rgb — цвет * интенсивность
};

struct SpotLight {
    vec4 position;  // xyz — позиция в мире
    vec4 direction; // xyz — нормализованное направление луча
    vec4 color;     // rgb — цвет * интенсивность
    vec4 angles;    // x = cos(внутреннего угла), y = cos(внешнего угла)
};

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 0) out vec4 outColor;

layout(std140, binding = 0) uniform Scene {
    mat4 view;
    mat4 proj;
    vec4 cameraPosition;
    vec4 ambientColor;
    vec4 materialAlbedo;
    vec4 materialSpecular;
    vec4 directionalDirection;
    vec4 directionalColor;
    float shininess;
} scene;

// Storage-буферы (SSBO): массивы без фиксированного максимального размера,
layout(std430, binding = 1) readonly buffer PointLightBuffer {
    PointLight lights[];
} pointLights;

layout(std430, binding = 2) readonly buffer SpotLightBuffer {
    SpotLight lights[];
} spotLights;

layout(push_constant) uniform Push {
    mat4 model;
    ivec4 counts; // x — число точечных источников, y — число прожекторов
} push;

// Уравнение Блинна-Фонга для одного источника.
// normal     — нормаль фрагмента,
// toLight    — направление из фрагмента на источник,
// toCamera   — направление из фрагмента на камеру,
// lightColor — цвет источника * интенсивность.
vec3 blinnPhong(vec3 normal, vec3 toLight, vec3 toCamera, vec3 lightColor) {
    const vec3 halfVector = normalize(toLight + toCamera);
    const float diffuse = max(dot(normal, toLight), 0.0);
    float specular = 0.0;
    if (diffuse > 0.0) {
        specular = pow(max(dot(normal, halfVector), 0.0), scene.shininess);
    }
    return scene.materialAlbedo.rgb * lightColor * diffuse
         + scene.materialSpecular.rgb * lightColor * specular;
}

void main() {
    const vec3 normal = normalize(inWorldNormal);
    const vec3 toCamera = normalize(scene.cameraPosition.xyz - inWorldPosition);

    // Рассеянный свет (ambient).
    vec3 color = scene.materialAlbedo.rgb * scene.ambientColor.rgb;

    const vec3 dirToLight = normalize(-scene.directionalDirection.xyz);
    color += blinnPhong(normal, dirToLight, toCamera, scene.directionalColor.rgb);

    for (int i = 0; i < push.counts.x; ++i) {
        const vec3 toLight = pointLights.lights[i].position.xyz - inWorldPosition;
        const float distanceSquared = max(dot(toLight, toLight), 1e-4);
        const float attenuation = 1.0 / distanceSquared;
        color += blinnPhong(normal, normalize(toLight), toCamera,
                            pointLights.lights[i].color.rgb) * attenuation;
    }

    for (int i = 0; i < push.counts.y; ++i) {
        const SpotLight light = spotLights.lights[i];

        const vec3 toLight = light.position.xyz - inWorldPosition;
        const float distanceSquared = max(dot(toLight, toLight), 1e-4);
        const vec3 dirToLightSpot = normalize(toLight);

        // cos угла между направлением луча прожектора и направлением на фрагмент
        const float theta = dot(-dirToLightSpot, light.direction.xyz);
        const float cosInner = light.angles.x;
        const float cosOuter = light.angles.y;

        // Мягкие края 1 внутри внутреннего угла, 0 за внешним углом,
        // линейная интерполяция между ними + сглаживание (smoothstep)
        float spotFactor = clamp((theta - cosOuter) / max(cosInner - cosOuter, 1e-4), 0.0, 1.0);
        spotFactor = spotFactor * spotFactor * (3.0 - 2.0 * spotFactor);

        const float attenuation = 1.0 / distanceSquared;
        color += blinnPhong(normal, dirToLightSpot, toCamera, light.color.rgb)
               * attenuation * spotFactor;
    }

    outColor = vec4(color, 1.0);
}
