#version 450

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0, std140) uniform ObjectBlock {
        mat4 model;
        vec4 baseColor;
        vec4 specular;
} object;

layout(set = 0, binding = 1, std140) uniform SceneBlock {
        mat4 view;
        mat4 proj;
        vec4 cameraPosition;
        vec4 ambientColor;
        vec4 directionalDirection;
        vec4 directionalColor;
        ivec4 counts;
} scene;

struct PointLight { vec4 position;  vec4 color; };
struct SpotLight { vec4 position; vec4 direction; vec4 color; vec4 angles; };

layout(set = 0, binding = 2, std430) readonly buffer PointLights { PointLight points[]; };
layout(set = 0, binding = 3, std430) readonly buffer SpotLights { SpotLight spots[]; };

layout(set = 0, binding = 4) uniform sampler2D albedoTexture;

layout(push_constant) uniform PushBlock {
        ivec4 flags; // x - режим сэмплирования: 0 - обычное,
                     //               1 - многократное с цветовой модуляцией,
                     //               2 - emissive (свечение)
        vec4  params; // x - масштаб детальной копии, y - фаза анимации,
                      // z - сила emissive, w - резерв
} pc;

vec3 blinnPhong(vec3 albedo, vec3 specularColor, float shininess,
                vec3 normal, vec3 toLight, vec3 toCamera, vec3 lightColor) {
        float diffuse = max(dot(normal, toLight), 0.0);
        vec3 halfVector = normalize(toLight + toCamera); // вектор половины пути
        float specular = pow(max(dot(normal, halfVector), 0.0), shininess);
        return albedo * lightColor * diffuse + specularColor * lightColor * specular;
}

vec3 sampleAlbedo(vec2 uv, out vec3 emissive) {
        emissive = vec3(0.0);
        vec3 base = texture(albedoTexture, uv).rgb; // тривиальное сэмплирование

        if (pc.flags.x == 1) {
                float scale = pc.params.x;
                vec2 shift = vec2(pc.params.y, 0.618 * pc.params.y);
                vec3 detail = texture(albedoTexture, uv * scale + shift).rgb;
                vec3 mirror = texture(albedoTexture, vec2(1.0 - uv.x, uv.y)).rgb;
                base = base * (2.0 * detail);
                base = mix(base, mirror, 0.25);
        } else if (pc.flags.x == 2) {
                emissive = texture(albedoTexture, uv).rgb * pc.params.z;
        }
        return base;
}

void main() {
        vec3 normal = normalize(inWorldNormal);
        vec3 toCamera = normalize(scene.cameraPosition.xyz - inWorldPosition);

        vec3 emissive;
        vec3 albedo = sampleAlbedo(inUV, emissive) * object.baseColor.rgb;
        vec3 specularColor = object.specular.rgb;
        float shininess = max(object.specular.a, 1.0);

        // Рассеянный свет
        vec3 result = albedo * scene.ambientColor.rgb;

        {
                vec3 toLight = normalize(-scene.directionalDirection.xyz);
                result += blinnPhong(albedo, specularColor, shininess, normal,
                                     toLight, toCamera, scene.directionalColor.rgb);
        }

        for (int i = 0; i < scene.counts.x; ++i) {
                vec3 toLight = points[i].position.xyz - inWorldPosition;
                float dist = length(toLight);
                toLight /= max(dist, 1e-4);
                float attenuation = 1.0 / max(dist * dist, 1e-4);
                result += blinnPhong(albedo, specularColor, shininess, normal,
                                     toLight, toCamera, points[i].color.rgb) * attenuation;
        }

        for (int i = 0; i < scene.counts.y; ++i) {
                vec3 toLight = spots[i].position.xyz - inWorldPosition;
                float dist = length(toLight);
                toLight /= max(dist, 1e-4);

                float theta = dot(-toLight, normalize(spots[i].direction.xyz));
                float t = clamp((theta - spots[i].angles.y) /
                                max(spots[i].angles.x - spots[i].angles.y, 1e-4), 0.0, 1.0);
                float spot = t * t * (3.0 - 2.0 * t); // сглаживание края

                float attenuation = 1.0 / max(dist * dist, 1e-4);
                result += blinnPhong(albedo, specularColor, shininess, normal,
                                     toLight, toCamera, spots[i].color.rgb) *
                          attenuation * spot;
        }

        result += emissive;

        outColor = vec4(result, 1.0);
}
