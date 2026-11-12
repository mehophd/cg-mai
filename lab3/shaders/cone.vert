#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outWorldNormal;
layout(location = 2) out vec2 outUV;

layout(set = 0, binding = 0, std140) uniform ObjectBlock {
        mat4 model;      // матрица модели
        vec4 baseColor;  // rgb - оттенок альбедо (умножается на текстуру)
        vec4 specular;   // rgb - цвет блика, a - коэффициент блика (shininess)
} object;

// Uniform-буфер сцены: общие данные кадра (как в лабораторной №2).
layout(set = 0, binding = 1, std140) uniform SceneBlock {
        mat4 view;
        mat4 proj;
        vec4 cameraPosition;
        vec4 ambientColor;
        vec4 directionalDirection;
        vec4 directionalColor;
        ivec4 counts; // x - число точечных источников, y - число прожекторов
} scene;

void main() {
        vec4 world = object.model * vec4(inPosition, 1.0);
        outWorldPosition = world.xyz;
        // Нормаль переводится в мировое пространство матрицей модели
        outWorldNormal = mat3(object.model) * inNormal;
        outUV = inUV;
        gl_Position = scene.proj * scene.view * world;
}
