#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

// Выходы во фрагментный шейдер
layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outWorldNormal;

// Uniform-блок сцены (std140, binding 0)
layout(std140, binding = 0) uniform Scene {
    mat4 view;
    mat4 proj;
    vec4 cameraPosition;       // xyz - позиция камеры в мире
    vec4 ambientColor;         // rgb - цвет рассеянного света * интенсивность
    vec4 materialAlbedo;       // rgb - альбедо материала
    vec4 materialSpecular;     // rgb - цвет блика материала
    vec4 directionalDirection; // xyz - направление лучей направленного источника
    vec4 directionalColor;     // rgb - цвет * интенсивность направленного источника
    float shininess;           // коэффициент блика
} scene;

// model  - матрица модели (используется здесь)
layout(push_constant) uniform Push {
    mat4 model;
    ivec4 counts;
} push;

void main() {
    const vec4 world = push.model * vec4(inPosition, 1.0);
    outWorldPosition = world.xyz;
    // Масштаб объекта равномерный, поэтому достаточно mat3(model)
    // нормализация выполняется во фрагментном шейдере
    outWorldNormal = mat3(push.model) * inNormal;
    gl_Position = scene.proj * scene.view * world;
}
