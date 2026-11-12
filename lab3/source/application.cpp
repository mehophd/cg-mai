#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "application.hpp"

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace application {
namespace {

constexpr uint32_t kMaxPointLights = 16;
constexpr uint32_t kMaxSpotLights = 16;
constexpr uint32_t kMaxObjects = 16;

struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
};
static_assert(sizeof(Vertex) == 32, "vertex layout mismatch");


struct ObjectUniforms {
        glm::mat4 model;    // 0   матрица модели
        glm::vec4 baseColor; // 64  rgb - оттенок альбедо (умножается на текстуру)
        glm::vec4 specular;  // 80  rgb - цвет блика, w - коэффициент блика (shininess)
};
static_assert(sizeof(ObjectUniforms) == 96, "std140 layout mismatch");

struct SceneUniforms {
        glm::mat4 view;                 // 0
        glm::mat4 proj;                 // 64
        glm::vec4 cameraPosition;       // 128  xyz - позиция камеры
        glm::vec4 ambientColor;         // 144  rgb - цвет ambient * интенсивность
        glm::vec4 directionalDirection; // 160  xyz - направление лучей направленного источника
        glm::vec4 directionalColor;     // 176  rgb - цвет * интенсивность
        glm::ivec4 counts;              // 192  x - точечные, y - прожекторы
};
static_assert(sizeof(SceneUniforms) == 208, "std140 layout mismatch");

struct GpuPointLight {
        glm::vec4 position; // xyz - позиция
        glm::vec4 color;    // rgb - цвет * интенсивность
};
static_assert(sizeof(GpuPointLight) == 32, "std430 layout mismatch");

struct GpuSpotLight {
        glm::vec4 position;  // xyz - позиция
        glm::vec4 direction; // xyz - нормализованное направление луча
        glm::vec4 color;     // rgb - цвет * интенсивность
        glm::vec4 angles;    // x = cos(внутренний угол), y = cos(внешний угол)
};
static_assert(sizeof(GpuSpotLight) == 64, "std430 layout mismatch");

struct Push {
        glm::ivec4 flags; // x - режим сэмплирования: 0 - обычное,
                          //               1 - многократное с модуляцией, 2 - emissive
        glm::vec4 params; // x - масштаб детальной копии, y - фаза анимации,
                          // z - сила emissive
};
static_assert(sizeof(Push) == 32, "push constant size mismatch");


VkPipelineLayout layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
VkBuffer object_buffer = VK_NULL_HANDLE;
VmaAllocation object_allocation = VK_NULL_HANDLE;
uint8_t* object_mapped = nullptr; // базовый адрес отображения
VkDeviceSize object_stride = 0;    // выровненный размер одного слота

VkBuffer scene_buffer = VK_NULL_HANDLE;
VmaAllocation scene_allocation = VK_NULL_HANDLE;
SceneUniforms* scene_mapped = nullptr;

VkBuffer point_light_buffer = VK_NULL_HANDLE;
VmaAllocation point_light_allocation = VK_NULL_HANDLE;
GpuPointLight* point_light_mapped = nullptr;

VkBuffer spot_light_buffer = VK_NULL_HANDLE;
VmaAllocation spot_light_allocation = VK_NULL_HANDLE;
GpuSpotLight* spot_light_mapped = nullptr;

VkCommandPool upload_command_pool = VK_NULL_HANDLE;
VkCommandBuffer upload_command_buffer = VK_NULL_HANDLE;
VkFence upload_fence = VK_NULL_HANDLE;

VkSampler texture_sampler = VK_NULL_HANDLE;


float camera_position[3] = {0.0f, 1.1f, 5.6f};
float camera_yaw = 0.0f;     // рыскание, рад
float camera_pitch = -0.13f; // тангаж, рад
float camera_fov = 45.0f;    // угол обзора, градусы (перспективная проекция)
float ortho_extent = 3.0f;   // полуразмер окна ортографической проекции
int projection = 0;          // 0 - перспективная, 1 - ортографическая

float ambient_color[3] = {1.0f, 1.0f, 1.0f};
float ambient_intensity = 0.12f;

float directional_yaw = 0.7f;    // куда светит: рыскание, рад
float directional_pitch = -0.9f; // куда светит: тангаж, рад
float directional_color[3] = {1.0f, 0.96f, 0.90f};
float directional_intensity = 1.0f;

struct PointLightState {
        float position[3];
        float color[3];
        float intensity;
};

std::vector<PointLightState> point_lights = {
        {{2.4f, 1.6f, 2.4f}, {0.25f, 0.55f, 1.0f}, 6.0f},
        {{-2.8f, 1.0f, -1.2f}, {1.0f, 0.25f, 0.15f}, 6.0f},
};

struct SpotLightState {
        float position[3];
        float yaw;         // направление луча: рыскание, рад
        float pitch;       // направление луча: тангаж, рад
        float color[3];
        float intensity;
        float inner_angle; // внутренний угол конуса, градусы
        float outer_angle; // внешний угол конуса, градусы
};

std::vector<SpotLightState> spot_lights = {
        {{0.0f, 3.2f, 0.0f}, 0.0f, -1.45f, {0.35f, 1.0f, 0.45f}, 14.0f, 12.0f, 22.0f},
};

enum class MeshKind { Cone = 0, Sphere = 1, Plane = 2 };

struct ObjectState {
        const char* name;
        MeshKind kind;
        float position[3];
        float rotation[3];
        float scale;
        bool auto_rotate;
        bool visible;
        float base_color[3];
        float specular_color[3];
        float shininess;
};

std::vector<ObjectState> objects = {
        {"Конус (объект варианта 9)", MeshKind::Cone,
         {0.0f, 0.0f, 0.0f}, {0.45f, 0.6f, 0.0f}, 1.0f, true, true,
         {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, 48.0f},

        {"Сфера (нетривиальное сэмплирование)", MeshKind::Sphere,
         {-2.8f, 0.4f, 0.6f}, {0.0f, 0.0f, 0.0f}, 0.9f, false, true,
         {1.0f, 1.0f, 1.0f}, {0.8f, 0.8f, 0.8f}, 32.0f},

        {"Плоскость (пол)", MeshKind::Plane,
         {0.0f, -1.7f, 0.0f}, {0.0f, 0.0f, 0.0f}, 1.0f, false, true,
         {1.0f, 1.0f, 1.0f}, {0.25f, 0.25f, 0.25f}, 8.0f},
};

int selected_object = 0;

int sampling_mode = 1;          // 0 - обычное, 1 - многократное с модуляцией, 2 - emissive
float sampling_detail_scale = 4.0f; // масштаб детальной копии (режим 1)
float sampling_emissive = 1.0f;     // сила emissive (режим 2)
float sampling_anim_speed = 0.2f;   // скорость анимации сдвига
float sampling_anim_phase = 0.0f;   // фаза (обновляется каждый кадр)

struct TextureAsset {
        std::string name;        // отображаемое имя (имя файла)
        std::string path;        // путь, по которому файл был найден
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        uint32_t mip_levels = 1;
};

std::vector<TextureAsset> textures;
int active_texture = 0;
bool texture_switch_pending = false; // применяется в render() после ожидания fence

glm::vec3 directionFromAngles(float yaw, float pitch) {
        const float cp = std::cos(pitch);
        return glm::normalize(glm::vec3(cp * std::sin(yaw), std::sin(pitch), -cp * std::cos(yaw)));
}

VkShaderModule loadShader(const char* path) {
        FILE* f = std::fopen(path, "rb");
        if (!f) return VK_NULL_HANDLE;
        std::fseek(f, 0, SEEK_END);
        long size = std::ftell(f);
        std::rewind(f);
        std::vector<uint32_t> code(size > 0 ? size / 4 : 0);
        bool ok = size > 0 && std::fread(code.data(), 1, size, f) == size_t(size);
        std::fclose(f);
        if (!ok) return VK_NULL_HANDLE;
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = size;
        ci.pCode = code.data();
        VkShaderModule m = VK_NULL_HANDLE;
        if (vkCreateShaderModule(graphics::internal::context.device, &ci, nullptr, &m) != VK_SUCCESS)
                return VK_NULL_HANDLE;
        return m;
}

bool findAsset(const char* file, std::string& out_path) {
        const char* prefixes[] = {"", "assets/", "../assets/", "../../assets/"};
        for (const char* prefix : prefixes) {
                std::string candidate = std::string(prefix) + file;
                FILE* f = std::fopen(candidate.c_str(), "rb");
                if (f) {
                        std::fclose(f);
                        out_path = candidate;
                        return true;
                }
        }
        return false;
}


bool createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage, void** mapped,
                      VkBuffer* buffer, VmaAllocation* allocation) {
        auto& c = graphics::internal::context;

        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_AUTO;
        alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo vma_info{};
        if (vmaCreateBuffer(c.allocator, &info, &alloc, buffer, allocation, &vma_info) != VK_SUCCESS)
                return false;

        *mapped = vma_info.pMappedData;
        return true;
}


bool createUploadContext() {
        auto& c = graphics::internal::context;

        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = c.graphics_queue_index;
        if (vkCreateCommandPool(c.device, &pool_info, nullptr, &upload_command_pool) != VK_SUCCESS)
                return false;

        VkCommandBufferAllocateInfo alloc_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc_info.commandPool = upload_command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(c.device, &alloc_info, &upload_command_buffer) != VK_SUCCESS)
                return false;

        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(c.device, &fence_info, nullptr, &upload_fence) != VK_SUCCESS)
                return false;

        return true;
}


bool immediateSubmit(const std::function<void(VkCommandBuffer)>& record) {
        auto& c = graphics::internal::context;

        vkWaitForFences(c.device, 1, &upload_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(c.device, 1, &upload_fence);
        vkResetCommandBuffer(upload_command_buffer, 0);

        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(upload_command_buffer, &begin) != VK_SUCCESS)
                return false;

        record(upload_command_buffer);

        if (vkEndCommandBuffer(upload_command_buffer) != VK_SUCCESS)
                return false;

        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &upload_command_buffer;
        if (vkQueueSubmit(c.graphics_queue, 1, &submit, upload_fence) != VK_SUCCESS)
                return false;

        vkWaitForFences(c.device, 1, &upload_fence, VK_TRUE, UINT64_MAX);
        return true;
}


bool createDeviceLocalBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                             VkBuffer* buffer, VmaAllocation* allocation) {
        auto& c = graphics::internal::context;

        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VmaAllocation staging_allocation = VK_NULL_HANDLE;
        void* staging_mapped = nullptr;
        if (!createHostBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              &staging_mapped, &staging_buffer, &staging_allocation)) {
                return false;
        }
        std::memcpy(staging_mapped, data, size_t(size));

        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE; 

        if (vmaCreateBuffer(c.allocator, &info, &alloc, buffer, allocation, nullptr) != VK_SUCCESS) {
                vmaDestroyBuffer(c.allocator, staging_buffer, staging_allocation);
                return false;
        }

        bool ok = immediateSubmit([&](VkCommandBuffer cmd) {
                VkBufferCopy copy{};
                copy.srcOffset = 0;
                copy.dstOffset = 0;
                copy.size = size;
                vkCmdCopyBuffer(cmd, staging_buffer, *buffer, 1, &copy);
        });

        vmaDestroyBuffer(c.allocator, staging_buffer, staging_allocation);
        return ok;
}

void imageMemoryBarrier(VkCommandBuffer cmd, VkImage image,
                        uint32_t base_level, uint32_t level_count,
                        VkAccessFlags src_access, VkAccessFlags dst_access,
                        VkImageLayout old_layout, VkImageLayout new_layout,
                        VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask = src_access;
        barrier.dstAccessMask = dst_access;
        barrier.oldLayout = old_layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, base_level, level_count, 0, 1};
        vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}


bool createTextureFromPixels(const uint8_t* pixels, int width, int height, TextureAsset& tex) {
        auto& c = graphics::internal::context;

        tex.mip_levels =
                uint32_t(std::floor(std::log2(float(std::max(width, height))))) + 1u;

        VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        image_info.extent = {uint32_t(width), uint32_t(height), 1};
        image_info.mipLevels = tex.mip_levels;
        image_info.arrayLayers = 1;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                           VK_IMAGE_USAGE_SAMPLED_BIT;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE; 
        if (vmaCreateImage(c.allocator, &image_info, &alloc,
                           &tex.image, &tex.allocation, nullptr) != VK_SUCCESS) {
                return false;
        }

        const VkDeviceSize pixel_data_size = VkDeviceSize(width) * height * 4;
        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VmaAllocation staging_allocation = VK_NULL_HANDLE;
        void* staging_mapped = nullptr;
        if (!createHostBuffer(pixel_data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              &staging_mapped, &staging_buffer, &staging_allocation)) {
                return false;
        }
        std::memcpy(staging_mapped, pixels, size_t(pixel_data_size));

        bool ok = immediateSubmit([&](VkCommandBuffer cmd) {
                imageMemoryBarrier(cmd, tex.image, 0, tex.mip_levels,
                                   0, VK_ACCESS_TRANSFER_WRITE_BIT,
                                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

                VkBufferImageCopy region{};
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageExtent = {uint32_t(width), uint32_t(height), 1};
                vkCmdCopyBufferToImage(cmd, staging_buffer, tex.image,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

                int32_t mip_width = width;
                int32_t mip_height = height;
                for (uint32_t level = 1; level < tex.mip_levels; ++level) {
                        imageMemoryBarrier(cmd, tex.image, level - 1, 1,
                                           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                           VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

                        VkImageBlit blit{};
                        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1};
                        blit.srcOffsets[1] = {mip_width, mip_height, 1};
                        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
                        blit.dstOffsets[1] = {std::max(1, mip_width / 2), std::max(1, mip_height / 2), 1};
                        vkCmdBlitImage(cmd, tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       1, &blit, VK_FILTER_LINEAR);

                        imageMemoryBarrier(cmd, tex.image, level - 1, 1,
                                           VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                           VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

                        mip_width = std::max(1, mip_width / 2);
                        mip_height = std::max(1, mip_height / 2);
                }

                imageMemoryBarrier(cmd, tex.image, tex.mip_levels - 1, 1,
                                   VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        });

        vmaDestroyBuffer(c.allocator, staging_buffer, staging_allocation);
        if (!ok) return false;

        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = tex.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, tex.mip_levels, 0, 1};
        if (vkCreateImageView(c.device, &view_info, nullptr, &tex.view) != VK_SUCCESS)
                return false;

        return true;
}

// Загрузка изображения из файловой системы через stb_image
bool loadTextureFromFile(const char* file, TextureAsset& tex) {
        std::string path;
        if (!findAsset(file, path)) {
                std::cerr << "Texture file not found: " << file << "\n";
                return false;
        }

        int width = 0, height = 0, channels = 0;
        uint8_t* pixels = stbi_load(path.c_str(), &width, &height, &channels, 4);
        if (!pixels) {
                std::cerr << "Failed to decode texture: " << path
                          << " (" << stbi_failure_reason() << ")\n";
                return false;
        }

        tex.name = file;
        tex.path = path;
        bool ok = createTextureFromPixels(pixels, width, height, tex);
        stbi_image_free(pixels);
        if (ok)
                std::cout << "Loaded texture: " << path << " (" << width << "x" << height
                          << ", " << tex.mip_levels << " mip levels)\n";
        return ok;
}


struct MeshData {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
};


MeshData makeCone() {
        MeshData mesh;

        const int segments = 48;
        const float radius = 1.0f;
        const float height = 2.0f;
        const float half_height = height / 2.0f;

        const glm::vec3 apex(0.0f, half_height, 0.0f);
        const glm::vec3 base_center(0.0f, -half_height, 0.0f);
        const glm::vec3 apex_normal(0.0f, 1.0f, 0.0f);
        const glm::vec3 base_normal(0.0f, -1.0f, 0.0f);

        std::vector<glm::vec3> rim;
        std::vector<glm::vec3> rim_normals;
        rim.reserve(segments);
        rim_normals.reserve(segments);
        for (int i = 0; i < segments; ++i) {
                const float angle = 2.0f * glm::pi<float>() * float(i) / float(segments);
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                rim.emplace_back(radius * c, -half_height, radius * s);
                rim_normals.emplace_back(glm::normalize(glm::vec3(height * c, radius, height * s)));
        }


        const uint32_t side_first = uint32_t(mesh.vertices.size());
        for (int i = 0; i <= segments; ++i) {
                const int k = i % segments;
                const float u = float(i) / float(segments);
                mesh.vertices.push_back({rim[k], rim_normals[k], {u, 1.0f}});
        }
        const uint32_t apex_first = uint32_t(mesh.vertices.size());
        for (int i = 0; i < segments; ++i) {
                const float u = (float(i) + 0.5f) / float(segments);
                mesh.vertices.push_back({apex, apex_normal, {u, 0.0f}});
        }
        mesh.vertices.push_back({base_center, base_normal, {0.5f, 0.5f}});
        const uint32_t base_center_index = uint32_t(mesh.vertices.size()) - 1;

        const uint32_t base_rim_first = uint32_t(mesh.vertices.size());
        for (int i = 0; i < segments; ++i) {
                const float u = 0.5f + 0.5f * rim[i].x / radius;
                const float v = 0.5f + 0.5f * rim[i].z / radius;
                mesh.vertices.push_back({rim[i], base_normal, {u, v}});
        }

        for (int i = 0; i < segments; ++i) {
                mesh.indices.push_back(apex_first + i);
                mesh.indices.push_back(side_first + i);
                mesh.indices.push_back(side_first + i + 1);
                mesh.indices.push_back(base_center_index);
                mesh.indices.push_back(base_rim_first + (i + 1) % segments);
                mesh.indices.push_back(base_rim_first + i);
        }

        return mesh;
}

MeshData makeSphere() {
        MeshData mesh;

        const int rings = 32;    // по широте
        const int segments = 48; // по долготе
        const float radius = 1.0f;

        for (int j = 0; j <= rings; ++j) {
                const float v = float(j) / float(rings);
                const float theta = glm::pi<float>() * v; 
                for (int i = 0; i <= segments; ++i) {
                        const float u = float(i) / float(segments);
                        const float phi = 2.0f * glm::pi<float>() * u;
                        const float st = std::sin(theta);
                        glm::vec3 pos(radius * st * std::cos(phi),
                                      radius * std::cos(theta),
                                      radius * st * std::sin(phi));
                        glm::vec3 normal = pos / radius; // нормаль сферы = нормированная позиция
                        mesh.vertices.push_back({pos, normal, {u, v}});
                }
        }

        const uint32_t row = uint32_t(segments) + 1;
        for (int j = 0; j < rings; ++j) {
                for (int i = 0; i < segments; ++i) {
                        const uint32_t a = uint32_t(j) * row + i;
                        const uint32_t b = a + 1;
                        const uint32_t c = a + row;
                        const uint32_t d = c + 1;
                        mesh.indices.push_back(a);
                        mesh.indices.push_back(c);
                        mesh.indices.push_back(b);
                        mesh.indices.push_back(b);
                        mesh.indices.push_back(c);
                        mesh.indices.push_back(d);
                }
        }

        return mesh;
}

MeshData makePlane(float tiling) {
        MeshData mesh;

        const float half = 7.0f;
        const glm::vec3 normal(0.0f, 1.0f, 0.0f);

        mesh.vertices.push_back({{-half, 0.0f, -half}, normal, {0.0f, 0.0f}});
        mesh.vertices.push_back({{half, 0.0f, -half}, normal, {tiling, 0.0f}});
        mesh.vertices.push_back({{half, 0.0f, half}, normal, {tiling, tiling}});
        mesh.vertices.push_back({{-half, 0.0f, half}, normal, {0.0f, tiling}});

        mesh.indices = {0, 1, 2, 0, 2, 3};

        return mesh;
}


struct GpuMesh {
        VkBuffer vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vertex_allocation = VK_NULL_HANDLE;
        VkBuffer index_buffer = VK_NULL_HANDLE;
        VmaAllocation index_allocation = VK_NULL_HANDLE;
        uint32_t index_count = 0;
};

GpuMesh meshes[3]; 

bool createGpuMesh(const MeshData& data, GpuMesh& mesh) {
        if (!createDeviceLocalBuffer(data.vertices.data(),
                                     VkDeviceSize(data.vertices.size() * sizeof(Vertex)),
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                     &mesh.vertex_buffer, &mesh.vertex_allocation)) {
                return false;
        }
        if (!createDeviceLocalBuffer(data.indices.data(),
                                     VkDeviceSize(data.indices.size() * sizeof(uint32_t)),
                                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                     &mesh.index_buffer, &mesh.index_allocation)) {
                return false;
        }
        mesh.index_count = uint32_t(data.indices.size());
        return true;
}

bool createMeshes() {
        if (!createGpuMesh(makeCone(), meshes[size_t(MeshKind::Cone)])) return false;
        if (!createGpuMesh(makeSphere(), meshes[size_t(MeshKind::Sphere)])) return false;
        if (!createGpuMesh(makePlane(6.0f), meshes[size_t(MeshKind::Plane)])) return false;
        return true;
}


bool createFrameBuffers() {
        auto& c = graphics::internal::context;

        void* mapped = nullptr;

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(c.physical_device, &props);
        const VkDeviceSize alignment = VkDeviceSize(props.limits.minUniformBufferOffsetAlignment);
        object_stride = (sizeof(ObjectUniforms) + alignment - 1) / alignment * alignment;

        if (!createHostBuffer(kMaxObjects * object_stride, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              &mapped, &object_buffer, &object_allocation)) {
                return false;
        }
        object_mapped = static_cast<uint8_t*>(mapped);

        if (!createHostBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              &mapped, &scene_buffer, &scene_allocation)) {
                return false;
        }
        scene_mapped = static_cast<SceneUniforms*>(mapped);

        if (!createHostBuffer(kMaxPointLights * sizeof(GpuPointLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              &mapped, &point_light_buffer, &point_light_allocation)) {
                return false;
        }
        point_light_mapped = static_cast<GpuPointLight*>(mapped);

        if (!createHostBuffer(kMaxSpotLights * sizeof(GpuSpotLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              &mapped, &spot_light_buffer, &spot_light_allocation)) {
                return false;
        }
        spot_light_mapped = static_cast<GpuSpotLight*>(mapped);

        return true;
}


bool createTextures() {
        auto& c = graphics::internal::context;

        const char* files[] = {"mouse.png", "checker.png", "uv_grid.png"};
        for (const char* file : files) {
                TextureAsset tex;
                if (loadTextureFromFile(file, tex))
                        textures.push_back(std::move(tex));
        }

        // Если ни один файл не найден - процедурная заглушка
        if (textures.empty()) {
                std::cerr << "No textures loaded, using a generated fallback\n";
                uint8_t pixels[4 * 4 * 4];
                for (int y = 0; y < 4; ++y) {
                        for (int x = 0; x < 4; ++x) {
                                uint8_t* px = pixels + (y * 4 + x) * 4;
                                const bool white = ((x / 2) + (y / 2)) % 2 == 0;
                                px[0] = px[1] = px[2] = white ? uint8_t(255) : uint8_t(60);
                                px[3] = 255;
                        }
                }
                TextureAsset tex;
                tex.name = "(заглушка)";
                if (!createTextureFromPixels(pixels, 4, 4, tex))
                        return false;
                textures.push_back(std::move(tex));
        }

        VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.mipLodBias = 0.0f;
        sampler_info.maxAnisotropy = 1.0f;
        sampler_info.minLod = 0.0f;
        sampler_info.maxLod = VK_LOD_CLAMP_NONE;
        if (vkCreateSampler(c.device, &sampler_info, nullptr, &texture_sampler) != VK_SUCCESS)
                return false;

        return true;
}

bool createDescriptors() {
        auto& c = graphics::internal::context;

        VkDescriptorSetLayoutBinding bindings[5]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[4].binding = 4;
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.bindingCount = 5;
        layout_info.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(c.device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS)
                return false;

        VkDescriptorPoolSize pool_sizes[4]{};
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pool_sizes[0].descriptorCount = 1;
        pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        pool_sizes[1].descriptorCount = 1;
        pool_sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[2].descriptorCount = 2;
        pool_sizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        pool_sizes[3].descriptorCount = 1;

        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 4;
        pool_info.pPoolSizes = pool_sizes;
        if (vkCreateDescriptorPool(c.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS)
                return false;

        VkDescriptorSetAllocateInfo alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc_info.descriptorPool = descriptor_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &descriptor_set_layout;
        if (vkAllocateDescriptorSets(c.device, &alloc_info, &descriptor_set) != VK_SUCCESS)
                return false;

        VkDescriptorBufferInfo buffer_infos[4]{};
        buffer_infos[0] = {object_buffer, 0, sizeof(ObjectUniforms)};
        buffer_infos[1] = {scene_buffer, 0, VK_WHOLE_SIZE};
        buffer_infos[2] = {point_light_buffer, 0, VK_WHOLE_SIZE};
        buffer_infos[3] = {spot_light_buffer, 0, VK_WHOLE_SIZE};

        VkDescriptorImageInfo image_info{};
        image_info.sampler = texture_sampler;
        image_info.imageView = textures[active_texture].view;
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet writes[5]{};
        for (uint32_t i = 0; i < 4; ++i) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = descriptor_set;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType =
                        (i == 0) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                                 : (i == 1) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &buffer_infos[i];
        }
        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = descriptor_set;
        writes[4].dstBinding = 4;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[4].pImageInfo = &image_info;

        vkUpdateDescriptorSets(c.device, 5, writes, 0, nullptr);
        return true;
}


bool createPipeline() {
        auto& c = graphics::internal::context;

        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        range.offset = 0;
        range.size = sizeof(Push);

        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &descriptor_set_layout;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(c.device, &layout_info, nullptr, &layout) != VK_SUCCESS)
                return false;

        VkShaderModule vs = loadShader("shaders/cone.vert.spv");
        VkShaderModule fs = loadShader("shaders/cone.frag.spv");
        if (!vs || !fs) {
                std::cerr << "Shader files missing: run the CMake build from project root.\n";
                return false;
        }

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", nullptr};

        VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attrs[3] = {
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
                {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
        };

        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = 3;
        vi.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState blend{};
        blend.colorWriteMask = 0xf;

        VkPipelineColorBlendStateCreateInfo bs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        bs.attachmentCount = 1;
        bs.pAttachments = &blend;

        VkDynamicState dyns[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = dyns;

        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pi.stageCount = 2;
        pi.pStages = stages;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vp;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &bs;
        pi.pDynamicState = &dyn;
        pi.layout = layout;
        pi.renderPass = c.render_pass;
        pi.subpass = 0;

        VkResult result = vkCreateGraphicsPipelines(c.device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline);
        vkDestroyShaderModule(c.device, vs, nullptr);
        vkDestroyShaderModule(c.device, fs, nullptr);

        return result == VK_SUCCESS;
}

} // namespace

bool initialize() {
        if (!createUploadContext()) {
                std::cerr << "Failed to create upload context\n";
                return false;
        }
        if (!createTextures()) {
                std::cerr << "Failed to create textures\n";
                return false;
        }
        if (!createMeshes()) {
                std::cerr << "Failed to upload vertex/index buffers to VRAM\n";
                return false;
        }
        if (!createFrameBuffers()) {
                std::cerr << "Failed to create application buffers\n";
                return false;
        }
        if (!createDescriptors()) {
                std::cerr << "Failed to create descriptor set\n";
                return false;
        }
        if (!createPipeline()) {
                std::cerr << "Failed to create graphics pipeline\n";
                return false;
        }
        return true;
}

void shutdown() {
        auto& c = graphics::internal::context;
        vkQueueWaitIdle(c.graphics_queue);
        if (pipeline) vkDestroyPipeline(c.device, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(c.device, layout, nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(c.device, descriptor_pool, nullptr);
        if (descriptor_set_layout) vkDestroyDescriptorSetLayout(c.device, descriptor_set_layout, nullptr);
        for (GpuMesh& mesh : meshes) {
                if (mesh.vertex_buffer)
                        vmaDestroyBuffer(c.allocator, mesh.vertex_buffer, mesh.vertex_allocation);
                if (mesh.index_buffer)
                        vmaDestroyBuffer(c.allocator, mesh.index_buffer, mesh.index_allocation);
        }
        for (TextureAsset& tex : textures) {
                if (tex.view) vkDestroyImageView(c.device, tex.view, nullptr);
                if (tex.image) vmaDestroyImage(c.allocator, tex.image, tex.allocation);
        }
        if (texture_sampler) vkDestroySampler(c.device, texture_sampler, nullptr);
        if (object_buffer) vmaDestroyBuffer(c.allocator, object_buffer, object_allocation);
        if (scene_buffer) vmaDestroyBuffer(c.allocator, scene_buffer, scene_allocation);
        if (point_light_buffer) vmaDestroyBuffer(c.allocator, point_light_buffer, point_light_allocation);
        if (spot_light_buffer) vmaDestroyBuffer(c.allocator, spot_light_buffer, spot_light_allocation);
        if (upload_fence) vkDestroyFence(c.device, upload_fence, nullptr);
        if (upload_command_pool) vkDestroyCommandPool(c.device, upload_command_pool, nullptr);
}

void update(double time) {
        ImGui::Begin("Лабораторная работа 3 - Текстурирование (Вариант 9)");
        ImGui::Text("Текстуры как альбедо");

        if (ImGui::CollapsingHeader("Текстура альбедо", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Загрузка из файлов: stb_image (PNG, JPEG, BMP, TGA...)");
                std::vector<const char*> texture_names;
                texture_names.reserve(textures.size());
                for (const TextureAsset& tex : textures)
                        texture_names.push_back(tex.name.c_str());
                if (!texture_names.empty() &&
                    ImGui::Combo("Текстура", &active_texture, texture_names.data(),
                                 int(texture_names.size()))) {
                        texture_switch_pending = true;
                }
                if (active_texture >= 0 && active_texture < int(textures.size()))
                        ImGui::Text("Файл: %s", textures[active_texture].path.c_str());
                ImGui::TextWrapped("Все объекты используют один набор дескрипторов: "
                                   "динамический uniform-буфер + один кортеж изображение-сэмплер.");
        }

        if (ImGui::CollapsingHeader("Камера", ImGuiTreeNodeFlags_DefaultOpen)) {
                const char* modes[] = {"Перспективная", "Ортографическая"};
                ImGui::Combo("Проекция", &projection, modes, 2);
                if (projection == 0) {
                        ImGui::SliderFloat("Угол обзора (FOV)", &camera_fov, 20.0f, 120.0f, "%.0f°");
                } else {
                        ImGui::SliderFloat("Полуширина вида (орто)", &ortho_extent, 1.0f, 8.0f);
                }
                ImGui::SliderFloat3("Позиция камеры", camera_position, -10.0f, 10.0f);
                ImGui::SliderFloat("Рыскание (yaw)", &camera_yaw, -3.14f, 3.14f, "%.2f рад");
                ImGui::SliderFloat("Тангаж (pitch)", &camera_pitch, -1.55f, 1.55f, "%.2f рад");
                if (ImGui::Button("Сброс камеры")) {
                        camera_position[0] = 0.0f;
                        camera_position[1] = 1.1f;
                        camera_position[2] = 5.6f;
                        camera_yaw = 0.0f;
                        camera_pitch = -0.13f;
                        camera_fov = 45.0f;
                        ortho_extent = 3.0f;
                }
        }

        if (ImGui::CollapsingHeader("Объекты сцены", ImGuiTreeNodeFlags_DefaultOpen)) {
                std::vector<const char*> object_names;
                object_names.reserve(objects.size());
                for (const ObjectState& obj : objects)
                        object_names.push_back(obj.name);
                ImGui::Combo("Объект", &selected_object, object_names.data(), int(object_names.size()));
                ObjectState& obj = objects[selected_object];

                ImGui::Checkbox("Показывать", &obj.visible);
                ImGui::SliderFloat3("Позиция", obj.position, -8.0f, 8.0f);
                ImGui::SliderFloat3("Поворот", obj.rotation, -3.14f, 3.14f, "%.2f рад");
                ImGui::SliderFloat("Масштаб", &obj.scale, 0.2f, 2.0f);
                ImGui::Checkbox("Вращать автоматически", &obj.auto_rotate);
                ImGui::Separator();
                ImGui::Text("Материал (передаётся динамическим uniform-буфером)");
                ImGui::ColorEdit3("Оттенок альбедо", obj.base_color);
                ImGui::ColorEdit3("Цвет блика (specular)", obj.specular_color);
                ImGui::SliderFloat("Коэффициент блика (shininess)", &obj.shininess, 2.0f, 256.0f,
                                   "%.0f", ImGuiSliderFlags_Logarithmic);
        }

        if (ImGui::CollapsingHeader("Нетривиальное сэмплирование (сфера)",
                                    ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::TextWrapped("Флаги и параметры передаются push-константами; "
                                   "в структуре свойств моделей этих данных нет.");
                const char* modes[] = {"Обычное", "Многократное с модуляцией", "Emissive (свечение)"};
                ImGui::Combo("Режим сэмплирования", &sampling_mode, modes, 3);
                if (sampling_mode == 1)
                        ImGui::SliderFloat("Масштаб детальной копии", &sampling_detail_scale, 1.0f, 16.0f);
                if (sampling_mode == 2)
                        ImGui::SliderFloat("Сила emissive", &sampling_emissive, 0.0f, 3.0f);
                ImGui::SliderFloat("Скорость анимации", &sampling_anim_speed, 0.0f, 1.0f);
        }

        if (ImGui::CollapsingHeader("Рассеянный свет (ambient)", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::ColorEdit3("Цвет##ambient", ambient_color);
                ImGui::SliderFloat("Интенсивность##ambient", &ambient_intensity, 0.0f, 1.0f);
        }

        if (ImGui::CollapsingHeader("Направленный источник", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Азимут и наклон задают, КУДА светит источник");
                ImGui::SliderFloat("Азимут##dir", &directional_yaw, -3.14f, 3.14f, "%.2f рад");
                ImGui::SliderFloat("Наклон##dir", &directional_pitch, -1.55f, 1.55f, "%.2f рад");
                ImGui::ColorEdit3("Цвет##dir", directional_color);
                ImGui::SliderFloat("Интенсивность##dir", &directional_intensity, 0.0f, 5.0f);
        }

        if (ImGui::CollapsingHeader("Точечные источники", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Количество: %u (ёмкость storage-буфера: %u)",
                            uint32_t(point_lights.size()), kMaxPointLights);
                for (size_t i = 0; i < point_lights.size(); ++i) {
                        auto& light = point_lights[i];
                        ImGui::PushID(int(100 + i));
                        ImGui::Separator();
                        ImGui::Text("Точечный источник #%u", uint32_t(i + 1));
                        ImGui::SliderFloat3("Позиция", light.position, -10.0f, 10.0f);
                        ImGui::ColorEdit3("Цвет", light.color);
                        ImGui::SliderFloat("Интенсивность", &light.intensity, 0.0f, 20.0f);
                        if (ImGui::Button("Удалить")) {
                                point_lights.erase(point_lights.begin() + i);
                                ImGui::PopID();
                                break;
                        }
                        ImGui::PopID();
                }
                if (ImGui::Button("Добавить точечный источник")) {
                        if (point_lights.size() < kMaxPointLights)
                                point_lights.push_back({{0.0f, 1.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, 5.0f});
                }
        }

        if (ImGui::CollapsingHeader("Прожекторы", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Количество: %u (ёмкость storage-буфера: %u)",
                            uint32_t(spot_lights.size()), kMaxSpotLights);
                for (size_t i = 0; i < spot_lights.size(); ++i) {
                        auto& spot = spot_lights[i];
                        ImGui::PushID(int(200 + i));
                        ImGui::Separator();
                        ImGui::Text("Прожектор #%u", uint32_t(i + 1));
                        ImGui::SliderFloat3("Позиция", spot.position, -10.0f, 10.0f);
                        ImGui::SliderFloat("Азимут", &spot.yaw, -3.14f, 3.14f, "%.2f рад");
                        ImGui::SliderFloat("Наклон", &spot.pitch, -1.55f, 1.55f, "%.2f рад");
                        ImGui::ColorEdit3("Цвет", spot.color);
                        ImGui::SliderFloat("Интенсивность", &spot.intensity, 0.0f, 30.0f);
                        if (ImGui::SliderFloat("Внутренний угол", &spot.inner_angle, 1.0f, 89.0f, "%.0f°"))
                                spot.outer_angle = std::max(spot.outer_angle, spot.inner_angle + 1.0f);
                        if (ImGui::SliderFloat("Внешний угол", &spot.outer_angle, 2.0f, 90.0f, "%.0f°"))
                                spot.inner_angle = std::min(spot.inner_angle, spot.outer_angle - 1.0f);
                        if (ImGui::Button("Удалить")) {
                                spot_lights.erase(spot_lights.begin() + i);
                                ImGui::PopID();
                                break;
                        }
                        ImGui::PopID();
                }
                if (ImGui::Button("Добавить прожектор")) {
                        if (spot_lights.size() < kMaxSpotLights)
                                spot_lights.push_back({{0.0f, 3.0f, 0.0f}, 0.0f, -1.5f, {1.0f, 1.0f, 1.0f}, 12.0f, 12.0f, 22.0f});
                }
        }

        ImGui::End();

        for (ObjectState& obj : objects) {
                if (obj.auto_rotate) obj.rotation[1] = float(time) * 0.45f;
        }
        sampling_anim_phase = float(time) * sampling_anim_speed;
}

void render(const graphics::internal::FrameData& fd) {
        if (!pipeline || !fd.command_buffer) return;
        auto& c = graphics::internal::context;
        auto cmd = fd.command_buffer;

        if (texture_switch_pending && !textures.empty()) {
                texture_switch_pending = false;
                if (active_texture < 0 || active_texture >= int(textures.size()))
                        active_texture = 0;

                VkDescriptorImageInfo image_info{};
                image_info.sampler = texture_sampler;
                image_info.imageView = textures[active_texture].view;
                image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = descriptor_set;
                write.dstBinding = 4;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write.pImageInfo = &image_info;
                vkUpdateDescriptorSets(c.device, 1, &write, 0, nullptr);
        }

        const float aspect = float(c.swapchain_extent.width) / std::max(1u, c.swapchain_extent.height);

        const glm::vec3 camera_pos(camera_position[0], camera_position[1], camera_position[2]);
        const glm::vec3 camera_forward = directionFromAngles(camera_yaw, camera_pitch);
        const glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_forward, glm::vec3(0.0f, 1.0f, 0.0f));

        glm::mat4 proj;
        if (projection == 0) {
                proj = glm::perspective(glm::radians(camera_fov), aspect, 0.1f, 100.0f);
        } else {
                proj = glm::ortho(-ortho_extent * aspect, ortho_extent * aspect,
                                  -ortho_extent, ortho_extent, 0.1f, 100.0f);
        }
        proj[1][1] *= -1.0f;

        scene_mapped->view = view;
        scene_mapped->proj = proj;
        scene_mapped->cameraPosition = glm::vec4(camera_pos, 1.0f);
        scene_mapped->ambientColor = glm::vec4(ambient_color[0] * ambient_intensity,
                                               ambient_color[1] * ambient_intensity,
                                               ambient_color[2] * ambient_intensity, 0.0f);
        const glm::vec3 directional_dir = directionFromAngles(directional_yaw, directional_pitch);
        scene_mapped->directionalDirection = glm::vec4(directional_dir, 0.0f);
        scene_mapped->directionalColor = glm::vec4(directional_color[0] * directional_intensity,
                                                   directional_color[1] * directional_intensity,
                                                   directional_color[2] * directional_intensity, 0.0f);

        const uint32_t point_count = uint32_t(std::min<size_t>(point_lights.size(), kMaxPointLights));
        for (uint32_t i = 0; i < point_count; ++i) {
                const auto& src = point_lights[i];
                auto& dst = point_light_mapped[i];
                dst.position = glm::vec4(src.position[0], src.position[1], src.position[2], 1.0f);
                dst.color = glm::vec4(src.color[0] * src.intensity,
                                      src.color[1] * src.intensity,
                                      src.color[2] * src.intensity, 0.0f);
        }

        const uint32_t spot_count = uint32_t(std::min<size_t>(spot_lights.size(), kMaxSpotLights));
        for (uint32_t i = 0; i < spot_count; ++i) {
                const auto& src = spot_lights[i];
                auto& dst = spot_light_mapped[i];
                dst.position = glm::vec4(src.position[0], src.position[1], src.position[2], 1.0f);
                dst.direction = glm::vec4(directionFromAngles(src.yaw, src.pitch), 0.0f);
                dst.color = glm::vec4(src.color[0] * src.intensity,
                                      src.color[1] * src.intensity,
                                      src.color[2] * src.intensity, 0.0f);
                dst.angles = glm::vec4(std::cos(glm::radians(src.inner_angle)),
                                       std::cos(glm::radians(src.outer_angle)), 0.0f, 0.0f);
        }
        scene_mapped->counts = glm::ivec4(int(point_count), int(spot_count), 0, 0);

        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &begin);

        VkClearValue clears[2]{};
        clears[0].color = {{.035f, .045f, .07f, 1.0f}};
        clears[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rp.renderPass = c.render_pass;
        rp.framebuffer = fd.framebuffer;
        rp.renderArea = {{0, 0}, c.swapchain_extent};
        rp.clearValueCount = 2;
        rp.pClearValues = clears;
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{0, 0, float(c.swapchain_extent.width), float(c.swapchain_extent.height), 0, 1};
        VkRect2D scissor{{0, 0}, c.swapchain_extent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);


        for (uint32_t i = 0; i < objects.size(); ++i) {
                const ObjectState& obj = objects[i];
                if (!obj.visible) continue;

                glm::mat4 model(1.0f);
                model = glm::translate(model, glm::vec3(obj.position[0], obj.position[1], obj.position[2]));
                model = glm::rotate(model, obj.rotation[0], glm::vec3(1.0f, 0.0f, 0.0f));
                model = glm::rotate(model, obj.rotation[1], glm::vec3(0.0f, 1.0f, 0.0f));
                model = glm::rotate(model, obj.rotation[2], glm::vec3(0.0f, 0.0f, 1.0f));
                model = glm::scale(model, glm::vec3(obj.scale));

                auto* slot = reinterpret_cast<ObjectUniforms*>(object_mapped + size_t(i) * object_stride);
                slot->model = model;
                slot->baseColor = glm::vec4(obj.base_color[0], obj.base_color[1],
                                            obj.base_color[2], 1.0f);
                slot->specular = glm::vec4(obj.specular_color[0], obj.specular_color[1],
                                           obj.specular_color[2], obj.shininess);

                const uint32_t dynamic_offset = uint32_t(size_t(i) * object_stride);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1,
                                        &descriptor_set, 1, &dynamic_offset);

                const GpuMesh& mesh = meshes[size_t(obj.kind)];
                VkDeviceSize vertex_offset = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertex_buffer, &vertex_offset);
                vkCmdBindIndexBuffer(cmd, mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);

                Push push{};
                if (obj.kind == MeshKind::Sphere) {
                        push.flags = glm::ivec4(sampling_mode, 0, 0, 0);
                        push.params = glm::vec4(sampling_detail_scale, sampling_anim_phase,
                                                sampling_emissive, 0.0f);
                }
                vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                   sizeof(push), &push);

                vkCmdDrawIndexed(cmd, mesh.index_count, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
}
} // namespace application

