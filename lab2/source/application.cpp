#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "application.hpp"

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

namespace application {
namespace {

constexpr uint32_t kMaxPointLights = 16;
constexpr uint32_t kMaxSpotLights = 16;

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
};

struct SceneUniforms {
	glm::mat4 view;                 // 0
	glm::mat4 proj;                 // 64
	glm::vec4 cameraPosition;       // 128  xyz - позиция камеры
	glm::vec4 ambientColor;         // 144  rgb - цвет ambient * интенсивность
	glm::vec4 materialAlbedo;       // 160  rgb - альбедо материала
	glm::vec4 materialSpecular;     // 176  rgb - цвет блика материала
	glm::vec4 directionalDirection; // 192  xyz - направление лучей направленного источника
	glm::vec4 directionalColor;     // 208  rgb - цвет * интенсивность
	float shininess;                // 224  коэффициент блика
	float _padding[3];              // дополнение размера до 240 байт
};

// Элемент storage-буфера точечных источников (std430).
struct GpuPointLight {
	glm::vec4 position; // xyz - позиция
	glm::vec4 color;    // rgb - цвет * интенсивность
};
static_assert(sizeof(GpuPointLight) == 32, "std430 layout mismatch");

// Элемент storage-буфера прожекторов (std430).
struct GpuSpotLight {
	glm::vec4 position;  // xyz - позиция
	glm::vec4 direction; // xyz - нормализованное направление луча
	glm::vec4 color;     // rgb - цвет * интенсивность
	glm::vec4 angles;    // x = cos(внутренний угол), y = cos(внешний угол)
};
static_assert(sizeof(GpuSpotLight) == 64, "std430 layout mismatch");

// Push-константы (общие для vertex- и fragment-стадий).
struct Push {
	glm::mat4 model;   // матрица модели (vertex)
	glm::ivec4 counts; // x - число точечных источников, y - число прожекторов (fragment)
};
static_assert(sizeof(Push) == 80, "push constant size mismatch");


VkPipelineLayout layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

VkBuffer uniform_buffer = VK_NULL_HANDLE;
VmaAllocation uniform_allocation = VK_NULL_HANDLE;
SceneUniforms* uniform_mapped = nullptr;

VkBuffer point_light_buffer = VK_NULL_HANDLE;
VmaAllocation point_light_allocation = VK_NULL_HANDLE;
GpuPointLight* point_light_mapped = nullptr;

VkBuffer spot_light_buffer = VK_NULL_HANDLE;
VmaAllocation spot_light_allocation = VK_NULL_HANDLE;
GpuSpotLight* spot_light_mapped = nullptr;

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_allocation = VK_NULL_HANDLE;
uint32_t vertex_count = 0;

float camera_position[3] = {0.0f, 0.8f, 4.2f};
float camera_yaw = 0.0f;     // рыскание, рад
float camera_pitch = -0.15f; // тангаж, рад
float camera_fov = 45.0f;    // угол обзора, градусы (перспективная проекция)
float ortho_extent = 2.0f;   // полуразмер окна ортографической проекции
int projection = 0;          // 0 - перспективная, 1 - ортографическая

// (модель Блинна-Фонга)
float material_albedo[3] = {0.75f, 0.75f, 0.80f};
float material_specular[3] = {1.0f, 1.0f, 1.0f};
float material_shininess = 48.0f;

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
	{{2.0f, 1.2f, 2.0f}, {0.25f, 0.55f, 1.0f}, 6.0f},
	{{-2.5f, 0.6f, -1.0f}, {1.0f, 0.25f, 0.15f}, 6.0f},
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
	{{0.0f, 2.6f, 0.0f}, 0.0f, -1.45f, {0.35f, 1.0f, 0.45f}, 12.0f, 12.0f, 22.0f},
};

float rotation[3] = {0.45f, 0.6f, 0.0f};
float scale = 1.0f;
bool auto_rotate = true;

// Направление по углам (рыскание вокруг оси Y, тангаж)
glm::vec3 directionFromAngles(float yaw, float pitch) {
	const float cp = std::cos(pitch);
	return glm::normalize(glm::vec3(cp * std::sin(yaw), std::sin(pitch), -cp * std::cos(yaw)));
}

std::vector<Vertex> makeCone() {
	std::vector<Vertex> out;

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
		// Нормаль боковой поверхности: перпендикуляр к образующей конуса
		rim_normals.emplace_back(glm::normalize(glm::vec3(height * c, radius, height * s)));
	}

	for (int i = 0; i < segments; ++i) {
		const int j = (i + 1) % segments;
		// Боковая поверхность
		out.push_back({apex, apex_normal});
		out.push_back({rim[i], rim_normals[i]});
		out.push_back({rim[j], rim_normals[j]});
		// Основание
		out.push_back({base_center, base_normal});
		out.push_back({rim[j], base_normal});
		out.push_back({rim[i], base_normal});
	}

	return out;
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

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, void** mapped,
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

bool createBuffers() {
	auto vertices = makeCone();
	vertex_count = uint32_t(vertices.size());

	void* mapped = nullptr;

	// Вершинный буфер
	if (!createBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                  &mapped, &vertex_buffer, &vertex_allocation)) {
		return false;
	}
	std::memcpy(mapped, vertices.data(), vertices.size() * sizeof(Vertex));

	// Uniform-буфер сцены: матрицы, камера, материал, ambient, направленный источник
	if (!createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                  &mapped, &uniform_buffer, &uniform_allocation)) {
		return false;
	}
	uniform_mapped = static_cast<SceneUniforms*>(mapped);

	// Storage-буферы массивов источников света
	if (!createBuffer(kMaxPointLights * sizeof(GpuPointLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                  &mapped, &point_light_buffer, &point_light_allocation)) {
		return false;
	}
	point_light_mapped = static_cast<GpuPointLight*>(mapped);

	if (!createBuffer(kMaxSpotLights * sizeof(GpuSpotLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                  &mapped, &spot_light_buffer, &spot_light_allocation)) {
		return false;
	}
	spot_light_mapped = static_cast<GpuSpotLight*>(mapped);

	return true;
}

bool createDescriptors() {
	auto& c = graphics::internal::context;

	// binding 0 - uniform-буфер сцены (vertex + fragment)
	// binding 1 - storage-буфер точечных источников (fragment)
	// binding 2 - storage-буфер прожекторов (fragment)
	VkDescriptorSetLayoutBinding bindings[3]{};
	bindings[0].binding = 0;
	bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	bindings[0].descriptorCount = 1;
	bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	bindings[1].binding = 1;
	bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	bindings[1].descriptorCount = 1;
	bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	bindings[2].binding = 2;
	bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	bindings[2].descriptorCount = 1;
	bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	layout_info.bindingCount = 3;
	layout_info.pBindings = bindings;
	if (vkCreateDescriptorSetLayout(c.device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS)
		return false;

	VkDescriptorPoolSize pool_sizes[2]{};
	pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	pool_sizes[0].descriptorCount = 1;
	pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	pool_sizes[1].descriptorCount = 2;

	VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	pool_info.maxSets = 1;
	pool_info.poolSizeCount = 2;
	pool_info.pPoolSizes = pool_sizes;
	if (vkCreateDescriptorPool(c.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS)
		return false;

	VkDescriptorSetAllocateInfo alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
	alloc_info.descriptorPool = descriptor_pool;
	alloc_info.descriptorSetCount = 1;
	alloc_info.pSetLayouts = &descriptor_set_layout;
	if (vkAllocateDescriptorSets(c.device, &alloc_info, &descriptor_set) != VK_SUCCESS)
		return false;

	VkDescriptorBufferInfo buffer_infos[3]{};
	buffer_infos[0] = {uniform_buffer, 0, VK_WHOLE_SIZE};
	buffer_infos[1] = {point_light_buffer, 0, VK_WHOLE_SIZE};
	buffer_infos[2] = {spot_light_buffer, 0, VK_WHOLE_SIZE};

	VkWriteDescriptorSet writes[3]{};
	for (uint32_t i = 0; i < 3; ++i) {
		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = descriptor_set;
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
		                                    : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].pBufferInfo = &buffer_infos[i];
	}
	vkUpdateDescriptorSets(c.device, 3, writes, 0, nullptr);

	return true;
}

bool createPipeline() {
	auto& c = graphics::internal::context;

	VkPushConstantRange range{};
	range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
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
	VkVertexInputAttributeDescription attrs[2] = {
		{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
		{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)}
	};

	VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
	vi.vertexBindingDescriptionCount = 1;
	vi.pVertexBindingDescriptions = &binding;
	vi.vertexAttributeDescriptionCount = 2;
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
	if (!createBuffers()) {
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
	if (vertex_buffer) vmaDestroyBuffer(c.allocator, vertex_buffer, vertex_allocation);
	if (uniform_buffer) vmaDestroyBuffer(c.allocator, uniform_buffer, uniform_allocation);
	if (point_light_buffer) vmaDestroyBuffer(c.allocator, point_light_buffer, point_light_allocation);
	if (spot_light_buffer) vmaDestroyBuffer(c.allocator, spot_light_buffer, spot_light_allocation);
}

void update(double time) {
	ImGui::Begin("Лабораторная работа 2 - Конус");
	ImGui::Text("Камера + освещение по Блинну-Фонгу | Vulkan + GLFW + ImGui");

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
			camera_position[1] = 0.8f;
			camera_position[2] = 4.2f;
			camera_yaw = 0.0f;
			camera_pitch = -0.15f;
			camera_fov = 45.0f;
			ortho_extent = 2.0f;
		}
	}

	if (ImGui::CollapsingHeader("Объект (конус)", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderFloat3("Поворот", rotation, -3.14f, 3.14f, "%.2f рад");
		ImGui::SliderFloat("Масштаб", &scale, 0.35f, 1.6f);
		ImGui::Checkbox("Вращать автоматически", &auto_rotate);
		if (ImGui::Button("Сбросить объект")) {
			rotation[0] = 0.45f;
			rotation[1] = 0.6f;
			rotation[2] = 0.0f;
			scale = 1.0f;
		}
	}

	if (ImGui::CollapsingHeader("Материал", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Альбедо (цвет поверхности)", material_albedo);
		ImGui::ColorEdit3("Цвет блика (specular)", material_specular);
		ImGui::SliderFloat("Коэффициент блика (shininess)", &material_shininess, 2.0f, 256.0f,
		                   "%.0f", ImGuiSliderFlags_Logarithmic);
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

	// ---------- Прожекторы ----------
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

	if (auto_rotate) {
		rotation[1] = float(time) * 0.45f;
	}
}

void render(const graphics::internal::FrameData& fd) {
	if (!pipeline || !fd.command_buffer) return;
	auto& c = graphics::internal::context;
	auto cmd = fd.command_buffer;

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
	// Инверсия оси Y для Vulkan (clip space Y направлен вниз)
	proj[1][1] *= -1.0f;

	uniform_mapped->view = view;
	uniform_mapped->proj = proj;
	uniform_mapped->cameraPosition = glm::vec4(camera_pos, 1.0f);
	uniform_mapped->ambientColor = glm::vec4(ambient_color[0] * ambient_intensity,
	                                         ambient_color[1] * ambient_intensity,
	                                         ambient_color[2] * ambient_intensity, 0.0f);
	uniform_mapped->materialAlbedo = glm::vec4(material_albedo[0], material_albedo[1],
	                                           material_albedo[2], 1.0f);
	uniform_mapped->materialSpecular = glm::vec4(material_specular[0], material_specular[1],
	                                             material_specular[2], 1.0f);
	const glm::vec3 directional_dir = directionFromAngles(directional_yaw, directional_pitch);
	uniform_mapped->directionalDirection = glm::vec4(directional_dir, 0.0f);
	uniform_mapped->directionalColor = glm::vec4(directional_color[0] * directional_intensity,
	                                             directional_color[1] * directional_intensity,
	                                             directional_color[2] * directional_intensity, 0.0f);
	uniform_mapped->shininess = material_shininess;

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

	glm::mat4 model = glm::rotate(glm::mat4(1.0f), rotation[0], {1, 0, 0});
	model = glm::rotate(model, rotation[1], {0, 1, 0});
	model = glm::rotate(model, rotation[2], {0, 0, 1});
	model = glm::scale(model, glm::vec3(scale));

	Push push{};
	push.model = model;
	push.counts = glm::ivec4(int(point_count), int(spot_count), 0, 0);

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
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &descriptor_set, 0, nullptr);

	VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &offset);
	vkCmdPushConstants(cmd, layout,
	                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	                   0, sizeof(push), &push);
	vkCmdDraw(cmd, vertex_count, 1, 0, 0);

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}
} // namespace application
