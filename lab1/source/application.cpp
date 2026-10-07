#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "application.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

namespace application {
namespace {

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
};

struct Push {
    glm::mat4 mvp;
};

VkPipelineLayout layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_allocation = VK_NULL_HANDLE;
uint32_t vertex_count = 0;

std::vector<Vertex> makeCone() {
    std::vector<Vertex> out;
    const int segments = 48;
    const float radius = 1.0f;
    const float height = 2.0f;
    const float half_height = height / 2.0f;

    glm::vec3 apex(0.0f, half_height, 0.0f);
    glm::vec3 base_center(0.0f, -half_height, 0.0f);
    glm::vec3 color_apex(1.0f, 0.75f, 0.2f);
    glm::vec3 color_base(0.2f, 0.5f, 0.9f);

    std::vector<glm::vec3> base_vertices;
    for (int i = 0; i < segments; ++i) {
        float angle = 2.0f * glm::pi<float>() * float(i) / float(segments);
        base_vertices.emplace_back(radius * std::cos(angle), -half_height, radius * std::sin(angle));
    }

    for (int i = 0; i < segments; ++i) {
        int next_i = (i + 1) % segments;
        glm::vec3 c_side = glm::mix(color_base, color_apex, 0.5f);
        out.push_back({apex, color_apex});
        out.push_back({base_vertices[i], c_side});
        out.push_back({base_vertices[next_i], c_side});
    }

    for (int i = 0; i < segments; ++i) {
        int next_i = (i + 1) % segments;
        out.push_back({base_center, color_base});
        out.push_back({base_vertices[next_i], color_base});
        out.push_back({base_vertices[i], color_base});
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
    bool ok = size > 0 && std::fread(code.data(), 1, static_cast<size_t>(size), f) == static_cast<size_t>(size);
    std::fclose(f);
    if (!ok) return VK_NULL_HANDLE;

    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = static_cast<size_t>(size);
    ci.pCode = code.data();
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(graphics::internal::context.device, &ci, nullptr, &m) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return m;
}

} // namespace

bool initialize() {
    auto& c = graphics::internal::context;
    auto vertices = makeCone();
    vertex_count = uint32_t(vertices.size());

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = vertices.size() * sizeof(Vertex);
    bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};

    if (vmaCreateBuffer(c.allocator, &bi, &ai, &vertex_buffer, &vertex_allocation, &info) != VK_SUCCESS)
        return false;
    std::memcpy(info.pMappedData, vertices.data(), bi.size);

    VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(c.device, &li, nullptr, &layout) != VK_SUCCESS)
        return false;

    VkShaderModule vs = loadShader("shaders/cone.vert.spv");
    VkShaderModule fs = loadShader("shaders/cone.frag.spv");
    if (!vs || !fs) {
        std::cerr << "Shader files missing.\n";
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", nullptr};

    VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attrs[2] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)}
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

void shutdown() {
    auto& c = graphics::internal::context;
    vkQueueWaitIdle(c.graphics_queue);
    if (pipeline) vkDestroyPipeline(c.device, pipeline, nullptr);
    if (layout) vkDestroyPipelineLayout(c.device, layout, nullptr);
    if (vertex_buffer) vmaDestroyBuffer(c.allocator, vertex_buffer, vertex_allocation);
}

void update(double time) {
    (void)time;
}

void render(const graphics::internal::FrameData& fd) {
    if (!pipeline || !fd.command_buffer) return;
    auto& c = graphics::internal::context;
    auto cmd = fd.command_buffer;

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

    static auto start_time = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(now - start_time).count();
    
    float rot_y = float(time) * 0.5f;
    float rot_x = 0.45f;

    glm::mat4 model = glm::rotate(glm::mat4(1.0f), rot_x, {1, 0, 0});
    model = glm::rotate(model, rot_y, {0, 1, 0});

    float aspect = float(c.swapchain_extent.width) / std::max(1u, c.swapchain_extent.height);
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 20.0f);
    proj[1][1] *= -1.0f;

    glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 3.4f), glm::vec3(0), glm::vec3(0, 1, 0));
    Push push{proj * view * model};

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &offset);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
    vkCmdDraw(cmd, vertex_count, 1, 0, 0);

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application
