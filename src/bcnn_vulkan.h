
/*
 * Copyright (c) 2019-present Jean-Noel Braun.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef BCNN_VULKAN_H
#define BCNN_VULKAN_H

#include <vulkan/vulkan.h>

typedef struct bcnn_vulkan_context {
    VkInstance instance;
    VkPhysicalDevice physical_device; /* GPU device */
    int32_t physical_device_index;    /* GPU id */
    VkDevice device;                  /* Logical device */
    VkPipeline pipeline;
    VkPipelineLayout pipeline_layout;
    VkShaderModule shader_module; /* Compute shader module */
    VkPipelineCache pipeline_cache;
    VkCommandPool command_pool; /* Pool of command buffers */
    VkCommandBuffer
        command_buffer; /* Record the commands to submitted to command queues */
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
    VkDescriptorSetLayout descriptor_set_layout;
    VkBuffer buffer;
    VkDeviceMemory buffer_memory;
    uint32_t buffer_size; /* Size of 'buffer' in bytes = workspace size */
    VkQueue queue;
    uint32_t queue_family_index;
    VkBool32 debug_utils_available;
    bcnn_log_context* log_ctx; /* Hold by net struct */
} bcnn_vk_context;

/* Layer pipeline */
typedef struct bcnn_vulkan_pipeline {
    const bcnn_vk_context* context;
    VkShaderModule shader_module; /* should be shared among each instance of a
                                     specific layer */
    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    uint32_t local_size_x;
    uint32_t local_size_y;
    uint32_t local_size_z;
} bcnn_vk_pipeline;

/**
 * Error codes.
 */
typedef enum {
    BVK_SUCCESS,
    BVK_VULKAN_INTERNAL_ERROR,
    BVK_OTHER_ERROR
} bvk_status;

/**
 * \brief Create a vulkan instance with a given application name.
 *
 * \param[in]   context     Pointer to vulkan context.
 * \param[in]   app_name    A name of the application.
 *
 * \return 0 if vkCreateInstance is successful.
 */
int32_t bcnn_vk_create_instance(bcnn_vk_context* context, const char* app_name);

/**
 * \brief Destroy a vulkan instance created by bcnn_vk_create_instance.
 */
void bcnn_vk_destroy_instance(bcnn_vk_context* context);

bcnn_vk_pipeline* bcnn_vk_pipeline_create(bcnn_vk_context* context);

void bcnn_vk_pipeline_destroy(bcnn_vk_pipeline* pipeline);

#endif  // BCNN_VULKAN_H