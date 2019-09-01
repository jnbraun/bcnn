
/*
 * Copyright (c) 2016-present Jean-Noel Braun.
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

#ifndef BCNN_NET_H
#define BCNN_NET_H

#ifdef BCNN_USE_VULKAN
#include <vulkan/vulkan.h>
#endif

#include <bcnn/bcnn.h>
#include "bcnn_data.h"
#include "bcnn_learner.h"
#include "bcnn_node.h"
#include "bcnn_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef BCNN_USE_CUDA
typedef struct bcnn_cuda_context {
    int workspace_size;
    float *workspace_gpu;
} bcnn_cuda_context;
#endif

#ifdef BCNN_USE_VULKAN
typedef struct bcnn_vulkan_context {
    VkInstance instance;
    VkPhysicalDevice physical_device; /* Graphic card */
    VkDevice device;                  /* Logical device */
    VkPipeline pipeline;
    VkPipelineLayout pipeline_layout;
    VkShaderModule shader_module; /* Compute shader module */
    VkCommandPool command_pool;   /* Pool of command buffers */
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
} bcnn_vulkan_context;
#endif

/**
 * Net definition
 */
struct bcnn_net {
    int batch_size;
    int num_nodes;   /* Number of nodes hold in the network */
    int num_tensors; /* Number of tensors hold in the network */
    int num_inputs;  /* Number of input tensors */
    int *inputs;     /* Indexes of the input tensors in the below 'tensors'
                            array */
    bcnn_mode mode;
    bcnn_log_context log_ctx; /* Logging stuff */
    bcnn_node *nodes;         /* Array of 'num_nodes' nodes */
    bcnn_tensor *tensors;     /* Array of 'num_tensors' tensors */
    bcnn_learner *learner;    /* Learner/optimizer parameters */
    bcnn_loader *data_loader; /* Handles the loading and iteration over training
                                 / testing datasets */
    bcnn_data_augmenter *data_aug; /* Handles the online data augmentation */
    void *gemm_ctx;
#ifdef BCNN_USE_CUDA
    void *cuda_ctx;
#endif
#ifdef BCNN_USE_VULKAN
    void *vulkan_context; /* Vulkan internal context */
#endif
    int num_threads; /* Number of threads (CPU only) */
};

bcnn_status bcnn_net_create_gemm_context(bcnn_net *net);
#ifdef BCNN_USE_CUDA
bcnn_status bcnn_net_create_cuda_context(bcnn_net *net);
#endif
#ifdef BCNN_USE_VULKAN
bcnn_status bcnn_net_create_vulkan_context(bcnn_net *net);
bcnn_status bcnn_net_destroy_vulkan_context(bcnn_net *net);
#endif
bcnn_status bcnn_net_add_node(bcnn_net *net, bcnn_node node);
bcnn_status bcnn_net_add_tensor(bcnn_net *net, bcnn_tensor tensor);
void bcnn_net_set_param(bcnn_net *net, const char *name, const char *val);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_NET_H