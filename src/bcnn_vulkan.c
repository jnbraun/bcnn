
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

#include "bcnn_vulkan.h"

#include <stdio.h>

#include <bh/bh_macros.h>
#include "bcnn_utils.h"

#define BCNN_VK_ENABLE_DEBUG

static const char* vkresult2string(VkResult result) {
    switch (result) {
#define STR(r)   \
    case VK_##r: \
        return #r
        STR(NOT_READY);
        STR(TIMEOUT);
        STR(EVENT_SET);
        STR(EVENT_RESET);
        STR(INCOMPLETE);
        STR(ERROR_OUT_OF_HOST_MEMORY);
        STR(ERROR_OUT_OF_DEVICE_MEMORY);
        STR(ERROR_INITIALIZATION_FAILED);
        STR(ERROR_DEVICE_LOST);
        STR(ERROR_MEMORY_MAP_FAILED);
        STR(ERROR_LAYER_NOT_PRESENT);
        STR(ERROR_EXTENSION_NOT_PRESENT);
        STR(ERROR_FEATURE_NOT_PRESENT);
        STR(ERROR_INCOMPATIBLE_DRIVER);
        STR(ERROR_TOO_MANY_OBJECTS);
        STR(ERROR_FORMAT_NOT_SUPPORTED);
        STR(ERROR_SURFACE_LOST_KHR);
        STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
        STR(SUBOPTIMAL_KHR);
        STR(ERROR_OUT_OF_DATE_KHR);
        STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
        STR(ERROR_VALIDATION_FAILED_EXT);
        STR(ERROR_INVALID_SHADER_NV);
#undef STR
        default:
            return "UNKNOWN_ERROR";
    }
}

#define VK_CHECK(f)                                             \
    do {                                                        \
        VkResult r = (f);                                       \
        if ((r) != VK_SUCCESS) {                                \
            printf("[ERROR][Vulkan] %s", (vkresult2string(r))); \
        }                                                       \
    } while (0)

#ifdef BCNN_VK_ENABLE_DEBUG
static VkDebugUtilsMessengerEXT callback;

static VKAPI_ATTR VkBool32 VKAPI_CALL user_callback(
    VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType,
    uint64_t object, size_t location, int32_t messageCode,
    const char* pLayerPrefix, const char* pMessage, void* pUserData) {
    printf("[VULKAN][DEBUG] %s: %s\n", pLayerPrefix, pMessage);
    return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pCallback) {
    PFN_vkCreateDebugUtilsMessengerEXT func =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkCreateDebugUtilsMessengerEXT");
    if (func) {
        return func(instance, pCreateInfo, pAllocator, pCallback);
    }
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT callback,
                                   const VkAllocationCallbacks* pAllocator) {
    PFN_vkDestroyDebugUtilsMessengerEXT func =
        (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func) {
        func(instance, callback, pAllocator);
    }
}

static int32_t get_validation_layers(uint32_t* layer_count,
                                     const char** layers) {
    uint32_t available_layer_count = 0;
    VkLayerProperties available_layers[16] = {};

    VK_CHECK(vkEnumerateInstanceLayerProperties(&available_layer_count, NULL));
    VK_CHECK(vkEnumerateInstanceLayerProperties(&available_layer_count,
                                                available_layers));
    available_layer_count = bh_min(16, available_layer_count);
    /* Prefer 'VK_LAYER_KHRONOS_validation' over older
     * 'VK_LAYER_LUNARG_standard_validation' */
    for (uint32_t i = 0; i < available_layer_count; ++i) {
        if (strcmp(available_layers[i].layerName,
                   "VK_LAYER_KHRONOS_validation") == 0) {
            *layer_count = 1;
            layers[0] = "VK_LAYER_KHRONOS_validation";
            break;
        } else if (strcmp(available_layers[i].layerName,
                          "VK_LAYER_LUNARG_standard_validation") == 0) {
            *layer_count = 1;
            layers[0] = "VK_LAYER_LUNARG_standard_validation";
        }
    }
    return 0;
}

static int32_t get_extensions(uint32_t* extension_count,
                              const char** extensions,
                              VkBool32* debug_utils_available) {
    uint32_t available_extension_count = 0;
    VkExtensionProperties available_extensions[16] = {};
    *debug_utils_available = VK_FALSE;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(
        NULL, &available_extension_count, NULL));
    VK_CHECK(vkEnumerateInstanceExtensionProperties(
        NULL, &available_extension_count, available_extensions));
    available_extension_count = bh_min(16, available_extension_count);

    /* Prefer 'VK_EXT_debug_utils' over 'VK_EXT_debug_report'.
     */
    for (uint32_t i = 0; i < available_extension_count; ++i) {
        if (!strcmp(available_extensions[i].extensionName,
                    VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
            *extension_count = 1;
            extensions[0] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
            *debug_utils_available = VK_TRUE;
            return 0;
        } else if (!strcmp(available_extensions[i].extensionName,
                           VK_EXT_DEBUG_REPORT_EXTENSION_NAME)) {
            *extension_count = 1;
            extensions[0] = VK_EXT_DEBUG_REPORT_EXTENSION_NAME;
        }
    }
    return 0;
}
#endif

int32_t bcnn_vk_create_instance(bcnn_vk_context* context,
                                const char* app_name) {
    const VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = NULL,
        .pApplicationName = app_name,
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0,
        .pEngineName = app_name,
    };

    uint32_t num_layers = 0;
    const char* layer_names[1] = {NULL};
    uint32_t num_extensions = 0;
    const char* enabled_extensions[1] = {NULL};
    context->debug_utils_available = VK_FALSE;
#ifdef BCNN_VK_ENABLE_DEBUG
    get_validation_layers(&num_layers, layer_names);
    get_extensions(&num_extensions, enabled_extensions,
                   &context->debug_utils_available);
#endif
    // Create instance
    const VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames =
            (const char* const*){VK_EXT_DEBUG_REPORT_EXTENSION_NAME},
        .ppEnabledLayerNames = layer_names,
        .enabledLayerCount = num_layers};

    VK_CHECK(vkCreateInstance(&instance_info, NULL, &context->instance));

#ifdef BCNN_VK_ENABLE_DEBUG
    if (context->debug_utils_available) {
        VkDebugUtilsMessengerCreateInfoEXT create_info = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = user_callback,
            .pUserData = 0};
        VK_CHECK(CreateDebugUtilsMessengerEXT(context->instance, &create_info,
                                              NULL, &callback));
    }
#endif

    /* Find physical devices */
    uint32_t num_phys_devices = 0;
    VK_CHECK(
        vkEnumeratePhysicalDevices(context->instance, &num_phys_devices, 0));
    num_phys_devices = bh_min(8, num_phys_devices);
    VkPhysicalDevice phys_devices[8] = {0};
    VkResult res = vkEnumeratePhysicalDevices(context->instance,
                                              &num_phys_devices, phys_devices);
    if (res != VK_SUCCESS && res != VK_INCOMPLETE) {
        fprintf(stderr, "[ERROR] vkEnumeratePhysicalDevices failed.\n");
        return (int32_t)(res);
    }
    VkBool32 found_valid_gpu = VK_FALSE;
    VkPhysicalDeviceType device_type[8] = {0};
    for (uint32_t i = 0; i < num_phys_devices; i++) {
        /* Retrieve the device type */
        VkPhysicalDeviceProperties phys_device_properties;
        vkGetPhysicalDeviceProperties(phys_devices[i], &phys_device_properties);
        device_type[i] = phys_device_properties.deviceType;
    }
    /* Pick GPU device index */
    context->physical_device_index = -1;
    /* Prefer discrete GPU */
    for (uint32_t i = 0; i < num_phys_devices; i++) {
        if (device_type[i] == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            context->physical_device = phys_devices[i];
            context->physical_device_index = i;
            found_valid_gpu = VK_TRUE;
            break;
        }
    }
    if (found_valid_gpu == VK_FALSE) {
        for (uint32_t i = 0; i < num_phys_devices; i++) {
            if (device_type[i] == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                context->physical_device = phys_devices[i];
                context->physical_device_index = i;
                found_valid_gpu = VK_TRUE;
                break;
            }
        }
    }
    return 0;
}

int32_t bcnn_vk_create_device(bcnn_vk_context* context) {
    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(context->physical_device,
                                             &queue_family_count, NULL);

    VkQueueFamilyProperties* queue_family;
    queue_family = (VkQueueFamilyProperties*)calloc(
        queue_family_count, sizeof(VkQueueFamilyProperties));
    if (queue_family == NULL) {
        bcnn_log(context->log_ctx, BCNN_LOG_ERROR,
                 "Internal allocation error\n");
        return -1;
    }

    vkGetPhysicalDeviceQueueFamilyProperties(context->physical_device,
                                             &queue_family_count, queue_family);

    uint32_t has_index = 0;
    context->queue_family_index = 0;
    for (int32_t i = 0; i < queue_family_count; ++i) {
        if (queue_family[i].queueCount > 0 &&
            queue_family[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            context->queue_family_index = i;
            has_index = 1;
            break;
        }
    }
    if (has_index == 0) {
        bcnn_log(context->log_ctx, BCNN_LOG_ERROR,
                 "vkGetPhysicalDeviceQueueFamilyProperties failed\n");
        goto fail_get_family_queue;
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .queueFamilyIndex = context->queue_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority};

    VkDeviceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_create_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = NULL,
        .enabledExtensionCount = extension_count,
        .ppEnabledExtensionNames = extension_names,
        .pEnabledFeatures = NULL};
    if (vkCreateDevice(context->physical_device, &create_info, NULL,
                       &context->device) != VK_SUCCESS) {
        bcnn_log(context->log_ctx, BCNN_LOG_ERROR, "vkCreateDevice failed\n");
        goto fail_create_device;
    }
    vkGetDeviceQueue(context->device, context->queue_family_index, 0,
                     &context->queue);

    bh_free(queue_family);
    return 0;

/* Failure cases */
fail_create_device:
fail_get_family_queue:
    bh_free(queue_family);
    return -1;
}

void bcnn_vk_destroy_instance(bcnn_vk_context* context) {
#ifdef BCNN_VK_ENABLE_DEBUG
    if (context->debug_utils_available) {
        DestroyDebugUtilsMessengerEXT(context->instance, callback, NULL);
    }
#endif
    vkDestroyInstance(context->instance, 0);
}

struct bcnn_vk_shader_map {
    const char* name;     /* shader name */
    const uint32_t* data; /* spv hex data */
    size_t data_size;     /* spv hex data size */
};

static const uint32_t* dense_data[] = {0};

static const bcnn_vk_shader_map shader_map[] = {
    {.name = "dense", .data = dense_data, .data_size = sizeof(dense_data)}};

bcnn_vk_pipeline* bcnn_vk_pipeline_create(bcnn_vk_context* context,
                                          const char* shader_name,
                                          int32_t num_descriptor_set_bindings) {
    bcnn_vk_pipeline* pipeline =
        (bcnn_vk_pipeline*)calloc(1, sizeof(bcnn_vk_pipeline));
    if (pipeline == NULL) {
        bcnn_log(context->log_ctx, BCNN_LOG_ERROR,
                 "Internal allocation error\n");
        return NULL;
    }

    /* Create descriptor set layout */
    if (num_descriptor_set_bindings == 0) {
        pipeline->descriptor_set_layout = 0;
        return pipeline;
    }

    VkDescriptorSetLayoutBinding* descriptor_set_bindings =
        (VkDescriptorSetLayoutBinding*)calloc(
            num_descriptor_set_bindings, sizeof(VkDescriptorSetLayoutBinding));
    for (int32_t i = 0; i < num_descriptor_set_bindings; ++i) {
        descriptor_set_bindings[i].binding = i;
        descriptor_set_bindings[i].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptor_set_bindings[i].descriptorCount = 1;
        descriptor_set_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        descriptor_set_bindings[i].pImmutableSamplers = 0;
    }

    VkDescriptorSetLayoutCreateInfo descriptor_set_create_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = 0,
        .flags = 0,
        .bindingCount = num_descriptor_set_bindings,
        .pBindings = descriptor_set_bindings};

    if (vkdev->info.support_VK_KHR_push_descriptor) {
        descriptor_set_create_info.flags |=
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    }

    VkResult ret = vkCreateDescriptorSetLayout(
        context->device, &descriptor_set_create_info, 0,
        &pipeline->descriptor_set_layout);
    if (ret != VK_SUCCESS) {
        fprintf(stderr, "vkCreateDescriptorSetLayout failed %d\n", ret);
        bh_free(descriptor_set_bindings);
        bh_free(pipeline);
        return NULL;
    }
    bh_free(descriptor_set_bindings);

    /* Create pipeline layout */
    VkPipelineLayoutCreateInfo pipeline_create_info = {
        pipeline_create_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        pipeline_create_info.pNext = 0,
        pipeline_create_info.flags = 0,
        pipeline_create_info.setLayoutCount = 1,
        pipeline_create_info.pSetLayouts = &descriptorset_layout,
        pipeline_create_info.pushConstantRangeCount = 0,
        pipeline_create_info.pPushConstantRanges = 0};

    VkResult ret = vkCreatePipelineLayout(
        context->device, &pipeline_create_info, 0, &pipeline->pipeline_layout);
    if (ret != VK_SUCCESS) {
        bcnn_log(context->log_ctx, BCNN_LOG_ERROR,
                 "vkCreatePipelineLayout failed: %d\n", ret);
        bh_free(pipeline);
        return NULL;
    }

    /* Create shader module */

    /* Create pipeline */

    return pipeline;
}

bcnn_vk_pipeline_destroy(bcnn_vk_pipeline* pipeline) {
    /* TODO destroy internal */
    bh_free(pipeline);
}
