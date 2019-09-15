
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

#include "bvk.h"

#include <stdio.h>

/* Standalone low-level wrapper around Vulkan API that provides the bricks to
 * build a bcnn_vulkan_context */

#ifdef BVK_ENABLE_VALIDATION_LAYERS
const VkBool32 enable_validation_layers = 1;
#else
const VkBool32 enable_validation_layers = 0;
#endif

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

static const char* bvkstatus2string(bvk_status status) {
    switch (status) {
#define STR(r)    \
    case BVK_##r: \
        return #r
        STR(VULKAN_INTERNAL_ERROR);
        STR(OTHER_ERROR);
#undef STR
        default:
            return "UNKNOWN_ERROR";
    }
}

#define VK_CHECK(res)                                             \
    do {                                                          \
        VkResult r = (res);                                       \
        if ((r) != VK_SUCCESS) {                                  \
            printf("[ERROR][Vulkan] %s", (vkresult2string(res))); \
            return (BVK_VULKAN_INTERNAL_ERROR);                   \
        }                                                         \
    } while (0)

#define BVK_CHECK(res)                                     \
    do {                                                   \
        bvk_status r = (res);                              \
        if ((r) != BVK_SUCCESS) {                          \
            printf("[ERROR] %s", (bvkstatus2string(res))); \
            return (r);                                    \
        }                                                  \
    } while (0)

#define bvk_max(a, b) (((a) > (b)) ? (a) : (b))

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType,
    uint64_t object, size_t location, int32_t messageCode,
    const char* pLayerPrefix, const char* pMessage, void* pUserData) {
    printf("[BVK][DEBUG] %s: %s\n", pLayerPrefix, pMessage);
    return VK_FALSE;
}

static bvk_status get_validation_layers(uint32_t* layer_count,
                                        const char** layers) {
    uint32_t available_layer_count = 0;
    VkLayerProperties available_layers[16] = {};

    VK_CHECK(vkEnumerateInstanceLayerProperties(&available_layer_count, NULL));
    VK_CHECK(vkEnumerateInstanceLayerProperties(&available_layer_count,
                                                available_layers));

    /* Prefer 'VK_LAYER_KHRONOS_validation' over older
     * 'VK_LAYER_LUNARG_standard_validation' */
    for (uint32_t i = 0; i < bvk_max(16, available_layer_count); ++i) {
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
    return BVK_SUCCESS;
}

static bvk_status get_extensions(uint32_t* extension_count,
                                 const char** extensions,
                                 VkBool32* debug_utils_available) {
    uint32_t available_extension_count = 0;
    VkExtensionProperties available_extensions[16] = {};
    *debug_utils_available = VK_FALSE;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(
        NULL, &available_extension_count, NULL));
    VK_CHECK(vkEnumerateInstanceExtensionProperties(
        NULL, &available_extension_count, available_extensions));

    /* Prefer 'VK_EXT_debug_utils' over 'VK_EXT_debug_report'.
     */
    for (uint32_t i = 0; i < available_extension_count; ++i) {
        if (!strcmp(available_extensions[i].extensionName,
                    VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
            *extension_count = 1;
            extensions[0] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
            *debug_utils_available = VK_TRUE;
            return BVK_SUCCESS;
        } else if (!strcmp(available_extensions[i].extensionName,
                           VK_EXT_DEBUG_REPORT_EXTENSION_NAME)) {
            *extension_count = 1;
            extensions[0] = VK_EXT_DEBUG_REPORT_EXTENSION_NAME;
        }
    }
    return BVK_SUCCESS;
}

bvk_status bvk_create_instance(const char* app_name, VkInstance* instance) {
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
    VkBool32 debug_utils_available = VK_FALSE;
#ifdef BVK_ENABLE_VALIDATION_LAYERS
    BVK_CHECK(get_validation_layers(&num_layers, layer_names));
    BVK_CHECK(get_extensions(&num_layers, layer_names, &debug_utils_available));
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
        .enabledLayerCount = num_layers,
    };

    VK_CHECK(vkCreateInstance(&instance_info, NULL, instance));

    if (enable_validation_layers) {
        VkDebugReportCallbackCreateInfoEXT create_info = {};
        create_info.sType =
            VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        create_info.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT |
                            VK_DEBUG_REPORT_WARNING_BIT_EXT |
                            VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
        create_info.pfnCallback = &debug_callback;

        PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT =
            (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(
                instance, "vkCreateDebugReportCallbackEXT");
        if (vkCreateDebugReportCallbackEXT == NULL) {
            fprintf(stderr,
                    "[ERROR] Could not load vkCreateDebugReportCallbackEXT");
            return BVK_FAILED_DEBUG_CALLBACK;
        }

        // Create and register callback.
        VK_CHECK(vkCreateDebugReportCallbackEXT(instance, &create_info, NULL,
                                                &debug_callback));
    }
    return BVK_SUCCESS;
}

bvk_status bvk_find_physical_device(VkInstance instance,
                                    VkPhysicalDevice* gpu) {
    uint32_t num_devices = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &num_devices, NULL));
    if (num_devices == 0) {
        fprintf(stderr, "[ERROR] Could not find a device with vulkan support");
        return BVK_NO_DEVICE_FOUND;
    } else {
    }
    return BVK_SUCCESS;
}
