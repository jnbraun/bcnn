
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

#define BVK_CHECK(res)                                            \
    do {                                                          \
        VkResult r = (res);                                       \
        if ((r) != VK_SUCCESS) {                                  \
            printf("[ERROR][VULKAN] %s", (vkresult2string(res))); \
            return (BVK_VULKAN_INTERNAL_ERROR);                   \
        }                                                         \
    } while (0)

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType,
    uint64_t object, size_t location, int32_t messageCode,
    const char* pLayerPrefix, const char* pMessage, void* pUserData) {
    printf("[BVK][DEBUG] %s: %s\n", pLayerPrefix, pMessage);
    return VK_FALSE;
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

    // TODO: check validation layers and extensions availability
    // Create instance
    const VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames =
            (const char* const*){VK_EXT_DEBUG_REPORT_EXTENSION_NAME},
#ifdef BVK_ENABLE_VALIDATION_LAYERS
        .ppEnabledLayerNames =
            (const char* const*){"VK_LAYER_LUNARG_standard_validation"},
        .enabledLayerCount = 1,
#else
        .ppEnabledLayerNames = NULL,
        .enabledLayerCount = 0,
#endif
    };

    BVK_CHECK(vkCreateInstance(&instance_info, NULL, instance));

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
        BVK_CHECK(vkCreateDebugReportCallbackEXT(instance, &create_info, NULL,
                                                 &debug_callback));
    }
    return BVK_SUCCESS;
}

bvk_status bvk_find_physical_device(VkInstance instance,
                                    VkPhysicalDevice* gpu) {
    uint32_t num_devices = 0;
    BVK_CHECK(vkEnumeratePhysicalDevices(instance, &num_devices, NULL));
    if (num_devices == 0) {
        fprintf(stderr, "[ERROR] Could not find a device with vulkan support");
        return BVK_NO_DEVICE_FOUND;
    } else {
        }
    return BVK_SUCCESS;
}
