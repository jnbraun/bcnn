
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
            return (int32_t)(r);                                \
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
    BVK_CHECK(get_validation_layers(&num_layers, layer_names));
    BVK_CHECK(get_extensions(&num_extensions, enabled_extensions,
                             &context->debug_utils_available));
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

    return 0;
}

void bcnn_vk_destroy_instance(bcnn_vk_context* context) {
#ifdef BCNN_VK_ENABLE_DEBUG
    if (context->debug_utils_available) {
        DestroyDebugUtilsMessengerEXT(context->instance, callback, NULL);
    }
#endif  // ENABLE_VALIDATION_LAYER
    vkDestroyInstance(context->instance, 0);
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
