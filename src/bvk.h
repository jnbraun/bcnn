
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

#ifndef BVK_H
#define BVK_H

#include <vulkan/vulkan.h>

/**
 * Error codes.
 */
typedef enum {
    BVK_SUCCESS,
    BVK_VULKAN_INTERNAL_ERROR,
    BVK_FAILED_DEBUG_CALLBACK,
    BVK_NO_DEVICE_FOUND
} bvk_status;

/**
 * \brief Create a vulkan instance with a given application name.
 *
 * \param[in]   app_name    A name of the application.
 * \param[out]  instance    Pointer to the created vulkan instance.
 *
 * \return BVK_SUCCESS if vkCreateInstance is successful.
 */
bvk_status bvk_create_instance(const char* app_name, VkInstance* instance);

#endif