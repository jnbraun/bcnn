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

#include "bcnn_tensor.h"
#include "bcnn_net.h"

#include <math.h>

#include <bh/bh_macros.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>

#include "bcnn_utils.h"
#ifdef BCNN_USE_OPENCL
#include "bcnn_ocl_utils.h"
#endif

#define BCNN_CHECK_ALLOC(p)                  \
    do {                                     \
        if (((void *)p) == ((void *)NULL)) { \
            return BCNN_FAILED_ALLOC;        \
        }                                    \
    } while (0)

// Alignment for align_malloc
static const size_t align_offset_ = 32;

void bcnn_tensor_create(bcnn_tensor *t, int n, int c, int h, int w,
                        int has_grad, const char *name, bcnn_net *net) {
    bcnn_tensor_set_shape(t, n, c, h, w, has_grad);
    bcnn_tensor_allocate(t, net);
    bh_strfill(&t->name, name);
}

void bcnn_tensor_fill(bcnn_tensor *t, bcnn_tensor_filler filler) {
    if (!t->data) {
        return;
    }
    switch (filler.type) {
        float std_init;
        case BCNN_FILLER_XAVIER:
            std_init = sqrtf(3.0f / filler.range);
            for (int i = 0; i < bcnn_tensor_size(t); ++i) {
                t->data[i] = std_init * (2 * ((float)rand() / RAND_MAX) - 1);
            }
            break;
        case BCNN_FILLER_MSRA:
            std_init = sqrtf(2.0f / filler.range);
            bcnn_gauss_gen g = {0};
            for (int i = 0; i < bcnn_tensor_size(t); ++i) {
                t->data[i] = std_init * bcnn_rng_gaussian(&g);
            }
            break;
        case BCNN_FILLER_FIXED:
            for (int i = 0; i < bcnn_tensor_size(t); ++i) {
                t->data[i] = filler.value;
            }
            break;
    }
#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_f32_noalloc(t->data, t->data_gpu, bcnn_tensor_size(t));
#endif
    return;
}

void bcnn_tensor_destroy(bcnn_tensor *t) {
    bcnn_tensor_free(t);
    t->n = 0;
    t->c = 0;
    t->h = 0;
    t->w = 0;
    t->has_grad = 0;
    bh_free(t->name);
}

void bcnn_tensor_set_shape(bcnn_tensor *t, int n, int c, int h, int w,
                           int has_grad) {
    t->n = n;
    t->c = c;
    t->h = h;
    t->w = w;
    t->has_grad = has_grad;
}

int bcnn_tensor_size(const bcnn_tensor *t) { return t->w * t->h * t->c * t->n; }

int bcnn_tensor_size3d(const bcnn_tensor *t) { return t->w * t->h * t->c; }

int bcnn_tensor_size2d(const bcnn_tensor *t) { return t->w * t->h; }

bcnn_status bcnn_tensor_allocate(bcnn_tensor *t, bcnn_net *net) {
    int size = t->n * t->c * t->h * t->w;

    bcnn_tensor_free(t);
    if (size <= 0) {
        return BCNN_INVALID_PARAMETER;
    }
    t->data = (float *)bh_align_calloc(size * sizeof(float), align_offset_);
    BCNN_CHECK_ALLOC(t->data);
    if (t->has_grad && net->mode == BCNN_MODE_TRAIN) {
        t->grad_data =
            (float *)bh_align_calloc(size * sizeof(float), align_offset_);
        BCNN_CHECK_ALLOC(t->grad_data);
    }
#if defined(BCNN_USE_CUDA)
    t->data_gpu = bcnn_cuda_memcpy_f32(t->data, size);
    if (t->has_grad && net->mode == BCNN_MODE_TRAIN) {
        t->grad_data_gpu = bcnn_cuda_memcpy_f32(t->grad_data, size);
    }
#elif defined(BCNN_USE_OPENCL)
    cl_int rc;
    t->data_gpu = (cl_mem)clCreateBuffer(
        net->opencl_ctx->ctx,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        size * sizeof(cl_float), t->data, &rc);
    BCNN_OPENCL_CHECK(rc);
    if (t->has_grad && net->mode == BCNN_MODE_TRAIN) {
        t->grad_data_gpu = (cl_mem)clCreateBuffer(
            net->opencl_ctx->ctx,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
            size * sizeof(cl_float), t->data, &rc);
        BCNN_OPENCL_CHECK(rc);
    }
#endif

    return BCNN_SUCCESS;
}

bcnn_status bcnn_tensor_free(bcnn_tensor *t) {
    bh_align_free(t->data);
    t->data = NULL;
    if (t->has_grad) {
        bh_align_free(t->grad_data);
        t->grad_data = NULL;
    }
#if defined(BCNN_USE_CUDA)
    bcnn_cuda_free(t->data_gpu);
    t->data_gpu = NULL;
    if (t->has_grad) {
        bcnn_cuda_free(t->grad_data_gpu);
        t->grad_data_gpu = NULL;
    }
#elif defined(BCNN_USE_OPENCL)
    if (t->data_gpu) {
        BCNN_OPENCL_CHECK(clReleaseMemObject((cl_mem)t->data_gpu));
        t->data_gpu = NULL;
    }
    if (t->has_grad) {
        if (t->grad_data_gpu) {
            BCNN_OPENCL_CHECK(clReleaseMemObject((cl_mem)t->grad_data_gpu));
            t->grad_data_gpu = NULL;
        }
    }
#endif
    return BCNN_SUCCESS;
}
