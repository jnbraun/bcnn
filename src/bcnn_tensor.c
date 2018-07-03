/*
 * Copyright (c) 2016-2018 Jean-Noel Braun.
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

#include <bh/bh_log.h>
#include <bh/bh_macros.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>

#include "bcnn/bcnn.h"
#include "bcnn_utils.h"

void bcnn_tensor_create(bcnn_tensor *t, int n, int c, int h, int w,
                        int has_grad, char *name) {
    bcnn_tensor_set_shape(t, n, c, h, w, has_grad);
    bcnn_tensor_allocate(t);
    bh_strfill(&t->name, name);
}

void bcnn_tensor_fill(bcnn_tensor *t, bcnn_tensor_filler filler) {
    if (!t->data) {
        return;
    }
    switch (filler.type) {
        float std_init;
        case XAVIER:
            std_init = sqrtf(3.0f / filler.range);
            for (int i = 0; i < bcnn_tensor_size(t); ++i) {
                t->data[i] = std_init * (2 * ((float)rand() / RAND_MAX) - 1);
            }
            break;
        case MSRA:
            std_init = sqrtf(2.0f / filler.range);
            bcnn_gauss_gen g = {0};
            for (int i = 0; i < bcnn_tensor_size(t); ++i) {
                t->data[i] = std_init * bcnn_rng_gaussian(&g);
            }
            break;
        case FIXED:
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

void bcnn_tensor_set_shape_from_tensor(bcnn_tensor *dst, bcnn_tensor *src) {
    dst->n = src->n;
    dst->c = src->c;
    dst->h = src->h;
    dst->w = src->w;
    dst->has_grad = src->has_grad;
}

int bcnn_tensor_size(bcnn_tensor *t) { return t->w * t->h * t->c * t->n; }

int bcnn_tensor_get_size3d(bcnn_tensor *t) { return t->w * t->h * t->c; }

int bcnn_tensor_get_size2d(bcnn_tensor *t) { return t->w * t->h; }

void bcnn_tensor_allocate(bcnn_tensor *t) {
    int size = t->n * t->c * t->h * t->w;

    bcnn_tensor_free(t);
    if (size <= 0) return;
    t->data = (float *)bh_align_calloc(size * sizeof(float), align_offset_);
#ifndef BCNN_DEPLOY_ONLY
    if (t->has_grad) {
        t->grad_data =
            (float *)bh_align_calloc(size * sizeof(float), align_offset_);
    }
#endif
#ifdef BCNN_USE_CUDA
    t->data_gpu = bcnn_cuda_memcpy_f32(t->data, size);
#ifndef BCNN_DEPLOY_ONLY
    if (t->has_grad) {
        t->grad_data_gpu = bcnn_cuda_memcpy_f32(t->grad_data, size);
    }
#endif
#endif
}

void bcnn_tensor_free(bcnn_tensor *t) {
    bh_align_free(t->data);
    t->data = NULL;
    bh_free(t->name);
#ifndef BCNN_DEPLOY_ONLY
    if (t->has_grad) {
        bh_align_free(t->grad_data);
        t->grad_data = NULL;
    }
#endif
#ifdef BCNN_USE_CUDA
    bcnn_cuda_free(t->data_gpu);
    t->data_gpu = NULL;
#ifndef BCNN_DEPLOY_ONLY
    if (t->has_grad) {
        bcnn_cuda_free(t->grad_data_gpu);
        t->grad_data_gpu = NULL;
    }
#endif
#endif
}

/* bcnn_tensor_assign carries out a shallow copy */
void bcnn_tensor_assign(bcnn_tensor *dst, bcnn_tensor *src) {
    dst->n = src->n;
    dst->c = src->c;
    dst->h = src->h;
    dst->w = src->w;
    dst->has_grad = src->has_grad;
    // Copy of the pointers
    dst->data = src->data;
#ifndef BCNN_DEPLOY_ONLY
    if (dst->has_grad) {
        dst->grad_data = src->grad_data;
    }
#endif
#ifdef BCNN_USE_CUDA
    dst->data_gpu = src->data_gpu;
#ifndef BCNN_DEPLOY_ONLY
    if (dst->has_grad) {
        dst->grad_data_gpu = src->grad_data_gpu;
    }
#endif
#endif
}
