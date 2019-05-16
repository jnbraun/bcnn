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

#ifndef BCNN_TENSOR_H
#define BCNN_TENSOR_H

#include <bcnn/bcnn.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tensor_filler {
    int range;
    float value;
    bcnn_filler_type type;
} bcnn_tensor_filler;

void bcnn_tensor_create(bcnn_tensor *t, int n, int c, int h, int w,
                        int has_grad, const char *name, int net_state);

void bcnn_tensor_fill(bcnn_tensor *t, bcnn_tensor_filler filler);

void bcnn_tensor_destroy(bcnn_tensor *t);

void bcnn_tensor_set_shape(bcnn_tensor *t, int n, int c, int h, int w,
                           int has_grad);

bcnn_status bcnn_tensor_allocate_buffer(bcnn_tensor *t, int net_state,
                                        size_t size);
bcnn_status bcnn_tensor_allocate(bcnn_tensor *t, int net_state);

void bcnn_tensor_free(bcnn_tensor *t);

/**
 * Tensor manipulation helpers
 */
int bcnn_tensor_size(const bcnn_tensor *t);
int bcnn_tensor_size3d(const bcnn_tensor *t);
int bcnn_tensor_size2d(const bcnn_tensor *t);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_TENSOR_H