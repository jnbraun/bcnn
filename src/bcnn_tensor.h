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

#ifndef BH_TENSOR_H
#define BH_TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

static const int align_offset_ = 32;

typedef struct {
    int n;        // Batch size
    int c;        // Number of channels = depth
    int h;        // Height
    int w;        // Width
    float *data;  // Pointer to data
#ifndef BCNN_DEPLOY_ONLY
    float *grad_data;  // Pointer to gradient data
#endif
#ifdef BCNN_USE_CUDA
    float *data_gpu;  // Pointer to data on gpu
#ifndef BCNN_DEPLOY_ONLY
    float *grad_data_gpu;  // Pointer to gradient data on gpu
#endif
#endif
    int has_grad;  // if has gradient data or not
} bcnn_tensor;

// The different type of tensor initialization.
// This is ususally used to randomly initialize the weights/bias of one layer
typedef enum bcnn_filler_type {
    FIXED,   // Fill with constant set by value
    XAVIER,  // Xavier init
    MSRA     // MSRA init
} bcnn_filler_type;

typedef struct tensor_filler {
    int range;
    float value;
    bcnn_filler_type type;
} bcnn_tensor_filler;

void bcnn_tensor_create(bcnn_tensor *t, int n, int c, int h, int w,
                        int has_grad);

void bcnn_tensor_fill(bcnn_tensor *t, bcnn_tensor_filler filler);

void bcnn_tensor_destroy(bcnn_tensor *t);

// User will be responsible to declare/allocate the tensor instance such as:
// On stack:
// tensor t = { 0 };
// bcnn_tensor_set_shape(&t, ...);
// // allocate internal tensor memory
// bcnn_tensor_allocate(&t);
// --- do some stuff ---
// bcnn_tensor_free(&t);

// On heap:
// tensor *t = NULL;
// t = (bcnn_tensor)malloc(1, sizeof(t));
// bcnn_tensor_set_shape(t, ...);
// // allocate internal tensor memory
// bcnn_tensor_allocate(t);
// --- do some stuff ---
// bcnn_tensor_free(t);
// bh_free(t);

void bcnn_tensor_set_shape(bcnn_tensor *t, int n, int c, int h, int w,
                           int has_grad);

void bcnn_tensor_set_shape_from_tensor(bcnn_tensor *dst, bcnn_tensor *src);

void bcnn_tensor_allocate(bcnn_tensor *t);

void bcnn_tensor_free(bcnn_tensor *t);

int bcnn_tensor_get_size(bcnn_tensor *tensor);

int bcnn_tensor_get_size3d(bcnn_tensor *t);

int bcnn_tensor_get_size2d(bcnn_tensor *t);

void bcnn_tensor_assign(bcnn_tensor *dst, bcnn_tensor *src);

#ifdef __cplusplus
}
#endif

#endif  // BH_TENSOR_H
