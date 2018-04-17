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

#ifndef BCNN_ACTIVATION_LAYER_H
#define BCNN_ACTIVATION_LAYER_H

#include <bcnn/bcnn.h>

#ifdef __cplusplus
extern "C" {
#endif

int bcnn_forward_activation_cpu(float *x, int sz, bcnn_activation a);
int bcnn_forward_activation_layer(bcnn_net *net, bcnn_node *node);
int bcnn_backward_activation_cpu(float *x, float *dx, int sz,
                                 bcnn_activation a);
int bcnn_backward_activation_layer(bcnn_net *net, bcnn_node *node);

#ifdef BCNN_USE_CUDA
int bcnn_forward_activation_gpu(float *x, int sz, bcnn_activation a);
int bcnn_forward_activation_layer_gpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                      bcnn_tensor *dst_tensor);
int bcnn_backward_activation_gpu(float *x, float *dx, int sz,
                                 bcnn_activation a);
int bcnn_backward_activation_layer_gpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                       bcnn_tensor *dst_tensor);
#endif

#ifdef __cplusplus
}
#endif

#endif // BCNN_ACTIVATION_LAYER_H