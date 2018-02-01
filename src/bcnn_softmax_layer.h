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

#ifndef BCNN_SOFTMAX_LAYER_H
#define BCNN_SOFTMAX_LAYER_H

#include <bcnn/bcnn.h>

#ifdef __cplusplus
extern "C" {
#endif

int bcnn_forward_softmax_layer(bcnn_net *net, bcnn_connection *conn);
int bcnn_backward_softmax_layer(bcnn_net *net, bcnn_connection *conn);

#ifdef BCNN_USE_CUDA
int bcnn_forward_softmax_layer_gpu(bcnn_layer *layer, bcnn_node *src_node,
                                   bcnn_node *dst_node);
int bcnn_backward_softmax_layer_gpu(bcnn_layer *layer, bcnn_node *src_node,
                                    bcnn_node *dst_node);
#endif

#ifdef __cplusplus
}
#endif

#endif  // BCNN_SOFTMAX_LAYER_H