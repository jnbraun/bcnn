/*
* Copyright (c) 2016 Jean-Noel Braun.
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
#include "bcnn_pooling_layer.h"

#include <bh/bh.h>
#include <bh/bh_string.h>
#include "bcnn_utils.h"
#include "bh_log.h"

int bcnn_add_maxpool_layer(bcnn_net *net, int size, int stride, char *src_id,
                           char *dst_id) {
    int sz, i;
    bcnn_node node = {0};
    bcnn_tensor dst_tensor = {0};

    if (net->num_nodes > 0) {
        int is_src_node_found = 0;
        for (i = net->num_tensors - 1; i >= 0; --i) {
            if (strcmp(net->tensors[i].name, src_id) == 0) {
                bcnn_node_add_input(&node, i);
                is_src_node_found = 1;
                break;
            }
        }
        bh_check(is_src_node_found, "Maxpool layer: invalid input node name %s",
                 src_id);
    } else {
        bcnn_node_add_input(&node, 0);
    }

    bcnn_tensor_set_shape(
        &dst_tensor,
        net->tensors[node.src[0]].n,  // batch size
        net->tensors[node.src[0]].c,  // depth
        (int)(ceil((float)(net->tensors[node.src[0]].h - size) / stride)) +
            1,  // height
        (int)(ceil((float)(net->tensors[node.src[0]].w - size) / stride)) +
            1,  // width
        1);
    bcnn_tensor_allocate(&dst_tensor);
    bh_strfill(&dst_tensor.name, dst_id);
    // Add node to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(&node, net->num_tensors - 1);

    node.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    node.layer->type = MAXPOOL;
    node.layer->size = size;
    node.layer->stride = stride;

    sz = bcnn_tensor_get_size(&net->tensors[node.dst[0]]);
    node.layer->indexes = (int *)calloc(sz, sizeof(int));
#ifdef BCNN_USE_CUDA
    node.layer->indexes_gpu = bcnn_cuda_malloc_i32(sz);
#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&node.layer->src_tensor_desc));
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&node.layer->dst_tensor_desc));
    bcnn_cudnn_check(cudnnCreatePoolingDescriptor(&node.layer->pooling_desc));
    bcnn_cudnn_check(cudnnSetPooling2dDescriptor(
        node.layer->pooling_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
        node.layer->size, node.layer->size, 0, 0, node.layer->stride,
        node.layer->stride));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(
        node.layer->src_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        net->tensors[node.src[0]].n, net->tensors[node.src[0]].c,
        net->tensors[node.src[0]].h, net->tensors[node.src[0]].w));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(
        node.layer->dst_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        net->tensors[node.dst[0]].n, net->tensors[node.dst[0]].c,
        net->tensors[node.dst[0]].h, net->tensors[node.dst[0]].w));
#endif
#endif

    bcnn_net_add_node(net, node);

    bh_log_info(
        "[Maxpool] input_shape= %dx%dx%d size= %d stride= %d ouput_shape= "
        "%dx%dx%d",
        net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
        net->tensors[node.src[0]].c, size, stride, net->tensors[node.dst[0]].w,
        net->tensors[node.dst[0]].h, net->tensors[node.dst[0]].c);
    return 0;
}

int bcnn_forward_maxpool_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                   bcnn_tensor *dst_tensor) {
    int b, i, j, k, m, n, dst_index, valid, src_index, cur_w, cur_h, max_i;
    float max_f = -FLT_MAX, val;

    int batch_size = dst_tensor->n;
    int offset0, offset1, offset2;

    for (b = 0; b < batch_size; ++b) {  // batch_size
        offset0 = dst_tensor->c * b;
        for (k = 0; k < dst_tensor->c; ++k) {  // depth
            offset1 = dst_tensor->h * (k + offset0);
            for (i = 0; i < dst_tensor->h; ++i) {  // height
                offset2 = dst_tensor->w * (offset1 + i);
                for (j = 0; j < dst_tensor->w; ++j) {  // width
                    dst_index = j + offset2;
                    max_f = -FLT_MAX;
                    max_i = -1;
                    for (n = 0; n < layer->size; ++n) {  // pooling window
                        for (m = 0; m < layer->size; ++m) {
                            cur_h = i * layer->stride + n;
                            cur_w = j * layer->stride + m;
                            src_index =
                                cur_w +
                                src_tensor->w *
                                    (cur_h +
                                     src_tensor->h * (k + b * src_tensor->c));
                            valid = (cur_h >= 0 && cur_h < src_tensor->h &&
                                     cur_w >= 0 && cur_w < src_tensor->w);
                            val = (valid != 0) ? src_tensor->data[src_index]
                                               : -FLT_MAX;
                            if (val > max_f) {
                                max_f = val;
                                max_i = src_index;
                            }
                        }
                    }
                    dst_tensor->data[dst_index] = max_f;
                    layer->indexes[dst_index] = max_i;
                }
            }
        }
    }
    return BCNN_SUCCESS;
}

int bcnn_forward_maxpool_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_maxpool_layer_gpu(node->layer, src, dst);
#else
    return bcnn_forward_maxpool_layer_cpu(node->layer, src, dst);
#endif
}

int bcnn_backward_maxpool_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                    bcnn_tensor *dst_tensor) {
    int i, index;

    int sz = bcnn_tensor_get_size(dst_tensor);

    for (i = 0; i < sz; ++i) {
        index = layer->indexes[i];
        src_tensor->grad_data[index] += dst_tensor->grad_data[i];
    }

    return BCNN_SUCCESS;
}

int bcnn_backward_maxpool_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_maxpool_layer_gpu(node->layer, src, dst);
#else
    return bcnn_backward_maxpool_layer_cpu(node->layer, src, dst);
#endif
    return 0;
}
