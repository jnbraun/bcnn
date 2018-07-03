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
#include "bcnn_maxpool_layer.h"

#include <bh/bh_log.h>
#include <bh/bh_macros.h>
#include <bh/bh_string.h>
#include "bcnn_utils.h"

#include <bh/bh_timer.h>
#ifdef BCNN_USE_NEON
#include <arm_neon.h>
#endif

bcnn_status bcnn_add_maxpool_layer(bcnn_net *net, int size, int stride,
                                   bcnn_padding padding, char *src_id,
                                   char *dst_id) {
    int sz, i;
    bcnn_node node = {0};
    bcnn_tensor dst_tensor = {0};

    if (net->num_nodes > 0) {
        int is_src_node_found = 0;
        for (i = net->num_tensors - 1; i >= 0; --i) {
            if (strcmp(net->tensors[i].name, src_id) == 0) {
                bcnn_node_add_input(net, &node, i);
                is_src_node_found = 1;
                break;
            }
        }
        BCNN_CHECK_AND_LOG(net->log_ctx, is_src_node_found,
                           BCNN_INVALID_PARAMETER,
                           "Maxpool layer: invalid input node name %s", src_id);
    } else {
        bcnn_node_add_input(net, &node, 0);
    }
    // Compute output size according to padding option
    int out_h =
        (padding == PADDING_SAME)
            ? (net->tensors[node.src[0]].h + stride - 1) / stride
            : (padding == PADDING_VALID)
                  ? (net->tensors[node.src[0]].h - size + stride) / stride
                  : (padding == PADDING_CAFFE)
                        ? ((int)(ceil(
                               (float)(net->tensors[node.src[0]].h - size) /
                               stride)) +
                           1)
                        : 0;
    int out_w =
        (padding == PADDING_SAME)
            ? (net->tensors[node.src[0]].w + stride - 1) / stride
            : (padding == PADDING_VALID)
                  ? (net->tensors[node.src[0]].w - size + stride) / stride
                  : (padding == PADDING_CAFFE)
                        ? ((int)(ceil(
                               (float)(net->tensors[node.src[0]].w - size) /
                               stride)) +
                           1)
                        : 0;
    bcnn_tensor_set_shape(&dst_tensor,
                          net->tensors[node.src[0]].n,  // batch size
                          net->tensors[node.src[0]].c,  // depth
                          out_h,                        // height
                          out_w,                        // width
                          1);
    bcnn_tensor_allocate(&dst_tensor);
    bh_strfill(&dst_tensor.name, dst_id);
    // Add node to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(net, &node, net->num_tensors - 1);

    node.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    node.layer->type = MAXPOOL;
    node.layer->size = size;
    node.layer->stride = stride;

    sz = bcnn_tensor_size(&net->tensors[node.dst[0]]);
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

    BCNN_INFO(
        net->log_ctx,
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

#ifdef BCNN_USE_NEON
int bcnn_forward_maxpool_layer_cpu_neon(bcnn_layer *layer,
                                        bcnn_tensor *src_tensor,
                                        bcnn_tensor *dst_tensor) {
    const int tail = src_tensor->w + (src_tensor->w - 2 * dst_tensor->w);
    for (int b = 0; b < dst_tensor->n; ++b) {  // batch_size
        int offset0 = dst_tensor->c * b;
        for (int k = 0; k < dst_tensor->c; ++k) {  // depth
            const float *p_src0 =
                src_tensor->data +
                src_tensor->h * src_tensor->w * (k + src_tensor->c * b);
            const float *p_src1 =
                src_tensor->data +
                src_tensor->h * src_tensor->w * (k + offset0) + src_tensor->w;
            float *outptr = dst_tensor->data +
                            dst_tensor->h * dst_tensor->w * (k + offset0);
            for (int i = 0; i < dst_tensor->h; i++) {
                int half = dst_tensor->w >> 2;
                int last = dst_tensor->w - (half << 2);
                for (int x = 0; x < (half << 2); ++x) {
                    float32x2_t max0, max1;
                    max0 = vld1_f32(p_src0);
                    p_src0 += 2;
                    max1 = vld1_f32(p_src1);
                    p_src1 += 2;
                    max0 = vmax_f32(max0, max1);
                    max0 = vpmax_f32(max0, max0);
                    vst1_lane_f32(outptr, max0, 0);
                    outptr++;
                }
                for (; last > 0; last--) {
                    float max0 = bh_max(p_src0[0], p_src0[1]);
                    float max1 = bh_max(p_src1[0], p_src1[1]);
                    *outptr = bh_max(max0, max1);
                    p_src0 += 2;
                    p_src1 += 2;
                    outptr++;
                }
                p_src0 += tail;
                p_src1 += tail;
            }
        }
    }
    return BCNN_SUCCESS;
}
#endif

int bcnn_forward_maxpool_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_maxpool_layer_gpu(node->layer, src, dst);
#else
#ifdef BCNN_USE_NEON
    if (node->layer->size == 2 && node->layer->stride == 2) {
        return bcnn_forward_maxpool_layer_cpu_neon(node->layer, src, dst);
    } else {
        return bcnn_forward_maxpool_layer_cpu(node->layer, src, dst);
    }
#else
    return bcnn_forward_maxpool_layer_cpu(node->layer, src, dst);
#endif
#endif
}

int bcnn_backward_maxpool_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                    bcnn_tensor *dst_tensor) {
    int i, index;

    int sz = bcnn_tensor_size(dst_tensor);

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
