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
#include "bcnn_maxpool_layer.h"

#include <float.h>
#include <math.h>

#include <bh/bh_log.h>
#include <bh/bh_string.h>

#include "bcnn_net.h"
#include "bcnn_tensor.h"
#include "bcnn_utils.h"

#include <bh/bh_timer.h>
#ifdef BCNN_USE_NEON
#include <arm_neon.h>
#include <bh/bh_macros.h>
#endif

bcnn_status bcnn_add_maxpool_layer(bcnn_net *net, int size, int stride,
                                   bcnn_padding padding, const char *src_id,
                                   const char *dst_id) {
    bcnn_node node = {0};
    bcnn_tensor dst_tensor = {0};

    if (net->num_nodes > 0) {
        int is_src_node_found = 0;
        for (int i = net->num_tensors - 1; i >= 0; --i) {
            if (strcmp(net->tensors[i].name, src_id) == 0) {
                bcnn_node_add_input(net, &node, i);
                is_src_node_found = 1;
                break;
            }
        }
        BCNN_CHECK_AND_LOG(
            net->log_ctx, is_src_node_found, BCNN_INVALID_PARAMETER,
            "Maxpool layer: invalid input node name %s\n", src_id);
    } else {
        bcnn_node_add_input(net, &node, 0);
    }
    // Compute output size according to padding option
    int out_h =
        (padding == BCNN_PADDING_SAME)
            ? (net->tensors[node.src[0]].h + stride - 1) / stride
            : (padding == BCNN_PADDING_VALID)
                  ? (net->tensors[node.src[0]].h - size + stride) / stride
                  : (padding == BCNN_PADDING_CAFFE)
                        ? ((int)(ceil(
                               (float)(net->tensors[node.src[0]].h - size) /
                               stride)) +
                           1)
                        : 0;
    int out_w =
        (padding == BCNN_PADDING_SAME)
            ? (net->tensors[node.src[0]].w + stride - 1) / stride
            : (padding == BCNN_PADDING_VALID)
                  ? (net->tensors[node.src[0]].w - size + stride) / stride
                  : (padding == BCNN_PADDING_CAFFE)
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
    bcnn_tensor_allocate(&dst_tensor, net->mode);
    bh_strfill(&dst_tensor.name, dst_id);
    // Add node to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(net, &node, net->num_tensors - 1);

    int sz = bcnn_tensor_size(&net->tensors[node.dst[0]]);
    node.type = BCNN_LAYER_MAXPOOL;
    node.param_size = sizeof(bcnn_maxpool_param);
    node.param = (bcnn_maxpool_param *)calloc(1, node.param_size);
    bcnn_maxpool_param *param = (bcnn_maxpool_param *)node.param;
    param->size = size;
    param->stride = stride;
    param->padding = padding;
    param->indexes = (int *)calloc(sz, sizeof(int));
#ifdef BCNN_USE_CUDA
    param->indexes_gpu = bcnn_cuda_malloc_i32(sz);
#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&param->src_tensor_desc));
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&param->dst_tensor_desc));
    bcnn_cudnn_check(cudnnCreatePoolingDescriptor(&param->pooling_desc));
    bcnn_cudnn_check(cudnnSetPooling2dDescriptor(
        param->pooling_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
        param->size, param->size, 0, 0, param->stride, param->stride));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(
        param->src_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        net->tensors[node.src[0]].n, net->tensors[node.src[0]].c,
        net->tensors[node.src[0]].h, net->tensors[node.src[0]].w));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(
        param->dst_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        net->tensors[node.dst[0]].n, net->tensors[node.dst[0]].c,
        net->tensors[node.dst[0]].h, net->tensors[node.dst[0]].w));
#endif
#endif
    node.forward = bcnn_forward_maxpool_layer;
    node.backward = bcnn_backward_maxpool_layer;
    node.release_param = bcnn_release_param_maxpool_layer;

    bcnn_net_add_node(net, node);

    char node_opname[256];
    snprintf(node_opname, 256,
             BH_LOG_BOLDBLUE "[Maxpool]" BH_LOG_RESET "        ");
    BCNN_INFO(
        net->log_ctx,
        "%-48s %-8s (%4d x%4d x%4d) -> %-8s (%4d x%4d x%4d) %12d x %2d / %2d\n",
        node_opname, net->tensors[node.src[0]].name,
        net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
        net->tensors[node.src[0]].c, net->tensors[node.dst[0]].name,
        net->tensors[node.dst[0]].w, net->tensors[node.dst[0]].h,
        net->tensors[node.dst[0]].c, size, size, stride);
    return 0;
}

void bcnn_forward_maxpool_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_maxpool_param *param = (bcnn_maxpool_param *)node->param;
    int size = param->size;
    int stride = param->stride;
    int *indexes = param->indexes;
    int batch_size = dst_tensor->n;
    for (int b = 0; b < batch_size; ++b) {  // batch_size
        int offset0 = dst_tensor->c * b;
#pragma omp parallel for num_threads(net->num_threads)
        for (int k = 0; k < dst_tensor->c; ++k) {  // depth
            int offset1 = dst_tensor->h * (k + offset0);
            for (int i = 0; i < dst_tensor->h; ++i) {  // height
                int offset2 = dst_tensor->w * (offset1 + i);
                for (int j = 0; j < dst_tensor->w; ++j) {  // width
                    int dst_index = j + offset2;
                    float max_f = -FLT_MAX;
                    int max_i = -1;
                    for (int n = 0; n < size; ++n) {  // pooling window
                        for (int m = 0; m < size; ++m) {
                            int cur_h = i * stride + n;
                            int cur_w = j * stride + m;
                            int src_index =
                                cur_w +
                                src_tensor->w *
                                    (cur_h +
                                     src_tensor->h * (k + b * src_tensor->c));
                            int valid = (cur_h >= 0 && cur_h < src_tensor->h &&
                                         cur_w >= 0 && cur_w < src_tensor->w);
                            float val = (valid != 0)
                                            ? src_tensor->data[src_index]
                                            : -FLT_MAX;
                            if (val > max_f) {
                                max_f = val;
                                max_i = src_index;
                            }
                        }
                    }
                    dst_tensor->data[dst_index] = max_f;
                    indexes[dst_index] = max_i;
                }
            }
        }
    }
    return;
}

#ifdef BCNN_USE_NEON
void bcnn_forward_maxpool_layer_cpu_neon_2x2(bcnn_tensor *src_tensor,
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
    return;
}
#endif

void bcnn_forward_maxpool_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_forward_maxpool_layer_gpu(net, node);
#else
#ifdef BCNN_USE_NEON
    bcnn_maxpool_param *param = (bcnn_maxpool_param *)node->param;
    if (param->size == 2 && param->stride == 2) {
        bcnn_tensor *src = &net->tensors[node->src[0]];
        bcnn_tensor *dst = &net->tensors[node->dst[0]];
        return bcnn_forward_maxpool_layer_cpu_neon_2x2(src, dst);
    } else {
        return bcnn_forward_maxpool_layer_cpu(net, node);
    }
#else
    return bcnn_forward_maxpool_layer_cpu(net, node);
#endif  // BCNN_USE_NEON
#endif  // BCNN_USE_CUDA
}

void bcnn_backward_maxpool_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_maxpool_param *param = (bcnn_maxpool_param *)node->param;
    int *indexes = param->indexes;
    int i, index;

    int sz = bcnn_tensor_size(dst_tensor);

    for (i = 0; i < sz; ++i) {
        index = indexes[i];
        src_tensor->grad_data[index] += dst_tensor->grad_data[i];
    }

    return;
}

void bcnn_backward_maxpool_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_backward_maxpool_layer_gpu(net, node);
#else
    return bcnn_backward_maxpool_layer_cpu(net, node);
#endif
    return;
}

void bcnn_release_param_maxpool_layer(bcnn_node *node) {
    bcnn_maxpool_param *param = (bcnn_maxpool_param *)node->param;
    bh_free(param->indexes);
#ifdef BCNN_USE_CUDA
    if (param->indexes_gpu) {
        bcnn_cuda_free(param->indexes_gpu);
    }
#ifdef BCNN_USE_CUDNN
    cudnnDestroyTensorDescriptor(param->src_tensor_desc);
    cudnnDestroyTensorDescriptor(param->dst_tensor_desc);
    cudnnDestroyPoolingDescriptor(param->pooling_desc);
#endif
#endif
    return;
}