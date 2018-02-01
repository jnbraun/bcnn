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
#include "bcnn_depthwise_conv_layer.h"
#include "bcnn_activation_layer.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"

#ifdef BCNN_USE_BLAS
#include "cblas.h"
#endif

#include <bh/bh_mem.h>
#include <bh/bh_string.h>
#include <bh/bh_timer.h>

#include "bh_log.h"

/* Depthwise Separable convolution */

int bcnn_add_depthwise_sep_conv_layer(bcnn_net *net, int size, int stride,
                                      int pad, int batch_norm,
                                      bcnn_weights_init init,
                                      bcnn_activation activation, char *src_id,
                                      char *dst_id) {
    int nb_connections = net->nb_connections + 1;
    int i, sz;
    bcnn_connection conn = {0};
    float std_init = 0.0f;
    bcnn_gauss_gen g = {0};
#ifdef BCNN_USE_CUDNN
    size_t cudnn_wrk_sz = 0;
#endif
    bcnn_node dst_node = {0};

    if (net->nb_connections > 0) {
        int is_src_node_found = 0;
        for (i = net->num_nodes - 1; i >= 0; --i) {
            if (strcmp(net->nodes[i].id, src_id) == 0) {
                bcnn_connection_add_src_node(&conn, i);
                is_src_node_found = 1;
                break;
            }
        }
        bh_check(is_src_node_found,
                 "Dephtwise convolution layer: invalid input node name %s",
                 src_id);
    } else {
        bcnn_connection_add_src_node(&conn, 0);
    }

    // Create layer
    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = DEPTHWISE_CONV;
    conn.layer->num = net->nodes[conn.src[0]].tensor.c;
    conn.layer->stride = stride;
    conn.layer->size = size;
    conn.layer->pad = pad;
    conn.layer->bias_size = net->nodes[conn.src[0]].tensor.c;
    conn.layer->weights_size = net->nodes[conn.src[0]].tensor.c * size * size;
    conn.layer->weight =
        (float *)calloc(conn.layer->weights_size, sizeof(float));
    conn.layer->weight_diff =
        (float *)calloc(conn.layer->weights_size, sizeof(float));
    conn.layer->bias = (float *)calloc(conn.layer->bias_size, sizeof(float));
    conn.layer->bias_diff =
        (float *)calloc(conn.layer->bias_size, sizeof(float));
    switch (init) {
        case XAVIER:
            std_init = (float)sqrt(
                3.0f / (size * size * net->nodes[conn.src[0]].tensor.c));
            for (i = 0; i < conn.layer->weights_size; ++i) {
                conn.layer->weight[i] =
                    std_init * (2 * ((float)rand() / RAND_MAX) - 1);
            }
            break;
        case MSRA:
            std_init = (float)sqrt(
                2.0f / (size * size * net->nodes[conn.src[0]].tensor.c));
            for (i = 0; i < conn.layer->weights_size; ++i) {
                conn.layer->weight[i] = std_init * bcnn_rng_gaussian(&g);
            }
            break;
    }
    if (net->learner.optimizer == ADAM) {
        conn.layer->adam_m =
            (float *)calloc(conn.layer->weights_size, sizeof(float));
        conn.layer->adam_v =
            (float *)calloc(conn.layer->weights_size, sizeof(float));
    }

    bh_strfill(&dst_node.id, dst_id);
    bcnn_tensor_set_shape(&dst_node.tensor, net->nodes[conn.src[0]].tensor.n,
                          net->nodes[conn.src[0]].tensor.c,
                          (net->nodes[conn.src[0]].tensor.h +
                           2 * conn.layer->pad - conn.layer->size) /
                                  conn.layer->stride +
                              1,
                          (net->nodes[conn.src[0]].tensor.w +
                           2 * conn.layer->pad - conn.layer->size) /
                                  conn.layer->stride +
                              1,
                          1);
    bcnn_tensor_allocate(&dst_node.tensor);
    // Add node to net
    bcnn_net_add_node(net, dst_node);
    // Add node pointer to connection
    bcnn_connection_add_dst_node(&conn, net->num_nodes - 1);

    sz = net->nodes[conn.dst[0]].tensor.w * net->nodes[conn.dst[0]].tensor.h *
         net->nodes[conn.src[0]].tensor.c * size * size;
    conn.layer->conv_workspace = (float *)calloc(sz, sizeof(float));

#ifdef BCNN_USE_CUDA
    conn.layer->weight_gpu =
        bcnn_cuda_memcpy_f32(conn.layer->weight, conn.layer->weights_size);
    conn.layer->weight_diff_gpu =
        bcnn_cuda_memcpy_f32(conn.layer->weight_diff, conn.layer->weights_size);
    conn.layer->bias_gpu =
        bcnn_cuda_memcpy_f32(conn.layer->bias, conn.layer->bias_size);
    conn.layer->bias_diff_gpu =
        bcnn_cuda_memcpy_f32(conn.layer->bias_diff, conn.layer->bias_size);
    if (net->learner.optimizer == ADAM) {
        conn.layer->adam_m_gpu =
            bcnn_cuda_memcpy_f32(conn.layer->adam_m, conn.layer->weights_size);
        conn.layer->adam_v_gpu =
            bcnn_cuda_memcpy_f32(conn.layer->adam_v, conn.layer->weights_size);
    }
    sz = net->nodes[conn.dst[0]].tensor.w * net->nodes[conn.dst[0]].tensor.h *
         net->nodes[conn.src[0]].tensor.c * size * size;
    conn.layer->conv_workspace_gpu =
        bcnn_cuda_memcpy_f32(conn.layer->conv_workspace, sz);
#endif
    conn.layer->activation = activation;

    bcnn_net_add_connection(net, conn);

    bh_log_info(
        "[DepthwiseConvolutional] input_shape= %dx%dx%d nb_filters= %d "
        "kernel_size= %d stride= %d padding= %d output_shape= %dx%dx%d\n",
        net->nodes[conn.src[0]].tensor.w, net->nodes[conn.src[0]].tensor.h,
        net->nodes[conn.src[0]].tensor.c, net->nodes[conn.src[0]].tensor.c,
        size, stride, pad, net->nodes[conn.dst[0]].tensor.w,
        net->nodes[conn.dst[0]].tensor.h, net->nodes[conn.dst[0]].tensor.c);

    return 0;
}

int bcnn_forward_depthwise_sep_conv_layer_cpu(bcnn_layer *layer,
                                              bcnn_node *src_node,
                                              bcnn_node *dst_node) {
    int n, sz, c, h, w, kh, kw, h_in, w_in, offset;
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = src.n;
    float *dst_data = NULL;
    const float *bias_data = NULL;
    const float *weight_data = NULL;
    float val = 0;
    /*bh_timer t = { 0 };
    bh_timer_start(&t);*/

    sz = bcnn_tensor_get_size(&dst);

    dst_data = dst.data;
    memset(dst_data, 0, sz * sizeof(float));

    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < dst.c; ++c) {
            for (h = 0; h < dst.h; ++h) {
                if (h * layer->stride - layer->pad >= 0 &&
                    (h * layer->stride - layer->pad + layer->size) < src.h) {
                    for (w = 0; w < dst.w; ++w) {
                        weight_data =
                            layer->weight + c * layer->size * layer->size;
                        val = 0;
                        if (w * layer->stride - layer->pad >= 0 &&
                            (w * layer->stride - layer->pad + layer->size) <
                                src.w) {
                            for (kh = 0; kh < layer->size; ++kh) {
                                for (kw = 0; kw < layer->size; ++kw) {
                                    h_in = -layer->pad + h * layer->stride + kh;
                                    w_in = -layer->pad + w * layer->stride + kw;
                                    offset = ((n * dst.c + c) * src.h + h_in) *
                                                 src.w +
                                             w_in;
                                    val += (*weight_data) * src.data[offset];
                                    ++weight_data;
                                }
                            }
                        } else {
                            for (kh = 0; kh < layer->size; ++kh) {
                                for (kw = 0; kw < layer->size; ++kw) {
                                    h_in = -layer->pad + h * layer->stride + kh;
                                    w_in = -layer->pad + w * layer->stride + kw;
                                    if ((w_in >= 0) && (w_in < src.w)) {
                                        offset =
                                            ((n * dst.c + c) * src.h + h_in) *
                                                src.w +
                                            w_in;
                                        val +=
                                            (*weight_data) * src.data[offset];
                                    }
                                    ++weight_data;
                                }
                            }
                        }
                        *dst_data++ = val;
                    }
                } else {
                    for (w = 0; w < dst.w; ++w) {
                        weight_data =
                            layer->weight + c * layer->size * layer->size;
                        val = 0;
                        if (w * layer->stride - layer->pad >= 0 &&
                            (w * layer->stride - layer->pad + layer->size) <
                                src.w) {
                            for (kh = 0; kh < layer->size; ++kh) {
                                for (kw = 0; kw < layer->size; ++kw) {
                                    h_in = -layer->pad + h * layer->stride + kh;
                                    w_in = -layer->pad + w * layer->stride + kw;
                                    if ((h_in >= 0) && (h_in < src.h)) {
                                        offset =
                                            ((n * dst.c + c) * src.h + h_in) *
                                                src.w +
                                            w_in;
                                        val +=
                                            (*weight_data) * src.data[offset];
                                    }
                                    ++weight_data;
                                }
                            }
                        } else {
                            for (kh = 0; kh < layer->size; ++kh) {
                                for (kw = 0; kw < layer->size; ++kw) {
                                    h_in = -layer->pad + h * layer->stride + kh;
                                    w_in = -layer->pad + w * layer->stride + kw;
                                    if ((h_in >= 0) && (h_in < src.h) &&
                                        (w_in >= 0) && (w_in < src.w)) {
                                        offset =
                                            ((n * dst.c + c) * src.h + h_in) *
                                                src.w +
                                            w_in;
                                        val +=
                                            (*weight_data) * src.data[offset];
                                    }
                                    ++weight_data;
                                }
                            }
                        }
                        *dst_data++ = val;
                    }
                }
            }
        }
    }

    bcnn_add_bias(dst.data, layer->bias, batch_size, dst.c, dst.w * dst.h);

    sz = dst.w * dst.h * dst.c * batch_size;
    bcnn_forward_activation_cpu(dst.data, sz, layer->activation);

    /*bh_timer_stop(&t);
    fprintf(stderr, "sep-conv-forward-time %lf sec\n", bh_timer_get_msec(&t) /
    1000);*/

    return BCNN_SUCCESS;
}

int bcnn_backward_depthwise_sep_conv_layer_cpu(bcnn_layer *layer,
                                               bcnn_node *src_node,
                                               bcnn_node *dst_node) {
    int sz, n, c, h, w, kh, kw, w_in, h_in, offset;
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = src.n;
    float *dst_grad_data = NULL;
    float *weight_diff_base = NULL, *weight_diff = NULL;
    float *weight_data_base = NULL, *weight_data = NULL;
    float *bias_diff = NULL;
    /*bh_timer t = { 0 };
    bh_timer_start(&t);*/

    sz = bcnn_tensor_get_size(&dst);

    bcnn_backward_activation_cpu(dst.data, dst.grad_data,
                                 dst.w * dst.h * dst.c * batch_size,
                                 layer->activation);

    bcnn_grad_bias(layer->bias_diff, dst.grad_data, batch_size, dst.c,
                   dst.w * dst.h);

    if (src.grad_data) {
        dst_grad_data = dst.grad_data;
        weight_diff_base = layer->weight_diff;
        ;
        for (n = 0; n < batch_size; ++n) {
            for (c = 0; c < dst.c; ++c) {
                for (h = 0; h < dst.h; ++h) {
                    if (h * layer->stride - layer->pad >= 0 &&
                        (h * layer->stride - layer->pad + layer->size) <
                            src.h) {
                        for (w = 0; w < dst.w; ++w) {
                            weight_diff = weight_diff_base +
                                          c * layer->size * layer->size;
                            if (w * layer->stride - layer->pad >= 0 &&
                                (w * layer->stride - layer->pad + layer->size) <
                                    src.w) {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride +
                                               kh;
                                        w_in = -layer->pad + w * layer->stride +
                                               kw;
                                        offset =
                                            ((n * dst.c + c) * src.h + h_in) *
                                                src.w +
                                            w_in;
                                        *weight_diff +=
                                            src.data[offset] * (*dst_grad_data);
                                        ++weight_diff;
                                    }
                                }
                            } else {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride +
                                               kh;
                                        w_in = -layer->pad + w * layer->stride +
                                               kw;
                                        if ((w_in >= 0) && (w_in < src.w)) {
                                            offset = ((n * dst.c + c) * src.h +
                                                      h_in) *
                                                         src.w +
                                                     w_in;
                                            *weight_diff += src.data[offset] *
                                                            (*dst_grad_data);
                                        }
                                        ++weight_diff;
                                    }
                                }
                            }
                            ++dst_grad_data;
                        }
                    } else {
                        for (w = 0; w < dst.w; ++w) {
                            weight_diff = weight_diff_base +
                                          c * layer->size * layer->size;
                            if (w * layer->stride - layer->pad >= 0 &&
                                (w * layer->stride - layer->pad + layer->size) <
                                    src.w) {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride +
                                               kh;
                                        w_in = -layer->pad + w * layer->stride +
                                               kw;
                                        if ((h_in >= 0) && (h_in < src.h)) {
                                            offset = ((n * dst.c + c) * src.h +
                                                      h_in) *
                                                         src.w +
                                                     w_in;
                                            *weight_diff += src.data[offset] *
                                                            (*dst_grad_data);
                                        }
                                        ++weight_diff;
                                    }
                                }
                            } else {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride +
                                               kh;
                                        w_in = -layer->pad + w * layer->stride +
                                               kw;
                                        if ((h_in >= 0) && (h_in < src.h) &&
                                            (w_in >= 0) && (w_in < src.w)) {
                                            offset = ((n * dst.c + c) * src.h +
                                                      h_in) *
                                                         src.w +
                                                     w_in;
                                            *weight_diff += src.data[offset] *
                                                            (*dst_grad_data);
                                        }
                                        ++weight_diff;
                                    }
                                }
                            }
                            ++dst_grad_data;
                        }
                    }
                }
            }
        }
    }
    if (src.grad_data) {
        dst_grad_data = dst.grad_data;
        weight_data_base = layer->weight;
        for (n = 0; n < batch_size; ++n) {
            for (c = 0; c < dst.c; ++c) {
                for (h = 0; h < dst.h; ++h) {
                    if (h * layer->stride - layer->pad >= 0 &&
                        (h * layer->stride - layer->pad + layer->size) <
                            src.h) {
                        for (w = 0; w < dst.w; ++w) {
                            weight_data = weight_data_base +
                                          c * layer->size * layer->size;
                            if (w * layer->stride - layer->pad >= 0 &&
                                (w * layer->stride - layer->pad + layer->size) <
                                    src.w) {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride +
                                               kh;
                                        w_in = -layer->pad + w * layer->stride +
                                               kw;
                                        offset =
                                            ((n * dst.c + c) * src.h + h_in) *
                                                src.w +
                                            w_in;
                                        src.grad_data[offset] +=
                                            (*weight_data) * (*dst_grad_data);
                                        ++weight_data;
                                    }
                                }
                            } else {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride +
                                               kh;
                                        w_in = -layer->pad + w * layer->stride +
                                               kw;
                                        if ((w_in >= 0) && (w_in < src.w)) {
                                            offset = ((n * dst.c + c) * src.h +
                                                      h_in) *
                                                         src.w +
                                                     w_in;
                                            src.grad_data[offset] +=
                                                (*weight_data) *
                                                (*dst_grad_data);
                                        }
                                        ++weight_data;
                                    }
                                }
                            }
                            ++dst_grad_data;
                        }
                    } else {
                        for (w = 0; w < dst.w; ++w) {
                            weight_data = weight_data_base +
                                          c * layer->size * layer->size;
                            if (w * layer->stride - layer->pad >= 0 &&
                                (w * layer->stride - layer->pad + layer->size) <
                                    src.w) {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride +
                                               kh;
                                        w_in = -layer->pad + w * layer->stride +
                                               kw;
                                        if ((h_in >= 0) && (h_in < src.h)) {
                                            offset = ((n * dst.c + c) * src.h +
                                                      h_in) *
                                                         src.w +
                                                     w_in;
                                            src.grad_data[offset] +=
                                                (*weight_data) *
                                                (*dst_grad_data);
                                        }
                                        ++weight_data;
                                    }
                                }
                            } else {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride +
                                               kh;
                                        w_in = -layer->pad + w * layer->stride +
                                               kw;
                                        if ((h_in >= 0) && (h_in < src.h) &&
                                            (w_in >= 0) && (w_in < src.w)) {
                                            offset = ((n * dst.c + c) * src.h +
                                                      h_in) *
                                                         src.w +
                                                     w_in;
                                            src.grad_data[offset] +=
                                                (*weight_data) *
                                                (*dst_grad_data);
                                        }
                                        ++weight_data;
                                    }
                                }
                            }
                            ++dst_grad_data;
                        }
                    }
                }
            }
        }
    }

    /*bh_timer_stop(&t);
    fprintf(stderr, "sep-conv-backward-time %lf sec\n", bh_timer_get_msec(&t) /
    1000);*/

    return BCNN_SUCCESS;
}

int bcnn_forward_depthwise_sep_conv_layer(bcnn_net *net,
                                          bcnn_connection *conn) {
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_depthwise_sep_conv_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_forward_depthwise_sep_conv_layer_cpu(conn->layer, src, dst);
#endif
}

int bcnn_backward_depthwise_sep_conv_layer(bcnn_net *net,
                                           bcnn_connection *conn) {
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_depthwise_sep_conv_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_backward_depthwise_sep_conv_layer_cpu(conn->layer, src, dst);
#endif
}