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

#include "bcnn_depthwise_conv_layer.h"

#include "bcnn_activation_layer.h"
#include "bcnn_learner.h"
#include "bcnn_mat.h"
#include "bcnn_net.h"
#include "bcnn_tensor.h"
#include "bcnn_utils.h"

#ifdef BCNN_USE_BLAS
#include "cblas.h"
#endif

#include <bh/bh_string.h>

/* Depthwise Separable convolution */

bcnn_status bcnn_add_depthwise_conv_layer(bcnn_net *net, int size, int stride,
                                          int pad, int batch_norm,
                                          bcnn_filler_type init,
                                          bcnn_activation activation,
                                          const char *src_id,
                                          const char *dst_id) {
    int num_nodes = net->num_nodes + 1;
    int i, sz;
    bcnn_node node = {0};
    float std_init = 0.0f;
    bcnn_gauss_gen g = {0};
#ifdef BCNN_USE_CUDNN
    size_t cudnn_wrk_sz = 0;
#endif
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
        BCNN_CHECK_AND_LOG(
            net->log_ctx, is_src_node_found, BCNN_INVALID_PARAMETER,
            "Dephtwise convolution layer: invalid input node name %s", src_id);
    } else {
        bcnn_node_add_input(net, &node, 0);
    }

    // Fill nodes param
    node.type = BCNN_LAYER_DEPTHWISE_CONV2D;
    node.param_size = sizeof(bcnn_depthwise_conv_param);
    node.param = (bcnn_depthwise_conv_param *)calloc(1, node.param_size);
    bcnn_depthwise_conv_param *param = (bcnn_depthwise_conv_param *)node.param;
    param->activation = activation;
    param->pad = pad;
    param->num = net->tensors[node.src[0]].c;
    param->size = size;
    param->stride = stride;
    node.forward = bcnn_forward_depthwise_conv_layer;
    node.backward = bcnn_backward_depthwise_conv_layer;
    node.update = bcnn_update_depthwise_conv_layer;
    node.release_param = bcnn_release_param_depthwise_conv_layer;

    // Create weights tensor
    bcnn_tensor weights = {0};
    char weights_name[256];
    sprintf(weights_name, "%s_w", src_id);
    bcnn_tensor_create(&weights, 1, 1, 1,
                       net->tensors[node.src[0]].c * size * size, 1,
                       weights_name, net->mode);
    bcnn_tensor_filler w_filler = {
        .range = (size * size * net->tensors[node.src[0]].c), .type = init};
    bcnn_tensor_fill(&weights, w_filler);
    bcnn_net_add_tensor(net, weights);
    bcnn_node_add_input(net, &node, net->num_tensors - 1);
    // Create bias tensor
    bcnn_tensor biases = {0};
    char biases_name[256];
    sprintf(biases_name, "%s_b", src_id);
    bcnn_tensor_create(&biases, 1, 1, 1, net->tensors[node.src[0]].c, 1,
                       biases_name, net->mode);
    bcnn_net_add_tensor(net, biases);
    bcnn_node_add_input(net, &node, net->num_tensors - 1);
    if (net->learner != NULL) {
        if (net->learner->optimizer == BCNN_OPTIM_ADAM) {
            int weights_size = bcnn_tensor_size(&weights);
            param->adam_m = (float *)calloc(weights_size, sizeof(float));
            param->adam_v = (float *)calloc(weights_size, sizeof(float));
        }
    }
#ifdef BCNN_USE_CUDA
    if (net->learner != NULL) {
        if (net->learner->optimizer == BCNN_OPTIM_ADAM) {
            int weights_size = bcnn_tensor_size(&weights);
            param->adam_m_gpu =
                bcnn_cuda_memcpy_f32(param->adam_m, weights_size);
            param->adam_v_gpu =
                bcnn_cuda_memcpy_f32(param->adam_v, weights_size);
        }
    }
#endif

    bcnn_tensor_set_shape(
        &dst_tensor, net->tensors[node.src[0]].n, net->tensors[node.src[0]].c,
        (net->tensors[node.src[0]].h + 2 * param->pad - param->size) /
                param->stride +
            1,
        (net->tensors[node.src[0]].w + 2 * param->pad - param->size) /
                param->stride +
            1,
        1);
    bcnn_tensor_allocate(&dst_tensor, net->mode);
    bh_strfill(&dst_tensor.name, dst_id);
    // Add node to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(net, &node, net->num_tensors - 1);

    sz = net->tensors[node.dst[0]].w * net->tensors[node.dst[0]].h *
         net->tensors[node.src[0]].c * size * size;

#ifdef BCNN_USE_CUDA
    if (net->learner->optimizer == BCNN_OPTIM_ADAM) {
        int weights_size = bcnn_tensor_size(&weights);
        param->adam_m_gpu = bcnn_cuda_memcpy_f32(param->adam_m, weights_size);
        param->adam_v_gpu = bcnn_cuda_memcpy_f32(param->adam_v, weights_size);
    }
    sz = net->tensors[node.dst[0]].w * net->tensors[node.dst[0]].h *
         net->tensors[node.src[0]].c * size * size;
#endif

    bcnn_net_add_node(net, node);

    BCNN_INFO(net->log_ctx,
              "[DepthwiseConvolutional] input_shape= %dx%dx%d nb_filters= %d "
              "kernel_size= %d stride= %d padding= %d output_shape= %dx%dx%d\n",
              net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
              net->tensors[node.src[0]].c, net->tensors[node.src[0]].c, size,
              stride, pad, net->tensors[node.dst[0]].w,
              net->tensors[node.dst[0]].h, net->tensors[node.dst[0]].c);

    return 0;
}

void bcnn_forward_depthwise_conv_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_depthwise_conv_param *param = (bcnn_depthwise_conv_param *)node->param;
    int n, sz, c, h, w, kh, kw, h_in, w_in, offset;

    int batch_size = src_tensor->n;
    float *dst_data = NULL;
    const float *bias_data = NULL;
    const float *weight_data = NULL;
    float val = 0;
    /*bh_timer t = { 0 };
    bh_timer_start(&t);*/

    sz = bcnn_tensor_size(dst_tensor);

    dst_data = dst_tensor->data;
    memset(dst_data, 0, sz * sizeof(float));

    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < dst_tensor->c; ++c) {
            for (h = 0; h < dst_tensor->h; ++h) {
                if (h * param->stride - param->pad >= 0 &&
                    (h * param->stride - param->pad + param->size) <
                        src_tensor->h) {
                    for (w = 0; w < dst_tensor->w; ++w) {
                        weight_data =
                            weights->data + c * param->size * param->size;
                        val = 0;
                        if (w * param->stride - param->pad >= 0 &&
                            (w * param->stride - param->pad + param->size) <
                                src_tensor->w) {
                            for (kh = 0; kh < param->size; ++kh) {
                                for (kw = 0; kw < param->size; ++kw) {
                                    h_in = -param->pad + h * param->stride + kh;
                                    w_in = -param->pad + w * param->stride + kw;
                                    offset = ((n * dst_tensor->c + c) *
                                                  src_tensor->h +
                                              h_in) *
                                                 src_tensor->w +
                                             w_in;
                                    val += (*weight_data) *
                                           src_tensor->data[offset];
                                    ++weight_data;
                                }
                            }
                        } else {
                            for (kh = 0; kh < param->size; ++kh) {
                                for (kw = 0; kw < param->size; ++kw) {
                                    h_in = -param->pad + h * param->stride + kh;
                                    w_in = -param->pad + w * param->stride + kw;
                                    if ((w_in >= 0) && (w_in < src_tensor->w)) {
                                        offset = ((n * dst_tensor->c + c) *
                                                      src_tensor->h +
                                                  h_in) *
                                                     src_tensor->w +
                                                 w_in;
                                        val += (*weight_data) *
                                               src_tensor->data[offset];
                                    }
                                    ++weight_data;
                                }
                            }
                        }
                        *dst_data++ = val;
                    }
                } else {
                    for (w = 0; w < dst_tensor->w; ++w) {
                        weight_data =
                            weights->data + c * param->size * param->size;
                        val = 0;
                        if (w * param->stride - param->pad >= 0 &&
                            (w * param->stride - param->pad + param->size) <
                                src_tensor->w) {
                            for (kh = 0; kh < param->size; ++kh) {
                                for (kw = 0; kw < param->size; ++kw) {
                                    h_in = -param->pad + h * param->stride + kh;
                                    w_in = -param->pad + w * param->stride + kw;
                                    if ((h_in >= 0) && (h_in < src_tensor->h)) {
                                        offset = ((n * dst_tensor->c + c) *
                                                      src_tensor->h +
                                                  h_in) *
                                                     src_tensor->w +
                                                 w_in;
                                        val += (*weight_data) *
                                               src_tensor->data[offset];
                                    }
                                    ++weight_data;
                                }
                            }
                        } else {
                            for (kh = 0; kh < param->size; ++kh) {
                                for (kw = 0; kw < param->size; ++kw) {
                                    h_in = -param->pad + h * param->stride + kh;
                                    w_in = -param->pad + w * param->stride + kw;
                                    if ((h_in >= 0) && (h_in < src_tensor->h) &&
                                        (w_in >= 0) && (w_in < src_tensor->w)) {
                                        offset = ((n * dst_tensor->c + c) *
                                                      src_tensor->h +
                                                  h_in) *
                                                     src_tensor->w +
                                                 w_in;
                                        val += (*weight_data) *
                                               src_tensor->data[offset];
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

    bcnn_add_bias(dst_tensor->data, biases->data, batch_size, dst_tensor->c,
                  dst_tensor->w * dst_tensor->h);

    sz = dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size;
    bcnn_forward_activation_cpu(dst_tensor->data, sz, param->activation);

    return;
}

void bcnn_backward_depthwise_conv_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_depthwise_conv_param *param = (bcnn_depthwise_conv_param *)node->param;
    int sz, n, c, h, w, kh, kw, w_in, h_in, offset;
    int batch_size = src_tensor->n;
    float *dst_grad_data = NULL;
    float *weight_diff_base = NULL, *weight_diff = NULL;
    float *weight_data_base = NULL, *weight_data = NULL;
    float *bias_diff = NULL;
    /*bh_timer t = { 0 };
    bh_timer_start(&t);*/

    sz = bcnn_tensor_size(dst_tensor);

    bcnn_backward_activation_cpu(
        dst_tensor->data, dst_tensor->grad_data,
        dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size,
        param->activation);

    bcnn_grad_bias(biases->grad_data, dst_tensor->grad_data, batch_size,
                   dst_tensor->c, dst_tensor->w * dst_tensor->h);

    if (src_tensor->grad_data) {
        dst_grad_data = dst_tensor->grad_data;
        weight_diff_base = weights->grad_data;
        for (n = 0; n < batch_size; ++n) {
            for (c = 0; c < dst_tensor->c; ++c) {
                for (h = 0; h < dst_tensor->h; ++h) {
                    if (h * param->stride - param->pad >= 0 &&
                        (h * param->stride - param->pad + param->size) <
                            src_tensor->h) {
                        for (w = 0; w < dst_tensor->w; ++w) {
                            weight_diff = weight_diff_base +
                                          c * param->size * param->size;
                            if (w * param->stride - param->pad >= 0 &&
                                (w * param->stride - param->pad + param->size) <
                                    src_tensor->w) {
                                for (kh = 0; kh < param->size; ++kh) {
                                    for (kw = 0; kw < param->size; ++kw) {
                                        h_in = -param->pad + h * param->stride +
                                               kh;
                                        w_in = -param->pad + w * param->stride +
                                               kw;
                                        offset = ((n * dst_tensor->c + c) *
                                                      src_tensor->h +
                                                  h_in) *
                                                     src_tensor->w +
                                                 w_in;
                                        *weight_diff +=
                                            src_tensor->data[offset] *
                                            (*dst_grad_data);
                                        ++weight_diff;
                                    }
                                }
                            } else {
                                for (kh = 0; kh < param->size; ++kh) {
                                    for (kw = 0; kw < param->size; ++kw) {
                                        h_in = -param->pad + h * param->stride +
                                               kh;
                                        w_in = -param->pad + w * param->stride +
                                               kw;
                                        if ((w_in >= 0) &&
                                            (w_in < src_tensor->w)) {
                                            offset = ((n * dst_tensor->c + c) *
                                                          src_tensor->h +
                                                      h_in) *
                                                         src_tensor->w +
                                                     w_in;
                                            *weight_diff +=
                                                src_tensor->data[offset] *
                                                (*dst_grad_data);
                                        }
                                        ++weight_diff;
                                    }
                                }
                            }
                            ++dst_grad_data;
                        }
                    } else {
                        for (w = 0; w < dst_tensor->w; ++w) {
                            weight_diff = weight_diff_base +
                                          c * param->size * param->size;
                            if (w * param->stride - param->pad >= 0 &&
                                (w * param->stride - param->pad + param->size) <
                                    src_tensor->w) {
                                for (kh = 0; kh < param->size; ++kh) {
                                    for (kw = 0; kw < param->size; ++kw) {
                                        h_in = -param->pad + h * param->stride +
                                               kh;
                                        w_in = -param->pad + w * param->stride +
                                               kw;
                                        if ((h_in >= 0) &&
                                            (h_in < src_tensor->h)) {
                                            offset = ((n * dst_tensor->c + c) *
                                                          src_tensor->h +
                                                      h_in) *
                                                         src_tensor->w +
                                                     w_in;
                                            *weight_diff +=
                                                src_tensor->data[offset] *
                                                (*dst_grad_data);
                                        }
                                        ++weight_diff;
                                    }
                                }
                            } else {
                                for (kh = 0; kh < param->size; ++kh) {
                                    for (kw = 0; kw < param->size; ++kw) {
                                        h_in = -param->pad + h * param->stride +
                                               kh;
                                        w_in = -param->pad + w * param->stride +
                                               kw;
                                        if ((h_in >= 0) &&
                                            (h_in < src_tensor->h) &&
                                            (w_in >= 0) &&
                                            (w_in < src_tensor->w)) {
                                            offset = ((n * dst_tensor->c + c) *
                                                          src_tensor->h +
                                                      h_in) *
                                                         src_tensor->w +
                                                     w_in;
                                            *weight_diff +=
                                                src_tensor->data[offset] *
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
    if (src_tensor->grad_data) {
        dst_grad_data = dst_tensor->grad_data;
        weight_data_base = weights->data;
        for (n = 0; n < batch_size; ++n) {
            for (c = 0; c < dst_tensor->c; ++c) {
                for (h = 0; h < dst_tensor->h; ++h) {
                    if (h * param->stride - param->pad >= 0 &&
                        (h * param->stride - param->pad + param->size) <
                            src_tensor->h) {
                        for (w = 0; w < dst_tensor->w; ++w) {
                            weight_data = weight_data_base +
                                          c * param->size * param->size;
                            if (w * param->stride - param->pad >= 0 &&
                                (w * param->stride - param->pad + param->size) <
                                    src_tensor->w) {
                                for (kh = 0; kh < param->size; ++kh) {
                                    for (kw = 0; kw < param->size; ++kw) {
                                        h_in = -param->pad + h * param->stride +
                                               kh;
                                        w_in = -param->pad + w * param->stride +
                                               kw;
                                        offset = ((n * dst_tensor->c + c) *
                                                      src_tensor->h +
                                                  h_in) *
                                                     src_tensor->w +
                                                 w_in;
                                        src_tensor->grad_data[offset] +=
                                            (*weight_data) * (*dst_grad_data);
                                        ++weight_data;
                                    }
                                }
                            } else {
                                for (kh = 0; kh < param->size; ++kh) {
                                    for (kw = 0; kw < param->size; ++kw) {
                                        h_in = -param->pad + h * param->stride +
                                               kh;
                                        w_in = -param->pad + w * param->stride +
                                               kw;
                                        if ((w_in >= 0) &&
                                            (w_in < src_tensor->w)) {
                                            offset = ((n * dst_tensor->c + c) *
                                                          src_tensor->h +
                                                      h_in) *
                                                         src_tensor->w +
                                                     w_in;
                                            src_tensor->grad_data[offset] +=
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
                        for (w = 0; w < dst_tensor->w; ++w) {
                            weight_data = weight_data_base +
                                          c * param->size * param->size;
                            if (w * param->stride - param->pad >= 0 &&
                                (w * param->stride - param->pad + param->size) <
                                    src_tensor->w) {
                                for (kh = 0; kh < param->size; ++kh) {
                                    for (kw = 0; kw < param->size; ++kw) {
                                        h_in = -param->pad + h * param->stride +
                                               kh;
                                        w_in = -param->pad + w * param->stride +
                                               kw;
                                        if ((h_in >= 0) &&
                                            (h_in < src_tensor->h)) {
                                            offset = ((n * dst_tensor->c + c) *
                                                          src_tensor->h +
                                                      h_in) *
                                                         src_tensor->w +
                                                     w_in;
                                            src_tensor->grad_data[offset] +=
                                                (*weight_data) *
                                                (*dst_grad_data);
                                        }
                                        ++weight_data;
                                    }
                                }
                            } else {
                                for (kh = 0; kh < param->size; ++kh) {
                                    for (kw = 0; kw < param->size; ++kw) {
                                        h_in = -param->pad + h * param->stride +
                                               kh;
                                        w_in = -param->pad + w * param->stride +
                                               kw;
                                        if ((h_in >= 0) &&
                                            (h_in < src_tensor->h) &&
                                            (w_in >= 0) &&
                                            (w_in < src_tensor->w)) {
                                            offset = ((n * dst_tensor->c + c) *
                                                          src_tensor->h +
                                                      h_in) *
                                                         src_tensor->w +
                                                     w_in;
                                            src_tensor->grad_data[offset] +=
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

    return;
}

void bcnn_forward_depthwise_conv_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_forward_depthwise_conv_layer_gpu(net, node);
#else
    return bcnn_forward_depthwise_conv_layer_cpu(net, node);
#endif
}

void bcnn_backward_depthwise_conv_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_backward_depthwise_conv_layer_gpu(net, node);
#else
    return bcnn_backward_depthwise_conv_layer_cpu(net, node);
#endif
}

void bcnn_update_depthwise_conv_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_depthwise_conv_param *param = (bcnn_depthwise_conv_param *)node->param;
    int batch_size = net->batch_size;
    int weights_size = bcnn_tensor_size(weights);
    int biases_size = bcnn_tensor_size(biases);
    switch (net->learner->optimizer) {
        case BCNN_OPTIM_ADAM: {
#ifdef BCNN_USE_CUDA
            bcnn_adam_update_gpu(
                weights->data_gpu, biases->data_gpu, weights->grad_data_gpu,
                biases->grad_data_gpu, param->adam_m_gpu, param->adam_v_gpu,
                weights_size, biases_size, batch_size, net->learner->seen,
                net->learner->beta1, net->learner->beta2,
                net->learner->learning_rate, net->learner->momentum,
                net->learner->decay);
#else
            bcnn_adam_update_cpu(weights->data, biases->data,
                                 weights->grad_data, biases->grad_data,
                                 param->adam_m, param->adam_v, weights_size,
                                 biases_size, batch_size, net->learner->seen,
                                 net->learner->beta1, net->learner->beta2,
                                 net->learner->learning_rate,
                                 net->learner->momentum, net->learner->decay);
#endif
            break;
        }
        case BCNN_OPTIM_SGD: {
#ifdef BCNN_USE_CUDA
            bcnn_sgd_update_gpu(weights->data_gpu, biases->data_gpu,
                                weights->grad_data_gpu, biases->grad_data_gpu,
                                weights_size, biases_size, batch_size,
                                net->learner->learning_rate,
                                net->learner->momentum, net->learner->decay);
#else
            bcnn_sgd_update_cpu(weights->data, biases->data, weights->grad_data,
                                biases->grad_data, weights_size, biases_size,
                                batch_size, net->learner->learning_rate,
                                net->learner->momentum, net->learner->decay);
#endif
            break;
        }
        default: { break; }
    }
}

void bcnn_release_param_depthwise_conv_layer(bcnn_node *node) {
    bcnn_depthwise_conv_param *param = (bcnn_depthwise_conv_param *)node->param;
    bh_free(param->adam_m);
    bh_free(param->adam_v);
#ifdef BCNN_USE_CUDA
    if (param->adam_m_gpu) {
        bcnn_cuda_free(param->adam_m_gpu);
    }
    if (param->adam_v_gpu) {
        bcnn_cuda_free(param->adam_v_gpu);
    }
#endif
    return;
}