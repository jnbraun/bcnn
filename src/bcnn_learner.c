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

/* include bh helpers */
#include <bh/bh.h>
#include <bh/bh_error.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>
#include <bh/bh_timer.h>

/* include bip image processing lib */
#include <bip/bip.h>

#include "bcnn/bcnn.h"
#include "bcnn_mat.h"

static float bcnn_update_learning_rate(bcnn_net *net) {
    int iter = net->seen / net->batch_size;

    switch (net->learner.policy) {
        case CONSTANT:
            return net->learner.learning_rate;
        case STEP:
            return net->learner.learning_rate *
                   (float)pow(net->learner.scale, iter / net->learner.step);
        case INV:
            return net->learner.learning_rate *
                   (float)pow(1.0f + net->learner.gamma * iter,
                              -net->learner.power);
        case EXP:
            return net->learner.learning_rate *
                   (float)pow(net->learner.gamma, iter);
        case POLY:
            return net->learner.learning_rate *
                   (float)pow(1 - (float)iter / net->max_batches,
                              net->learner.power);
        case SIGMOID:
            return net->learner.learning_rate *
                   (1.0f / (1.0f + (float)exp(net->learner.gamma *
                                              (iter - net->learner.step))));
        default:
            return net->learner.learning_rate;
    }
}

int bcnn_sgd_optimizer(bcnn_net *net, bcnn_node *node, int batch_size,
                       float learning_rate, float momentum, float decay) {
    bcnn_layer *layer = node->layer;
    int biases_size = bcnn_tensor_size(&layer->biases);
    int weights_size = bcnn_tensor_size(&layer->weights);
#ifdef BCNN_USE_CUDA
    if (layer->biases.data_gpu && layer->biases.grad_data_gpu) {
        bcnn_cuda_axpy(biases_size, -learning_rate / batch_size,
                       layer->biases.grad_data_gpu, 1, layer->biases.data_gpu,
                       1);
        bcnn_cuda_scal(biases_size, momentum, layer->biases.grad_data_gpu, 1);
    }
    if (layer->weights.data_gpu && layer->weights.grad_data_gpu) {
        bcnn_cuda_axpy(weights_size, decay * batch_size,
                       layer->weights.data_gpu, 1, layer->weights.grad_data_gpu,
                       1);
        bcnn_cuda_axpy(weights_size, -learning_rate / batch_size,
                       layer->weights.grad_data_gpu, 1, layer->weights.data_gpu,
                       1);
        bcnn_cuda_scal(weights_size, momentum, layer->weights.grad_data_gpu, 1);
    }
#else
    if (layer->biases.data && layer->biases.grad_data) {
        bcnn_axpy(biases_size, -learning_rate / batch_size,
                  layer->biases.grad_data, layer->biases.data);
        bcnn_scal(biases_size, momentum, layer->biases.grad_data);
    }
    if (layer->weights.data && layer->weights.grad_data) {
        bcnn_axpy(weights_size, decay * batch_size, layer->weights.data,
                  layer->weights.grad_data);
        bcnn_axpy(weights_size, -learning_rate / batch_size,
                  layer->weights.grad_data, layer->weights.data);
        bcnn_scal(weights_size, momentum, layer->weights.grad_data);
    }
#endif
    return 0;
}

int bcnn_sgd_optimizer_graph(bcnn_net *net, bcnn_node *node, int batch_size,
                             float learning_rate, float momentum, float decay) {
    bcnn_layer *layer = node->layer;
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    int biases_size = bcnn_tensor_size(biases);
    int weights_size = bcnn_tensor_size(weights);
#ifdef BCNN_USE_CUDA
    if (biases->data_gpu && biases->grad_data_gpu) {
        bcnn_cuda_axpy(biases_size, -learning_rate / batch_size,
                       biases->grad_data_gpu, 1, biases->data_gpu, 1);
        bcnn_cuda_scal(biases_size, momentum, biases->grad_data_gpu, 1);
    }
    if (weights->data_gpu && weights->grad_data_gpu) {
        bcnn_cuda_axpy(weights_size, decay * batch_size, weights->data_gpu, 1,
                       weights->grad_data_gpu, 1);
        bcnn_cuda_axpy(weights_size, -learning_rate / batch_size,
                       weights->grad_data_gpu, 1, weights->data_gpu, 1);
        bcnn_cuda_scal(weights_size, momentum, weights->grad_data_gpu, 1);
    }
#else
    if (biases->data && biases->grad_data) {
        bcnn_axpy(biases_size, -learning_rate / batch_size, biases->grad_data,
                  biases->data);
        bcnn_scal(biases_size, momentum, biases->grad_data);
    }
    if (weights->data && weights->grad_data) {
        bcnn_axpy(weights_size, decay * batch_size, weights->data,
                  weights->grad_data);
        bcnn_axpy(weights_size, -learning_rate / batch_size, weights->grad_data,
                  weights->data);
        bcnn_scal(weights_size, momentum, weights->grad_data);
    }
#endif
    return 0;
}

int bcnn_adam_optimizer(bcnn_net *net, bcnn_node *node, int iter,
                        int batch_size, float beta1, float beta2,
                        float learning_rate, float momentum, float decay) {
    bcnn_layer *layer = node->layer;
    float mu_correction = sqrtf(1.0f - powf(beta2, (float)iter + 1)) /
                          (1.0f - powf(beta1, (float)iter + 1));
    int biases_size = bcnn_tensor_size(&layer->biases);
    int weights_size = bcnn_tensor_size(&layer->weights);
#ifdef BCNN_USE_CUDA
    if (layer->biases.data_gpu && layer->biases.grad_data_gpu) {
        bcnn_cuda_axpy(biases_size, -learning_rate / batch_size,
                       layer->biases.grad_data_gpu, 1, layer->biases.data_gpu,
                       1);
        bcnn_cuda_scal(biases_size, momentum, layer->biases.grad_data_gpu, 1);
    }
    if (layer->weights.data_gpu && layer->weights.grad_data_gpu) {
        bcnn_cuda_axpy(weights_size, decay * batch_size,
                       layer->weights.data_gpu, 1, layer->weights.grad_data_gpu,
                       1);
        bcnn_cuda_axpby(weights_size, 1.0f - beta1,
                        layer->weights.grad_data_gpu, beta1, layer->adam_m_gpu);
        bcnn_cuda_vmul(weights_size, layer->weights.grad_data_gpu,
                       layer->weights.grad_data_gpu,
                       layer->weights.grad_data_gpu);
        bcnn_cuda_axpby(weights_size, 1.0f - beta2,
                        layer->weights.grad_data_gpu, beta2, layer->adam_v_gpu);
        bcnn_cuda_pow(weights_size, layer->adam_v_gpu, 0.5f,
                      layer->weights.grad_data_gpu);
        bcnn_cuda_add_scalar(weights_size, 0.0000001f,
                             layer->weights.grad_data_gpu);
        bcnn_cuda_vdiv(weights_size, layer->adam_m_gpu,
                       layer->weights.grad_data_gpu,
                       layer->weights.grad_data_gpu);
        bcnn_cuda_axpy(
            weights_size, -learning_rate / batch_size * mu_correction,
            layer->weights.grad_data_gpu, 1, layer->weights.data_gpu, 1);
        bcnn_cuda_fill_f32(weights_size, 0.0f, layer->weights.grad_data_gpu, 1);
    }
#else
    if (layer->biases.data && layer->biases.grad_data) {
        bcnn_axpy(biases_size, -learning_rate / batch_size,
                  layer->biases.grad_data, layer->biases.data);
        bcnn_scal(biases_size, momentum, layer->biases.grad_data);
    }

    if (layer->weights.data && layer->weights.grad_data) {
        bcnn_axpy(weights_size, decay * batch_size, layer->weights.data,
                  layer->weights.grad_data);
        bcnn_axpby(weights_size, 1.0f - beta1, layer->weights.grad_data, beta1,
                   layer->adam_m);
        bcnn_vmul(weights_size, layer->weights.grad_data,
                  layer->weights.grad_data, layer->weights.grad_data);
        bcnn_axpby(weights_size, 1.0f - beta2, layer->weights.grad_data, beta2,
                   layer->adam_v);
        bcnn_pow(weights_size, layer->adam_v, 0.5f, layer->weights.grad_data);
        bcnn_add_scalar(weights_size, 0.0000001f, layer->weights.grad_data);
        bcnn_vdiv(weights_size, layer->adam_m, layer->weights.grad_data,
                  layer->weights.grad_data);
        bcnn_axpy(weights_size, -learning_rate / batch_size * mu_correction,
                  layer->weights.grad_data, layer->weights.data);
        memset(layer->weights.grad_data, 0, weights_size * sizeof(float));
    }
#endif
    return 0;
}

int bcnn_adam_optimizer_graph(bcnn_net *net, bcnn_node *node, int iter,
                              int batch_size, float beta1, float beta2,
                              float learning_rate, float momentum,
                              float decay) {
    bcnn_layer *layer = node->layer;
    float mu_correction = sqrtf(1.0f - powf(beta2, (float)iter + 1)) /
                          (1.0f - powf(beta1, (float)iter + 1));
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    int biases_size = bcnn_tensor_size(biases);
    int weights_size = bcnn_tensor_size(weights);
#ifdef BCNN_USE_CUDA
    if (biases->data_gpu && biases->grad_data_gpu) {
        bcnn_cuda_axpy(biases_size, -learning_rate / batch_size,
                       biases->grad_data_gpu, 1, biases->data_gpu, 1);
        bcnn_cuda_scal(biases_size, momentum, biases->grad_data_gpu, 1);
    }
    if (weights->data_gpu && weights->grad_data_gpu) {
        bcnn_cuda_axpy(weights_size, decay * batch_size, weights->data_gpu, 1,
                       weights->grad_data_gpu, 1);
        bcnn_cuda_axpby(weights_size, 1.0f - beta1, weights->grad_data_gpu,
                        beta1, layer->adam_m_gpu);
        bcnn_cuda_vmul(weights_size, weights->grad_data_gpu,
                       weights->grad_data_gpu, weights->grad_data_gpu);
        bcnn_cuda_axpby(weights_size, 1.0f - beta2, weights->grad_data_gpu,
                        beta2, layer->adam_v_gpu);
        bcnn_cuda_pow(weights_size, layer->adam_v_gpu, 0.5f,
                      weights->grad_data_gpu);
        bcnn_cuda_add_scalar(weights_size, 0.0000001f, weights->grad_data_gpu);
        bcnn_cuda_vdiv(weights_size, layer->adam_m_gpu, weights->grad_data_gpu,
                       weights->grad_data_gpu);
        bcnn_cuda_axpy(weights_size,
                       -learning_rate / batch_size * mu_correction,
                       weights->grad_data_gpu, 1, weights->data_gpu, 1);
        bcnn_cuda_fill_f32(weights_size, 0.0f, weights->grad_data_gpu, 1);
    }
#else
    if (biases->data && biases->grad_data) {
        bcnn_axpy(biases_size, -learning_rate / batch_size, biases->grad_data,
                  biases->data);
        bcnn_scal(biases_size, momentum, biases->grad_data);
    }

    if (weights->data && weights->grad_data) {
        bcnn_axpy(weights_size, decay * batch_size, weights->data,
                  weights->grad_data);
        bcnn_axpby(weights_size, 1.0f - beta1, weights->grad_data, beta1,
                   layer->adam_m);
        bcnn_vmul(weights_size, weights->grad_data, weights->grad_data,
                  weights->grad_data);
        bcnn_axpby(weights_size, 1.0f - beta2, weights->grad_data, beta2,
                   layer->adam_v);
        bcnn_pow(weights_size, layer->adam_v, 0.5f, weights->grad_data);
        bcnn_add_scalar(weights_size, 0.0000001f, weights->grad_data);
        bcnn_vdiv(weights_size, layer->adam_m, weights->grad_data,
                  weights->grad_data);
        bcnn_axpy(weights_size, -learning_rate / batch_size * mu_correction,
                  weights->grad_data, weights->data);
        memset(weights->grad_data, 0, weights_size * sizeof(float));
    }
#endif
    return 0;
}

int bcnn_update(bcnn_net *net) {
    int i;
    float lr = bcnn_update_learning_rate(net);
    bcnn_layer_type type;

    if (net->learner.optimizer == SGD) {
        for (i = 0; i < net->num_nodes; ++i) {
            type = net->nodes[i].layer->type;
#ifndef GRAPH_TOPOLOGY
            if ((type == CONVOLUTIONAL || type == DECONVOLUTIONAL ||
                 type == DEPTHWISE_CONV || type == FULL_CONNECTED ||
                 (type == ACTIVATION &&
                  net->nodes[i].layer->activation == PRELU))) {
                bcnn_sgd_optimizer(net, &net->nodes[i], net->batch_size, lr,
                                   net->learner.momentum, net->learner.decay);
            }
#else
            if ((type == CONVOLUTIONAL || type == DECONVOLUTIONAL ||
                 type == DEPTHWISE_CONV || type == FULL_CONNECTED ||
                 (type == ACTIVATION &&
                  net->nodes[i].layer->activation == PRELU))) {
                bcnn_sgd_optimizer_graph(net, &net->nodes[i], net->batch_size,
                                         lr, net->learner.momentum,
                                         net->learner.decay);
            }
#endif  // GRAPH_TOPOLOGY
        }
    } else if (net->learner.optimizer == ADAM) {
        for (i = 0; i < net->num_nodes; ++i) {
            type = net->nodes[i].layer->type;
#ifndef GRAPH_TOPOLOGY
            if ((type == CONVOLUTIONAL || type == DECONVOLUTIONAL ||
                 type == DEPTHWISE_CONV || type == FULL_CONNECTED ||
                 (type == ACTIVATION &&
                  net->nodes[i].layer->activation == PRELU))) {
                bcnn_adam_optimizer(net, &net->nodes[i], net->seen,
                                    net->batch_size, net->learner.beta1,
                                    net->learner.beta2, lr,
                                    net->learner.momentum, net->learner.decay);
            }
#else
            if ((type == CONVOLUTIONAL || type == DECONVOLUTIONAL ||
                 type == DEPTHWISE_CONV || type == FULL_CONNECTED ||
                 (type == ACTIVATION &&
                  net->nodes[i].layer->activation == PRELU))) {
                bcnn_adam_optimizer_graph(
                    net, &net->nodes[i], net->seen, net->batch_size,
                    net->learner.beta1, net->learner.beta2, lr,
                    net->learner.momentum, net->learner.decay);
            }
#endif  // GRAPH_TOPOLOGY
        }
    }

    return BCNN_SUCCESS;
}