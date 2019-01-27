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

int bcnn_sgd_optimizer_graph(bcnn_net *net, bcnn_node *node, int batch_size,
                             float learning_rate, float momentum, float decay) {
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    int weights_size = bcnn_tensor_size(weights);
    bcnn_tensor *biases = NULL;
    int biases_size = 0;
    if (node->type != ACTIVATION) {
        biases = &net->tensors[node->src[2]];
        biases_size = bcnn_tensor_size(biases);
    }
#ifdef BCNN_USE_CUDA
    if (biases) {
        if (biases->data_gpu && biases->grad_data_gpu) {
            bcnn_cuda_axpy(biases_size, -learning_rate / batch_size,
                           biases->grad_data_gpu, 1, biases->data_gpu, 1);
            bcnn_cuda_scal(biases_size, momentum, biases->grad_data_gpu, 1);
        }
    }
    if (weights->data_gpu && weights->grad_data_gpu) {
        bcnn_cuda_axpy(weights_size, decay * batch_size, weights->data_gpu, 1,
                       weights->grad_data_gpu, 1);
        bcnn_cuda_axpy(weights_size, -learning_rate / batch_size,
                       weights->grad_data_gpu, 1, weights->data_gpu, 1);
        bcnn_cuda_scal(weights_size, momentum, weights->grad_data_gpu, 1);
    }
#else
    if (biases) {
        if (biases->data && biases->grad_data) {
            bcnn_axpy(biases_size, -learning_rate / batch_size,
                      biases->grad_data, biases->data);
            bcnn_scal(biases_size, momentum, biases->grad_data);
        }
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

void bcnn_sgd_update_cpu(float *weights, float *biases, float *weights_grad,
                         float *biases_grad, int weights_size, int biases_size,
                         int batch_size, float learning_rate, float momentum,
                         float decay) {
    if (biases && biases_grad) {
        bcnn_axpy(biases_size, -learning_rate / batch_size, biases_grad,
                  biases);
        bcnn_scal(biases_size, momentum, biases_grad);
    }
    if (weights && weights_grad) {
        bcnn_axpy(weights_size, decay * batch_size, weights, weights_grad);
        bcnn_axpy(weights_size, -learning_rate / batch_size, weights_grad,
                  weights);
        bcnn_scal(weights_size, momentum, weights_grad);
    }
    return;
}

#ifdef BCNN_USE_CUDA
void bcnn_sgd_update_gpu(float *weights, float *biases, float *weights_grad,
                         float *biases_grad, int weights_size, int biases_size,
                         int batch_size, float learning_rate, float momentum,
                         float decay) {
    if (biases && biases_grad) {
        bcnn_cuda_axpy(biases_size, -learning_rate / batch_size, biases_grad, 1,
                       biases, 1);
        bcnn_cuda_scal(biases_size, momentum, biases_grad, 1);
    }
    if (weights && weights_grad) {
        bcnn_cuda_axpy(weights_size, decay * batch_size, weights, 1,
                       weights_grad, 1);
        bcnn_cuda_axpy(weights_size, -learning_rate / batch_size, weights_grad,
                       1, weights, 1);
        bcnn_cuda_scal(weights_size, momentum, weights_grad, 1);
    }
    return;
}
#endif

void bcnn_adam_update_cpu(float *weights, float *biases, float *weights_grad,
                          float *biases_grad, float *adam_m, float *adam_v,
                          int weights_size, int biases_size, int batch_size,
                          int iter, float beta1, float beta2,
                          float learning_rate, float momentum, float decay) {
    float mu_correction = sqrtf(1.0f - powf(beta2, (float)iter + 1)) /
                          (1.0f - powf(beta1, (float)iter + 1));
    if (biases && biases_grad) {
        bcnn_axpy(biases_size, -learning_rate / batch_size, biases_grad,
                  biases);
        bcnn_scal(biases_size, momentum, biases_grad);
    }
    if (weights && weights_grad) {
        bcnn_axpy(weights_size, decay * batch_size, weights, weights_grad);
        bcnn_axpby(weights_size, 1.0f - beta1, weights_grad, beta1, adam_m);
        bcnn_vmul(weights_size, weights_grad, weights_grad, weights_grad);
        bcnn_axpby(weights_size, 1.0f - beta2, weights_grad, beta2, adam_v);
        bcnn_pow(weights_size, adam_v, 0.5f, weights_grad);
        bcnn_add_scalar(weights_size, 0.0000001f, weights_grad);
        bcnn_vdiv(weights_size, adam_m, weights_grad, weights_grad);
        bcnn_axpy(weights_size, -learning_rate / batch_size * mu_correction,
                  weights_grad, weights);
        memset(weights_grad, 0, weights_size * sizeof(float));
    }
    return;
}

#ifdef BCNN_USE_CUDA
void bcnn_adam_update_gpu(float *weights, float *biases, float *weights_grad,
                          float *biases_grad, float *adam_m, float *adam_v,
                          int weights_size, int biases_size, int batch_size,
                          int iter, float beta1, float beta2,
                          float learning_rate, float momentum, float decay) {
    float mu_correction = sqrtf(1.0f - powf(beta2, (float)iter + 1)) /
                          (1.0f - powf(beta1, (float)iter + 1));
    if (biases && biases_grad) {
        bcnn_cuda_axpy(biases_size, -learning_rate / batch_size, biases_grad, 1,
                       biases, 1);
        bcnn_cuda_scal(biases_size, momentum, biases_grad, 1);
    }
    if (weights && weights_grad) {
        bcnn_cuda_axpy(weights_size, decay * batch_size, weights, 1,
                       weights_grad, 1);
        bcnn_cuda_axpby(weights_size, 1.0f - beta1, weights_grad, beta1,
                        adam_m);
        bcnn_cuda_vmul(weights_size, weights_grad, weights_grad, weights_grad);
        bcnn_cuda_axpby(weights_size, 1.0f - beta2, weights_grad, beta2,
                        adam_v);
        bcnn_cuda_pow(weights_size, adam_v, 0.5f, weights_grad);
        bcnn_cuda_add_scalar(weights_size, 0.0000001f, weights_grad);
        bcnn_cuda_vdiv(weights_size, adam_m, weights_grad, weights_grad);
        bcnn_cuda_axpy(weights_size,
                       -learning_rate / batch_size * mu_correction,
                       weights_grad, 1, weights, 1);
        bcnn_cuda_fill_f32(weights_size, 0.0f, weights_grad, 1);
    }
    return;
}
#endif

int bcnn_adam_optimizer_graph(bcnn_net *net, bcnn_node *node, int iter,
                              int batch_size, float beta1, float beta2,
                              float learning_rate, float momentum,
                              float decay) {
    bcnn_layer *layer = NULL;
    float mu_correction = sqrtf(1.0f - powf(beta2, (float)iter + 1)) /
                          (1.0f - powf(beta1, (float)iter + 1));
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    int weights_size = bcnn_tensor_size(weights);
    bcnn_tensor *biases = NULL;
    int biases_size = 0;
    if (node->type != ACTIVATION) {
        biases = &net->tensors[node->src[2]];
        biases_size = bcnn_tensor_size(biases);
    }
#ifdef BCNN_USE_CUDA
    if (biases) {
        if (biases->data_gpu && biases->grad_data_gpu) {
            bcnn_cuda_axpy(biases_size, -learning_rate / batch_size,
                           biases->grad_data_gpu, 1, biases->data_gpu, 1);
            bcnn_cuda_scal(biases_size, momentum, biases->grad_data_gpu, 1);
        }
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
    if (biases) {
        if (biases->data && biases->grad_data) {
            bcnn_axpy(biases_size, -learning_rate / batch_size,
                      biases->grad_data, biases->data);
            bcnn_scal(biases_size, momentum, biases->grad_data);
        }
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

    for (int i = 0; i < net->num_nodes; ++i) {
        bcnn_node *node = &net->nodes[i];
        if (node->update) {
            node->update(net, node);
        }
    }
    return BCNN_SUCCESS;
}

#if 0
int bcnn_update(bcnn_net *net) {
    int i;
    float lr = bcnn_update_learning_rate(net);
    bcnn_layer_type type;

    if (net->learner.optimizer == SGD) {
        for (i = 0; i < net->num_nodes; ++i) {
            type = net->nodes[i].layer->type;
            if ((type == CONVOLUTIONAL || type == DECONVOLUTIONAL ||
                 type == DEPTHWISE_CONV || type == FULL_CONNECTED ||
                 (type == ACTIVATION &&
                  net->nodes[i].layer->activation == PRELU))) {
                bcnn_sgd_optimizer_graph(net, &net->nodes[i], net->batch_size,
                                         lr, net->learner.momentum,
                                         net->learner.decay);
            }
        }
    } else if (net->learner.optimizer == ADAM) {
        for (i = 0; i < net->num_nodes; ++i) {
            type = net->nodes[i].layer->type;
            if ((type == CONVOLUTIONAL || type == DECONVOLUTIONAL ||
                 type == DEPTHWISE_CONV || type == FULL_CONNECTED ||
                 (type == ACTIVATION &&
                  net->nodes[i].layer->activation == PRELU))) {
                bcnn_adam_optimizer_graph(
                    net, &net->nodes[i], net->seen, net->batch_size,
                    net->learner.beta1, net->learner.beta2, lr,
                    net->learner.momentum, net->learner.decay);
            }
        }
    }

    return BCNN_SUCCESS;
}
#endif