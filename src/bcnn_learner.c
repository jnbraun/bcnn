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

#include <math.h>
#include <string.h>

#include "bcnn/bcnn.h"
#include "bcnn_mat.h"

static float bcnn_update_learning_rate(bcnn_net *net) {
    net->learner.seen += net->batch_size;
    int iter = net->learner.seen / net->batch_size;

    switch (net->learner.decay_type) {
        case BCNN_LR_DECAY_CONSTANT:
            return net->learner.learning_rate;
        case BCNN_LR_DECAY_STEP:
            return net->learner.learning_rate *
                   (float)pow(net->learner.scale, iter / net->learner.step);
        case BCNN_LR_DECAY_INV:
            return net->learner.learning_rate *
                   (float)pow(1.0f + net->learner.gamma * iter,
                              -net->learner.power);
        case BCNN_LR_DECAY_EXP:
            return net->learner.learning_rate *
                   (float)pow(net->learner.gamma, iter);
        case BCNN_LR_DECAY_POLY:
            return net->learner.learning_rate *
                   (float)pow(1 - (float)iter / net->learner.max_batches,
                              net->learner.power);
        case BCNN_LR_DECAY_SIGMOID:
            return net->learner.learning_rate *
                   (1.0f / (1.0f + (float)exp(net->learner.gamma *
                                              (iter - net->learner.step))));
        default:
            return net->learner.learning_rate;
    }
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

void bcnn_update(bcnn_net *net) {
    float lr = bcnn_update_learning_rate(net);

    for (int i = 0; i < net->num_nodes; ++i) {
        bcnn_node *node = &net->nodes[i];
        if (node->update) {
            node->update(net, node);
        }
    }
}