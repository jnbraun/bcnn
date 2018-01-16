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


static float bcnn_update_learning_rate(bcnn_net *net)
{
    int iter = net->seen / net->batch_size;

    switch (net->learner.policy) {
        case CONSTANT:
            return net->learner.learning_rate;
        case STEP:
            return net->learner.learning_rate * (float)pow(net->learner.scale, iter / net->learner.step);
        case INV:
            return net->learner.learning_rate * (float)pow(1.0f + net->learner.gamma * iter, -net->learner.power);
        case EXP:
            return net->learner.learning_rate * (float)pow(net->learner.gamma, iter);
        case POLY:
            return net->learner.learning_rate * (float)pow(1 - (float)iter / net->max_batches, net->learner.power);
        case SIGMOID:
            return net->learner.learning_rate * (1.0f / (1.0f + (float)exp(net->learner.gamma * (iter - net->learner.step))));
        default:
            return net->learner.learning_rate;
    }
}


int bcnn_sgd_optimizer(bcnn_connection *conn, int batch_size, float learning_rate, float momentum, float decay)
{
    bcnn_layer *layer = conn->layer;

#ifdef BCNN_USE_CUDA
    if (layer->bias_gpu && layer->bias_diff_gpu) {
        bcnn_cuda_axpy(layer->bias_size, -learning_rate / batch_size, layer->bias_diff_gpu, 1, layer->bias_gpu, 1);
        bcnn_cuda_scal(layer->bias_size, momentum, layer->bias_diff_gpu, 1);
    }
    if (layer->weight_gpu && layer->weight_diff_gpu) {
        bcnn_cuda_axpy(layer->weights_size, decay * batch_size, layer->weight_gpu, 1, layer->weight_diff_gpu, 1);
        bcnn_cuda_axpy(layer->weights_size, -learning_rate / batch_size, layer->weight_diff_gpu, 1, layer->weight_gpu, 1);
        bcnn_cuda_scal(layer->weights_size, momentum, layer->weight_diff_gpu, 1);
    }
#else
    if (layer->bias && layer->bias_diff) {
        bcnn_axpy(layer->bias_size, -learning_rate / batch_size, layer->bias_diff, layer->bias);
        bcnn_scal(layer->bias_size, momentum, layer->bias_diff);
    }
    if (layer->weight && layer->weight_diff) {
        bcnn_axpy(layer->weights_size, decay * batch_size, layer->weight, layer->weight_diff);
        bcnn_axpy(layer->weights_size, -learning_rate / batch_size, layer->weight_diff, layer->weight);
        bcnn_scal(layer->weights_size, momentum, layer->weight_diff);
    }
#endif
    return 0;
}


int bcnn_adam_optimizer(bcnn_connection *conn, int iter, int batch_size, float beta1, float beta2, float learning_rate, float momentum, float decay)
{
    bcnn_layer *layer = conn->layer;
    float mu_correction = sqrtf(1.0f - powf(beta2, (float)iter + 1)) / (1.0f - powf(beta1, (float)iter + 1));

#ifdef BCNN_USE_CUDA
    if (layer->bias_gpu && layer->bias_diff_gpu) {
        bcnn_cuda_axpy(layer->bias_size, -learning_rate / batch_size, layer->bias_diff_gpu, 1, layer->bias_gpu, 1);
        bcnn_cuda_scal(layer->bias_size, momentum, layer->bias_diff_gpu, 1);
    }
    if (layer->weight_gpu && layer->weight_diff_gpu) {
        bcnn_cuda_axpy(layer->weights_size, decay * batch_size, layer->weight_gpu, 1, layer->weight_diff_gpu, 1);
        bcnn_cuda_axpby(layer->weights_size, 1.0f - beta1, layer->weight_diff_gpu, beta1, layer->adam_m_gpu);
        bcnn_cuda_vmul(layer->weights_size, layer->weight_diff_gpu, layer->weight_diff_gpu, layer->weight_diff_gpu);
        bcnn_cuda_axpby(layer->weights_size, 1.0f - beta2, layer->weight_diff_gpu, beta2, layer->adam_v_gpu);
        bcnn_cuda_pow(layer->weights_size, layer->adam_v_gpu, 0.5f, layer->weight_diff_gpu);
        bcnn_cuda_add_scalar(layer->weights_size, 0.0000001f, layer->weight_diff_gpu);
        bcnn_cuda_vdiv(layer->weights_size, layer->adam_m_gpu, layer->weight_diff_gpu, layer->weight_diff_gpu);
        bcnn_cuda_axpy(layer->weights_size, -learning_rate / batch_size * mu_correction, layer->weight_diff_gpu, 1, layer->weight_gpu, 1);
        bcnn_cuda_fill_f32(layer->weights_size, 0.0f, layer->weight_diff_gpu, 1);
    }
#else
    if (layer->bias && layer->bias_diff) {
        bcnn_axpy(layer->bias_size, -learning_rate / batch_size, layer->bias_diff, layer->bias);
        bcnn_scal(layer->bias_size, momentum, layer->bias_diff);
    }

    if (layer->weight && layer->weight_diff) {
        bcnn_axpy(layer->weights_size, decay * batch_size, layer->weight, layer->weight_diff);
        bcnn_axpby(layer->weights_size, 1.0f - beta1, layer->weight_diff, beta1, layer->adam_m);
        bcnn_vmul(layer->weights_size, layer->weight_diff, layer->weight_diff, layer->weight_diff);
        bcnn_axpby(layer->weights_size, 1.0f - beta2, layer->weight_diff, beta2, layer->adam_v);
        bcnn_pow(layer->weights_size, layer->adam_v, 0.5f, layer->weight_diff);
        bcnn_add_scalar(layer->weights_size, 0.0000001f, layer->weight_diff);
        bcnn_vdiv(layer->weights_size, layer->adam_m, layer->weight_diff, layer->weight_diff);	
        bcnn_axpy(layer->weights_size, -learning_rate / batch_size * mu_correction, layer->weight_diff, layer->weight);
        memset(layer->weight_diff, 0, layer->weights_size * sizeof(float));
    }
#endif
    return 0;
}


int bcnn_update(bcnn_net *net)
{
    int i;
    float lr = bcnn_update_learning_rate(net);
    bcnn_layer_type	type;

    if (net->learner.optimizer == SGD) {
        for (i = 0; i < net->nb_connections; ++i) {
            type = net->connections[i].layer->type;
            if ((type == CONVOLUTIONAL || 
                type == DECONVOLUTIONAL ||
                type == DEPTHWISE_CONV ||
                type == FULL_CONNECTED)) {
                bcnn_sgd_optimizer(&net->connections[i],
                        net->batch_size, lr, net->learner.momentum, net->learner.decay);
            }
        }
    }
    else if (net->learner.optimizer == ADAM) {
        for (i = 0; i < net->nb_connections; ++i) {
            type = net->connections[i].layer->type;
            if ((type == CONVOLUTIONAL || 
                type == DECONVOLUTIONAL || 
                type == DEPTHWISE_CONV ||
                type == FULL_CONNECTED)) {
                bcnn_adam_optimizer(&net->connections[i], net->seen, net->batch_size, net->learner.beta1, net->learner.beta2,
                    lr, net->learner.momentum, net->learner.decay);
            }
        }
    }

    return BCNN_SUCCESS;
}