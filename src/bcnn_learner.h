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

#ifndef BCNN_LEARNER_H
#define BCNN_LEARNER_H

/**
 *  Structure to handle learner method and parameters.
 */
typedef struct {
    int step;
    int seen;            /* Number of instances seen by the network */
    int max_batches;     /* Maximum number of batches for training */
    float momentum;      /* Momentum parameter */
    float decay;         /* Decay parameter */
    float learning_rate; /* Base learning rate */
    float gamma;
    float scale;
    float power;
    float beta1;              /* Parameter for Adam optimizer */
    float beta2;              /* Parameter for Adam optimizer */
    bcnn_optimizer optimizer; /* Optimization method */
    bcnn_lr_decay decay_type; /* Learning rate decay type */
} bcnn_learner;

void bcnn_sgd_update_cpu(float *weights, float *biases, float *weights_grad,
                         float *biases_grad, int weights_size, int biases_size,
                         int batch_size, float learning_rate, float momentum,
                         float decay);
void bcnn_adam_update_cpu(float *weights, float *biases, float *weights_grad,
                          float *biases_grad, float *adam_m, float *adam_v,
                          int weights_size, int biases_size, int batch_size,
                          int iter, float beta1, float beta2,
                          float learning_rate, float momentum, float decay);

#ifdef BCNN_USE_CUDA
void bcnn_sgd_update_gpu(float *weights, float *biases, float *weights_grad,
                         float *biases_grad, int weights_size, int biases_size,
                         int batch_size, float learning_rate, float momentum,
                         float decay);
void bcnn_adam_update_gpu(float *weights, float *biases, float *weights_grad,
                          float *biases_grad, float *adam_m, float *adam_v,
                          int weights_size, int biases_size, int batch_size,
                          int iter, float beta1, float beta2,
                          float learning_rate, float momentum, float decay);
#endif

#endif  // BCNN_LEARNER_H