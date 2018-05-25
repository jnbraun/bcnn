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

#ifdef BCNN_USE_CUDA

#include "bcnn_batchnorm_layer.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"

__global__ void fast_mean_kernel(float *x, int batch, int filters, int spatial,
                                 float *mean) {
    const int threads = 512;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for (j = 0; j < batch; ++j) {
        for (i = 0; i < spatial; i += threads) {
            int index = j * spatial * filters + filter * spatial + i + id;
            local[id] += (i + id < spatial) ? x[index] : 0;
        }
    }

    __syncthreads();

    if (id == 0) {
        mean[filter] = 0;
        for (i = 0; i < threads; ++i) {
            mean[filter] += local[i];
        }
        mean[filter] /= spatial * batch;
    }
}

__global__ void fast_variance_kernel(float *x, float *mean, int batch,
                                     int filters, int spatial,
                                     float *variance) {
    const int threads = 512;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for (j = 0; j < batch; ++j) {
        for (i = 0; i < spatial; i += threads) {
            int index = j * spatial * filters + filter * spatial + i + id;

            local[id] +=
                (i + id < spatial)
                    ? ((x[index] - mean[filter]) * (x[index] - mean[filter]))
                    : 0;
        }
    }

    __syncthreads();

    if (id == 0) {
        variance[filter] = 0;
        for (i = 0; i < threads; ++i) {
            variance[filter] += local[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}

extern "C" void fast_mean_gpu(float *x, int batch, int filters, int spatial,
                              float *mean) {
    fast_mean_kernel<<<filters, BCNN_CUDA_THREADS>>>(x, batch, filters, spatial,
                                                     mean);
    bcnn_cuda_check(cudaPeekAtLastError());
}

extern "C" void fast_variance_gpu(float *x, float *mean, int batch, int filters,
                                  int spatial, float *variance) {
    fast_variance_kernel<<<filters, BCNN_CUDA_THREADS>>>(
        x, mean, batch, filters, spatial, variance);
    bcnn_cuda_check(cudaPeekAtLastError());
}

__global__ void fast_mean_delta_kernel(float *delta, float *variance, int batch,
                                       int filters, int spatial,
                                       float *mean_delta) {
    const int threads = 512;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for (j = 0; j < batch; ++j) {
        for (i = 0; i < spatial; i += threads) {
            int index = j * spatial * filters + filter * spatial + i + id;
            local[id] += (i + id < spatial) ? delta[index] : 0;
        }
    }

    __syncthreads();

    if (id == 0) {
        mean_delta[filter] = 0;
        for (i = 0; i < threads; ++i) {
            mean_delta[filter] += local[i];
        }
        mean_delta[filter] *= (-1.f / sqrtf(variance[filter] + .00001f));
    }
}

__global__ void fast_variance_delta_kernel(float *x, float *delta, float *mean,
                                           float *variance, int batch,
                                           int filters, int spatial,
                                           float *variance_delta) {
    const int threads = 512;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for (j = 0; j < batch; ++j) {
        for (i = 0; i < spatial; i += threads) {
            int index = j * spatial * filters + filter * spatial + i + id;

            local[id] += (i + id < spatial)
                             ? delta[index] * (x[index] - mean[filter])
                             : 0;
        }
    }

    __syncthreads();

    if (id == 0) {
        variance_delta[filter] = 0;
        for (i = 0; i < threads; ++i) {
            variance_delta[filter] += local[i];
        }
        variance_delta[filter] *=
            -.5f * powf(variance[filter] + .00001f, (float)(-3.f / 2.f));
    }
}

extern "C" void fast_mean_delta_gpu(float *delta, float *variance, int batch,
                                    int filters, int spatial,
                                    float *mean_delta) {
    fast_mean_delta_kernel<<<filters, BCNN_CUDA_THREADS>>>(
        delta, variance, batch, filters, spatial, mean_delta);
    bcnn_cuda_check(cudaPeekAtLastError());
}

extern "C" void fast_variance_delta_gpu(float *x, float *delta, float *mean,
                                        float *variance, int batch, int filters,
                                        int spatial, float *variance_delta) {
    fast_variance_delta_kernel<<<filters, BCNN_CUDA_THREADS>>>(
        x, delta, mean, variance, batch, filters, spatial, variance_delta);
    bcnn_cuda_check(cudaPeekAtLastError());
}

int bcnn_forward_batchnorm_layer_gpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                     bcnn_tensor *dst_tensor) {
    
    
#ifndef BCNN_USE_CUDNN
    int batch_size = src_tensor->n;
    int sz = dst_tensor->w * dst_tensor->h * dst_tensor->c;
#else
    float alpha = 1.0f;
    float beta = 0.0f;
#endif

    if (layer->net_state) {
#ifdef BCNN_USE_CUDNN
        bcnn_cudnn_check(cudnnBatchNormalizationForwardTraining(
            bcnn_cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
            layer->dst_tensor_desc, src_tensor->data_gpu, layer->dst_tensor_desc,
            dst_tensor->data_gpu, layer->bias_desc, layer->scales.data_gpu,
            layer->biases.data_gpu, 0.1, layer->running_mean.data_gpu,
            layer->running_variance.data_gpu, 0.0001,
            layer->saved_mean.data_gpu, layer->saved_variance.data_gpu));
#else
        bcnn_cuda_copy_f32(sz * batch_size, src_tensor->data_gpu, 1, dst_tensor->data_gpu, 1);
        bcnn_cuda_copy_f32(sz * batch_size, dst_tensor->data_gpu, 1,
                           layer->bn_workspace_gpu, 1);

        // bcnn_cuda_mean_variance_forward(dst_tensor->data_gpu, batch_size, dst_tensor->c,
        // dst_tensor->h * dst_tensor->w, layer->saved_mean.data_gpu,
        // layer->saved_variance.data_gpu);
        fast_mean_gpu(dst_tensor->data_gpu, batch_size, dst_tensor->c, dst_tensor->h * dst_tensor->w,
                      layer->saved_mean.data_gpu);
        fast_variance_gpu(dst_tensor->data_gpu, layer->saved_mean.data_gpu, batch_size,
                          dst_tensor->c, dst_tensor->h * dst_tensor->w, layer->saved_variance.data_gpu);

        bcnn_cuda_scal(dst_tensor->c, 0.9f, layer->running_mean.data_gpu, 1);
        bcnn_cuda_axpy(dst_tensor->c, 0.1f, layer->saved_mean.data_gpu, 1,
                       layer->running_mean.data_gpu, 1);
        bcnn_cuda_scal(dst_tensor->c, 0.9f, layer->running_variance.data_gpu, 1);
        bcnn_cuda_axpy(dst_tensor->c, 0.1f, layer->saved_variance.data_gpu, 1,
                       layer->running_variance.data_gpu, 1);

        // bcnn_cuda_copy_f32(batch_size * sz, dst_tensor->data_gpu, 1,
        // layer->bn_workspace_gpu, 1);
        bcnn_cuda_norm_forward(dst_tensor->data_gpu, layer->saved_mean.data_gpu,
                               layer->saved_variance.data_gpu, batch_size,
                               dst_tensor->c, dst_tensor->h * dst_tensor->w);
        bcnn_cuda_copy_f32(batch_size * sz, dst_tensor->data_gpu, 1, layer->x_norm_gpu,
                           1);
        bcnn_scales_gpu(dst_tensor->data_gpu, layer->scales.data_gpu, batch_size,
                               dst_tensor->c, dst_tensor->h * dst_tensor->w);
        bcnn_cuda_add_bias(dst_tensor->data_gpu, layer->biases.data_gpu, batch_size,
                               dst_tensor->c, dst_tensor->h * dst_tensor->w);
#endif
    } else {
#ifdef BCNN_USE_CUDNN
        bcnn_cudnn_check(cudnnBatchNormalizationForwardInference(
            bcnn_cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
            layer->dst_tensor_desc, src_tensor->data_gpu, layer->dst_tensor_desc,
            dst_tensor->data_gpu, layer->bias_desc, layer->scales.data_gpu,
            layer->biases.data_gpu, layer->running_mean.data_gpu,
            layer->running_variance.data_gpu, 0.0001));
#else
        bcnn_cuda_copy_f32(sz * batch_size, src_tensor->data_gpu, 1, dst_tensor->data_gpu, 1);
        bcnn_cuda_copy_f32(sz * batch_size, dst_tensor->data_gpu, 1,
                           layer->bn_workspace_gpu, 1);
        // Normalize with global mean / variance
        bcnn_cuda_norm_forward(dst_tensor->data_gpu, layer->running_mean.data_gpu,
                               layer->running_variance.data_gpu, batch_size,
                               dst_tensor->c, dst_tensor->h * dst_tensor->w);
        bcnn_scales_gpu(dst_tensor->data_gpu, layer->scales.data_gpu, batch_size,
                               dst_tensor->c, dst_tensor->h * dst_tensor->w);
        bcnn_cuda_add_bias(dst_tensor->data_gpu, layer->biases.data_gpu, batch_size,
                               dst_tensor->c, dst_tensor->h * dst_tensor->w);
#endif
    }

    return BCNN_SUCCESS;
}

int bcnn_backward_batchnorm_layer_gpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                      bcnn_tensor *dst_tensor) {
    
    
#ifndef BCNN_USE_CUDNN
    int batch_size = src_tensor->n;
    int sz = dst_tensor->w * dst_tensor->h * dst_tensor->c;
#else
    float a_data = 1.0f, a_param = 1.0f;
    float b_data = 0.0f, b_param = 1.0f;
#endif

    if (!layer->net_state) {
        layer->saved_mean.data_gpu = layer->running_mean.data_gpu;
        layer->saved_variance.data_gpu = layer->running_variance.data_gpu;
    }

#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnBatchNormalizationBackward(
        bcnn_cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, &a_data, &b_data,
        &a_param, &b_param, layer->dst_tensor_desc, src_tensor->data_gpu,
        layer->dst_tensor_desc, dst_tensor->grad_data_gpu, layer->dst_tensor_desc,
        src_tensor->grad_data_gpu, layer->bias_desc, layer->scales.data_gpu,
        layer->scales.grad_data_gpu, layer->biases.grad_data_gpu, 0.0001,
        layer->saved_mean.data_gpu, layer->saved_variance.data_gpu));
#else
    bcnn_cuda_grad_bias(layer->biases.grad_data_gpu, dst_tensor->grad_data_gpu, batch_size,
                   dst_tensor->c, dst_tensor->h * dst_tensor->w);
    bcnn_grad_scales_gpu(layer->x_norm_gpu, dst_tensor->grad_data_gpu, batch_size,
                     dst_tensor->c, dst_tensor->h * dst_tensor->w,
                     layer->scales.grad_data_gpu);
    bcnn_scales_gpu(dst_tensor->grad_data_gpu, layer->scales.data_gpu, batch_size,
                dst_tensor->c, dst_tensor->h * dst_tensor->w);

    fast_mean_delta_gpu(dst_tensor->grad_data_gpu, layer->saved_variance.data_gpu,
                        batch_size, dst_tensor->c, dst_tensor->w * dst_tensor->h,
                        layer->saved_mean.grad_data_gpu);
    fast_variance_delta_gpu(layer->bn_workspace_gpu, dst_tensor->grad_data_gpu,
                            layer->saved_mean.data_gpu,
                            layer->saved_variance.data_gpu, batch_size, dst_tensor->c,
                            dst_tensor->w * dst_tensor->h, layer->saved_variance.grad_data_gpu);
    bcnn_cuda_norm_backward(layer->bn_workspace_gpu, layer->saved_mean.data_gpu,
                            layer->saved_variance.data_gpu,
                            layer->saved_mean.grad_data_gpu,
                            layer->saved_variance.grad_data_gpu, src_tensor->n, dst_tensor->c,
                            dst_tensor->w * dst_tensor->h, dst_tensor->grad_data_gpu);

    if (src_tensor->grad_data_gpu)
        bcnn_cuda_copy_f32(sz * batch_size, dst_tensor->grad_data_gpu, 1,
                           src_tensor->grad_data_gpu, 1);
#endif

    return BCNN_SUCCESS;
}

#endif