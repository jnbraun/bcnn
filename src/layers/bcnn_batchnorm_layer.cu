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

#ifdef BCNN_USE_CUDA

#include "bcnn_batchnorm_layer.h"
#include "bcnn_mat.h"

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

void bcnn_forward_batchnorm_gpu(bcnn_tensor *src_tensor,
                                bcnn_tensor *dst_tensor, bcnn_tensor *bn_mean,
                                bcnn_tensor *bn_var, bcnn_tensor *bn_scales,
                                bcnn_tensor *bn_biases, bcnn_tensor *saved_mean,
                                bcnn_tensor *saved_variance, float *x_norm_gpu,
                                float *workspace_gpu, bcnn_mode mode
#ifdef BCNN_USE_CUDNN
                                ,
                                cudnnTensorDescriptor_t dst_tensor_desc,
                                cudnnTensorDescriptor_t bias_desc
#endif
                                ) {
    int batch_size = src_tensor->n;
    int sz = dst_tensor->w * dst_tensor->h * dst_tensor->c;
#ifdef BCNN_USE_CUDNN
    float alpha = 1.0f;
    float beta = 0.0f;
#endif

    if (src_tensor != dst_tensor) {
        bcnn_cuda_copy_f32(sz * batch_size, (float *)src_tensor->data_gpu, 1,
                           (float *)dst_tensor->data_gpu, 1);
    }
    bcnn_cuda_copy_f32(sz * batch_size, (float *)dst_tensor->data_gpu, 1,
                       workspace_gpu, 1);

    if (mode == BCNN_MODE_TRAIN) {
#ifdef BCNN_USE_CUDNN
        bcnn_cudnn_check(cudnnBatchNormalizationForwardTraining(
            bcnn_cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
            dst_tensor_desc,
            /*src_tensor->data_gpu*/ workspace_gpu, dst_tensor_desc,
            (float *)dst_tensor->data_gpu, bias_desc,
            (float *)bn_scales->data_gpu, (float *)bn_biases->data_gpu, 0.1,
            (float *)bn_mean->data_gpu, (float *)bn_var->data_gpu, 0.0001,
            (float *)saved_mean->data_gpu, (float *)saved_variance->data_gpu));
#else
        fast_mean_gpu((float *)dst_tensor->data_gpu, batch_size, dst_tensor->c,
                      dst_tensor->h * dst_tensor->w,
                      (float *)saved_mean->data_gpu);
        fast_variance_gpu((float *)dst_tensor->data_gpu,
                          (float *)saved_mean->data_gpu, batch_size,
                          dst_tensor->c, dst_tensor->h * dst_tensor->w,
                          (float *)saved_variance->data_gpu);

        bcnn_cuda_scal(dst_tensor->c, 0.9f, (float *)bn_mean->data_gpu, 1);
        bcnn_cuda_axpy(dst_tensor->c, 0.1f, (float *)saved_mean->data_gpu, 1,
                       (float *)bn_mean->data_gpu, 1);
        bcnn_cuda_scal(dst_tensor->c, 0.9f, (float *)bn_var->data_gpu, 1);
        bcnn_cuda_axpy(dst_tensor->c, 0.1f, (float *)saved_variance->data_gpu,
                       1, (float *)bn_var->data_gpu, 1);
        bcnn_cuda_norm_forward((float *)dst_tensor->data_gpu,
                               (float *)saved_mean->data_gpu,
                               (float *)saved_variance->data_gpu, batch_size,
                               dst_tensor->c, dst_tensor->h * dst_tensor->w);
        bcnn_cuda_copy_f32(batch_size * sz, (float *)dst_tensor->data_gpu, 1,
                           x_norm_gpu, 1);
        bcnn_scales_gpu((float *)dst_tensor->data_gpu,
                        (float *)bn_scales->data_gpu, batch_size, dst_tensor->c,
                        dst_tensor->h * dst_tensor->w);
        bcnn_cuda_add_bias((float *)dst_tensor->data_gpu,
                           (float *)bn_biases->data_gpu, batch_size,
                           dst_tensor->c, dst_tensor->h * dst_tensor->w);
#endif
    } else {
#ifdef BCNN_USE_CUDNN
        bcnn_cudnn_check(cudnnBatchNormalizationForwardInference(
            bcnn_cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
            dst_tensor_desc, (float *)src_tensor->data_gpu, dst_tensor_desc,
            (float *)dst_tensor->data_gpu, bias_desc,
            (float *)bn_scales->data_gpu, (float *)bn_biases->data_gpu,
            (float *)bn_mean->data_gpu, (float *)bn_var->data_gpu, 0.0001));
#else
        bcnn_cuda_copy_f32(sz * batch_size, (float *)src_tensor->data_gpu, 1,
                           (float *)dst_tensor->data_gpu, 1);
        bcnn_cuda_copy_f32(sz * batch_size, (float *)dst_tensor->data_gpu, 1,
                           workspace_gpu, 1);
        // Normalize with global mean / variance
        bcnn_cuda_norm_forward((float *)dst_tensor->data_gpu,
                               (float *)bn_mean->data_gpu,
                               (float *)bn_var->data_gpu, batch_size,
                               dst_tensor->c, dst_tensor->h * dst_tensor->w);
        bcnn_scales_gpu((float *)dst_tensor->data_gpu,
                        (float *)bn_scales->data_gpu, batch_size, dst_tensor->c,
                        dst_tensor->h * dst_tensor->w);
        bcnn_cuda_add_bias((float *)dst_tensor->data_gpu,
                           (float *)bn_biases->data_gpu, batch_size,
                           dst_tensor->c, dst_tensor->h * dst_tensor->w);
#endif
    }
    return;
}

void bcnn_forward_batchnorm_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *bn_mean = &net->tensors[node->src[1]];
    bcnn_tensor *bn_var = &net->tensors[node->src[2]];
    bcnn_tensor *bn_scales = &net->tensors[node->src[3]];
    bcnn_tensor *bn_biases = &net->tensors[node->src[4]];
    bcnn_batchnorm_param *param = (bcnn_batchnorm_param *)node->param;

    bcnn_forward_batchnorm_gpu(src_tensor, dst_tensor, bn_mean, bn_var,
                               bn_scales, bn_biases, &param->saved_mean,
                               &param->saved_variance, param->x_norm_gpu,
                               param->workspace_gpu, net->mode
#ifdef BCNN_USE_CUDNN
                               ,
                               param->dst_tensor_desc, param->bias_desc
#endif
                               );
    return;
}

void bcnn_backward_batchnorm_gpu(
    bcnn_tensor *src_tensor, bcnn_tensor *dst_tensor, bcnn_tensor *bn_mean,
    bcnn_tensor *bn_var, bcnn_tensor *bn_scales, bcnn_tensor *bn_biases,
    bcnn_tensor *saved_mean, bcnn_tensor *saved_variance, float *x_norm_gpu,
    float *workspace_gpu, bcnn_mode mode
#ifdef BCNN_USE_CUDNN
    ,
    cudnnTensorDescriptor_t dst_tensor_desc, cudnnTensorDescriptor_t bias_desc
#endif
    ) {
    int batch_size = src_tensor->n;
    int sz = dst_tensor->w * dst_tensor->h * dst_tensor->c;
#ifdef BCNN_USE_CUDNN
    float a_data = 1.0f, a_param = 1.0f;
    float b_data = 0.0f, b_param = 1.0f;
#endif
    if (mode != BCNN_MODE_TRAIN) {
        saved_mean->data_gpu = bn_mean->data_gpu;
        saved_variance->data_gpu = bn_var->data_gpu;
    }

#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnBatchNormalizationBackward(
        bcnn_cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, &a_data, &b_data,
        &a_param, &b_param, dst_tensor_desc,
        /*src_tensor->data_gpu*/ workspace_gpu, dst_tensor_desc,
        (float *)dst_tensor->grad_data_gpu, dst_tensor_desc,
        /*(float *src_tensor->grad_data_gpu*/ x_norm_gpu, bias_desc,
        (float *)bn_scales->data_gpu, (float *)bn_scales->grad_data_gpu,
        (float *)bn_biases->grad_data_gpu, 0.0001,
        (float *)saved_mean->data_gpu, (float *)saved_variance->data_gpu));
    bcnn_cuda_copy_f32(sz * batch_size, x_norm_gpu, 1,
                       (float *)dst_tensor->grad_data_gpu, 1);
#else
    bcnn_cuda_grad_bias((float *)bn_biases->grad_data_gpu,
                        (float *)dst_tensor->grad_data_gpu, batch_size,
                        dst_tensor->c, dst_tensor->h * dst_tensor->w);
    bcnn_grad_scales_gpu(x_norm_gpu, (float *)dst_tensor->grad_data_gpu,
                         batch_size, dst_tensor->c,
                         dst_tensor->h * dst_tensor->w,
                         (float *)bn_scales->grad_data_gpu);
    bcnn_scales_gpu((float *)dst_tensor->grad_data_gpu,
                    (float *)bn_scales->data_gpu, batch_size, dst_tensor->c,
                    dst_tensor->h * dst_tensor->w);

    fast_mean_delta_gpu((float *)dst_tensor->grad_data_gpu,
                        (float *)saved_variance->data_gpu, batch_size,
                        dst_tensor->c, dst_tensor->w * dst_tensor->h,
                        (float *)saved_mean->grad_data_gpu);
    fast_variance_delta_gpu(workspace_gpu, (float *)dst_tensor->grad_data_gpu,
                            (float *)saved_mean->data_gpu,
                            (float *)saved_variance->data_gpu, batch_size,
                            dst_tensor->c, dst_tensor->w * dst_tensor->h,
                            (float *)saved_variance->grad_data_gpu);
    bcnn_cuda_norm_backward(
        workspace_gpu, (float *)saved_mean->data_gpu,
        (float *)saved_variance->data_gpu, (float *)saved_mean->grad_data_gpu,
        (float *)saved_variance->grad_data_gpu, src_tensor->n, dst_tensor->c,
        dst_tensor->w * dst_tensor->h, (float *)dst_tensor->grad_data_gpu);
#endif
    if (src_tensor->grad_data_gpu && src_tensor != dst_tensor) {
        bcnn_cuda_copy_f32(sz * batch_size, (float *)dst_tensor->grad_data_gpu,
                           1, (float *)src_tensor->grad_data_gpu, 1);
    }
    return;
}

void bcnn_backward_batchnorm_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *bn_mean = &net->tensors[node->src[1]];
    bcnn_tensor *bn_var = &net->tensors[node->src[2]];
    bcnn_tensor *bn_scales = &net->tensors[node->src[3]];
    bcnn_tensor *bn_biases = &net->tensors[node->src[4]];
    bcnn_batchnorm_param *param = (bcnn_batchnorm_param *)node->param;

    bcnn_backward_batchnorm_gpu(src_tensor, dst_tensor, bn_mean, bn_var,
                                bn_scales, bn_biases, &param->saved_mean,
                                &param->saved_variance, param->x_norm_gpu,
                                param->workspace_gpu, net->mode
#ifdef BCNN_USE_CUDNN
                                ,
                                param->dst_tensor_desc, param->bias_desc
#endif
                                );

    return;
}

#endif