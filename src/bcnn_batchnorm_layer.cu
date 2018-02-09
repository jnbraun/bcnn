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

__global__ void  fast_mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
    const int threads = 512;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? x[index] : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        mean[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean[filter] += local[i];
        }
        mean[filter] /= spatial * batch;
    }
}

__global__ void  fast_variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    const int threads = 512;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? ((x[index] - mean[filter]) * (x[index] - mean[filter])) : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance[filter] += local[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}

extern "C" void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    fast_mean_kernel<<<filters, BCNN_CUDA_THREADS>>>(x, batch, filters, spatial, mean);
    bcnn_cuda_check(cudaPeekAtLastError());
}

extern "C" void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    fast_variance_kernel<<<filters, BCNN_CUDA_THREADS>>>(x, mean, batch, filters, spatial, variance);
    bcnn_cuda_check(cudaPeekAtLastError());
}

__global__ void fast_mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    const int threads = 512;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? delta[index] : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        mean_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean_delta[filter] += local[i];
        }
        mean_delta[filter] *= (-1.f/sqrtf(variance[filter] + .00001f));
    }
}

__global__ void  fast_variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    const int threads = 512;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance_delta[filter] += local[i];
        }
        variance_delta[filter] *= -.5f * powf(variance[filter] + .00001f, (float)(-3.f/2.f));
    }
}

extern "C" void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    fast_mean_delta_kernel<<<filters, BCNN_CUDA_THREADS>>>(delta, variance, batch, filters, spatial, mean_delta);
    bcnn_cuda_check(cudaPeekAtLastError());
}

extern "C" void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    fast_variance_delta_kernel<<<filters, BCNN_CUDA_THREADS>>>(x, delta, mean, variance, batch, filters, spatial, variance_delta);
    bcnn_cuda_check(cudaPeekAtLastError());
}



int bcnn_forward_batchnorm_layer_gpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
#ifndef BCNN_USE_CUDNN
    int batch_size = src.n;
    int sz = dst.w * dst.h * dst.c;
#else
    float alpha = 1.0f;
    float beta = 0.0f;
#endif
    
    if (layer->net_state) {
#ifdef BCNN_USE_CUDNN
        bcnn_cudnn_check(cudnnBatchNormalizationForwardTraining(bcnn_cudnn_handle(),
                CUDNN_BATCHNORM_SPATIAL,
                &alpha,
                &beta,
                layer->src_tensor_desc,
                src.data_gpu,
                layer->src_tensor_desc,
                dst.data_gpu,
                layer->dst_tensor_desc,
                layer->bn_scale_gpu,
                layer->biases.data_gpu,
                0.1,
                layer->global_mean_gpu,
                layer->global_variance_gpu,
                0.0001,
                layer->mean_gpu,
                layer->variance_gpu));
#else
        bcnn_cuda_copy_f32(sz * batch_size, src.data_gpu, 1, dst.data_gpu, 1);
        bcnn_cuda_copy_f32(sz * batch_size, dst.data_gpu, 1, layer->bn_workspace_gpu, 1);
        
        //bcnn_cuda_mean_variance_forward(dst.data_gpu, batch_size, dst.c, dst.h * dst.w, layer->mean_gpu, layer->variance_gpu);
        fast_mean_gpu(dst.data_gpu, batch_size, dst.c, dst.h * dst.w, layer->mean_gpu);
        fast_variance_gpu(dst.data_gpu, layer->mean_gpu, batch_size, dst.c, dst.h * dst.w, layer->variance_gpu);
        
        bcnn_cuda_scal(dst.c, 0.9f, layer->global_mean_gpu, 1);
        bcnn_cuda_axpy(dst.c, 0.1f, layer->mean_gpu, 1, layer->global_mean_gpu, 1);
        bcnn_cuda_scal(dst.c, 0.9f, layer->global_variance_gpu, 1);
        bcnn_cuda_axpy(dst.c, 0.1f, layer->variance_gpu, 1, layer->global_variance_gpu, 1);
        
        //bcnn_cuda_copy_f32(batch_size * sz, dst.data_gpu, 1, layer->bn_workspace_gpu, 1);
        bcnn_cuda_norm_forward(dst.data_gpu, layer->mean_gpu, layer->variance_gpu, batch_size, dst.c, dst.h * dst.w);   
        bcnn_cuda_copy_f32(batch_size * sz, dst.data_gpu, 1, layer->x_norm_gpu, 1);
#endif
    }
    else {
#ifdef BCNN_USE_CUDNN
        bcnn_cudnn_check(cudnnBatchNormalizationForwardInference(bcnn_cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &beta,
            layer->src_tensor_desc,
            src.data_gpu,
            layer->src_tensor_desc,
            dst.data_gpu,
            layer->dst_tensor_desc,
            layer->bn_scale_gpu,
            layer->biases.data_gpu,
            layer->global_mean_gpu,
            layer->global_variance_gpu,
            0.0001));
#else
        bcnn_cuda_copy_f32(sz * batch_size, src.data_gpu, 1, dst.data_gpu, 1);
        bcnn_cuda_copy_f32(sz * batch_size, dst.data_gpu, 1, layer->bn_workspace_gpu, 1);
        // Normalize with global mean / variance
        bcnn_cuda_norm_forward(dst.data_gpu, layer->global_mean_gpu, layer->global_variance_gpu, batch_size, dst.c, dst.h * dst.w);  
#endif
    }

    return BCNN_SUCCESS;
}


int bcnn_backward_batchnorm_layer_gpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
#ifndef BCNN_USE_CUDNN
    int batch_size = src.n;
    int sz = dst.w * dst.h * dst.c;
#else
    float a_data = 1.0f, a_param = 1.0f;
    float b_data = 0.0f, b_param = 1.0f;
#endif

    if (!layer->net_state) {
        layer->mean_gpu = layer->global_mean_gpu;
        layer->variance_gpu = layer->global_variance_gpu;
    }

#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnBatchNormalizationBackward(bcnn_cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &a_data,
            &b_data,
            &a_param,
            &b_param,
            layer->src_tensor_desc,
            src.data_gpu,
            layer->src_tensor_desc,
            dst.grad_data_gpu,
            layer->src_tensor_desc,
            src.grad_data_gpu,
            layer->dst_tensor_desc,
            layer->bn_scale_gpu,
            layer->bn_scale_diff_gpu,
            layer->biases.grad_data_gpu,
            0.0001,
            layer->mean_gpu,
            layer->variance_gpu));
#else
    fast_mean_delta_gpu(dst.grad_data_gpu, layer->variance_gpu, batch_size, dst.c, dst.w * dst.h, layer->diff_mean_gpu);
    fast_variance_delta_gpu(layer->bn_workspace_gpu, dst.grad_data_gpu, layer->mean_gpu, layer->variance_gpu, batch_size, dst.c, dst.w * dst.h, layer->diff_variance_gpu);
    bcnn_cuda_norm_backward(layer->bn_workspace_gpu, layer->mean_gpu, layer->variance_gpu, layer->diff_mean_gpu, layer->diff_variance_gpu, src.n,
        dst.c, dst.w * dst.h, dst.grad_data_gpu);
    
    if (src.grad_data_gpu)
        bcnn_cuda_copy_f32(sz * batch_size, dst.grad_data_gpu, 1, src.grad_data_gpu, 1);	
#endif

    

    return BCNN_SUCCESS;
}

#endif