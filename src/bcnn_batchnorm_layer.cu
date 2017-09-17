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

#include "bcnn/bcnn.h"


__global__ void _bcnn_forward_scale_kernel(float *output, float *scale, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if (offset < size)
		output[(batch * n + filter) * size + offset] += scale[filter];
}

int bcnn_forward_scale_gpu(float *output, float *scale, int batch, int n, int size)
{
    dim3 dimGrid((size - 1) / BCNN_CUDA_THREADS + 1, n, batch);
    dim3 dimBlock(BCNN_CUDA_THREADS, 1, 1);

    _bcnn_forward_scale_kernel<<<dimGrid, dimBlock>>>(output, scale, n, size);
    bcnn_cuda_check(cudaPeekAtLastError());
	return BCNN_SUCCESS;
}


__global__ void _bcnn_backward_scale_kernel(float *scale_diff, float *diff, float *dx, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if (offset < size)
		scale_diff[filter] += diff[(batch * n + filter) * size + offset] * dx[(batch * n + filter) * size + offset];
}

int bcnn_backward_scale_gpu(float *scale_diff, float *diff, float *dx, int batch, int n, int size)
{
    dim3 dimGrid((size - 1) / BCNN_CUDA_THREADS + 1, n, batch);
    dim3 dimBlock(BCNN_CUDA_THREADS, 1, 1);

    _bcnn_backward_scale_kernel<<<dimGrid, dimBlock>>>(scale_diff, diff, dx, n, size);
    bcnn_cuda_check(cudaPeekAtLastError());
	return BCNN_SUCCESS;
}

int bcnn_forward_batchnorm_layer_gpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_tensor src = conn->src_tensor;
	bcnn_tensor dst = conn->dst_tensor;
	int batch_size = src.b;
	int sz = dst.w * dst.h * dst.c;
#ifdef BCNN_USE_CUDNN
	float alpha = 1.0f;
    float beta = 0.0f;
#endif
	
	if (conn->state) {
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
                layer->bias_gpu,
                0.1,
                layer->global_mean_gpu,
                layer->global_variance_gpu,
                0.0001,
                layer->mean_gpu,
                layer->variance_gpu));
#else
		bcnn_cuda_copy_f32(sz * batch_size, src.data_gpu, 1, dst.data_gpu, 1);
		bcnn_cuda_copy_f32(sz * batch_size, dst.data_gpu, 1, layer->bn_workspace_gpu, 1);
		
		bcnn_cuda_mean_variance_forward(dst.data_gpu, batch_size, dst.c, dst.h * dst.w, layer->mean_gpu, layer->variance_gpu);
		
		bcnn_cuda_scal(dst.c, 0.9f, layer->global_mean_gpu, 1);
		bcnn_cuda_axpy(dst.c, 0.1f, layer->mean_gpu, 1, layer->global_mean_gpu, 1);
		bcnn_cuda_scal(dst.c, 0.9f, layer->global_variance_gpu, 1);
		bcnn_cuda_axpy(dst.c, 0.1f, layer->variance_gpu, 1, layer->global_variance_gpu, 1);
		
		bcnn_cuda_copy_f32(batch_size * sz, dst.data_gpu, 1, layer->bn_workspace_gpu, 1);
		bcnn_cuda_norm_forward(dst.data_gpu, layer->mean_gpu, layer->variance_gpu, batch_size, dst.c, dst.h * dst.w);   
		bcnn_cuda_copy_f32(batch_size * sz, dst.data_gpu, 1, layer->x_norm_gpu, 1);

		bcnn_forward_scale_gpu(dst.data_gpu, layer->bn_scale_gpu, batch_size, dst.c, dst.h * dst.w);
		bcnn_forward_bias_gpu(dst.data_gpu, layer->bias_gpu, batch_size, dst.c, dst.h * dst.w);
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
            layer->bias_gpu,
            layer->global_mean_gpu,
            layer->global_variance_gpu,
            0.0001));
#else
		bcnn_cuda_copy_f32(sz * batch_size, src.data_gpu, 1, dst.data_gpu, 1);
		bcnn_cuda_copy_f32(sz * batch_size, dst.data_gpu, 1, layer->bn_workspace_gpu, 1);
		// Normalize with global mean / variance
		bcnn_cuda_norm_forward(dst.data_gpu, layer->global_mean_gpu, layer->global_variance_gpu, batch_size, dst.c, dst.h * dst.w);  
		bcnn_forward_scale_gpu(dst.data_gpu, layer->bn_scale_gpu, batch_size, dst.c, dst.h * dst.w);
		bcnn_forward_bias_gpu(dst.data_gpu, layer->bias_gpu, batch_size, dst.c, dst.h * dst.w);
#endif
	}

	return BCNN_SUCCESS;
}


int bcnn_backward_batchnorm_layer_gpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_tensor src = conn->src_tensor;
	bcnn_tensor dst = conn->dst_tensor;
	int batch_size = src.b;
	int sz = dst.w * dst.h * dst.c;
#ifdef BCNN_USE_CUDNN
	float a_data = 1.0f, a_param = 1.0f;
    float b_data = 0.0f, b_param = 1.0f;
#endif

	if (!conn->state) {
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
            layer->bias_diff_gpu,
            0.0001,
            layer->mean_gpu,
            layer->variance_gpu));
#else
	bcnn_backward_bias_gpu(layer->bias_diff_gpu, dst.grad_data_gpu, batch_size, dst.c, dst.w * dst.h);
	bcnn_backward_scale_gpu(layer->bn_scale_diff_gpu, dst.grad_data_gpu, layer->x_norm_gpu, batch_size, dst.c, dst.w * dst.h);

	bcnn_cuda_mean_variance_backward(layer->bn_workspace_gpu, dst.grad_data_gpu, layer->mean_gpu,
		layer->variance_gpu, src.b, dst.c, dst.w * dst.h, layer->diff_mean_gpu, layer->diff_variance_gpu);
	bcnn_cuda_norm_backward(layer->bn_workspace_gpu, layer->mean_gpu, layer->variance_gpu, layer->diff_mean_gpu, layer->diff_variance_gpu, src.b,
		dst.c, dst.w * dst.h, dst.grad_data_gpu);
	
	if (src.grad_data_gpu)
		bcnn_cuda_copy_f32(sz * batch_size, dst.grad_data_gpu, 1, src.grad_data_gpu, 1);	
#endif

	

	return BCNN_SUCCESS;
}

#endif