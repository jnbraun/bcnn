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

int bcnn_forward_batchnorm_layer_gpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int batch_size = src.b;
	int sz = dst.w * dst.h * dst.c;
	int spatial_dim = dst.w * dst.h;
	
	if (conn->state) {
		bcnn_cuda_vmul(batch_size * sz, src.data_gpu, src.data_gpu, layer->bn_workspace_gpu);
		// Compute mean
		bcnn_cuda_gemv(0, dst.c * batch_size, spatial_dim,
			1.0f / (spatial_dim), src.data_gpu,
			layer->spatial_sum_multiplier_gpu, 0.0f,
			layer->spatial_stats_gpu);
		bcnn_cuda_gemv(1, batch_size, dst.c, 1.0f / batch_size,
			layer->spatial_stats_gpu, layer->batch_sum_multiplier_gpu, 0.0f,
			layer->mean_gpu);

		bcnn_cuda_scal(dst.c, 0.9f, layer->global_mean_gpu, 1);
		bcnn_cuda_axpy(dst.c, 0.1f, layer->mean_gpu, 1, layer->global_mean_gpu, 1);

		// E(X^2) across spatial
		bcnn_cuda_gemv(0, dst.c * batch_size, spatial_dim, 1.0f / (spatial_dim), layer->bn_workspace_gpu,
			layer->spatial_sum_multiplier_gpu, 0.0f, layer->spatial_stats_gpu);
		// E(X^2) across batch
		bcnn_cuda_gemv(1, batch_size, dst.c, 1.0f / batch_size, layer->spatial_stats_gpu,
			layer->batch_sum_multiplier_gpu, 0.0f, layer->variance_gpu);
 
		bcnn_cuda_vmul(dst.c, layer->mean_gpu, layer->mean_gpu, layer->bn_workspace_gpu); // (EX)^2
		bcnn_cuda_vsub(dst.c, layer->variance_gpu, layer->bn_workspace_gpu, layer->variance_gpu);  // variance

		bcnn_cuda_scal(dst.c, 0.9f, layer->global_variance_gpu, 1);
		bcnn_cuda_axpy(dst.c, 0.1f, layer->variance_gpu, 1, layer->global_variance_gpu, 1);
		
		bcnn_cuda_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
		  layer->batch_sum_multiplier_gpu, 1, layer->mean_gpu, dst.c, 0.0f,
		  layer->spatial_stats_gpu, dst.c);
		bcnn_cuda_gemm(0, 0, dst.c * batch_size,
		  spatial_dim, 1, -1.0f, layer->spatial_stats_gpu, 1,
		  layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f, layer->bn_workspace_gpu, spatial_dim);
		bcnn_cuda_vadd(batch_size * sz, src.data_gpu, layer->bn_workspace_gpu, dst.data_gpu);
		
		// normalize variance
		bcnn_cuda_add_scalar(dst.c, 0.00001f, layer->variance_gpu);
		bcnn_cuda_pow(dst.c, layer->variance_gpu, 0.5f, layer->variance_gpu);
		bcnn_cuda_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->variance_gpu, dst.c, 0.0f,
			layer->spatial_stats_gpu, dst.c);
		bcnn_cuda_gemm(0, 0, dst.c * batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f,
			layer->bn_workspace_gpu, spatial_dim);
		bcnn_cuda_vdiv(batch_size * sz, dst.data_gpu, layer->bn_workspace_gpu, dst.data_gpu);
			
		// save x_norm
		bcnn_cuda_copy_f32(batch_size * sz, dst.data_gpu, 1, layer->x_norm_gpu, 1);

		// scale
		bcnn_cuda_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->bn_scale_gpu, dst.c, 0.0f,
			layer->spatial_stats_gpu, dst.c);
		bcnn_cuda_gemm(0, 0, dst.c * batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f,
			layer->bn_workspace_gpu, spatial_dim);
		bcnn_cuda_vmul(batch_size * sz, dst.data_gpu, layer->bn_workspace_gpu, dst.data_gpu);

		// shift
		bcnn_cuda_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->bn_shift_gpu, dst.c, 0.0f,
			layer->spatial_stats_gpu, dst.c);
		bcnn_cuda_gemm(0, 0, dst.c * batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f,
			layer->bn_workspace_gpu, spatial_dim);
		bcnn_cuda_vadd(batch_size * sz, dst.data_gpu, layer->bn_workspace_gpu, dst.data_gpu);
	}
	else {
		// Normalize with global mean / variance
		bcnn_cuda_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
		  layer->batch_sum_multiplier_gpu, 1, layer->global_mean_gpu, dst.c, 0.0f,
		  layer->spatial_stats_gpu, dst.c);
		bcnn_cuda_gemm(0, 0, dst.c * batch_size,
		  spatial_dim, 1, -1.0f, layer->spatial_stats_gpu, 1,
		  layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f, layer->bn_workspace_gpu, spatial_dim);
		bcnn_cuda_vadd(batch_size * sz, src.data_gpu, layer->bn_workspace_gpu, dst.data_gpu);
		
		bcnn_cuda_fill_f32(sz * batch_size, 0.0f, layer->bn_workspace_gpu, 1);
		bcnn_cuda_copy_f32(dst.c, layer->global_variance_gpu, 1, layer->bn_workspace_gpu, 1);
		bcnn_cuda_add_scalar(dst.c, 0.00001f, layer->bn_workspace_gpu);
		bcnn_cuda_pow(dst.c, layer->bn_workspace_gpu, 0.5f, layer->bn_workspace_gpu);
		bcnn_cuda_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->bn_workspace_gpu, dst.c, 0.0f,
			layer->spatial_stats_gpu, dst.c);
		bcnn_cuda_gemm(0, 0, dst.c * batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f,
			layer->bn_workspace_gpu, spatial_dim);
		bcnn_cuda_vdiv(batch_size * sz, dst.data_gpu, layer->bn_workspace_gpu, dst.data_gpu);
	}

	return BCNN_SUCCESS;
}


int bcnn_backward_batchnorm_layer_gpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int batch_size = src.b;
	int sz = dst.w * dst.h * dst.c;
	int spatial_dim = dst.w * dst.h;
	

	bcnn_cuda_vmul(sz * batch_size, layer->x_norm_gpu, dst.grad_data_gpu, layer->bn_workspace_gpu);
	// EX across spatial
	bcnn_cuda_gemv(0, batch_size * dst.c, spatial_dim, 1.0f, layer->bn_workspace_gpu,
		layer->spatial_sum_multiplier_gpu, 0.0f, layer->spatial_stats_gpu);
	// EX across batch
	bcnn_cuda_gemv(1, batch_size, dst.c, 1.0f, layer->spatial_stats_gpu,
		layer->batch_sum_multiplier_gpu, 0.0f, layer->bn_scale_diff_gpu);
	
	// gradient w.r.t. shift
	// EX across spatial
	bcnn_cuda_gemv(0, batch_size * dst.c, spatial_dim, 1.0f, dst.grad_data_gpu,
		layer->spatial_sum_multiplier_gpu, 0.0f, layer->spatial_stats_gpu);
	// EX across batch
	bcnn_cuda_gemv(1, batch_size, dst.c, 1.0f, layer->spatial_stats_gpu,
		layer->batch_sum_multiplier_gpu, 0.0f, layer->bn_shift_diff_gpu);

	if (src.grad_data_gpu) {
		bcnn_cuda_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->bn_scale_gpu, dst.c, 0.0f,
			layer->spatial_stats_gpu, dst.c);
		bcnn_cuda_gemm(0, 0, batch_size * dst.c, spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f,
			layer->bn_workspace_gpu, spatial_dim);
		bcnn_cuda_vmul(batch_size * sz, dst.grad_data_gpu, layer->bn_workspace_gpu, layer->bn_workspace_gpu);

		// use new top conv_workspace_gpu for computation
		bcnn_cuda_vmul(batch_size * sz, layer->x_norm_gpu, layer->bn_workspace_gpu, src.grad_data_gpu);
		// EX across spatial
		bcnn_cuda_gemv(0, batch_size * dst.c, spatial_dim, 1.0f, src.grad_data_gpu,
			layer->spatial_sum_multiplier_gpu, 0.0f, layer->spatial_stats_gpu);
		// EX across batch
		bcnn_cuda_gemv(1, batch_size, dst.c, 1.0f, layer->spatial_stats_gpu,
			layer->batch_sum_multiplier_gpu, 0.0f, layer->mean_gpu);
		//bcnn_cuda_gemm(0, 0, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
		bcnn_cuda_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->mean_gpu, dst.c, 0.0f,
			layer->spatial_stats_gpu, dst.c);
		bcnn_cuda_gemm(0, 0, batch_size * dst.c, spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f,
			src.grad_data_gpu, spatial_dim);

		bcnn_cuda_vmul(sz * batch_size, layer->x_norm_gpu, src.grad_data_gpu, src.grad_data_gpu);

		//
		// EX across spatial
		bcnn_cuda_gemv(0,  batch_size * dst.c, spatial_dim, 1.0f, layer->bn_workspace_gpu,
			layer->spatial_sum_multiplier_gpu, 0.0f, layer->spatial_stats_gpu);
		// EX across batch
		bcnn_cuda_gemv(1, batch_size, dst.c, 1.0f, layer->spatial_stats_gpu,
			layer->batch_sum_multiplier_gpu, 0.0f, layer->mean_gpu);

		bcnn_cuda_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->mean_gpu, dst.c, 0.0f,
			layer->spatial_stats_gpu, dst.c);
		bcnn_cuda_gemm(0, 0, batch_size * dst.c, spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 1.0f, src.grad_data_gpu, spatial_dim);

		bcnn_cuda_axpby(sz * batch_size, 1.0f, layer->bn_workspace_gpu, -1.0f / (batch_size * spatial_dim),
			src.grad_data_gpu);
        
		// variance normalization
		bcnn_cuda_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->variance_gpu, dst.c, 0.0f,
			layer->spatial_stats_gpu, dst.c);
		bcnn_cuda_gemm(0, 0, batch_size * dst.c, spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f,
			layer->bn_workspace_gpu, spatial_dim);

		bcnn_cuda_vdiv(sz * batch_size, src.grad_data_gpu, layer->bn_workspace_gpu, src.grad_data_gpu);
	}

	return BCNN_SUCCESS;
}

#endif