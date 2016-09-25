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

int bcnn_forward_batchnorm_layer_gpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int sz = layer->output_shape[0] * layer->output_shape[1] * layer->output_shape[2];
	int spatial_dim = layer->output_shape[0] * layer->output_shape[1];
	
	if (wrk->train) {
		bcnn_cuda_vmul(wrk->batch_size * sz, wrk->input_gpu, wrk->input_gpu, layer->bn_workspace_gpu);
		// Compute mean
		bcnn_cuda_gemv(0, layer->output_shape[2] * wrk->batch_size, spatial_dim,
			1.0f / (spatial_dim), wrk->input_gpu,
			layer->spatial_sum_multiplier_gpu, 0.0f,
			layer->spatial_stats_gpu);
		bcnn_cuda_gemv(1, wrk->batch_size, layer->output_shape[2], 1.0f / wrk->batch_size,
			layer->spatial_stats_gpu, layer->batch_sum_multiplier_gpu, 0.0f,
			layer->mean_gpu);

		bcnn_cuda_scal(layer->output_shape[2], 0.9f, layer->global_mean_gpu, 1);
		bcnn_cuda_axpy(layer->output_shape[2], 0.1f, layer->mean_gpu, 1, layer->global_mean_gpu, 1);

		// E(X^2) across spatial
		bcnn_cuda_gemv(0, layer->output_shape[2] * wrk->batch_size, spatial_dim, 1.0f / (spatial_dim), layer->bn_workspace_gpu,
			layer->spatial_sum_multiplier_gpu, 0.0f, layer->spatial_stats_gpu);
		// E(X^2) across batch
		bcnn_cuda_gemv(1, wrk->batch_size, layer->output_shape[2], 1.0f / wrk->batch_size, layer->spatial_stats_gpu,
			layer->batch_sum_multiplier_gpu, 0.0f, layer->variance_gpu);
 
		bcnn_cuda_vmul(layer->output_shape[2], layer->mean_gpu, layer->mean_gpu, layer->bn_workspace_gpu); // (EX)^2
		bcnn_cuda_vsub(layer->output_shape[2], layer->variance_gpu, layer->bn_workspace_gpu, layer->variance_gpu);  // variance

		bcnn_cuda_scal(layer->output_shape[2], 0.9f, layer->global_variance_gpu, 1);
		bcnn_cuda_axpy(layer->output_shape[2], 0.1f, layer->variance_gpu, 1, layer->global_variance_gpu, 1);
		
		bcnn_cuda_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
		  layer->batch_sum_multiplier_gpu, 1, layer->mean_gpu, layer->output_shape[2], 0.0f,
		  layer->spatial_stats_gpu, layer->output_shape[2]);
		bcnn_cuda_gemm(0, 0, layer->output_shape[2] * wrk->batch_size,
		  spatial_dim, 1, -1.0f, layer->spatial_stats_gpu, 1,
		  layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f, layer->bn_workspace_gpu, spatial_dim);
		bcnn_cuda_vadd(wrk->batch_size * sz, wrk->input_gpu, layer->bn_workspace_gpu, layer->output_gpu);
		
		// normalize variance
		bcnn_cuda_add_scalar(layer->output_shape[2], 0.00001f, layer->variance_gpu);
		bcnn_cuda_pow(layer->output_shape[2], layer->variance_gpu, 0.5f, layer->variance_gpu);
		bcnn_cuda_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->variance_gpu, layer->output_shape[2], 0.0f,
			layer->spatial_stats_gpu, layer->output_shape[2]);
		bcnn_cuda_gemm(0, 0, layer->output_shape[2] * wrk->batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f,
			layer->bn_workspace_gpu, spatial_dim);
		bcnn_cuda_vdiv(wrk->batch_size * sz, layer->output_gpu, layer->bn_workspace_gpu, layer->output_gpu);
			
		// save x_norm
		bcnn_cuda_copy_f32(wrk->batch_size * sz, layer->output_gpu, 1, layer->x_norm_gpu, 1);

		// scale
		bcnn_cuda_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->bn_scale_gpu, layer->output_shape[2], 0.0f,
			layer->spatial_stats_gpu, layer->output_shape[2]);
		bcnn_cuda_gemm(0, 0, layer->output_shape[2] * wrk->batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f,
			layer->bn_workspace_gpu, spatial_dim);
		bcnn_cuda_vmul(wrk->batch_size * sz, layer->output_gpu, layer->bn_workspace_gpu, layer->output_gpu);

		// shift
		bcnn_cuda_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->bn_shift_gpu, layer->output_shape[2], 0.0f,
			layer->spatial_stats_gpu, layer->output_shape[2]);
		bcnn_cuda_gemm(0, 0, layer->output_shape[2] * wrk->batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f,
			layer->bn_workspace_gpu, spatial_dim);
		bcnn_cuda_vadd(wrk->batch_size * sz, layer->output_gpu, layer->bn_workspace_gpu, layer->output_gpu);
	}
	else {
		// Normalize with global mean / variance
		bcnn_cuda_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
		  layer->batch_sum_multiplier_gpu, 1, layer->global_mean_gpu, layer->output_shape[2], 0.0f,
		  layer->spatial_stats_gpu, layer->output_shape[2]);
		bcnn_cuda_gemm(0, 0, layer->output_shape[2] * wrk->batch_size,
		  spatial_dim, 1, -1.0f, layer->spatial_stats_gpu, 1,
		  layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f, layer->bn_workspace_gpu, spatial_dim);
		bcnn_cuda_vadd(wrk->batch_size * sz, wrk->input_gpu, layer->bn_workspace_gpu, layer->output_gpu);
		
		bcnn_cuda_fill_f32(sz * wrk->batch_size, 0.0f, layer->bn_workspace_gpu, 1);
		bcnn_cuda_copy_f32(layer->output_shape[2], layer->global_variance_gpu, 1, layer->bn_workspace_gpu, 1);
		bcnn_cuda_add_scalar(layer->output_shape[2], 0.00001f, layer->bn_workspace_gpu);
		bcnn_cuda_pow(layer->output_shape[2], layer->bn_workspace_gpu, 0.5f, layer->bn_workspace_gpu);
		bcnn_cuda_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->bn_workspace_gpu, layer->output_shape[2], 0.0f,
			layer->spatial_stats_gpu, layer->output_shape[2]);
		bcnn_cuda_gemm(0, 0, layer->output_shape[2] * wrk->batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f,
			layer->bn_workspace_gpu, spatial_dim);
		bcnn_cuda_vdiv(wrk->batch_size * sz, layer->output_gpu, layer->bn_workspace_gpu, layer->output_gpu);
	}

	return BCNN_SUCCESS;
}


int bcnn_backward_batchnorm_layer_gpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int sz = layer->output_shape[0] * layer->output_shape[1] * layer->output_shape[2];
	int spatial_dim = layer->output_shape[0] * layer->output_shape[1];
	

	bcnn_cuda_vmul(sz * wrk->batch_size, layer->x_norm_gpu, layer->diff_gpu, layer->bn_workspace_gpu);
	// EX across spatial
	bcnn_cuda_gemv(0, wrk->batch_size * layer->output_shape[2], spatial_dim, 1.0f, layer->bn_workspace_gpu,
		layer->spatial_sum_multiplier_gpu, 0.0f, layer->spatial_stats_gpu);
	// EX across batch
	bcnn_cuda_gemv(1, wrk->batch_size, layer->output_shape[2], 1.0f, layer->spatial_stats_gpu,
		layer->batch_sum_multiplier_gpu, 0.0f, layer->bn_scale_diff_gpu);
	
	// gradient w.r.t. shift
	// EX across spatial
	bcnn_cuda_gemv(0, wrk->batch_size * layer->output_shape[2], spatial_dim, 1.0f, layer->diff_gpu,
		layer->spatial_sum_multiplier_gpu, 0.0f, layer->spatial_stats_gpu);
	// EX across batch
	bcnn_cuda_gemv(1, wrk->batch_size, layer->output_shape[2], 1.0f, layer->spatial_stats_gpu,
		layer->batch_sum_multiplier_gpu, 0.0f, layer->bn_shift_diff_gpu);

	if (wrk->diff) {
		bcnn_cuda_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->bn_scale_gpu, layer->output_shape[2], 0.0f,
			layer->spatial_stats_gpu, layer->output_shape[2]);
		bcnn_cuda_gemm(0, 0, wrk->batch_size * layer->output_shape[2], spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f,
			layer->bn_workspace_gpu, spatial_dim);
		bcnn_cuda_vmul(wrk->batch_size * sz, layer->diff_gpu, layer->bn_workspace_gpu, layer->bn_workspace_gpu);

		// use new top conv_workspace_gpu for computation
		bcnn_cuda_vmul(wrk->batch_size * sz, layer->x_norm_gpu, layer->bn_workspace_gpu, wrk->diff);
		// EX across spatial
		bcnn_cuda_gemv(0, wrk->batch_size * layer->output_shape[2], spatial_dim, 1.0f, wrk->diff,
			layer->spatial_sum_multiplier_gpu, 0.0f, layer->spatial_stats_gpu);
		// EX across batch
		bcnn_cuda_gemv(1, wrk->batch_size, layer->output_shape[2], 1.0f, layer->spatial_stats_gpu,
			layer->batch_sum_multiplier_gpu, 0.0f, layer->mean_gpu);
		//bcnn_cuda_gemm(0, 0, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
		bcnn_cuda_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->mean_gpu, layer->output_shape[2], 0.0f,
			layer->spatial_stats_gpu, layer->output_shape[2]);
		bcnn_cuda_gemm(0, 0, wrk->batch_size * layer->output_shape[2], spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f,
			wrk->diff, spatial_dim);

		bcnn_cuda_vmul(sz * wrk->batch_size, layer->x_norm_gpu, wrk->diff, wrk->diff);

		//
		// EX across spatial
		bcnn_cuda_gemv(0,  wrk->batch_size * layer->output_shape[2], spatial_dim, 1.0f, layer->bn_workspace_gpu,
			layer->spatial_sum_multiplier_gpu, 0.0f, layer->spatial_stats_gpu);
		// EX across batch
		bcnn_cuda_gemv(1, wrk->batch_size, layer->output_shape[2], 1.0f, layer->spatial_stats_gpu,
			layer->batch_sum_multiplier_gpu, 0.0f, layer->mean_gpu);

		bcnn_cuda_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->mean_gpu, layer->output_shape[2], 0.0f,
			layer->spatial_stats_gpu, layer->output_shape[2]);
		bcnn_cuda_gemm(0, 0, wrk->batch_size * layer->output_shape[2], spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 1.0f, wrk->diff, spatial_dim);

		bcnn_cuda_axpby(sz * wrk->batch_size, 1.0f, layer->bn_workspace_gpu, -1.0f / (wrk->batch_size * spatial_dim),
			wrk->diff);
        
		// variance normalization
		bcnn_cuda_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier_gpu, 1, layer->variance_gpu, layer->output_shape[2], 0.0f,
			layer->spatial_stats_gpu, layer->output_shape[2]);
		bcnn_cuda_gemm(0, 0, wrk->batch_size * layer->output_shape[2], spatial_dim, 1, 1.0f,
			layer->spatial_stats_gpu, 1, layer->spatial_sum_multiplier_gpu, spatial_dim, 0.0f,
			layer->bn_workspace_gpu, spatial_dim);

		bcnn_cuda_vdiv(sz * wrk->batch_size, wrk->diff, layer->bn_workspace_gpu, wrk->diff);
	}

	return BCNN_SUCCESS;
}

#endif