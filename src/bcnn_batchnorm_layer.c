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


#include <bh/bh_error.h>
#include <bh/bh_mem.h>

#include "bcnn/bcnn.h"

int bcnn_add_batchnorm_layer(bcnn_net *net)
{
	int nb_layers = net->nb_layers + 1, i;
	int sz = 0;
	bcnn_layer layer = { 0 };
	char type_name[256] = { 0 };

	bh_assert(nb_layers >= 2,
		"Batchnorm layer can't be the first layer of the network", BCNN_INTERNAL_ERROR);

	layer.type = BATCHNORM;
	//memcpy(layer.input_shape, input_shape, 3 * sizeof(int));
	memcpy(layer.input_shape, net->layers[net->nb_layers - 1].output_shape, 3 * sizeof(int));
	memcpy(layer.output_shape, layer.input_shape, 3 * sizeof(int));

	sz = layer.output_shape[0] * layer.output_shape[1] * layer.output_shape[2] * net->batch_size;

	layer.output = (float *)calloc(sz, sizeof(float));
	layer.diff = (float *)calloc(sz, sizeof(float));
	layer.bn_scale = (float *)calloc(layer.output_shape[2], sizeof(float));
	for (i = 0; i < layer.output_shape[2]; ++i) {
        layer.bn_scale[i] = 1;
    }
	layer.bn_shift = (float *)calloc(layer.output_shape[2], sizeof(float));
	layer.mean = (float *)calloc(layer.output_shape[2], sizeof(float));
	layer.variance = (float *)calloc(layer.output_shape[2], sizeof(float));
    layer.global_mean = (float *)calloc(layer.output_shape[2], sizeof(float));
    layer.global_variance = (float *)calloc(layer.output_shape[2], sizeof(float));
	layer.diff_mean = (float *)calloc(layer.output_shape[2], sizeof(float));
    layer.diff_variance = (float *)calloc(layer.output_shape[2], sizeof(float));
	layer.x_norm = (float *)calloc(sz, sizeof(float));
	layer.bn_workspace = (float *)calloc(sz, sizeof(float));
	layer.bn_scale_diff = (float *)calloc(layer.output_shape[2], sizeof(float));
	layer.bn_shift_diff = (float *)calloc(layer.output_shape[2], sizeof(float));
	layer.batch_sum_multiplier = (float *)calloc(net->batch_size, sizeof(float));
	layer.spatial_sum_multiplier = (float *)calloc(layer.output_shape[0] * layer.output_shape[1],
		sizeof(float));
	layer.spatial_stats = (float *)calloc(net->batch_size * layer.output_shape[2], sizeof(float));
	bcnn_fill_f32(net->batch_size, 1.0f, layer.batch_sum_multiplier);
	bcnn_fill_f32(layer.output_shape[0] * layer.output_shape[1], 1.0f, layer.spatial_sum_multiplier);
#ifdef BCNN_USE_CUDA
	layer.output_gpu = bcnn_cuda_memcpy_f32(layer.output, sz);
	layer.diff_gpu = bcnn_cuda_memcpy_f32(layer.diff, sz);
	layer.mean_gpu = bcnn_cuda_memcpy_f32(layer.mean, layer.output_shape[2]);
    layer.variance_gpu = bcnn_cuda_memcpy_f32(layer.variance, layer.output_shape[2]);
    layer.global_mean_gpu = bcnn_cuda_memcpy_f32(layer.global_mean, layer.output_shape[2]);
    layer.global_variance_gpu = bcnn_cuda_memcpy_f32(layer.global_variance, layer.output_shape[2]);
	layer.diff_mean_gpu = bcnn_cuda_memcpy_f32(layer.diff_mean, layer.output_shape[2]);
    layer.diff_variance_gpu = bcnn_cuda_memcpy_f32(layer.diff_variance, layer.output_shape[2]);
	layer.x_norm_gpu = bcnn_cuda_memcpy_f32(layer.output, sz);
	layer.bn_scale_gpu = bcnn_cuda_memcpy_f32(layer.bn_scale, layer.output_shape[2]);
	layer.bn_scale_diff_gpu = bcnn_cuda_memcpy_f32(layer.bn_scale_diff, layer.output_shape[2]);
	layer.bn_shift_gpu = bcnn_cuda_memcpy_f32(layer.bn_shift, layer.output_shape[2]);
	layer.bn_shift_diff_gpu = bcnn_cuda_memcpy_f32(layer.bn_shift_diff, layer.output_shape[2]);
	layer.batch_sum_multiplier_gpu = bcnn_cuda_memcpy_f32(layer.batch_sum_multiplier, net->batch_size);
	layer.spatial_sum_multiplier_gpu = bcnn_cuda_memcpy_f32(layer.spatial_sum_multiplier,
		layer.output_shape[0] * layer.output_shape[1]);
	layer.spatial_stats_gpu = bcnn_cuda_memcpy_f32(layer.spatial_stats, net->batch_size * layer.output_shape[2]);
	layer.bn_workspace_gpu = bcnn_cuda_memcpy_f32(layer.bn_workspace, sz);
#endif

	bcnn_realloc(net, nb_layers);
	net->layers[nb_layers - 1] = layer;

	fprintf(stderr, "[Batchnorm] input_shape= %dx%dx%d output_shape= %dx%dx%d\n",
		layer.input_shape[0], layer.input_shape[1], layer.input_shape[2],
		layer.output_shape[0], layer.output_shape[1], layer.output_shape[2]);

	return BCNN_SUCCESS;
}


int bcnn_forward_batchnorm_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int sz = layer->output_shape[0] * layer->output_shape[1] * layer->output_shape[2];
	int spatial_dim = layer->output_shape[0] * layer->output_shape[1];

	if (wrk->train) {
		bcnn_vmul(wrk->batch_size * sz, wrk->input, wrk->input, layer->bn_workspace);
		// Compute mean
		bcnn_gemv(0, layer->output_shape[2] * wrk->batch_size, spatial_dim,
			1.0f / (spatial_dim), wrk->input,
			layer->spatial_sum_multiplier, 0.0f,
			layer->spatial_stats);
		bcnn_gemv(1, wrk->batch_size, layer->output_shape[2], 1.0f / wrk->batch_size,
			layer->spatial_stats, layer->batch_sum_multiplier, 0.0f,
			layer->mean);

		bcnn_scal(layer->output_shape[2], 0.9f, layer->global_mean);
		bcnn_axpy(layer->output_shape[2], 0.1f, layer->mean, layer->global_mean);

		// E(X^2) across spatial
		bcnn_gemv(0, layer->output_shape[2] * wrk->batch_size, spatial_dim, 1.0f / (spatial_dim), layer->bn_workspace,
			layer->spatial_sum_multiplier, 0.0f, layer->spatial_stats);
		// E(X^2) across batch
		bcnn_gemv(1, wrk->batch_size, layer->output_shape[2], 1.0f / wrk->batch_size, layer->spatial_stats,
			layer->batch_sum_multiplier, 0.0f, layer->variance);
 
		bcnn_vmul(layer->output_shape[2], layer->mean, layer->mean, layer->bn_workspace); // (EX)^2
		bcnn_vsub(layer->output_shape[2], layer->variance, layer->bn_workspace, layer->variance);  // variance

		bcnn_scal(layer->output_shape[2], 0.9f, layer->global_variance);
		bcnn_axpy(layer->output_shape[2], 0.1f, layer->variance, layer->global_variance);
		
		bcnn_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
		  layer->batch_sum_multiplier, 1, layer->mean, layer->output_shape[2], 0.0f,
		  layer->spatial_stats, layer->output_shape[2]);
		bcnn_gemm(0, 0, layer->output_shape[2] * wrk->batch_size,
		  spatial_dim, 1, -1.0f, layer->spatial_stats, 1,
		  layer->spatial_sum_multiplier, spatial_dim, 0.0f, layer->bn_workspace, spatial_dim);
		bcnn_vadd(wrk->batch_size * sz, wrk->input, layer->bn_workspace, layer->output);
		
		// normalize variance
		bcnn_add_scalar(layer->output_shape[2], 0.00001f, layer->variance);
		bcnn_pow(layer->output_shape[2], layer->variance, 0.5f, layer->variance);
		bcnn_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->variance, layer->output_shape[2], 0.0f,
			layer->spatial_stats, layer->output_shape[2]);
		bcnn_gemm(0, 0, layer->output_shape[2] * wrk->batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 0.0f,
			layer->bn_workspace, spatial_dim);
		bcnn_vdiv(wrk->batch_size * sz, layer->output, layer->bn_workspace, layer->output);
			
		// save x_norm
		bcnn_copy_f32(wrk->batch_size * sz, layer->output, layer->x_norm);

		// scale
		bcnn_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->bn_scale, layer->output_shape[2], 0.0f,
			layer->spatial_stats, layer->output_shape[2]);
		bcnn_gemm(0, 0, layer->output_shape[2] * wrk->batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 0.0f,
			layer->bn_workspace, spatial_dim);
		bcnn_vmul(wrk->batch_size * sz, layer->output, layer->bn_workspace, layer->output);

		// shift
		bcnn_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->bn_shift, layer->output_shape[2], 0.0f,
			layer->spatial_stats, layer->output_shape[2]);
		bcnn_gemm(0, 0, layer->output_shape[2] * wrk->batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 0.0f,
			layer->bn_workspace, spatial_dim);
		bcnn_vadd(wrk->batch_size * sz, layer->output, layer->bn_workspace, layer->output);
	}
	else {
		// Normalize with global mean / variance
		bcnn_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
		  layer->batch_sum_multiplier, 1, layer->global_mean, layer->output_shape[2], 0.0f,
		  layer->spatial_stats, layer->output_shape[2]);
		bcnn_gemm(0, 0, layer->output_shape[2] * wrk->batch_size,
		  spatial_dim, 1, -1.0f, layer->spatial_stats, 1,
		  layer->spatial_sum_multiplier, spatial_dim, 0.0f, layer->bn_workspace, spatial_dim);
		bcnn_vadd(wrk->batch_size * sz, wrk->input, layer->bn_workspace, layer->output);
		
		
		memset(layer->bn_workspace, 0, sz * wrk->batch_size * sizeof(float));
		memcpy(layer->bn_workspace, layer->global_variance, layer->output_shape[2] * sizeof(float));
		bcnn_add_scalar(layer->output_shape[2], 0.00001f, layer->bn_workspace);
		bcnn_pow(layer->output_shape[2], layer->bn_workspace, 0.5f, layer->bn_workspace);
		bcnn_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->bn_workspace, layer->output_shape[2], 0.0f,
			layer->spatial_stats, layer->output_shape[2]);
		bcnn_gemm(0, 0, layer->output_shape[2] * wrk->batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 0.0f,
			layer->bn_workspace, spatial_dim);
		bcnn_vdiv(wrk->batch_size * sz, layer->output, layer->bn_workspace, layer->output);
	}

	return BCNN_SUCCESS;
}

int bcnn_backward_batchnorm_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int sz = layer->output_shape[0] * layer->output_shape[1] * layer->output_shape[2];
	int spatial_dim = layer->output_shape[0] * layer->output_shape[1];
	

	bcnn_vmul(sz * wrk->batch_size, layer->x_norm, layer->diff, layer->bn_workspace);
	// EX across spatial
	bcnn_gemv(0, wrk->batch_size * layer->output_shape[2], spatial_dim, 1.0f, layer->bn_workspace,
		layer->spatial_sum_multiplier, 0.0f, layer->spatial_stats);
	// EX across batch
	bcnn_gemv(1, wrk->batch_size, layer->output_shape[2], 1.0f, layer->spatial_stats,
		layer->batch_sum_multiplier, 0.0f, layer->bn_scale_diff);
	
	// gradient w.r.t. shift
	// EX across spatial
	bcnn_gemv(0, wrk->batch_size * layer->output_shape[2], spatial_dim, 1.0f, layer->diff,
		layer->spatial_sum_multiplier, 0.0f, layer->spatial_stats);
	// EX across batch
	bcnn_gemv(1, wrk->batch_size, layer->output_shape[2], 1.0f, layer->spatial_stats,
		layer->batch_sum_multiplier, 0.0f, layer->bn_shift_diff);

	if (wrk->diff) {
		bcnn_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->bn_scale, layer->output_shape[2], 0.0f,
			layer->spatial_stats, layer->output_shape[2]);
		bcnn_gemm(0, 0, wrk->batch_size * layer->output_shape[2], spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 0.0f,
			layer->bn_workspace, spatial_dim);
		bcnn_vmul(wrk->batch_size * sz, layer->diff, layer->bn_workspace, layer->bn_workspace);

		// use new top conv_workspace_gpu for computation
		bcnn_vmul(wrk->batch_size * sz, layer->x_norm, layer->bn_workspace, wrk->diff);
		// EX across spatial
		bcnn_gemv(0, wrk->batch_size * layer->output_shape[2], spatial_dim, 1.0f, wrk->diff,
			layer->spatial_sum_multiplier, 0.0f, layer->spatial_stats);
		// EX across batch
		bcnn_gemv(1, wrk->batch_size, layer->output_shape[2], 1.0f, layer->spatial_stats,
			layer->batch_sum_multiplier, 0.0f, layer->mean);
		//bcnn_gemm(0, 0, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
		bcnn_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->mean, layer->output_shape[2], 0.0f,
			layer->spatial_stats, layer->output_shape[2]);
		bcnn_gemm(0, 0, wrk->batch_size * layer->output_shape[2], spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 0.0f,
			wrk->diff, spatial_dim);

		bcnn_vmul(sz * wrk->batch_size, layer->x_norm, wrk->diff, wrk->diff);

		//
		// EX across spatial
		bcnn_gemv(0,  wrk->batch_size * layer->output_shape[2], spatial_dim, 1.0f, layer->bn_workspace,
			layer->spatial_sum_multiplier, 0.0f, layer->spatial_stats);
		// EX across batch
		bcnn_gemv(1, wrk->batch_size, layer->output_shape[2], 1.0f, layer->spatial_stats,
			layer->batch_sum_multiplier, 0.0f, layer->mean);

		bcnn_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->mean, layer->output_shape[2], 0.0f,
			layer->spatial_stats, layer->output_shape[2]);
		bcnn_gemm(0, 0, wrk->batch_size * layer->output_shape[2], spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 1.0f, wrk->diff, spatial_dim);

		bcnn_axpby(sz * wrk->batch_size, 1.0f, layer->bn_workspace, -1.0f / (wrk->batch_size * spatial_dim),
			wrk->diff);
        
		// variance normalization
		bcnn_gemm(0, 0, wrk->batch_size, layer->output_shape[2], 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->variance, layer->output_shape[2], 0.0f,
			layer->spatial_stats, layer->output_shape[2]);
		bcnn_gemm(0, 0, wrk->batch_size * layer->output_shape[2], spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 0.0f,
			layer->bn_workspace, spatial_dim);

		bcnn_vdiv(sz * wrk->batch_size, wrk->diff, layer->bn_workspace, wrk->diff);
	}

	return BCNN_SUCCESS;
}


int bcnn_forward_batchnorm_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_batchnorm_layer_gpu(layer, wrk);
#else
	return bcnn_forward_batchnorm_layer_cpu(layer, wrk);
#endif
}

int bcnn_backward_batchnorm_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_batchnorm_layer_gpu(layer, wrk);
#else
	return bcnn_backward_batchnorm_layer_cpu(layer, wrk);
#endif	
}
