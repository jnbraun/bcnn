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
	int nb_connections = net->nb_connections + 1;
	int i, sz;
	bcnn_connection conn = { 0 };

	bh_assert(nb_connections >= 2,
		"Batchnorm layer can't be the first layer of the network", BCNN_INTERNAL_ERROR);

	conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
	conn.layer->type = BATCHNORM;
	if (nb_connections > 1)
		conn.src_node = net->connections[nb_connections - 2].dst_node;
	else
		conn.src_node = net->input_node;

	conn.dst_node.w = conn.src_node.w;
	conn.dst_node.h = conn.src_node.h;
	conn.dst_node.c = conn.src_node.c;
	conn.dst_node.b = conn.src_node.b;

	sz = bcnn_node_size(&conn.dst_node);

	conn.dst_node.data = (float *)calloc(sz, sizeof(float));
	conn.dst_node.grad_data = (float *)calloc(sz, sizeof(float));
	conn.layer->bn_scale = (float *)calloc(conn.dst_node.c, sizeof(float));
	for (i = 0; i < conn.dst_node.c; ++i) {
        conn.layer->bn_scale[i] = 1;
    }
	conn.layer->bn_shift = (float *)calloc(conn.dst_node.c, sizeof(float));
	conn.layer->mean = (float *)calloc(conn.dst_node.c, sizeof(float));
	conn.layer->variance = (float *)calloc(conn.dst_node.c, sizeof(float));
    conn.layer->global_mean = (float *)calloc(conn.dst_node.c, sizeof(float));
    conn.layer->global_variance = (float *)calloc(conn.dst_node.c, sizeof(float));
	conn.layer->diff_mean = (float *)calloc(conn.dst_node.c, sizeof(float));
    conn.layer->diff_variance = (float *)calloc(conn.dst_node.c, sizeof(float));
	conn.layer->x_norm = (float *)calloc(sz, sizeof(float));
	conn.layer->bn_workspace = (float *)calloc(sz, sizeof(float));
	conn.layer->bn_scale_diff = (float *)calloc(conn.dst_node.c, sizeof(float));
	conn.layer->bn_shift_diff = (float *)calloc(conn.dst_node.c, sizeof(float));
	conn.layer->batch_sum_multiplier = (float *)calloc(conn.dst_node.b, sizeof(float));
	conn.layer->spatial_sum_multiplier = (float *)calloc(conn.dst_node.w * conn.dst_node.h,
		sizeof(float));
	conn.layer->spatial_stats = (float *)calloc(conn.dst_node.b * conn.dst_node.c, sizeof(float));
	bcnn_fill_f32(conn.dst_node.b, 1.0f, conn.layer->batch_sum_multiplier);
	bcnn_fill_f32(conn.dst_node.w * conn.dst_node.h, 1.0f, conn.layer->spatial_sum_multiplier);
#ifdef BCNN_USE_CUDA
	conn.dst_node.data_gpu = bcnn_cuda_memcpy_f32(conn.dst_node.data, sz);
	conn.dst_node.grad_data_gpu = bcnn_cuda_memcpy_f32(conn.dst_node.grad_data, sz);
	conn.layer->mean_gpu = bcnn_cuda_memcpy_f32(conn.layer->mean, conn.dst_node.c);
    conn.layer->variance_gpu = bcnn_cuda_memcpy_f32(conn.layer->variance, conn.dst_node.c);
    conn.layer->global_mean_gpu = bcnn_cuda_memcpy_f32(conn.layer->global_mean, conn.dst_node.c);
    conn.layer->global_variance_gpu = bcnn_cuda_memcpy_f32(conn.layer->global_variance, conn.dst_node.c);
	conn.layer->diff_mean_gpu = bcnn_cuda_memcpy_f32(conn.layer->diff_mean, conn.dst_node.c);
    conn.layer->diff_variance_gpu = bcnn_cuda_memcpy_f32(conn.layer->diff_variance, conn.dst_node.c);
	conn.layer->x_norm_gpu = bcnn_cuda_memcpy_f32(conn.dst_node.data, sz);
	conn.layer->bn_scale_gpu = bcnn_cuda_memcpy_f32(conn.layer->bn_scale, conn.dst_node.c);
	conn.layer->bn_scale_diff_gpu = bcnn_cuda_memcpy_f32(conn.layer->bn_scale_diff, conn.dst_node.c);
	conn.layer->bn_shift_gpu = bcnn_cuda_memcpy_f32(conn.layer->bn_shift, conn.dst_node.c);
	conn.layer->bn_shift_diff_gpu = bcnn_cuda_memcpy_f32(conn.layer->bn_shift_diff, conn.dst_node.c);
	conn.layer->batch_sum_multiplier_gpu = bcnn_cuda_memcpy_f32(conn.layer->batch_sum_multiplier, conn.dst_node.b);
	conn.layer->spatial_sum_multiplier_gpu = bcnn_cuda_memcpy_f32(conn.layer->spatial_sum_multiplier,
		conn.dst_node.w * conn.dst_node.h);
	conn.layer->spatial_stats_gpu = bcnn_cuda_memcpy_f32(conn.layer->spatial_stats, conn.dst_node.b * conn.dst_node.c);
	conn.layer->bn_workspace_gpu = bcnn_cuda_memcpy_f32(conn.layer->bn_workspace, sz);
#endif
	net->nb_connections = nb_connections;
	bcnn_net_add_connection(net, conn);

	fprintf(stderr, "[Batchnorm] input_shape= %dx%dx%d output_shape= %dx%dx%d\n",
		conn.src_node.w, conn.src_node.h, conn.src_node.c,
		conn.dst_node.w, conn.dst_node.h, conn.dst_node.c);

	return BCNN_SUCCESS;
}


int bcnn_forward_batchnorm_layer_cpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int batch_size = src.b;
	int sz = dst.w * dst.h * dst.c;
	int spatial_dim = dst.w * dst.h;

	if (conn->state) {
		bcnn_vmul(batch_size * sz, src.data, src.data, layer->bn_workspace);
		// Compute mean
		bcnn_gemv(0, dst.c * batch_size, spatial_dim,
			1.0f / (spatial_dim), src.data,
			layer->spatial_sum_multiplier, 0.0f,
			layer->spatial_stats);
		bcnn_gemv(1, batch_size, dst.c, 1.0f / batch_size,
			layer->spatial_stats, layer->batch_sum_multiplier, 0.0f,
			layer->mean);

		bcnn_scal(dst.c, 0.9f, layer->global_mean);
		bcnn_axpy(dst.c, 0.1f, layer->mean, layer->global_mean);

		// E(X^2) across spatial
		bcnn_gemv(0, dst.c * batch_size, spatial_dim, 1.0f / (spatial_dim), layer->bn_workspace,
			layer->spatial_sum_multiplier, 0.0f, layer->spatial_stats);
		// E(X^2) across batch
		bcnn_gemv(1, batch_size, dst.c, 1.0f / batch_size, layer->spatial_stats,
			layer->batch_sum_multiplier, 0.0f, layer->variance);
 
		bcnn_vmul(dst.c, layer->mean, layer->mean, layer->bn_workspace); // (EX)^2
		bcnn_vsub(dst.c, layer->variance, layer->bn_workspace, layer->variance);  // variance

		bcnn_scal(dst.c, 0.9f, layer->global_variance);
		bcnn_axpy(dst.c, 0.1f, layer->variance, layer->global_variance);
		
		bcnn_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
		  layer->batch_sum_multiplier, 1, layer->mean, dst.c, 0.0f,
		  layer->spatial_stats, dst.c);
		bcnn_gemm(0, 0, dst.c * batch_size,
		  spatial_dim, 1, -1.0f, layer->spatial_stats, 1,
		  layer->spatial_sum_multiplier, spatial_dim, 0.0f, layer->bn_workspace, spatial_dim);
		bcnn_vadd(batch_size * sz, src.data, layer->bn_workspace, dst.data);
		
		// normalize variance
		bcnn_add_scalar(dst.c, 0.00001f, layer->variance);
		bcnn_pow(dst.c, layer->variance, 0.5f, layer->variance);
		bcnn_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->variance, dst.c, 0.0f,
			layer->spatial_stats, dst.c);
		bcnn_gemm(0, 0, dst.c * batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 0.0f,
			layer->bn_workspace, spatial_dim);
		bcnn_vdiv(batch_size * sz, dst.data, layer->bn_workspace, dst.data);
			
		// save x_norm
		bcnn_copy_f32(batch_size * sz, dst.data, layer->x_norm);

		// scale
		bcnn_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->bn_scale, dst.c, 0.0f,
			layer->spatial_stats, dst.c);
		bcnn_gemm(0, 0, dst.c * batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 0.0f,
			layer->bn_workspace, spatial_dim);
		bcnn_vmul(batch_size * sz, dst.data, layer->bn_workspace, dst.data);

		// shift
		bcnn_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->bn_shift, dst.c, 0.0f,
			layer->spatial_stats, dst.c);
		bcnn_gemm(0, 0, dst.c * batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 0.0f,
			layer->bn_workspace, spatial_dim);
		bcnn_vadd(batch_size * sz, dst.data, layer->bn_workspace, dst.data);
	}
	else {
		// Normalize with global mean / variance
		bcnn_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
		  layer->batch_sum_multiplier, 1, layer->global_mean, dst.c, 0.0f,
		  layer->spatial_stats, dst.c);
		bcnn_gemm(0, 0, dst.c * batch_size,
		  spatial_dim, 1, -1.0f, layer->spatial_stats, 1,
		  layer->spatial_sum_multiplier, spatial_dim, 0.0f, layer->bn_workspace, spatial_dim);
		bcnn_vadd(batch_size * sz, src.data, layer->bn_workspace, dst.data);
		
		
		memset(layer->bn_workspace, 0, sz * batch_size * sizeof(float));
		memcpy(layer->bn_workspace, layer->global_variance, dst.c * sizeof(float));
		bcnn_add_scalar(dst.c, 0.00001f, layer->bn_workspace);
		bcnn_pow(dst.c, layer->bn_workspace, 0.5f, layer->bn_workspace);
		bcnn_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->bn_workspace, dst.c, 0.0f,
			layer->spatial_stats, dst.c);
		bcnn_gemm(0, 0, dst.c * batch_size, spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 0.0f,
			layer->bn_workspace, spatial_dim);
		bcnn_vdiv(batch_size * sz, dst.data, layer->bn_workspace, dst.data);
	}

	return BCNN_SUCCESS;
}

int bcnn_backward_batchnorm_layer_cpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int batch_size = src.b;
	int sz = dst.w * dst.h * dst.c;
	int spatial_dim = dst.w * dst.h;
	

	bcnn_vmul(sz * batch_size, layer->x_norm, dst.grad_data, layer->bn_workspace);
	// EX across spatial
	bcnn_gemv(0, batch_size * dst.c, spatial_dim, 1.0f, layer->bn_workspace,
		layer->spatial_sum_multiplier, 0.0f, layer->spatial_stats);
	// EX across batch
	bcnn_gemv(1, batch_size, dst.c, 1.0f, layer->spatial_stats,
		layer->batch_sum_multiplier, 0.0f, layer->bn_scale_diff);
	
	// gradient w.r.t. shift
	// EX across spatial
	bcnn_gemv(0, batch_size * dst.c, spatial_dim, 1.0f, dst.grad_data,
		layer->spatial_sum_multiplier, 0.0f, layer->spatial_stats);
	// EX across batch
	bcnn_gemv(1, batch_size, dst.c, 1.0f, layer->spatial_stats,
		layer->batch_sum_multiplier, 0.0f, layer->bn_shift_diff);

	if (src.grad_data) {
		bcnn_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->bn_scale, dst.c, 0.0f,
			layer->spatial_stats, dst.c);
		bcnn_gemm(0, 0, batch_size * dst.c, spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 0.0f,
			layer->bn_workspace, spatial_dim);
		bcnn_vmul(batch_size * sz, dst.grad_data, layer->bn_workspace, layer->bn_workspace);

		// use new top conv_workspace_gpu for computation
		bcnn_vmul(batch_size * sz, layer->x_norm, layer->bn_workspace, src.grad_data);
		// EX across spatial
		bcnn_gemv(0, batch_size * dst.c, spatial_dim, 1.0f, src.grad_data,
			layer->spatial_sum_multiplier, 0.0f, layer->spatial_stats);
		// EX across batch
		bcnn_gemv(1, batch_size, dst.c, 1.0f, layer->spatial_stats,
			layer->batch_sum_multiplier, 0.0f, layer->mean);
		//bcnn_gemm(0, 0, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
		bcnn_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->mean, dst.c, 0.0f,
			layer->spatial_stats, dst.c);
		bcnn_gemm(0, 0, batch_size * dst.c, spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 0.0f,
			src.grad_data, spatial_dim);

		bcnn_vmul(sz * batch_size, layer->x_norm, src.grad_data, src.grad_data);

		//
		// EX across spatial
		bcnn_gemv(0,  batch_size * dst.c, spatial_dim, 1.0f, layer->bn_workspace,
			layer->spatial_sum_multiplier, 0.0f, layer->spatial_stats);
		// EX across batch
		bcnn_gemv(1, batch_size, dst.c, 1.0f, layer->spatial_stats,
			layer->batch_sum_multiplier, 0.0f, layer->mean);

		bcnn_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->mean, dst.c, 0.0f,
			layer->spatial_stats, dst.c);
		bcnn_gemm(0, 0, batch_size * dst.c, spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 1.0f, src.grad_data, spatial_dim);

		bcnn_axpby(sz * batch_size, 1.0f, layer->bn_workspace, -1.0f / (batch_size * spatial_dim),
			src.grad_data);
        
		// variance normalization
		bcnn_gemm(0, 0, batch_size, dst.c, 1, 1.0f,
			layer->batch_sum_multiplier, 1, layer->variance, dst.c, 0.0f,
			layer->spatial_stats, dst.c);
		bcnn_gemm(0, 0, batch_size * dst.c, spatial_dim, 1, 1.0f,
			layer->spatial_stats, 1, layer->spatial_sum_multiplier, spatial_dim, 0.0f,
			layer->bn_workspace, spatial_dim);

		bcnn_vdiv(sz * batch_size, src.grad_data, layer->bn_workspace, src.grad_data);
	}

	return BCNN_SUCCESS;
}


int bcnn_forward_batchnorm_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_batchnorm_layer_gpu(conn);
#else
	return bcnn_forward_batchnorm_layer_cpu(conn);
#endif
}

int bcnn_backward_batchnorm_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_batchnorm_layer_gpu(conn);
#else
	return bcnn_backward_batchnorm_layer_cpu(conn);
#endif	
}
