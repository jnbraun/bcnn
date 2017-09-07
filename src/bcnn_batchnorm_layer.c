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
#include <bh/bh_string.h>

#include "bcnn/bcnn.h"

int bcnn_add_batchnorm_layer(bcnn_net *net, char *id)
{
	int nb_connections = net->nb_connections + 1;
	int sz;
	bcnn_connection conn = { 0 };

	bh_assert(nb_connections >= 2,
		"Batchnorm layer can't be the first layer of the network", BCNN_INTERNAL_ERROR);

	if (id != NULL)
		bh_fill_option(&conn.id, id);
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
	conn.layer->mean = (float *)calloc(conn.dst_node.c, sizeof(float));
	conn.layer->variance = (float *)calloc(conn.dst_node.c, sizeof(float));
    	conn.layer->global_mean = (float *)calloc(conn.dst_node.c, sizeof(float));
    	conn.layer->global_variance = (float *)calloc(conn.dst_node.c, sizeof(float));
	conn.layer->diff_mean = (float *)calloc(conn.dst_node.c, sizeof(float));
    	conn.layer->diff_variance = (float *)calloc(conn.dst_node.c, sizeof(float));
	conn.layer->x_norm = (float *)calloc(sz, sizeof(float));
	conn.layer->bn_workspace = (float *)calloc(sz, sizeof(float));
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
	conn.layer->bn_workspace_gpu = bcnn_cuda_memcpy_f32(conn.layer->bn_workspace, sz);
#endif
	net->nb_connections = nb_connections;
	bcnn_net_add_connection(net, conn);

	fprintf(stderr, "[Batchnorm] input_shape= %dx%dx%d output_shape= %dx%dx%d\n",
		conn.src_node.w, conn.src_node.h, conn.src_node.c,
		conn.dst_node.w, conn.dst_node.h, conn.dst_node.c);

	return BCNN_SUCCESS;
}

static void _mean_variance_forward(float *x, int b, int c, int wxh, float *mean, float *var)
{
    float scale = 1.0f / (b * wxh);
    int i, j, k;
	float s = 0.0f;

    for (i = 0; i < c; ++i) {
        mean[i] = 0;
		var[i] = 0;
        for (j = 0; j < b; ++j) {
			k = j * c * wxh + i * wxh;
			bcnn_vsum(wxh, x + k, &s);
			mean[i] += s;
			var[i] += bcnn_dot(wxh, x + k, x + k);
        }
		// TODO: check which option is faster here
        //mean[i] *= scale;
        //var[i] = var[i] * scale - mean[i] * mean[i];
    }
	bcnn_scal(c, scale, mean);
	bcnn_varmean(c, mean, scale, var);
}

static void _norm_forward(float *x, float *mean, float *variance, int b, int c, int wxh)
{
    int k, j, i, ind;

    for (k = 0; k < b; ++k) {
        for (j = 0; j < c; ++j) {
            for (i = 0; i < wxh; ++i) {
                ind = k * c * wxh + j * wxh + i;
                x[ind] = (x[ind] - mean[j]) / (sqrtf(variance[j]) + 0.000001f);
            }
        }
    }
}


int bcnn_forward_batchnorm_layer_cpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int batch_size = src.b;
	int sz = dst.w * dst.h * dst.c;
	
	bcnn_copy_f32(sz * batch_size, src.data, dst.data);
	bcnn_copy_f32(sz * batch_size, dst.data, layer->bn_workspace);

	if (conn->state) {
		_mean_variance_forward(dst.data, batch_size, dst.c, dst.h * dst.w, layer->mean, layer->variance);

		bcnn_scal(dst.c, 0.9f, layer->global_mean);
		bcnn_axpy(dst.c, 0.1f, layer->mean, layer->global_mean);
		bcnn_scal(dst.c, 0.9f, layer->global_variance);
		bcnn_axpy(dst.c, 0.1f, layer->variance, layer->global_variance);
		
		_norm_forward(dst.data, layer->mean, layer->variance, batch_size, dst.c, dst.h * dst.w);   
		bcnn_copy_f32(batch_size * sz, dst.data, layer->x_norm);
	}
	else {
		// Normalize with global mean / variance
		_norm_forward(dst.data, layer->global_mean, layer->global_variance, batch_size, dst.c, dst.h * dst.w);  
	}

	return BCNN_SUCCESS;
}

static void _mean_variance_backward(float *x, float *grad, float *mean, float *var, int b, int c, int wxh, float *mean_diff, float *var_diff)
{
    int i, j, k;
	float s = 0.0f;

    for(i = 0; i < c; ++i){
        mean_diff[i] = 0;
		var_diff[i] = 0;
        for (j = 0; j < b; ++j) {
			k = j * c * wxh + i * wxh;
			bcnn_vsum(wxh, grad + k, &s);
			mean_diff[i] += s;
			var_diff[i] += bcnn_shiftdot(wxh, x + k, mean[i], grad + k, 0.0f);
        }
        mean_diff[i] *= (-1.0f / sqrtf(var[i] + 0.00001f));
    }
	bcnn_varnorm(c, var, -0.5f, var_diff);
}

static void _normalize_backward(float *x, float *mean, float *var, float *mean_delta, float *var_diff, int b, int c, int wxh, float *grad)
{
	int i, j, k, ind;

	for (j = 0; j < b; ++j) {
		for (i = 0; i < c; ++i) {
			for (k = 0; k < wxh; ++k) {
				ind = j * c * wxh + i * wxh + k;
				grad[ind] = grad[ind] * 1.0f / (sqrtf(var[i] + 0.00001f)) + var_diff[i] * 2.0f * (x[ind] - mean[i]) / (wxh * b) + mean_delta[i] / (wxh * b);
			}
		}
	}
}

int bcnn_backward_batchnorm_layer_cpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int batch_size = src.b;
	int sz = dst.w * dst.h * dst.c;
	
	if (!conn->state) {
		layer->mean = layer->global_mean;
		layer->variance = layer->global_variance;
	}

	_mean_variance_backward(layer->bn_workspace, dst.grad_data, layer->mean, layer->variance, batch_size, dst.c, dst.w * dst.h, layer->diff_mean, layer->diff_variance);
	_normalize_backward(layer->bn_workspace, layer->mean, layer->variance, layer->diff_mean, layer->diff_variance, batch_size, dst.c, dst.w * dst.h, dst.grad_data);

	if (src.grad_data)
		bcnn_copy_f32(sz * batch_size, dst.grad_data, src.grad_data);

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
