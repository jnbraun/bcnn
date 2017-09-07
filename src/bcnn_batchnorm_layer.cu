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
	
	bcnn_cuda_copy_f32(sz * batch_size, src.data_gpu, 1, dst.data_gpu, 1);
	bcnn_cuda_copy_f32(sz * batch_size, dst.data_gpu, 1, layer->bn_workspace_gpu, 1);

	if (conn->state) {
		bcnn_cuda_mean_variance_forward(dst.data_gpu, batch_size, dst.c, dst.h * dst.w, layer->mean_gpu, layer->variance_gpu);
		
		bcnn_cuda_scal(dst.c, 0.9f, layer->global_mean_gpu, 1);
		bcnn_cuda_axpy(dst.c, 0.1f, layer->mean_gpu, 1, layer->global_mean_gpu, 1);
		bcnn_cuda_scal(dst.c, 0.9f, layer->global_variance_gpu, 1);
		bcnn_cuda_axpy(dst.c, 0.1f, layer->variance_gpu, 1, layer->global_variance_gpu, 1);
		
		bcnn_cuda_norm_forward(dst.data_gpu, layer->mean_gpu, layer->variance_gpu, batch_size, dst.c, dst.h * dst.w);   
		bcnn_cuda_copy_f32(batch_size * sz, dst.data_gpu, 1, layer->x_norm_gpu, 1);
	}
	else {
		// Normalize with global mean / variance
		bcnn_cuda_norm_forward(dst.data_gpu, layer->global_mean_gpu, layer->global_variance_gpu, batch_size, dst.c, dst.h * dst.w);  
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

	if (!conn->state) {
        layer->mean_gpu = layer->global_mean_gpu;
        layer->variance_gpu = layer->global_variance_gpu;
	}

	bcnn_cuda_mean_variance_backward(layer->bn_workspace_gpu, dst.grad_data_gpu, layer->mean_gpu,
		layer->variance_gpu, src.b, dst.c, dst.w * dst.h, layer->diff_mean_gpu, layer->diff_variance_gpu);
	bcnn_cuda_norm_backward(layer->bn_workspace_gpu, layer->mean_gpu, layer->variance_gpu, layer->diff_mean_gpu, layer->diff_variance_gpu, src.b,
		dst.c, dst.w * dst.h, dst.grad_data_gpu);

	if (src.grad_data_gpu)
		bcnn_cuda_copy_f32(sz * batch_size, dst.grad_data_gpu, 1, src.grad_data_gpu, 1);

	return BCNN_SUCCESS;
}

#endif