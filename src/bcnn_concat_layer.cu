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

int bcnn_forward_concat_layer_gpu(bcnn_net *net, bcnn_connection *conn)
{
	int j, b = conn->src_tensor.b;
	float *data_concat = net->connections[conn->layer->concat_index].dst_tensor.data_gpu;
	bcnn_tensor src = conn->src_tensor;
	bcnn_tensor dst = conn->dst_tensor;
	int concat_sz = net->connections[conn->layer->concat_index].dst_tensor.w * 
		net->connections[conn->layer->concat_index].dst_tensor.h * 
		net->connections[conn->layer->concat_index].dst_tensor.c;
	int src_sz = src.c * src.w * src.h;
	int dst_sz = dst.c * dst.w * dst.h;

    for (j = 0; j < b; ++j) {
		bcnn_cuda_copy_f32(src_sz, src.data_gpu + j * src_sz, 1, dst.data_gpu + j * dst_sz, 1);
	}
	for (j = 0; j < b; ++j) {
        bcnn_cuda_copy_f32(concat_sz, data_concat + j * concat_sz, 1, dst.data_gpu + src_sz + j * dst_sz, 1);
    }

	return BCNN_SUCCESS;
}

int bcnn_backward_concat_layer_gpu(bcnn_net *net, bcnn_connection *conn)
{
	int j, b = conn->src_tensor.b;
	float *grad_concat = net->connections[conn->layer->concat_index].dst_tensor.grad_data_gpu;
	bcnn_tensor src = conn->src_tensor;
	bcnn_tensor dst = conn->dst_tensor;
	int concat_sz = net->connections[conn->layer->concat_index].dst_tensor.w * 
		net->connections[conn->layer->concat_index].dst_tensor.h * 
		net->connections[conn->layer->concat_index].dst_tensor.c;
	int src_sz = src.c * src.w * src.h;
	int dst_sz = dst.c * dst.w * dst.h;

    for (j = 0; j < b; ++j) {
		bcnn_cuda_axpy(src_sz, 1.0f, dst.grad_data_gpu + j * dst_sz, 1, src.grad_data_gpu + j * src_sz, 1);
	}
	for (j = 0; j < b; ++j) {
        bcnn_cuda_axpy(concat_sz, 1.0f, dst.grad_data_gpu + src_sz + j * dst_sz, 1, grad_concat + j * concat_sz, 1);
    }

	return BCNN_SUCCESS;
}

#endif