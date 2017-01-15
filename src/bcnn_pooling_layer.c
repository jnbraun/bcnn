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

#include <bh/bh.h>
#include <bh/bh_string.h>

#include "bcnn/bcnn.h"

int bcnn_add_maxpool_layer(bcnn_net *net, int size, int stride, char *id)
{
	int nb_connections = net->nb_connections + 1;
	int sz;
	bcnn_connection conn = { 0 };

	if (id != NULL)
		bh_fill_option(&conn.id, id);
	conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
	conn.layer->type = MAXPOOL;
	if (nb_connections > 1)
		conn.src_node = net->connections[nb_connections - 2].dst_node;
	else
		conn.src_node = net->input_node;

	conn.dst_node.w = (conn.src_node.w - 1) / stride + 1;
	conn.dst_node.h = (conn.src_node.h - 1) / stride + 1;
	conn.dst_node.c = conn.src_node.c;
	conn.dst_node.b = conn.src_node.b;
	
	conn.layer->size = size;
	conn.layer->stride = stride;

	sz = bcnn_node_size(&conn.dst_node);
	conn.layer->indexes = (int *)calloc(sz, sizeof(int));
	conn.dst_node.data = (float *)calloc(sz, sizeof(float));
	conn.dst_node.grad_data = (float *)calloc(sz, sizeof(float));
#ifdef BCNN_USE_CUDA
	conn.layer->indexes_gpu = bcnn_cuda_malloc_i32(sz);
	conn.dst_node.data_gpu = bcnn_cuda_memcpy_f32(conn.dst_node.data, sz);
	conn.dst_node.grad_data_gpu = bcnn_cuda_memcpy_f32(conn.dst_node.grad_data, sz);
#ifdef BCNN_USE_CUDNN
	bcnn_cudnn_check(cudnnCreateTensorDescriptor(&conn.layer->src_tensor_desc));
	bcnn_cudnn_check(cudnnCreateTensorDescriptor(&conn.layer->dst_tensor_desc));
	bcnn_cudnn_check(cudnnCreatePoolingDescriptor(&conn.layer->pooling_desc));
	bcnn_cudnn_check(cudnnSetPooling2dDescriptor(conn.layer->pooling_desc,
                                            CUDNN_POOLING_MAX,
											CUDNN_NOT_PROPAGATE_NAN,
											conn.layer->size,
											conn.layer->size,
                                            0,
                                            0,
											conn.layer->stride,
											conn.layer->stride));
	bcnn_cudnn_check(cudnnSetTensor4dDescriptor(conn.layer->src_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		conn.src_node.b, conn.src_node.c, conn.src_node.h, conn.src_node.w)); 
	bcnn_cudnn_check(cudnnSetTensor4dDescriptor(conn.layer->dst_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		conn.dst_node.b, conn.dst_node.c, conn.dst_node.h, conn.dst_node.w)); 
#endif
#endif
	net->nb_connections = nb_connections;
	bcnn_net_add_connection(net, conn);

	fprintf(stderr, "[Maxpool] input_shape= %dx%dx%d size= %d stride= %d ouput_shape= %dx%dx%d\n",
		conn.src_node.w, conn.src_node.h, conn.src_node.c, size, stride,
		conn.dst_node.w, conn.dst_node.h, conn.dst_node.c);
	return 0;
}


int bcnn_forward_maxpool_layer_cpu(bcnn_connection *conn)
{
	int b, i, j, k, m, n, dst_index, valid, src_index, cur_w, cur_h, max_i;
	float max_f = -FLT_MAX, val;
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int batch_size = dst.b;
	int offset_pool = (-layer->size - 1) / 2 + 1;
	int offset0, offset1, offset2;

	for (b = 0; b < batch_size; ++b) { // batch_size
		offset0 = dst.c * b;
		for (k = 0; k < dst.c; ++k) {	// depth
			offset1 = dst.h * (k + offset0);
			for (i = 0; i < dst.h; ++i) {	// height
				offset2 = dst.w * (offset1 + i);
				for (j = 0; j < dst.w; ++j) {	// width
					dst_index = j + offset2;
					max_f = -FLT_MAX;
					max_i = -1;
					for (n = 0; n < layer->size; ++n) {	// pooling window
						for (m = 0; m < layer->size; ++m) {
							cur_h = offset_pool + i * layer->stride + n;
							cur_w = offset_pool + j * layer->stride + m;
							src_index = cur_w + src.w * (cur_h + src.h * (k + b * src.c));
							valid = (cur_h >= 0 && cur_h < src.h &&
								cur_w >= 0 && cur_w < src.w);
							val = (valid != 0) ? src.data[src_index] : -FLT_MAX;
							if (val > max_f) {
								max_f = val;
								max_i = src_index;
							}
						}
					}
					dst.data[dst_index] = max_f;
					layer->indexes[dst_index] = max_i;
				}
			}
		}
	}
	return BCNN_SUCCESS;
}


int bcnn_forward_maxpool_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_maxpool_layer_gpu(conn);
#else
	return bcnn_forward_maxpool_layer_cpu(conn);
#endif
}

int bcnn_backward_maxpool_layer_cpu(bcnn_connection *conn)
{
	int i, index;
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int sz = bcnn_node_size(&dst);

	for (i = 0; i < sz; ++i) {
		index = layer->indexes[i];
		src.grad_data[index] += dst.grad_data[i];
	}

	return BCNN_SUCCESS;
}

int bcnn_backward_maxpool_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_maxpool_layer_gpu(conn);
#else
	return bcnn_backward_maxpool_layer_cpu(conn);
#endif
	return 0;
}
