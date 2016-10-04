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


#include <bh/bh_mem.h>

#include "bcnn/bcnn.h"


int bcnn_add_softmax_layer(bcnn_net *net)
{
	int nb_connections = net->nb_connections + 1;
	int sz;
	float std_init = 0.0f;
	bcnn_connection conn = { 0 };

	conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
	conn.layer->type = SOFTMAX;
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

#ifdef BCNN_USE_CUDA
	conn.dst_node.data_gpu = bcnn_cuda_memcpy_f32(conn.dst_node.data, sz);
	conn.dst_node.grad_data_gpu = bcnn_cuda_memcpy_f32(conn.dst_node.grad_data, sz);
#endif
	net->nb_connections = nb_connections;
	bcnn_net_add_connection(net, conn);

	fprintf(stderr, "[Softmax] input_shape= %dx%dx%d output_shape= %dx%dx%d\n",
		conn.src_node.w, conn.src_node.h, conn.src_node.c,
		conn.dst_node.w, conn.dst_node.h, conn.dst_node.c);

	return BCNN_SUCCESS;
}

int bcnn_forward_softmax_layer_cpu(bcnn_connection *conn)
{
	int b, i, batch_size = conn->dst_node.b;
	int src_size = conn->src_node.w * conn->src_node.h * conn->src_node.c;
	float vmax = -FLT_MAX;
	float sum = 0.0f;
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;

	for (b = 0; b < batch_size; ++b) {
		vmax = -FLT_MAX;
		sum = 0.0f;
		for (i = 0; i < src_size; ++i) {
			if (src.data[b * src_size + i] > vmax)
				vmax = src.data[b * src_size + i];
		}
		for (i = 0; i < src_size; ++i){
			sum += (float)exp(src.data[b * src_size + i] - vmax);
		}
		if (sum)
			sum = vmax + (float)log(sum);
		else
			sum = vmax - 100.0f;
		for (i = 0; i < src_size; ++i){
			dst.data[b * src_size + i] = (float)exp(src.data[b * src_size + i] - sum);
		}
	}
	return BCNN_SUCCESS;
}

int bcnn_backward_softmax_layer_cpu(bcnn_connection *conn)
{
	int i;
	int sz = conn->src_node.w * conn->src_node.h * conn->src_node.c * 
		conn->src_node.b;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;

	for (i = 0; i < sz; ++i)
		src.grad_data[i] += dst.grad_data[i];

	return BCNN_SUCCESS;
}


int bcnn_forward_softmax_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_softmax_layer_gpu(conn);
#else
	return bcnn_forward_softmax_layer_cpu(conn);
#endif
}


int bcnn_backward_softmax_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_softmax_layer_gpu(conn);
#else
	return bcnn_backward_softmax_layer_cpu(conn);
#endif
}