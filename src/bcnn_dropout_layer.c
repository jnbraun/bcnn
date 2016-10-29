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

int bcnn_add_dropout_layer(bcnn_net *net, float rate, char *id)
{
	int sz = 0;
	int nb_connections = net->nb_connections + 1;
	bcnn_connection conn = { 0 };

	bh_assert(nb_connections > 2,
		"Dropout layer can't be the first layer of the network", BCNN_INTERNAL_ERROR);

	if (id != NULL)
		bh_fill_option(&conn.id, id);
	conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
	conn.layer->type = DROPOUT;
	if (nb_connections > 1)
		conn.src_node = net->connections[nb_connections - 2].dst_node;
	else
		conn.src_node = net->input_node;

	conn.dst_node.w = conn.src_node.w;
	conn.dst_node.h = conn.src_node.h;
	conn.dst_node.c = conn.src_node.c;
	conn.dst_node.b = conn.src_node.b;

	conn.layer->dropout_rate = rate;
	sz = bcnn_node_size(&conn.src_node);
	conn.layer->rand = (float *)calloc(sz, sizeof(float));
	conn.layer->scale = 1.0f / (1.0f - rate);
#ifdef BCNN_USE_CUDA
	conn.layer->rand_gpu = bcnn_cuda_memcpy_f32(conn.layer->rand, sz);
#endif
	conn.dst_node.data = conn.src_node.data;
	conn.dst_node.grad_data = conn.src_node.grad_data;
#ifdef BCNN_USE_CUDA
	conn.dst_node.data_gpu = conn.src_node.data_gpu;
	conn.dst_node.grad_data_gpu = conn.src_node.grad_data_gpu;
#endif
	net->nb_connections = nb_connections;
	bcnn_net_add_connection(net, conn);

	fprintf(stderr, "[Dropout] input_shape= %dx%dx%d rate= %f output_shape= %dx%dx%d\n",
		conn.src_node.w, conn.src_node.h, conn.src_node.c, rate,
		conn.dst_node.w, conn.dst_node.h, conn.dst_node.c);
	return 0;
}


int bcnn_forward_dropout_layer_cpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	int i, sz = bcnn_node_size(&src);
	float r;

	if (!conn->state) // state != train
		return BCNN_SUCCESS;

	for (i = 0; i < sz; ++i) {
		r = (float)rand() / RAND_MAX;
		layer->rand[i] = r;
		if (r < layer->dropout_rate)
			src.data[i] = 0;
		else
			src.data[i] *= layer->scale;
	}
	return BCNN_SUCCESS;
}

int bcnn_forward_dropout_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_dropout_layer_gpu(conn);
#else
	return bcnn_forward_dropout_layer_cpu(conn);
#endif
}


int bcnn_backward_dropout_layer_cpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	int i, sz = bcnn_node_size(&src);
	float r;

	if (!src.grad_data)
		return BCNN_SUCCESS;

	for (i = 0; i < sz; ++i) {
		r = layer->rand[i];
		if (r < layer->dropout_rate)
			src.grad_data[i] = 0;
		else
			src.grad_data[i] *= layer->scale;
	}
	return BCNN_SUCCESS;
}

int bcnn_backward_dropout_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_dropout_layer_gpu(conn);
#else
	return bcnn_backward_dropout_layer_cpu(conn);
#endif
}