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


int bcnn_add_concat_layer(bcnn_net *net, char *concat, char *id)
{
	int nb_connections = net->nb_connections + 1;
	int i, sz, ind_concat = -1;
	bcnn_connection conn = { 0 };

	bh_assert(nb_connections >= 2,
		"Concat layer can't be the first layer of the network", BCNN_INTERNAL_ERROR);

	if (id != NULL)
		bh_fill_option(&conn.id, id);
	conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
	conn.layer->type = CONCAT;
	if (nb_connections > 1)
		conn.src_tensor = net->connections[nb_connections - 2].dst_tensor;
	else
		conn.src_tensor = net->input_node;

	for (i = 0; i < nb_connections; ++i) {
		if (strcmp(concat, net->connections[i].id) == 0)
			ind_concat = i;
	}
	
	bh_assert(ind_concat != -1, "Unknown id layer used in concat layer", BCNN_INVALID_PARAMETER);

	bh_assert(conn.src_tensor.w == net->connections[ind_concat].dst_tensor.w &&
		conn.src_tensor.h == net->connections[ind_concat].dst_tensor.h,
		"Concat layer: concatenated features maps must have the same spatial dimension",
		BCNN_INVALID_PARAMETER);

	conn.layer->concat_index = ind_concat;

	conn.dst_tensor.w = conn.src_tensor.w;
	conn.dst_tensor.h = conn.src_tensor.h;
	conn.dst_tensor.c = conn.src_tensor.c + net->connections[ind_concat].dst_tensor.c;
	conn.dst_tensor.b = conn.src_tensor.b;
	
	sz = bcnn_get_tensor_size(&conn.dst_tensor);
	conn.dst_tensor.data = (float *)calloc(sz, sizeof(float));
	conn.dst_tensor.grad_data = (float *)calloc(sz, sizeof(float));
#ifdef BCNN_USE_CUDA
	conn.dst_tensor.data_gpu = bcnn_cuda_memcpy_f32(conn.dst_tensor.data, sz);
	conn.dst_tensor.grad_data_gpu = bcnn_cuda_memcpy_f32(conn.dst_tensor.grad_data, sz);
#endif
	net->nb_connections = nb_connections;
	bcnn_net_add_connection(net, conn);

	fprintf(stderr, "[Concat] input_shape= %dx%dx%d output_shape= %dx%dx%d\n",
		conn.src_tensor.w, conn.src_tensor.h, conn.src_tensor.c,
		conn.dst_tensor.w, conn.dst_tensor.h, conn.dst_tensor.c);

	return BCNN_SUCCESS;
}


int bcnn_forward_concat_layer_cpu(bcnn_net *net, bcnn_connection *conn)
{
	int j, b = conn->src_tensor.b;
	float *data_concat = net->connections[conn->layer->concat_index].dst_tensor.data;
	bcnn_tensor src = conn->src_tensor;
	bcnn_tensor dst = conn->dst_tensor;
	int concat_sz = net->connections[conn->layer->concat_index].dst_tensor.w * 
		net->connections[conn->layer->concat_index].dst_tensor.h * 
		net->connections[conn->layer->concat_index].dst_tensor.c;
	int src_sz = src.c * src.w * src.h;
	int dst_sz = dst.c * dst.w * dst.h;

    for (j = 0; j < b; ++j) {
		bcnn_copy_f32(src_sz, src.data + j * src_sz, dst.data + j * dst_sz);
	}
	for (j = 0; j < b; ++j) {
        bcnn_copy_f32(concat_sz, data_concat + j * concat_sz, dst.data + src_sz + j * dst_sz);
    }

	return BCNN_SUCCESS;
}

int bcnn_backward_concat_layer_cpu(bcnn_net *net, bcnn_connection *conn)
{
	int j, b = conn->src_tensor.b;
	float *grad_concat = net->connections[conn->layer->concat_index].dst_tensor.grad_data;
	bcnn_tensor src = conn->src_tensor;
	bcnn_tensor dst = conn->dst_tensor;
	int concat_sz = net->connections[conn->layer->concat_index].dst_tensor.w * 
		net->connections[conn->layer->concat_index].dst_tensor.h * 
		net->connections[conn->layer->concat_index].dst_tensor.c;
	int src_sz = src.c * src.w * src.h;
	int dst_sz = dst.c * dst.w * dst.h;

    for (j = 0; j < b; ++j) {
		bcnn_axpy(src_sz, 1.0f, dst.grad_data + j * dst_sz, src.grad_data + j * src_sz);
	}
	for (j = 0; j < b; ++j) {
        bcnn_axpy(concat_sz, 1.0f, dst.grad_data + src_sz + j * dst_sz, grad_concat + j * concat_sz);
    }

	return BCNN_SUCCESS;
}


int bcnn_forward_concat_layer(bcnn_net *net, bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_concat_layer_gpu(net, conn);
#else
	return bcnn_forward_concat_layer_cpu(net, conn);
#endif
}

int bcnn_backward_concat_layer(bcnn_net *net, bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_concat_layer_gpu(net, conn);
#else
	return bcnn_backward_concat_layer_cpu(net, conn);
#endif
}



