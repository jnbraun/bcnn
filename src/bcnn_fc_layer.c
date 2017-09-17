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
#include <bh/bh_string.h>

#include "bcnn/bcnn.h"

int bcnn_add_fullc_layer(bcnn_net *net, int output_size, bcnn_weights_init init, bcnn_activation activation, int quantize, char *id)
{
	int nb_connections = net->nb_connections + 1;
	int i;
	float std_init = 0.0f;
	bcnn_gauss_gen g = { 0 };
	bcnn_connection conn = { 0 };
	int input_size = 0;
	
	if (id != NULL)
		bh_fill_option(&conn.id, id);
	conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
	conn.layer->type = FULL_CONNECTED;
	if (nb_connections > 1)
		conn.src_tensor = net->connections[nb_connections - 2].dst_tensor;
	else
		conn.src_tensor = net->input_node;

	conn.dst_tensor.w = 1;
	conn.dst_tensor.h = 1;
	conn.dst_tensor.c = output_size;
	conn.dst_tensor.b = conn.src_tensor.b;

	conn.layer->bias_size = output_size;
	input_size = conn.src_tensor.w * conn.src_tensor.h * conn.src_tensor.c;
	conn.layer->weights_size = input_size * output_size;
	
	conn.dst_tensor.data = (float *)calloc(conn.dst_tensor.b * output_size, sizeof(float));
	conn.dst_tensor.grad_data = (float *)calloc(conn.dst_tensor.b * output_size, sizeof(float));
	conn.layer->weight_diff = (float *)calloc(input_size * output_size, sizeof(float));
	conn.layer->bias_diff = (float *)calloc(output_size, sizeof(float));
	conn.layer->weight = (float *)calloc(output_size * input_size, sizeof(float));
	conn.layer->bias = (float *)calloc(output_size, sizeof(float));

	conn.layer->quantize = quantize;
	if (conn.layer->quantize == 1) {
		bh_assert((input_size % BITS_IN_UINT32 == 0), "Number of channels in input must be a multiple of 32", BCNN_INVALID_PARAMETER);
		conn.layer->binary_workspace = (uint32_t *)calloc(input_size * conn.src_tensor.b / (sizeof(float) * 8), sizeof(float));
		conn.layer->binary_weight = (uint32_t *)calloc(conn.layer->weights_size / BITS_IN_UINT32, sizeof(uint32_t));
	}

	switch (init) {
	case XAVIER:
		std_init = (float)sqrt(3.0f / input_size);
		for (i = 0; i < conn.layer->weights_size; ++i)
			conn.layer->weight[i] = std_init * (2 * ((float)rand() / RAND_MAX) - 1);
		break;
	case MSRA:
		std_init = (float)sqrt(2.0f / input_size);
		for (i = 0; i < conn.layer->weights_size; ++i)
			conn.layer->weight[i] = std_init * bcnn_rng_gaussian(&g);
		break;
	}
	
	/*for (i = 0; i < output_size; ++i)
		conn.layer->bias[i] = std_init;*/
	if (net->learner.optimizer == ADAM) {
		conn.layer->adam_m = (float *)calloc(conn.layer->weights_size, sizeof(float));
		conn.layer->adam_v = (float *)calloc(conn.layer->weights_size, sizeof(float));
	}

#ifdef BCNN_USE_CUDA
	conn.layer->weight_gpu = bcnn_cuda_memcpy_f32(conn.layer->weight, output_size * input_size);
	conn.layer->bias_gpu = bcnn_cuda_memcpy_f32(conn.layer->bias, output_size);

	conn.layer->weight_diff_gpu = bcnn_cuda_memcpy_f32(conn.layer->weight_diff, output_size * input_size);
	conn.layer->bias_diff_gpu = bcnn_cuda_memcpy_f32(conn.layer->bias_diff, output_size);

	conn.dst_tensor.data_gpu = bcnn_cuda_memcpy_f32(conn.dst_tensor.data, output_size * conn.dst_tensor.b);
	conn.dst_tensor.grad_data_gpu = bcnn_cuda_memcpy_f32(conn.dst_tensor.grad_data, output_size * conn.dst_tensor.b);
	if (net->learner.optimizer == ADAM) {
		conn.layer->adam_m_gpu = bcnn_cuda_memcpy_f32(conn.layer->adam_m, conn.layer->weights_size);
		conn.layer->adam_v_gpu = bcnn_cuda_memcpy_f32(conn.layer->adam_v, conn.layer->weights_size);
	}
#endif
	conn.layer->activation = activation;
	net->nb_connections = nb_connections;
	bcnn_net_add_connection(net, conn);
	
	fprintf(stderr, "[Connected] input_shape= %dx%dx%d output_shape= %dx%dx%d\n",
		conn.src_tensor.w, conn.src_tensor.h, conn.src_tensor.c,
		conn.dst_tensor.w, conn.dst_tensor.h, conn.dst_tensor.c);

	return 0;
}

int bcnn_forward_fullc_layer_cpu(bcnn_connection *conn)
{
	int i, batch_size = conn->dst_tensor.b;
	int src_size = conn->src_tensor.w * conn->src_tensor.h * conn->src_tensor.c;
	int dst_size = conn->dst_tensor.w * conn->dst_tensor.h * conn->dst_tensor.c;
	int sz = dst_size * conn->dst_tensor.b;
	bcnn_layer *layer = conn->layer;
	bcnn_tensor src = conn->src_tensor;
	bcnn_tensor dst = conn->dst_tensor;

	/*for (i = 0; i < batch_size; ++i)
		bcnn_copy_f32(dst_size, layer->bias, dst.data + i * dst_size);*/
	memset(dst.data, 0, dst_size * batch_size * sizeof(float));

	if (layer->quantize) {
		for (i = 0; i < layer->weights_size; ++i) {
			layer->weight[i] = (layer->weight[i] > 0) ? 1.0f : -1.0f;
		}
		for (i = 0; i < batch_size * src_size; ++i) {
			src.data[i] = (src.data[i] > 0) ? 1.0f : -1.0f;
		}
		if (conn->state == 0) {
			get_binary_col_unrolled(layer->weight, layer->binary_weight, src_size, dst_size);
			get_binary_row(src.data, layer->binary_workspace, batch_size * src_size);
			bcnn_xnor_gemm(0, 0, batch_size, dst_size, src_size / BITS_IN_UINT32, 1.0f,
					layer->binary_workspace, src_size / BITS_IN_UINT32,
					layer->binary_weight, dst_size,
					1.0f,
					dst.data, dst_size);
		}
		else {
			/*bcnn_gemm(0, 1, batch_size, dst_size, src_size, 1.0f,
				src.data, src_size, layer->weight, src_size, 1.0f, dst.data, dst_size);*/
			bcnn_gemm(0, 0, batch_size, dst_size, src_size, 1.0f,
				src.data, src_size, layer->weight, dst_size, 1.0f, dst.data, dst_size);
			// Mapping to obtain similar range output than xnor gemm
			for (i = 0; i < batch_size * dst_size; ++i) {
				dst.data[i] = (dst.data[i] + src_size) / 2;
			}
		}
	}
	else {
		/*bcnn_gemm(0, 1, batch_size, dst_size, src_size, 1.0f,
			src.data, src_size, layer->weight, src_size, 1.0f, dst.data, dst_size);*/
		bcnn_gemm(0, 0, batch_size, dst_size, src_size, 1.0f,
				src.data, src_size, layer->weight, dst_size, 1.0f, dst.data, dst_size);
	}
		
	for (i = 0; i < batch_size; ++i)
		bcnn_axpy(dst_size, 1, layer->bias, dst.data + i * dst_size);

	bcnn_forward_activation_cpu(dst.data, sz, layer->activation);

	return BCNN_SUCCESS;
}

int bcnn_backward_fullc_layer_cpu(bcnn_connection *conn)
{
	int i, batch_size = conn->dst_tensor.b;
	int src_size = conn->src_tensor.w * conn->src_tensor.h * conn->src_tensor.c;
	int dst_size = conn->dst_tensor.w * conn->dst_tensor.h * conn->dst_tensor.c;
	int sz = dst_size * conn->dst_tensor.b;
	bcnn_layer *layer = conn->layer;
	bcnn_tensor src = conn->src_tensor;
	bcnn_tensor dst = conn->dst_tensor;

	bcnn_backward_activation_cpu(dst.data, dst.grad_data, sz, layer->activation);

	for (i = 0; i < batch_size; ++i)
		bcnn_axpy(dst_size, 1, dst.grad_data + i * dst_size, layer->bias_diff);

	/*bcnn_gemm(1, 0, dst_size, src_size, batch_size, 1,
		dst.grad_data, dst_size, src.data, src_size, 1, layer->weight_diff, src_size);*/
	bcnn_gemm(1, 0, src_size, dst_size, batch_size, 1,
		src.data, src_size, dst.grad_data, dst_size, 1, layer->weight_diff, dst_size);

	if (src.grad_data)
		bcnn_gemm(0, 1, batch_size, src_size, dst_size, 1,
			dst.grad_data, dst_size, layer->weight, dst_size, 1, src.grad_data, src_size);
		/*bcnn_gemm(0, 0, batch_size, src_size, dst_size, 1,
			dst.grad_data, dst_size, layer->weight, src_size, 1, src.grad_data, src_size);*/

	if (layer->quantize && src.grad_data) {
		for (i = 0; i < batch_size * src_size; ++i) {
			src.grad_data[i] = src.grad_data[i] * ((fabs(src.data[i]) <= 1.0f) ? 1.0f : 0.0f);
		}
	}

	return BCNN_SUCCESS;
}

int bcnn_forward_fullc_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_fullc_layer_gpu(conn);
#else
	return bcnn_forward_fullc_layer_cpu(conn);
#endif
}

int bcnn_backward_fullc_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_fullc_layer_gpu(conn);
#else
	return bcnn_backward_fullc_layer_cpu(conn);
#endif
}