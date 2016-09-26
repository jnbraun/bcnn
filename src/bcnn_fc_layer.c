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

int bcnn_add_fullc_layer(bcnn_net *net, int output_size, bcnn_weights_init init, bcnn_activation activation)
{
	int nb_layers = net->nb_layers + 1;
	int i;
	float std_init = 0.0f;
	bcnn_layer layer = { 0 };
	int input_size = 0;

	layer.type = FULL_CONNECTED;
	/*layer.input_shape[0] = 1;
	layer.input_shape[1] = 1;
	layer.input_shape[2] = input_size;*/
	if (net->nb_layers == 0) {
		layer.input_shape[0] = 1;
		layer.input_shape[1] = 1;
		layer.input_shape[2] = net->w * net->h * net->c;
	}
	else {
		layer.input_shape[0] = 1;
		layer.input_shape[1] = 1;
		layer.input_shape[2] = net->layers[net->nb_layers - 1].output_shape[0]
			 * net->layers[net->nb_layers - 1].output_shape[1] * net->layers[net->nb_layers - 1].output_shape[2];
	}
	input_size = layer.input_shape[0] * layer.input_shape[1] * layer.input_shape[2];
	layer.output_shape[0] = 1;
	layer.output_shape[1] = 1;
	layer.output_shape[2] = output_size;
	layer.bias_size = output_size;
	layer.weights_size = input_size * output_size;
	
	layer.output = (float *)calloc(net->batch_size * output_size, sizeof(float));
	layer.diff = (float *)calloc(net->batch_size * output_size, sizeof(float));
	layer.weight_diff = (float *)calloc(input_size * output_size, sizeof(float));
	layer.bias_diff = (float *)calloc(output_size, sizeof(float));
	layer.weight = (float *)calloc(output_size * input_size, sizeof(float));
	layer.bias = (float *)calloc(output_size, sizeof(float));

	switch (init) {
	case XAVIER:
		std_init = (float)sqrt(3.0f / input_size);
		break;
	case MSRA:
		std_init = (float)sqrt(2.0f / input_size);
		break;
	}
	
	for (i = 0; i < output_size * input_size; ++i)
		layer.weight[i] = std_init * (2 * ((float)rand() / RAND_MAX) - 1);
	for (i = 0; i < output_size; ++i)
		layer.bias[i] = std_init;

#ifdef BCNN_USE_CUDA
	layer.weight_gpu = bcnn_cuda_memcpy_f32(layer.weight, output_size * input_size);
	layer.bias_gpu = bcnn_cuda_memcpy_f32(layer.bias, output_size);

	layer.weight_diff_gpu = bcnn_cuda_memcpy_f32(layer.weight_diff, output_size * input_size);
	layer.bias_diff_gpu = bcnn_cuda_memcpy_f32(layer.bias_diff, output_size);

	layer.output_gpu = bcnn_cuda_memcpy_f32(layer.output, output_size * net->batch_size);
	layer.diff_gpu = bcnn_cuda_memcpy_f32(layer.diff, output_size * net->batch_size);
#endif
	layer.activation = activation;

	bcnn_realloc(net, nb_layers);

	net->layers[nb_layers - 1] = layer;

	fprintf(stderr, "[Connected] input_shape= %dx%dx%d output_shape= %dx%dx%d\n",
		layer.input_shape[0], layer.input_shape[1], layer.input_shape[2],
		layer.output_shape[0], layer.output_shape[1], layer.output_shape[2]);

	return 0;
}

int bcnn_forward_fullc_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i, input_size = layer->input_shape[2], output_size = layer->output_shape[2];

	for (i = 0; i < wrk->batch_size; ++i)
		bcnn_copy_f32(output_size, layer->bias, layer->output + i * output_size);

	bcnn_gemm(0, 1, wrk->batch_size, output_size, input_size, 1,
		wrk->input, input_size, layer->weight, input_size, 1, layer->output, output_size);

	return BCNN_SUCCESS;
}


int bcnn_backward_fullc_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i, input_size = layer->input_shape[2], output_size = layer->output_shape[2];

	for (i = 0; i < wrk->batch_size; ++i)
		bcnn_axpy(output_size, 1, layer->diff + i * output_size, layer->bias_diff);

	bcnn_gemm(1, 0, output_size, input_size, wrk->batch_size, 1,
		layer->diff, output_size, wrk->input, input_size, 1, layer->weight_diff, input_size);

	if (wrk->diff)
		bcnn_gemm(0, 0, wrk->batch_size, input_size, output_size, 1,
			layer->diff, output_size, layer->weight, input_size, 1, wrk->diff, input_size);

	return BCNN_SUCCESS;
}

int bcnn_forward_fullc_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_fullc_layer_gpu(layer, wrk);
#else
	return bcnn_forward_fullc_layer_cpu(layer, wrk);
#endif
}

int bcnn_backward_fullc_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_fullc_layer_gpu(layer, wrk);
#else
	return bcnn_backward_fullc_layer_cpu(layer, wrk);
#endif
}