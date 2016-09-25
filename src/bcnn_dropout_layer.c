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

int bcnn_add_dropout_layer(bcnn_net *net, float rate)
{
	int nb_layers = net->nb_layers + 1;
	int sz = 0;
	bcnn_layer layer = { 0 };

	bh_assert(nb_layers > 2,
		"Dropout layer can't be the first layer of the network", BCNN_INTERNAL_ERROR);

	layer.type = DROPOUT;
	//memcpy(layer.input_shape, input_shape, 3 * sizeof(int));
	memcpy(layer.input_shape, net->layers[net->nb_layers - 1].output_shape, 3 * sizeof(int));
	memcpy(layer.output_shape, layer.input_shape, 3 * sizeof(int));
	layer.dropout_rate = rate;
	sz = layer.input_shape[0] * layer.input_shape[1] * layer.input_shape[2] * net->batch_size;
	layer.rand = (float *)calloc(sz, sizeof(float));
	layer.scale = 1.0f / (1.0f - rate);
#ifdef BCNN_USE_CUDA
	layer.rand_gpu = bcnn_cuda_memcpy_f32(layer.rand, sz);
#endif

	layer.output = net->layers[nb_layers - 2].output;
	layer.diff = net->layers[nb_layers - 2].diff;
#ifdef BCNN_USE_CUDA
	layer.output_gpu = net->layers[nb_layers - 2].output_gpu;
	layer.diff_gpu = net->layers[nb_layers - 2].diff_gpu;
#endif

	bcnn_realloc(net, nb_layers);
	net->layers[nb_layers - 1] = layer;

	fprintf(stderr, "[Dropout] input_shape= %dx%dx%d rate= %f output_shape= %dx%dx%d\n",
		layer.input_shape[0], layer.input_shape[1], layer.input_shape[2], rate,
		layer.output_shape[0], layer.output_shape[1], layer.output_shape[2]);
	return 0;
}


int bcnn_forward_dropout_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i, sz = layer->input_shape[0] * layer->input_shape[1] * layer->input_shape[2] * wrk->batch_size;
	float r;

	if (!wrk->train)
		return 0;

	for (i = 0; i < sz; ++i) {
		r = (float)rand() / RAND_MAX;
		layer->rand[i] = r;
		if (r < layer->dropout_rate)
			wrk->input[i] = 0;
		else
			wrk->input[i] *= layer->scale;
	}
	return BCNN_SUCCESS;
}

int bcnn_forward_dropout_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_dropout_layer_gpu(layer, wrk);
#else
	return bcnn_forward_dropout_layer_cpu(layer, wrk);
#endif
}


int bcnn_backward_dropout_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i, sz = layer->input_shape[0] * layer->input_shape[1] * layer->input_shape[2] * wrk->batch_size;
	float r;

	if (!wrk->diff)
		return 0;

	for (i = 0; i < sz; ++i) {
		r = layer->rand[i];
		if (r < layer->dropout_rate)
			wrk->diff[i] = 0;
		else
			wrk->diff[i] *= layer->scale;
	}
	return BCNN_SUCCESS;
}

int bcnn_backward_dropout_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_dropout_layer_gpu(layer, wrk);
#else
	return bcnn_backward_dropout_layer_cpu(layer, wrk);
#endif
}