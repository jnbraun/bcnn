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
	int nb_layers = net->nb_layers + 1;
	bcnn_layer layer = { 0 };
	int sz = 0;

	layer.type = SOFTMAX;
	/*layer.input_shape[0] = input_shape[0];
	layer.input_shape[1] = input_shape[1];
	layer.input_shape[2] = input_shape[2];*/
	if (net->nb_layers == 0) {
		layer.input_shape[0] = net->w;
		layer.input_shape[1] = net->h;
		layer.input_shape[2] = net->c;
	}
	else
		memcpy(layer.input_shape, net->layers[net->nb_layers - 1].output_shape, 3 * sizeof(int));
	memcpy(layer.output_shape, layer.input_shape, 3 * sizeof(int));
	sz = layer.input_shape[0] * layer.input_shape[1] * layer.input_shape[2] * net->batch_size;
	layer.output = (float *)calloc(sz, sizeof(float));
	layer.diff = (float *)calloc(sz, sizeof(float));
#ifdef BCNN_USE_CUDA
	layer.output_gpu = bcnn_cuda_memcpy_f32(layer.output, sz);
	layer.diff_gpu = bcnn_cuda_memcpy_f32(layer.diff, sz);
#endif

	bcnn_realloc(net, nb_layers);
	net->layers[nb_layers - 1] = layer;

	fprintf(stderr, "[Softmax] input_shape= %dx%dx%d output_shape= %dx%dx%d\n",
		layer.input_shape[0], layer.input_shape[1], layer.input_shape[2],
		layer.output_shape[0], layer.output_shape[1], layer.output_shape[2]);
	return 0;
}

int bcnn_forward_softmax_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int b, i;
	int input_size = layer->input_shape[0] * layer->input_shape[1] * layer->input_shape[2];
	float largest = -FLT_MAX;
	float sum = 0;

	for (b = 0; b < wrk->batch_size; ++b) {
		largest = -FLT_MAX;
		sum = 0;
		for (i = 0; i < input_size; ++i) {
			if (wrk->input[b * input_size + i] > largest)
				largest = wrk->input[b * input_size + i];
		}
		for (i = 0; i < input_size; ++i){
			sum += (float)exp(wrk->input[b * input_size + i] - largest);
		}
		if (sum)
			sum = largest + (float)log(sum);
		else
			sum = largest - 100.0f;
		for (i = 0; i < input_size; ++i){
			layer->output[b * input_size + i] = (float)exp(wrk->input[b * input_size + i] - sum);
		}
	}
	return BCNN_SUCCESS;
}

int bcnn_forward_softmax_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_softmax_layer_gpu(layer, wrk);
#else
	return bcnn_forward_softmax_layer_cpu(layer, wrk);
#endif
}


int bcnn_backward_softmax_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i, sz = layer->input_shape[0] * layer->input_shape[1] * layer->input_shape[2];
	for (i = 0; i < sz * wrk->batch_size; ++i)
		wrk->diff[i] += layer->diff[i];
	return BCNN_SUCCESS;
}

int bcnn_backward_softmax_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_softmax_layer_gpu(layer, wrk);
#else
	return bcnn_backward_softmax_layer_cpu(layer, wrk);
#endif
}