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
#include <bh/bh_error.h>

#include "bcnn/bcnn.h"

int bcnn_add_concat_layer(bcnn_net *net, int layer_index)
{
	int nb_layers = net->nb_layers + 1;
	int sz = 0;
	bcnn_layer layer = { 0 };
	char type_name[256] = { 0 };

	bh_assert(nb_layers >= 2,
		"Concat layer can't be the first layer of the network", BCNN_INTERNAL_ERROR);

	layer.type = CONCAT;
	layer.concat_index = layer_index;
	//memcpy(layer.input_shape, input_shape, 3 * sizeof(int));
	memcpy(layer.input_shape, net->layers[net->nb_layers - 1].output_shape, 3 * sizeof(int));
	layer.output_shape[0] = layer.input_shape[0];
	layer.output_shape[1] = layer.input_shape[1];
	layer.output_shape[2] = layer.input_shape[2] + net->layers[layer_index].output_shape[2];

	sz = layer.output_shape[0] * layer.output_shape[1] * layer.output_shape[2] * net->batch_size;

	layer.output = (float *)calloc(sz, sizeof(float));
	layer.diff = (float *)calloc(sz, sizeof(float));
#ifdef BCNN_USE_CUDA
	layer.output_gpu = bcnn_cuda_memcpy_f32(layer.output, sz);
	layer.diff_gpu = bcnn_cuda_memcpy_f32(layer.diff, sz);
#endif

	bcnn_realloc(net, nb_layers);
	net->layers[nb_layers - 1] = layer;

	fprintf(stderr, "[Concat %d with %d] input_shape= %dx%dx%d output_shape= %dx%dx%d\n",
		nb_layers - 2, layer_index,
		layer.input_shape[0], layer.input_shape[1], layer.input_shape[2],
		layer.output_shape[0], layer.output_shape[1], layer.output_shape[2]);

	return BCNN_SUCCESS;
}

int bcnn_forward_concat_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i;
	int spatial_sz = layer->input_shape[0] * layer->input_shape[1];
	int sz = spatial_sz * layer->input_shape[2];
	int sz2 =  spatial_sz * (layer->output_shape[2] - layer->input_shape[2]);
	int dst_sz = sz + sz2;

	for (i = 0; i < wrk->batch_size; ++i) {
		bcnn_copy_f32(sz, wrk->input + i * sz, layer->output + i * dst_sz);
		bcnn_copy_f32(sz2, wrk->input2 + i * sz2, layer->output + i * dst_sz + sz);
	}

	return BCNN_SUCCESS;
}


int bcnn_backward_concat_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i;
	int spatial_sz = layer->input_shape[0] * layer->input_shape[1];
	int sz = spatial_sz * layer->input_shape[2];
	int sz2 =  spatial_sz * (layer->output_shape[2] - layer->input_shape[2]);
	int dst_sz = sz + sz2;

	for (i = 0; i < wrk->batch_size; ++i) {
		bcnn_copy_f32(sz, layer->diff + i * dst_sz, wrk->diff + i * sz);
		bcnn_copy_f32(sz2, layer->diff + i * dst_sz + sz, wrk->diff2 + i * sz2);
	}

	return BCNN_SUCCESS;
}


int bcnn_forward_concat_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_concat_layer_gpu(layer, wrk);
#else
	return bcnn_forward_concat_layer_cpu(layer, wrk);
#endif
}

int bcnn_backward_concat_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_concat_layer_gpu(layer, wrk);
#else
	return bcnn_backward_concat_layer_cpu(layer, wrk);
#endif
}