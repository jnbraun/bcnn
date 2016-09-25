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
#include <bh/bh_mem.h>

#include "bcnn/bcnn.h"

int bcnn_add_cost_layer(bcnn_net *net, bcnn_loss_metric cost_type, float scale)
{
	int nb_layers = net->nb_layers + 1, sz;
	bcnn_layer layer = { 0 };

	layer.type = COST;
	layer.scale = scale;
	//memcpy(layer.input_shape, input_shape, 3 * sizeof(int));
	memcpy(layer.input_shape, net->layers[net->nb_layers - 1].output_shape, 3 * sizeof(int));
	memcpy(layer.output_shape, layer.input_shape, 3 * sizeof(int));
	layer.cost_type = cost_type;
	sz = layer.input_shape[0] * layer.input_shape[1] * layer.input_shape[2] * net->batch_size;
	layer.diff = (float *)calloc(sz, sizeof(float));
	layer.output = (float *)calloc(1, sizeof(float));
#ifdef BCNN_USE_CUDA
	layer.diff_gpu = bcnn_cuda_memcpy_f32(layer.diff, sz);
#endif

	bcnn_realloc(net, nb_layers);
	net->layers[nb_layers - 1] = layer;

	return 0;
}


int bcnn_forward_cost_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i, j, offset, j_best, n, d;
	int input_size = layer->input_shape[0] * layer->input_shape[1] * layer->input_shape[2];
	int sz = wrk->batch_size * input_size;
	float p_max;
	float *input_cpu = NULL;
	// If no truth available, do nothing
	if (!wrk->truth)
		return 0;

	switch (layer->cost_type) {
	case COST_ERROR:
		*(layer->output) = 0.0f;
		for (i = 0; i < wrk->batch_size; ++i) {
			offset = i * input_size;
			p_max = FLT_MIN;
			j_best = 0;
			for (j = 0; j < input_size; ++j) {
				if (wrk->input[offset + j] > p_max) {
					p_max = wrk->input[offset + j];
					j_best = j;
				}
			}
			if (wrk->truth[offset + j_best] == 0)
				*(layer->output) += 1.0f;
		}
		bcnn_copy_f32(sz, wrk->input, layer->diff);
		bcnn_axpy(sz, -1, wrk->truth, layer->diff);
		break;
	case COST_SSE:
		bcnn_copy_f32(sz, wrk->input, layer->diff);
		bcnn_axpy(sz, -1, wrk->truth, layer->diff);
		*(layer->output) = bcnn_dot(sz, layer->diff, layer->diff);
		break;
	case COST_MSE:
		bcnn_copy_f32(sz, wrk->input, layer->diff);
		bcnn_axpy(sz, -1, wrk->truth, layer->diff);
		*(layer->output) = bcnn_dot(sz, layer->diff, layer->diff);
		*(layer->output) /= input_size;
		break;
	case COST_CRPS:
		*(layer->output) = 0.0f;
		input_cpu = (float *)calloc(sz, sizeof(float));
		for (i = 0; i < wrk->batch_size; ++i) {
			offset = i * input_size;
			for (j = 1; j < input_size; ++j) {
				if (wrk->input[offset + j] < wrk->input[offset + j - 1]) {
					input_cpu[offset + j] = wrk->input[offset + j - 1];
				}
			}
		}
		bcnn_copy_f32(sz, input_cpu, layer->diff);
		bcnn_axpy(sz, -1, wrk->truth, layer->diff);
		*(layer->output) = bcnn_dot(sz, layer->diff, layer->diff);
		bh_free(input_cpu);
		break;
	case COST_LOGLOSS:
		*(layer->output) = 0.0f;
		for (i = 0; i < wrk->batch_size; ++i) {
			offset = i * input_size;
			for (j = 0; j < input_size; ++j) {
				if (wrk->truth[offset + j] > 0.0f) {
					*(layer->output) += (float)-log(bh_clamp(wrk->input[offset + j], 1e-8f, 1.0f - 1e-8f));
				}
			}
		}
		bcnn_copy_f32(sz, wrk->input, layer->diff);
		bcnn_axpy(sz, -1, wrk->truth, layer->diff);
		break;
	case COST_DICE:
		*(layer->output) = 0.0f;
		for (i = 0; i < wrk->batch_size; ++i) {
			offset = i * input_size;
			n = 0;
			d = 0;
			for (j = 0; j < input_size; ++j) {
				n += (int)(wrk->truth[offset + j] * (wrk->input[offset + j] > 0.5f));
				d += (int)(wrk->truth[offset + j] + (wrk->input[offset + j] > 0.5f));
			}
			*(layer->output) += (float)(2.0f * n + 1.0f) / (d + 1.0f);

		}
		bcnn_copy_f32(sz, wrk->input, layer->diff);
		bcnn_axpy(sz, -1, wrk->truth, layer->diff);
		break;
	}

	return BCNN_SUCCESS;
}


int bcnn_backward_cost_layer_cpu(const bcnn_layer *layer, bcnn_workload *wrk)
{
	int input_size = layer->input_shape[0] * layer->input_shape[1] * layer->input_shape[2];
	int sz = wrk->batch_size * input_size;

	bcnn_axpy(sz, layer->scale, layer->diff, wrk->diff);

	return BCNN_SUCCESS;
}

int bcnn_forward_cost_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_cost_layer_gpu(layer, wrk);
#else
	return bcnn_forward_cost_layer_cpu(layer, wrk);
#endif
}

int bcnn_backward_cost_layer(const bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_cost_layer_gpu(layer, wrk);
#else
	return bcnn_backward_cost_layer_cpu(layer, wrk);
#endif
}