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


int bcnn_add_activation_layer(bcnn_net *net, bcnn_activation type)
{
	int nb_layers = net->nb_layers + 1;
	int sz = 0;
	bcnn_layer layer = { 0 };
	char type_name[256] = { 0 };

	bh_assert(nb_layers >= 2,
		"Activation layer can't be the first layer of the network", BCNN_INTERNAL_ERROR);

	layer.type = ACTIVATION;
	//memcpy(layer.input_shape, input_shape, 3 * sizeof(int));
	memcpy(layer.input_shape, net->layers[net->nb_layers - 1].output_shape, 3 * sizeof(int));
	memcpy(layer.output_shape, layer.input_shape, 3 * sizeof(int));
	layer.activation = type;
	sz = layer.input_shape[0] * layer.input_shape[1] * layer.input_shape[2] * net->batch_size;

	layer.output = net->layers[nb_layers - 2].output;
	layer.diff = net->layers[nb_layers - 2].diff;
#ifdef BCNN_USE_CUDA
	layer.output_gpu = net->layers[nb_layers - 2].output_gpu;
	layer.diff_gpu = net->layers[nb_layers - 2].diff_gpu;
#endif

	bcnn_realloc(net, nb_layers);
	net->layers[nb_layers - 1] = layer;

	switch (type) {
	case TANH:		sprintf(type_name, "Tanh");			break;
	case RELU:		sprintf(type_name, "Relu");			break;
	case RAMP:		sprintf(type_name, "Ramp");			break;
	case SOFTPLUS:	sprintf(type_name, "Softplus");		break;
	case LRELU:		sprintf(type_name, "Leaky-Relu");	break;
	case ABS:		sprintf(type_name, "AbsVal");		break;
	case CLAMP:		sprintf(type_name, "Clamp");		break;
	}

	fprintf(stderr, "[Activation] input_shape= %dx%dx%d type= %s output_shape= %dx%dx%d\n",
		layer.input_shape[0], layer.input_shape[1], layer.input_shape[2], type_name,
		layer.output_shape[0], layer.output_shape[1], layer.output_shape[2]);

	return BCNN_SUCCESS;
}


int bcnn_forward_activation_cpu(float *x, int sz, bcnn_activation a)
{
	int i;

	switch (a) {
	case TANH:
		for (i = 0; i < sz; ++i)
			x[i] = (float)(exp(2 * x[i]) - 1) /
			((float)exp(2 * x[i]) + 1);
		break;
	case RELU:
		for (i = 0; i < sz; ++i)
			x[i] = x[i] * (x[i] > 0);
		break;
	case LRELU:
		for (i = 0; i < sz; ++i)
			x[i] = (x[i] > 0 ? 
				x[i] : 0.01f * x[i]);
		break;
	case RAMP:
		for (i = 0; i < sz; ++i)
			x[i] = x[i] * (x[i] > 0) + 0.1f * x[i];
		break;
	case SOFTPLUS:
		for (i = 0; i < sz; ++i)
			x[i] = (float)log(1.0f + (float)exp(x[i]));
		break;
	case ABS:
		for (i = 0; i < sz; ++i)
			x[i] = (float)fabs(x[i]);
		break;
	case CLAMP:
		for (i = 0; i < sz; ++i)
			x[i] = bh_clamp(x[i], 0, 1);
		break;
	}
	return BCNN_SUCCESS;
}

int bcnn_forward_activation_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int sz = layer->output_shape[0] * layer->output_shape[1] * layer->output_shape[2] *
		wrk->batch_size;

	layer->output = wrk->input;
	bcnn_forward_activation_cpu(layer->output, sz, layer->activation);

	return BCNN_SUCCESS;
}


int bcnn_backward_activation_cpu(float *x, float *dx, int sz, bcnn_activation a)
{
	int i;

	switch (a) {
	case TANH:
		for (i = 0; i < sz; ++i)
			dx[i] *= (1 - x[i] * x[i]);
		break;
	case RELU:
		for (i = 0; i < sz; ++i)
			dx[i] *= ((float)(x[i] > 0));
		break;
	case LRELU:
		for (i = 0; i < sz; ++i)
			dx[i] *= (x[i] > 0 ? 1.0f : 0.01f);
		break;
	case RAMP:
		for (i = 0; i < sz; ++i)
			dx[i] *= ((float)(x[i] > 0) + 0.1f);
		break;
	case SOFTPLUS:
		for (i = 0; i < sz; ++i)
			dx[i] *= 1.0f / (1.0f + (float)exp(-x[i]));
		break;
	case ABS:
		for (i = 0; i < sz; ++i)
			dx[i] *= (x[i] >= 0 ? 1.0f : -1.0f);
		break;
	case CLAMP:
		for (i = 0; i < sz; ++i)
			dx[i] *= ((float)(x[i] > 0.0f && x[i] < 1.0f));
		break;
	}
	return 0;
}

int bcnn_backward_activation_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int sz = layer->output_shape[0] * layer->output_shape[1] * layer->output_shape[2] *
		wrk->batch_size;
	
	bcnn_backward_activation_cpu(layer->output, layer->diff, sz, layer->activation);
	wrk->diff = layer->diff;

	return BCNN_SUCCESS;
}


int bcnn_forward_activation_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_activation_layer_gpu(layer, wrk);
#else
	return bcnn_forward_activation_layer_cpu(layer, wrk);
#endif
}

int bcnn_backward_activation_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_activation_layer_gpu(layer, wrk);
#else
	return bcnn_backward_activation_layer_cpu(layer, wrk);
#endif
}