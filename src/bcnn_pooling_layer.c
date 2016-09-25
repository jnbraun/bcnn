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

#include "bcnn/bcnn.h"

int bcnn_add_maxpool_layer(bcnn_net *net, int size, int stride)
{
	int nb_layers = net->nb_layers + 1;
	int output_size, sz;
	bcnn_layer layer = { 0 };

	layer.type = MAXPOOL;
	//memcpy(layer.input_shape, input_shape, 3 * sizeof(int));
	if (net->nb_layers == 0) {
		layer.input_shape[0] = net->w;
		layer.input_shape[1] = net->h;
		layer.input_shape[2] = net->c;
	}
	else
		memcpy(layer.input_shape, net->layers[net->nb_layers - 1].output_shape, 3 * sizeof(int));
	layer.output_shape[0] = (layer.input_shape[0] - 1) / stride + 1;
	layer.output_shape[1] = (layer.input_shape[1] - 1) / stride + 1;
	layer.output_shape[2] = layer.input_shape[2];
	output_size = layer.output_shape[0] * layer.output_shape[1] * layer.output_shape[2];
	layer.size = size;
	layer.stride = stride;

	sz = output_size * net->batch_size;
	layer.indexes = (int *)calloc(sz, sizeof(int));
	layer.output = (float *)calloc(sz, sizeof(float));
	layer.diff = (float *)calloc(sz, sizeof(float));

#ifdef BCNN_USE_CUDA
	layer.indexes_gpu = bcnn_cuda_malloc_i32(sz);
	layer.output_gpu = bcnn_cuda_memcpy_f32(layer.output, sz);
	layer.diff_gpu = bcnn_cuda_memcpy_f32(layer.diff, sz);
#ifdef BCNN_USE_CUDNN
	bcnn_cudnn_check(cudnnCreateTensorDescriptor(&layer.src_tensor_desc));
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&layer.dst_tensor_desc));
	bcnn_cudnn_check(cudnnCreatePoolingDescriptor(&layer.pooling_desc));
	bcnn_cudnn_check(cudnnSetPooling2dDescriptor(layer.pooling_desc,
                                            CUDNN_POOLING_MAX,
                                            layer.size,
                                            layer.size,
                                            0,
                                            0,
                                            layer.stride,
                                            layer.stride));
	bcnn_cudnn_check(cudnnSetTensor4dDescriptor(layer.src_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		net->batch_size, layer.input_shape[2], layer.input_shape[1], layer.input_shape[0])); 
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(layer.dst_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		net->batch_size, layer.output_shape[2], layer.output_shape[1], layer.output_shape[0])); 
#endif
#endif

	bcnn_realloc(net, nb_layers);
	net->layers[nb_layers - 1] = layer;

	fprintf(stderr, "[Maxpool] input_shape= %dx%dx%d size= %d stride= %d ouput_shape= %dx%dx%d\n",
		layer.input_shape[0], layer.input_shape[1], layer.input_shape[2], size, stride,
		layer.output_shape[0], layer.output_shape[1], layer.output_shape[2]);
	return 0;
}


int bcnn_forward_maxpool_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int b, i, j, k, m, n, dst_index, valid, src_index, cur_w, cur_h, max_i;
	float max_f = -FLT_MAX, val;
	int offset_pool = (-layer->size - 1) / 2 + 1;
	int offset0, offset1, offset2;
	int src_w = layer->input_shape[0], src_h = layer->input_shape[1], src_c = layer->input_shape[2];
	int dst_w = layer->output_shape[0], dst_h = layer->output_shape[1], dst_c = layer->output_shape[2];

	for (b = 0; b < wrk->batch_size; ++b) { // batch_size
		offset0 = dst_c * b;
		for (k = 0; k < dst_c; ++k) {	// depth
			offset1 = dst_h * (k + offset0);
			for (i = 0; i < dst_h; ++i) {	// height
				offset2 = dst_w * (offset1 + i);
				for (j = 0; j < dst_w; ++j) {	// width
					dst_index = j + offset2;
					max_f = -FLT_MAX;
					max_i = -1;
					for (n = 0; n < layer->size; ++n) {	// pooling window
						for (m = 0; m < layer->size; ++m) {
							cur_h = offset_pool + i * layer->stride + n;
							cur_w = offset_pool + j * layer->stride + m;
							src_index = cur_w + src_w * (cur_h + src_h * (k + b * src_c));
							valid = (cur_h >= 0 && cur_h < src_h &&
								cur_w >= 0 && cur_w < src_w);
							val = (valid != 0) ? wrk->input[src_index] : -FLT_MAX;
							if (val > max_f) {
								max_f = val;
								max_i = src_index;
							}
						}
					}
					layer->output[dst_index] = max_f;
					layer->indexes[dst_index] = max_i;
				}
			}
		}
	}
	return BCNN_SUCCESS;
}


int bcnn_forward_maxpool_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_maxpool_layer_gpu(layer, wrk);
#else
	return bcnn_forward_maxpool_layer_cpu(layer, wrk);
#endif
}

int bcnn_backward_maxpool_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i, index;
	int sz = layer->output_shape[0] * layer->output_shape[1] * layer->output_shape[2] * wrk->batch_size;
	for (i = 0; i < sz; ++i) {
		index = layer->indexes[i];
		wrk->diff[index] += layer->diff[i];
	}
	return BCNN_SUCCESS;
}

int bcnn_backward_maxpool_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_maxpool_layer_gpu(layer, wrk);
#else
	return bcnn_backward_maxpool_layer_cpu(layer, wrk);
#endif
	return 0;
}
