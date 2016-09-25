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

static int _bcnn_compute_conv_output_shape(bcnn_layer *layer)
{
	layer->output_shape[0] = (layer->input_shape[0] + 2 * layer->pad - layer->size) / layer->stride + 1;
	layer->output_shape[1] = (layer->input_shape[1] + 2 * layer->pad - layer->size) / layer->stride + 1;
	return 0;
}

static bh_inline int is_a_positive_and_inferior_to_b(int a, int b)
{
	return (unsigned int)a < (unsigned int)b;
}

static int _bcnn_im2col(const float* data_im, const int channels, const int height, const int width,
	const int kernel_size, const int pad, const int stride, float* data_col)
{
	int channel, kernel_row, kernel_col, output_rows, output_cols, input_col, input_row, output_col;
	const int output_h = (height + 2 * pad - kernel_size) / stride + 1;
	const int output_w = (width + 2 * pad - kernel_size) / stride + 1;
	const int channel_size = height * width;

	for (channel = channels; channel--; data_im += channel_size) {
		for (kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
			for (kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
				input_row = -pad + kernel_row;
				for (output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_positive_and_inferior_to_b(input_row, height)) {
						for (output_cols = output_w; output_cols; output_cols--) {
							*(data_col++) = 0;
						}
					}
					else {
						input_col = -pad + kernel_col;
						for (output_col = output_w; output_col; output_col--) {
							if (is_a_positive_and_inferior_to_b(input_col, width)) {
								*(data_col++) = data_im[input_row * width + input_col];
							} 
							else {
								*(data_col++) = 0;
							}
							input_col += stride;
						}
					}
					input_row += stride;
				}
			}
		}
	}
	return 0;
}


static int _bcnn_col2im(const float* data_col, const int channels, const int height, const int width,
	const int kernel, const int pad, const int stride, float* data_im) 
{
	int channel, kernel_row, kernel_col, output_rows, input_col, input_row, output_col;
	const int output_h = (height + 2 * pad - kernel) / stride + 1;
	const int output_w = (width + 2 * pad - kernel) / stride + 1;
	const int channel_size = height * width;

	bcnn_fill_f32(height * width * channels, 0.0f, data_im);
	
	for (channel = channels; channel--; data_im += channel_size) {
		for (kernel_row = 0; kernel_row < kernel; kernel_row++) {
			for (kernel_col = 0; kernel_col < kernel; kernel_col++) {
				input_row = -pad + kernel_row;
				for (output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_positive_and_inferior_to_b(input_row, height)) {
						data_col += output_w;
					}
					else {
						input_col = -pad + kernel_col;
						for (output_col = output_w; output_col; output_col--) {
							if (is_a_positive_and_inferior_to_b(input_col, width)) {
								data_im[input_row * width + input_col] += *data_col;
							}
							data_col++;
							input_col += stride;
						}
					}
					input_row += stride;
				}
			}
		}
	}
	return 0;
}

static int _bcnn_add_bias(float *output, float *bias, int batch_size, int n, int size)
{
	int i, j, b;

	for (b = 0; b < batch_size; ++b) {
		for (i = 0; i < n; ++i) {
			for (j = 0; j < size; ++j) {
				output[(b * n + i) * size + j] += bias[i];
			}
		}
	}
	return 0;
}

static int _bcnn_backward_bias(float *bias_diff, float *diff, int batch_size, int n, int size)
{
    int i, j, b;
	float *p = NULL;

    for (b = 0; b < batch_size; ++b) {
        for (i = 0; i < n; ++i) {
			p = diff + size * (i + b * n);
			for (j = 0; j < size; ++j)
				bias_diff[i] += p[j];
        }
    }
	return 0;
}


int bcnn_add_convolutional_layer(bcnn_net *net, int n, int size, int stride, int pad,
	int batch_norm, bcnn_weights_init init, bcnn_activation activation)
{
	int nb_layers = net->nb_layers + 1;
	int i, sz;
	float std_init = 0.0f;
	bcnn_layer layer = { 0 };
#ifdef BCNN_USE_CUDNN
	size_t cudnn_wrk_sz = 0;
#endif

	layer.type = CONVOLUTIONAL;

	//memcpy(layer.input_shape, input_shape, 3 * sizeof(int));
	if (net->nb_layers == 0) {
		layer.input_shape[0] = net->w;
		layer.input_shape[1] = net->h;
		layer.input_shape[2] = net->c;
	}
	else
		memcpy(layer.input_shape, net->layers[net->nb_layers - 1].output_shape, 3 * sizeof(int));

	layer.num = n;
	layer.stride = stride;
	layer.size = size;
	layer.pad = pad;
	layer.bias_size = n;
	layer.weights_size = layer.input_shape[2] * n * size * size;

	layer.weight = (float *)calloc(layer.weights_size, sizeof(float));
	layer.weight_diff = (float *)calloc(layer.weights_size, sizeof(float));
	layer.bias = (float *)calloc(layer.bias_size, sizeof(float));
	layer.bias_diff = (float *)calloc(layer.bias_size, sizeof(float));

	switch (init) {
	case XAVIER:
		std_init = (float)sqrt(3.0f / (size * size * layer.input_shape[2]));
		break;
	case MSRA:
		std_init = (float)sqrt(2.0f / (size * size * layer.input_shape[2]));
		break;
	}
	
	for (i = 0; i < layer.weights_size; ++i)
		layer.weight[i] = std_init * (2 * ((float)rand() / RAND_MAX) - 1);
	
	_bcnn_compute_conv_output_shape(&layer);
	layer.output_shape[2] = n;

	sz = layer.output_shape[0] * layer.output_shape[1] * layer.input_shape[2] * size * size;
	layer.conv_workspace = (float *)calloc(sz, sizeof(float));
	sz = net->batch_size * layer.output_shape[0] * layer.output_shape[1] * n;
	layer.output = (float *)calloc(sz, sizeof(float));
	layer.diff = (float *)calloc(sz, sizeof(float));

#ifdef BCNN_USE_CUDA
	layer.weight_gpu = bcnn_cuda_memcpy_f32(layer.weight, layer.weights_size);
	layer.weight_diff_gpu = bcnn_cuda_memcpy_f32(layer.weight_diff, layer.weights_size);
	layer.bias_gpu = bcnn_cuda_memcpy_f32(layer.bias, layer.bias_size);
	layer.bias_diff_gpu = bcnn_cuda_memcpy_f32(layer.bias_diff, layer.bias_size);

	sz = net->batch_size * layer.output_shape[0] * layer.output_shape[1] * n;
	layer.output_gpu = bcnn_cuda_memcpy_f32(layer.output, sz);
	layer.diff_gpu = bcnn_cuda_memcpy_f32(layer.diff, sz);
#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&layer.src_tensor_desc));
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&layer.dst_tensor_desc));
    bcnn_cudnn_check(cudnnCreateFilterDescriptor(&layer.filter_desc));
	bcnn_cudnn_check(cudnnCreateTensorDescriptor(&layer.bias_desc));
    bcnn_cudnn_check(cudnnCreateConvolutionDescriptor(&layer.conv_desc));  
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(layer.src_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		net->batch_size, layer.input_shape[2], layer.input_shape[1], layer.input_shape[0])); 
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(layer.dst_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		net->batch_size, layer.output_shape[2], layer.output_shape[1], layer.output_shape[0])); 
    bcnn_cudnn_check(cudnnSetFilter4dDescriptor(layer.filter_desc, CUDNN_DATA_FLOAT,
		layer.num, layer.input_shape[2], layer.size, layer.size));
	bcnn_cudnn_check(cudnnSetTensor4dDescriptor(layer.bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		1, layer.output_shape[2], 1, 1));
    bcnn_cudnn_check(cudnnSetConvolution2dDescriptor(layer.conv_desc, layer.pad, layer.pad, layer.stride, layer.stride, 1, 1, CUDNN_CROSS_CORRELATION));
    bcnn_cudnn_check(cudnnGetConvolutionForwardAlgorithm(bcnn_cudnn_handle(),
            layer.src_tensor_desc,
            layer.filter_desc,
            layer.conv_desc,
            layer.dst_tensor_desc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &layer.fwd_algo));
    bcnn_cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm(bcnn_cudnn_handle(),
            layer.filter_desc,
            layer.dst_tensor_desc_diff,
            layer.conv_desc,
            layer.src_tensor_desc_diff,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &layer.bwd_data_algo));
    bcnn_cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithm(bcnn_cudnn_handle(),
            layer.src_tensor_desc,
            layer.dst_tensor_desc,
            layer.conv_desc,
            layer.filter_desc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &layer.bwd_filter_algo));
    bcnn_cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(bcnn_cudnn_handle(),
            layer.src_tensor_desc,
            layer.filter_desc,
            layer.conv_desc,
            layer.dst_tensor_desc,
            layer.fwd_algo,
            &cudnn_wrk_sz));
    layer.workspace_size = bh_max(layer.workspace_size, cudnn_wrk_sz);
    bcnn_cudnn_check(cudnnGetConvolutionBackwardFilterWorkspaceSize(bcnn_cudnn_handle(),
            layer.src_tensor_desc,
            layer.dst_tensor_desc,
            layer.conv_desc,
            layer.filter_desc,
            layer.bwd_filter_algo,
            &cudnn_wrk_sz));
    layer.workspace_size = bh_max(layer.workspace_size, cudnn_wrk_sz);
    bcnn_cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(bcnn_cudnn_handle(),
            layer.filter_desc,
            layer.dst_tensor_desc,
            layer.conv_desc,
            layer.src_tensor_desc,
            layer.bwd_data_algo,
            &cudnn_wrk_sz));
    layer.workspace_size = bh_max(layer.workspace_size, cudnn_wrk_sz);
	layer.conv_workspace_gpu = bcnn_cuda_malloc_f32(layer.workspace_size);
#else
	sz = layer.output_shape[0] * layer.output_shape[1] * layer.input_shape[2] * size * size;
	layer.conv_workspace_gpu = bcnn_cuda_memcpy_f32(layer.conv_workspace, sz);
#endif
#endif
	//layer.activation = activation;

	bcnn_realloc(net, nb_layers);

	net->layers[nb_layers - 1] = layer;

	fprintf(stderr, "[Convolutional] input_shape= %dx%dx%d nb_filters= %d kernel_size= %d stride= %d padding= %d output_shape= %dx%dx%d\n",
		layer.input_shape[0], layer.input_shape[1], layer.input_shape[2], n, size, stride, pad,
		layer.output_shape[0], layer.output_shape[1], layer.output_shape[2]);

	return 0;
}


int bcnn_forward_conv_layer_cpu(const bcnn_layer *layer, bcnn_workload *wrk)
{
	int i, m, n, k, sz;
	float *a = NULL, *b = NULL, *c = NULL;

	sz = wrk->batch_size * layer->output_shape[0] * layer->output_shape[1] * layer->output_shape[2];

	bcnn_fill_f32(sz, 0.0f, layer->output);

	m = layer->num;
	k = layer->size * layer->size * layer->input_shape[2];
	n = layer->output_shape[0] * layer->output_shape[1];

	sz = layer->input_shape[2] * layer->input_shape[1] * layer->input_shape[0];

	a = layer->weight;
	b = layer->conv_workspace;
	c = layer->output;
	
	for (i = 0; i < wrk->batch_size; ++i) {
		_bcnn_im2col(wrk->input, layer->input_shape[2], layer->input_shape[1], layer->input_shape[0],
			layer->size, layer->pad, layer->stride, b);
		bcnn_gemm(0, 0, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
		c += n * m;
		wrk->input += sz;
	}

	_bcnn_add_bias(layer->output, layer->bias, wrk->batch_size, layer->num, layer->output_shape[0] * layer->output_shape[1]);

	return BCNN_SUCCESS;
}




int bcnn_backward_conv_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i, sz = layer->input_shape[0] * layer->input_shape[1] * layer->input_shape[2];
	int m = layer->num;
	int n = layer->size * layer->size * layer->input_shape[2];
	int k = layer->output_shape[0] * layer->output_shape[1];
	float *a = NULL, *b = NULL, *c = NULL, *src = NULL;

	_bcnn_backward_bias(layer->bias_diff, layer->diff, wrk->batch_size, layer->num, k);

	for (i = 0; i < wrk->batch_size; ++i){
		a = layer->diff + i * m * k;
		b = layer->conv_workspace;
		c = layer->weight_diff;

		src = wrk->input + i * sz;
		_bcnn_im2col(src, layer->input_shape[2], layer->input_shape[1], layer->input_shape[0],
			layer->size, layer->pad, layer->stride, b);
		bcnn_gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

		if (wrk->diff) {
			a = layer->weight;
			b = layer->diff + i * m * k;
			c = layer->conv_workspace;

			bcnn_gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);
			_bcnn_col2im(layer->conv_workspace, layer->input_shape[2], layer->input_shape[1], layer->input_shape[0],
				layer->size, layer->pad, layer->stride, wrk->diff + i * sz);
		}
	}

	return BCNN_SUCCESS;
}


int bcnn_forward_conv_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_conv_layer_gpu(layer, wrk);
#else
	return bcnn_forward_conv_layer_cpu(layer, wrk);
#endif
}

int bcnn_backward_conv_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_conv_layer_gpu(layer, wrk);
#else
	return bcnn_backward_conv_layer_cpu(layer, wrk);
#endif
}


/* Deconv layer */
static int _bcnn_compute_deconv_output_shape(bcnn_layer *layer)
{
	layer->output_shape[0] = layer->stride * (layer->input_shape[0] - 1) + layer->size - 2 * layer->pad;
	layer->output_shape[1] = layer->stride * (layer->input_shape[1] - 1) + layer->size - 2 * layer->pad;
	return BCNN_SUCCESS;
}

int bcnn_add_deconvolutional_layer(bcnn_net *net, int n, int size, int stride, int pad,
	bcnn_weights_init init, bcnn_activation activation)
{
	int nb_layers = net->nb_layers + 1;
	int i, sz;
	float std_init = 0.0f;
	bcnn_layer layer = { 0 };
	layer.type = DECONVOLUTIONAL;

	if (net->nb_layers == 0) {
		layer.input_shape[0] = net->w;
		layer.input_shape[1] = net->h;
		layer.input_shape[2] = net->c;
	}
	else
		memcpy(layer.input_shape, net->layers[net->nb_layers - 1].output_shape, 3 * sizeof(int));
	layer.num = n;
	layer.pad = pad;
	layer.stride = stride;
	layer.size = size;
	layer.bias_size = n;
	layer.weights_size = layer.input_shape[2] * n * size * size;

	layer.weight = (float *)calloc(layer.weights_size, sizeof(float));
	layer.weight_diff = (float *)calloc(layer.weights_size, sizeof(float));
	layer.bias = (float *)calloc(layer.bias_size, sizeof(float));
	layer.bias_diff = (float *)calloc(layer.bias_size, sizeof(float));

	switch (init) {
	case XAVIER:
		std_init = (float)sqrt(3.0f / (size * size * layer.input_shape[2]));
		break;
	case MSRA:
		std_init = (float)sqrt(2.0f / (size * size * layer.input_shape[2]));
		break;
	}
	
	for (i = 0; i < layer.weights_size; ++i)
		layer.weight[i] = std_init * (2 * ((float)rand() / RAND_MAX) - 1);
	
	_bcnn_compute_deconv_output_shape(&layer);
	layer.output_shape[2] = n;

	sz = layer.output_shape[0] * layer.output_shape[1] * layer.input_shape[2] * size * size;
	layer.conv_workspace = (float *)calloc(sz, sizeof(float));
	sz = net->batch_size * layer.output_shape[0] * layer.output_shape[1] * n;
	layer.output = (float *)calloc(sz, sizeof(float));
	layer.diff = (float *)calloc(sz, sizeof(float));

#ifdef BCNN_USE_CUDA
	layer.weight_gpu = bcnn_cuda_memcpy_f32(layer.weight, layer.weights_size);
	layer.weight_diff_gpu = bcnn_cuda_memcpy_f32(layer.weight_diff, layer.weights_size);
	layer.bias_gpu = bcnn_cuda_memcpy_f32(layer.bias, layer.bias_size);
	layer.bias_diff_gpu = bcnn_cuda_memcpy_f32(layer.bias_diff, layer.bias_size);

	sz = layer.output_shape[0] * layer.output_shape[1] * layer.input_shape[2] * size * size;
	layer.conv_workspace_gpu = bcnn_cuda_memcpy_f32(layer.conv_workspace, sz);
	sz = net->batch_size * layer.output_shape[0] * layer.output_shape[1] * n;
	layer.output_gpu = bcnn_cuda_memcpy_f32(layer.output, sz);
	layer.diff_gpu = bcnn_cuda_memcpy_f32(layer.diff, sz);
	
#endif
	layer.activation = activation;

	bcnn_realloc(net, nb_layers);

	net->layers[nb_layers - 1] = layer;

	fprintf(stderr, "[Deconvolutional] input_shape= %dx%dx%d nb_filters= %d kernel_size= %d stride= %d output_shape= %dx%dx%d\n",
		layer.input_shape[0], layer.input_shape[1], layer.input_shape[2], n, size, stride,
		layer.output_shape[0], layer.output_shape[1], layer.output_shape[2]);

	return BCNN_SUCCESS;
}


int bcnn_forward_deconv_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i, m, n, k, sz;

	sz = wrk->batch_size * layer->output_shape[0] * layer->output_shape[1] * layer->output_shape[2];

	bcnn_fill_f32(sz, 0.0f, layer->output);

	m = layer->num * layer->size * layer->size;
	k = layer->input_shape[2];
	n = layer->input_shape[0] * layer->input_shape[1];
	sz = layer->input_shape[2] * layer->input_shape[1] * layer->input_shape[0];
	for (i = 0; i < wrk->batch_size; ++i) {
		bcnn_gemm(1, 0, m, n, k, 1.0f, layer->weight, m, wrk->input + i * sz, n, 0.0f, layer->conv_workspace, n);
		_bcnn_col2im(layer->conv_workspace, layer->num, layer->output_shape[1], layer->output_shape[0], layer->size,
			0, layer->stride, layer->output + i * layer->num * layer->output_shape[0] * layer->output_shape[1]);
	}

	_bcnn_add_bias(layer->output, layer->bias, wrk->batch_size, layer->num, layer->output_shape[0] * layer->output_shape[1]);

	return BCNN_SUCCESS;
}


int bcnn_backward_deconv_layer_cpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i, sz = layer->input_shape[0] * layer->input_shape[1] * layer->input_shape[2];
	int m = layer->input_shape[2];
	int n = layer->size * layer->size * layer->output_shape[2];
	int k = layer->input_shape[0] * layer->input_shape[1];
	float *src = NULL;
	float alpha = 1.0f / wrk->batch_size;

	_bcnn_backward_bias(layer->bias_diff, layer->diff, wrk->batch_size, layer->num,
		layer->output_shape[0] * layer->output_shape[1]);

	for (i = 0; i < wrk->batch_size; ++i) {
		src = layer->diff + i * layer->num * layer->output_shape[0] * layer->output_shape[1];
		_bcnn_im2col(src, layer->output_shape[2], layer->output_shape[1], layer->output_shape[0],
			layer->size, 0, layer->stride, layer->conv_workspace);
		bcnn_gemm(0, 1, m, n, k, alpha, wrk->input + i * layer->input_shape[2] * layer->size * layer->size * layer->num,
			k, layer->conv_workspace, k, 1.0f, layer->weight_diff, n);

		if (wrk->diff) {
			bcnn_gemm(0, 0, layer->input_shape[2], k, n, 1.0f, layer->weight, n, layer->conv_workspace, k, 0.0f, wrk->diff + i * sz, k);
		}
	}
	return BCNN_SUCCESS;
}


int bcnn_forward_deconv_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_deconv_layer_gpu(layer, wrk);
#else
	return bcnn_forward_deconv_layer_cpu(layer, wrk);
#endif
}

int bcnn_backward_deconv_layer(bcnn_layer *layer, bcnn_workload *wrk)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_deconv_layer_gpu(layer, wrk);
#else
	return bcnn_backward_deconv_layer_cpu(layer, wrk);
#endif
}