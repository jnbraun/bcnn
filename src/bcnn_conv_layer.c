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
	int nb_connections = net->nb_connections + 1;
	int i, sz;
	bcnn_connection conn = { 0 };
	float std_init = 0.0f;
#ifdef BCNN_USE_CUDNN
	size_t cudnn_wrk_sz = 0;
#endif

	conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
	conn.layer->type = CONVOLUTIONAL;
	if (nb_connections > 1)
		conn.src_node = net->connections[nb_connections - 2].dst_node;
	else
		conn.src_node = net->input_node;

	conn.layer->num = n;
	conn.layer->stride = stride;
	conn.layer->size = size;
	conn.layer->pad = pad;
	conn.layer->bias_size = n;
	conn.layer->weights_size = conn.src_node.c * n * size * size;

	conn.layer->weight = (float *)calloc(conn.layer->weights_size, sizeof(float));
	conn.layer->weight_diff = (float *)calloc(conn.layer->weights_size, sizeof(float));
	conn.layer->bias = (float *)calloc(conn.layer->bias_size, sizeof(float));
	conn.layer->bias_diff = (float *)calloc(conn.layer->bias_size, sizeof(float));

	switch (init) {
	case XAVIER:
		std_init = (float)sqrt(3.0f / (size * size * conn.src_node.c));
		break;
	case MSRA:
		std_init = (float)sqrt(2.0f / (size * size * conn.src_node.c));
		break;
	}
	
	for (i = 0; i < conn.layer->weights_size; ++i)
		conn.layer->weight[i] = std_init * (2 * ((float)rand() / RAND_MAX) - 1);
	
	conn.dst_node.w = (conn.src_node.w + 2 * conn.layer->pad - conn.layer->size) / conn.layer->stride + 1;
	conn.dst_node.h = (conn.src_node.h + 2 * conn.layer->pad - conn.layer->size) / conn.layer->stride + 1;
	conn.dst_node.c = n;
	conn.dst_node.b = conn.src_node.b;
	sz = conn.dst_node.w * conn.dst_node.h * conn.src_node.c * size * size;
	conn.layer->conv_workspace = (float *)calloc(sz, sizeof(float));
	sz = conn.dst_node.b * conn.dst_node.w * conn.dst_node.h * n;
	conn.dst_node.data = (float *)calloc(sz, sizeof(float));
	conn.dst_node.grad_data = (float *)calloc(sz, sizeof(float));

#ifdef BCNN_USE_CUDA
	conn.layer->weight_gpu = bcnn_cuda_memcpy_f32(conn.layer->weight, conn.layer->weights_size);
	conn.layer->weight_diff_gpu = bcnn_cuda_memcpy_f32(conn.layer->weight_diff, conn.layer->weights_size);
	conn.layer->bias_gpu = bcnn_cuda_memcpy_f32(conn.layer->bias, conn.layer->bias_size);
	conn.layer->bias_diff_gpu = bcnn_cuda_memcpy_f32(conn.layer->bias_diff, conn.layer->bias_size);

	sz = conn.dst_node.b * conn.dst_node.w * conn.dst_node.h * n;
	conn.dst_node.data_gpu = bcnn_cuda_memcpy_f32(conn.dst_node.data, sz);
	conn.dst_node.grad_data_gpu = bcnn_cuda_memcpy_f32(conn.dst_node.grad_data, sz);
#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&conn.layer->src_tensor_desc));
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&conn.layer->dst_tensor_desc));
    bcnn_cudnn_check(cudnnCreateFilterDescriptor(&conn.layer->filter_desc));
	bcnn_cudnn_check(cudnnCreateTensorDescriptor(&conn.layer->bias_desc));
    bcnn_cudnn_check(cudnnCreateConvolutionDescriptor(&conn.layer->conv_desc));  
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(conn.layer->src_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		net->batch_size, conn.src.c, conn.src.h, conn.src.w)); 
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(conn.layer->dst_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		net->batch_size, conn.dst.c, conn.dst.h, conn.layer->output_shape[0])); 
    bcnn_cudnn_check(cudnnSetFilter4dDescriptor(conn.layer->filter_desc, CUDNN_DATA_FLOAT,
		conn.layer->num, conn.src.c, conn.layer->size, conn.layer->size));
	bcnn_cudnn_check(cudnnSetTensor4dDescriptor(conn.layer->bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		1, conn.dst.c, 1, 1));
    bcnn_cudnn_check(cudnnSetConvolution2dDescriptor(conn.layer->conv_desc, conn.layer->pad, conn.layer->pad, conn.layer->stride, conn.layer->stride, 1, 1, CUDNN_CROSS_CORRELATION));
    bcnn_cudnn_check(cudnnGetConvolutionForwardAlgorithm(bcnn_cudnn_handle(),
            conn.layer->src_tensor_desc,
            conn.layer->filter_desc,
            conn.layer->conv_desc,
            conn.layer->dst_tensor_desc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &conn.layer->fwd_algo));
    bcnn_cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm(bcnn_cudnn_handle(),
            conn.layer->filter_desc,
            conn.layer->dst_tensor_desc_diff,
            conn.layer->conv_desc,
            conn.layer->src_tensor_desc_diff,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &conn.layer->bwd_data_algo));
    bcnn_cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithm(bcnn_cudnn_handle(),
            conn.layer->src_tensor_desc,
            conn.layer->dst_tensor_desc,
            conn.layer->conv_desc,
            conn.layer->filter_desc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &conn.layer->bwd_filter_algo));
    bcnn_cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(bcnn_cudnn_handle(),
            conn.layer->src_tensor_desc,
            conn.layer->filter_desc,
            conn.layer->conv_desc,
            conn.layer->dst_tensor_desc,
            conn.layer->fwd_algo,
            &cudnn_wrk_sz));
    conn.layer->workspace_size = bh_max(conn.layer->workspace_size, cudnn_wrk_sz);
    bcnn_cudnn_check(cudnnGetConvolutionBackwardFilterWorkspaceSize(bcnn_cudnn_handle(),
            conn.layer->src_tensor_desc,
            conn.layer->dst_tensor_desc,
            conn.layer->conv_desc,
            conn.layer->filter_desc,
            conn.layer->bwd_filter_algo,
            &cudnn_wrk_sz));
    conn.layer->workspace_size = bh_max(conn.layer->workspace_size, cudnn_wrk_sz);
    bcnn_cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(bcnn_cudnn_handle(),
            conn.layer->filter_desc,
            conn.layer->dst_tensor_desc,
            conn.layer->conv_desc,
            conn.layer->src_tensor_desc,
            conn.layer->bwd_data_algo,
            &cudnn_wrk_sz));
    conn.layer->workspace_size = bh_max(conn.layer->workspace_size, cudnn_wrk_sz);
	conn.layer->conv_workspace_gpu = bcnn_cuda_malloc_f32(conn.layer->workspace_size);
#else
	sz = conn.dst_node.w * conn.dst_node.h * conn.src_node.c * size * size;
	conn.layer->conv_workspace_gpu = bcnn_cuda_memcpy_f32(conn.layer->conv_workspace, sz);
#endif
#endif
	conn.layer->activation = activation;
	net->nb_connections = nb_connections;
	bcnn_net_add_connection(net, conn);

	fprintf(stderr, "[Convolutional] input_shape= %dx%dx%d nb_filters= %d kernel_size= %d stride= %d padding= %d output_shape= %dx%dx%d\n",
		conn.src_node.w, conn.src_node.h, conn.src_node.c, n, size, stride, pad,
		conn.dst_node.w, conn.dst_node.h, conn.dst_node.c);

	return 0;
}


int bcnn_forward_conv_layer_cpu(bcnn_connection *conn)
{
	int i, m, n, k, sz;
	float *a = NULL, *b = NULL, *c = NULL;
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int batch_size = src.b;
	sz = bcnn_node_size(&dst);
	bcnn_fill_f32(sz, 0.0f, dst.data);

	m = layer->num;
	k = layer->size * layer->size * src.c;
	n = dst.w * dst.h;

	sz = src.c * src.h * src.w;

	a = layer->weight;
	b = layer->conv_workspace;
	c = dst.data;
	
	for (i = 0; i < batch_size; ++i) {
		_bcnn_im2col(src.data, src.c, src.h, src.w,
			layer->size, layer->pad, layer->stride, b);
		bcnn_gemm(0, 0, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
		c += n * m;
		src.data += sz;
	}

	_bcnn_add_bias(dst.data, layer->bias, batch_size, layer->num, dst.w * dst.h);

	sz = dst.w * dst.h * dst.c * batch_size;
	bcnn_forward_activation_cpu(dst.data, sz, layer->activation);

	return BCNN_SUCCESS;
}




int bcnn_backward_conv_layer_cpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int batch_size = src.b;
	int i, sz = src.w * src.h * src.c;
	int m = layer->num;
	int n = layer->size * layer->size * src.c;
	int k = dst.w * dst.h;
	float *a = NULL, *b = NULL, *c = NULL, *psrc = NULL;

	bcnn_backward_activation_cpu(dst.data, dst.grad_data,
		dst.w * dst.h * dst.c * batch_size,
		layer->activation);

	_bcnn_backward_bias(layer->bias_diff, dst.grad_data, batch_size, layer->num, k);

	for (i = 0; i < batch_size; ++i){
		a = dst.grad_data + i * m * k;
		b = layer->conv_workspace;
		c = layer->weight_diff;

		_bcnn_im2col(src.data + i * sz, src.c, src.h, src.w,
			layer->size, layer->pad, layer->stride, b);
		bcnn_gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

		if (src.grad_data) {
			a = layer->weight;
			b = dst.grad_data + i * m * k;
			c = layer->conv_workspace;

			bcnn_gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);
			_bcnn_col2im(layer->conv_workspace, src.c, src.h, src.w,
				layer->size, layer->pad, layer->stride, src.grad_data + i * sz);
		}
	}

	return BCNN_SUCCESS;
}


int bcnn_forward_conv_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_conv_layer_gpu(conn);
#else
	return bcnn_forward_conv_layer_cpu(conn);
#endif
}

int bcnn_backward_conv_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_conv_layer_gpu(conn);
#else
	return bcnn_backward_conv_layer_cpu(conn);
#endif
}


/* Deconv layer */
int bcnn_add_deconvolutional_layer(bcnn_net *net, int n, int size, int stride, int pad,
	bcnn_weights_init init, bcnn_activation activation)
{
	int nb_connections = net->nb_connections + 1;
	int i, sz;
	float std_init = 0.0f;
	bcnn_connection conn = { 0 };

	conn.layer->type = DECONVOLUTIONAL;
	if (nb_connections > 1)
		conn.src_node = net->connections[nb_connections - 2].dst_node;
	else
		conn.src_node = net->input_node;
	conn.layer->num = n;
	conn.layer->stride = stride;
	conn.layer->size = size;
	conn.layer->pad = pad;
	conn.layer->bias_size = n;
	conn.layer->weights_size = conn.src_node.c * n * size * size;

	conn.layer->weight = (float *)calloc(conn.layer->weights_size, sizeof(float));
	conn.layer->weight_diff = (float *)calloc(conn.layer->weights_size, sizeof(float));
	conn.layer->bias = (float *)calloc(conn.layer->bias_size, sizeof(float));
	conn.layer->bias_diff = (float *)calloc(conn.layer->bias_size, sizeof(float));

	switch (init) {
	case XAVIER:
		std_init = (float)sqrt(3.0f / (size * size * conn.src_node.c));
		break;
	case MSRA:
		std_init = (float)sqrt(2.0f / (size * size * conn.src_node.c));
		break;
	}

	for (i = 0; i < conn.layer->weights_size; ++i)
		conn.layer->weight[i] = std_init * (2 * ((float)rand() / RAND_MAX) - 1);

	conn.dst_node.w = conn.layer->stride * (conn.src_node.w - 1) + conn.layer->size - 2 * conn.layer->pad;
	conn.dst_node.h = conn.layer->stride * (conn.src_node.h - 1) + conn.layer->size - 2 * conn.layer->pad;
	conn.dst_node.c = n;
	conn.dst_node.b = conn.src_node.b;

	sz = conn.dst_node.w * conn.dst_node.h * conn.src_node.c * size * size;
	conn.layer->conv_workspace = (float *)calloc(sz, sizeof(float));
	sz = conn.dst_node.b * conn.dst_node.w * conn.dst_node.h * n;
	conn.dst_node.data = (float *)calloc(sz, sizeof(float));
	conn.dst_node.grad_data = (float *)calloc(sz, sizeof(float));

#ifdef BCNN_USE_CUDA
	conn.layer->weight_gpu = bcnn_cuda_memcpy_f32(conn.layer->weight, conn.layer->weights_size);
	conn.layer->weight_diff_gpu = bcnn_cuda_memcpy_f32(conn.layer->weight_diff, conn.layer->weights_size);
	conn.layer->bias_gpu = bcnn_cuda_memcpy_f32(conn.layer->bias, conn.layer->bias_size);
	conn.layer->bias_diff_gpu = bcnn_cuda_memcpy_f32(conn.layer->bias_diff, conn.layer->bias_size);
	sz = conn.dst_node.b * conn.dst_node.w * conn.dst_node.h * n;
	conn.dst_node.data_gpu = bcnn_cuda_memcpy_f32(conn.dst_node.data, sz);
	conn.dst_node.grad_data_gpu = bcnn_cuda_memcpy_f32(conn.dst_node.grad_data, sz);
	sz = conn.dst_node.w * conn.dst_node.h * conn.src_node.c * size * size;
	conn.layer->conv_workspace_gpu = bcnn_cuda_memcpy_f32(conn.layer->conv_workspace, sz);
#endif
	conn.layer->activation = activation;

	bcnn_net_add_connection(net, conn);

	fprintf(stderr, "[Deconvolutional] input_shape= %dx%dx%d nb_filters= %d kernel_size= %d stride= %d output_shape= %dx%dx%d\n",
		conn.src_node.w, conn.src_node.h, conn.src_node.c, n, size, stride,
		conn.dst_node.w, conn.dst_node.h, conn.dst_node.c);

	return BCNN_SUCCESS;
}


int bcnn_forward_deconv_layer_cpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int batch_size = src.b;
	int i, m, n, k, sz;

	sz = batch_size * dst.w * dst.h * dst.c;

	bcnn_fill_f32(sz, 0.0f, dst.data);

	m = layer->num * layer->size * layer->size;
	k = src.c;
	n = src.w * src.h;
	sz = src.c * src.h * src.w;
	for (i = 0; i < batch_size; ++i) {
		bcnn_gemm(1, 0, m, n, k, 1.0f, layer->weight, m, src.data + i * sz, n, 0.0f, layer->conv_workspace, n);
		_bcnn_col2im(layer->conv_workspace, layer->num, dst.h, dst.w, layer->size,
			0, layer->stride, dst.data + i * layer->num * dst.w * dst.h);
	}

	_bcnn_add_bias(dst.data, layer->bias, batch_size, layer->num, dst.w * dst.h);

	sz = dst.w * dst.h * dst.c * batch_size;
	bcnn_forward_activation_cpu(dst.data, sz, layer->activation);

	return BCNN_SUCCESS;
}


int bcnn_backward_deconv_layer_cpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int batch_size = src.b;
	int i, sz = src.w * src.h * src.c;
	int m = src.c;
	int n = layer->size * layer->size * dst.c;
	int k = src.w * src.h;
	float *pdst = NULL;
	float alpha = 1.0f / batch_size;

	bcnn_backward_activation_cpu(dst.data, dst.grad_data,
		dst.w * dst.h * dst.c * batch_size,
		layer->activation);

	_bcnn_backward_bias(layer->bias_diff, dst.grad_data, batch_size, layer->num,
		dst.w * dst.h);

	for (i = 0; i < batch_size; ++i) {
		pdst = dst.grad_data + i * layer->num * dst.w * dst.h;
		_bcnn_im2col(pdst, dst.c, dst.h, dst.w,
			layer->size, 0, layer->stride, layer->conv_workspace);
		bcnn_gemm(0, 1, m, n, k, alpha, src.data + i * src.c * layer->size * layer->size * layer->num,
			k, layer->conv_workspace, k, 1.0f, layer->weight_diff, n);

		if (src.grad_data) {
			bcnn_gemm(0, 0, src.c, k, n, 1.0f, layer->weight, n, layer->conv_workspace, k, 0.0f, src.grad_data + i * sz, k);
		}
	}
	return BCNN_SUCCESS;
}


int bcnn_forward_deconv_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_forward_deconv_layer_gpu(conn);
#else
	return bcnn_forward_deconv_layer_cpu(conn);
#endif
}

int bcnn_backward_deconv_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
	return bcnn_backward_deconv_layer_gpu(conn);
#else
	return bcnn_backward_deconv_layer_cpu(conn);
#endif
}