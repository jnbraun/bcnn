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


#ifdef BCNN_USE_CUDA

#include "bcnn/bcnn.h"

// im2col and col2im functions from caffe
// Reference https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) 
{
	int i, j, w, h, w_out, h_index, h_out, channel_in, channel_out;
	int h_in, w_in;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
	float *data_col_ptr = NULL;
	const float *data_im_ptr = NULL;

    for(; index < n; index += blockDim.x * gridDim.x) {
        w_out = index % width_col;
        h_index = index / width_col;
        h_out = h_index % height_col;
        channel_in = h_index / height_col;
        channel_out = channel_in * ksize * ksize;
        h_in = h_out * stride - pad;
        w_in = w_out * stride - pad;
        data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (i = 0; i < ksize; ++i) {
            for (j = 0; j < ksize; ++j) {
                h = h_in + i;
                w = w_in + j;
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void bcnn_im2col_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col)
{
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    pad = pad ? ksize/2 : 0;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    im2col_gpu_kernel<<<(num_kernels + BCNN_CUDA_THREADS - 1) / BCNN_CUDA_THREADS,
							BCNN_CUDA_THREADS>>>(
							num_kernels, im, height, width, ksize, pad,
							stride, height_col,
							width_col, data_col);
}


__global__ void col2im_gpu_kernel(const int n, const float* data_col,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_im)
{
	int w, h, c, w_col_start, w_col_end, h_col_start, h_col_end;
	int offset, coeff_h_col, coeff_w_col, h_col, w_col;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
	float val;

    for (; index < n; index += blockDim.x * gridDim.x) {
        val = 0;
        w = index % width + pad;
        h = (index / width) % height + pad;
        c = index / (width * height);

        // compute the start and end of the output
        w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        w_col_end = min(w / stride + 1, width_col);
        h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        h_col_end = min(h / stride + 1, height_col);

        // equivalent implementation
        offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
        coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        coeff_w_col = (1 - stride * height_col * width_col);
        for (h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        data_im[index] += val;
    }
}

void bcnn_col2im_gpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im)
{
	int height_col, width_col, num_kernels;
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    pad = pad ? ksize/2 : 0;
    height_col = (height + 2 * pad - ksize) / stride + 1;
    width_col = (width + 2 * pad - ksize) / stride + 1;
    num_kernels = channels * height * width;

    col2im_gpu_kernel<<<(num_kernels + BCNN_CUDA_THREADS - 1) / BCNN_CUDA_THREADS,
							 BCNN_CUDA_THREADS>>>(
							num_kernels, data_col, height, width, ksize, pad,
							stride, height_col,
							width_col, data_im);
}


__global__ void _bcnn_forward_bias_kernel(float *output, float *bias, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch * n+filter) * size + offset] += bias[filter];
}

int bcnn_forward_bias_gpu(float *output, float *bias, int batch, int n, int size)
{
    dim3 dimGrid((size - 1) / BCNN_CUDA_THREADS + 1, n, batch);
    dim3 dimBlock(BCNN_CUDA_THREADS, 1, 1);

    _bcnn_forward_bias_kernel<<<dimGrid, dimBlock>>>(output, bias, n, size);
    bcnn_cuda_check(cudaPeekAtLastError());
	return BCNN_SUCCESS;
}


__global__ void _bcnn_backward_bias_kernel(float *bias_diff, float *diff, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if (offset < size)
		bias_diff[filter] += diff[(batch*n+filter)*size + offset];
}

int bcnn_backward_bias_gpu(float *bias_diff, float *diff, int batch, int n, int size)
{
    dim3 dimGrid((size - 1) / BCNN_CUDA_THREADS + 1, n, batch);
    dim3 dimBlock(BCNN_CUDA_THREADS, 1, 1);

    _bcnn_backward_bias_kernel<<<dimGrid, dimBlock>>>(bias_diff, diff, n, size);
    bcnn_cuda_check(cudaPeekAtLastError());
	return BCNN_SUCCESS;
}


int bcnn_forward_conv_layer_gpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int batch_size = dst.b;
#ifdef BCNN_USE_CUDNN
	float alpha = 1.0f, beta = 1.0f;
	//fprintf(stderr, "%d %d\n", layer->fwd_algo, layer->workspace_size);
    bcnn_cudnn_check(cudnnConvolutionForward(bcnn_cudnn_handle(),
										&alpha,
										layer->src_tensor_desc,
										src.data_gpu,
										layer->filter_desc,
										layer->weight_gpu,
										layer->conv_desc,
										layer->fwd_algo,
										layer->conv_workspace_gpu,
										layer->workspace_size,
										&beta,
										layer->dst_tensor_desc,
										dst.data_gpu));

	bcnn_cudnn_check(cudnnAddTensor(bcnn_cudnn_handle(), CUDNN_ADD_SAME_C, &alpha,
              layer->bias_desc, layer->bias_gpu,
              &beta,
              layer->dst_tensor_desc, dst.data_gpu));
	//bcnn_forward_bias_gpu(dst.data_gpu, layer->bias_gpu, batch_size, layer->num, out_spatial_dim);
#else
	int i, w_sz, sz, out_sz, out_spatial_dim;

	out_sz = batch_size * dst.w * dst.h * dst.c;
	w_sz = layer->size * layer->size * src.c;
	out_spatial_dim = dst.w * dst.h;
	sz = src.c * src.h * src.w;

	bcnn_cuda_fill_f32(out_sz, 0, dst.data_gpu, 1);
	for (i = 0; i < batch_size; ++i) {
		bcnn_im2col_gpu(src.data_gpu + i * sz,
			src.c, src.h, src.w,
			layer->size, layer->stride, layer->pad, layer->conv_workspace_gpu);
		bcnn_cuda_gemm(0, 0, layer->num, out_spatial_dim, w_sz, 1.0f,
			layer->weight_gpu, w_sz, layer->conv_workspace_gpu, out_spatial_dim, 1.0f,
			dst.data_gpu + i * layer->num * out_spatial_dim, out_spatial_dim);
	}
	bcnn_forward_bias_gpu(dst.data_gpu, layer->bias_gpu, batch_size, layer->num, out_spatial_dim);
#endif

	sz = dst.w * dst.h * dst.c * batch_size;
	bcnn_forward_activation_gpu(dst.data_gpu, sz, layer->activation);

	return BCNN_SUCCESS;
}


int bcnn_backward_conv_layer_gpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int i, sz = src.w * src.h * src.c;
	int w_sz = layer->size * layer->size * src.c;
	int out_spatial_dim = dst.w * dst.h;
	int batch_size = dst.b;
#ifdef BCNN_USE_CUDNN
	float one = 1.0f, zero = 0.0f;
#endif

	bcnn_backward_activation_gpu(dst.data_gpu, dst.grad_data_gpu,
		dst.w * dst.h * dst.c * batch_size,
		layer->activation);

#ifdef BCNN_USE_CUDNN
	bcnn_cudnn_check(cudnnConvolutionBackwardBias(bcnn_cudnn_handle(),
              &one,
              layer->dst_tensor_desc,  dst.grad_data_gpu,
              &one,
              layer->bias_desc, layer->bias_diff_gpu));
	//bcnn_backward_bias_gpu(layer->bias_diff_gpu, dst.grad_data_gpu, batch_size, layer->num, out_spatial_dim);
    bcnn_cudnn_check(cudnnConvolutionBackwardFilter_v3(bcnn_cudnn_handle(),
											&one,
											layer->src_tensor_desc,
											src.data_gpu,
											layer->dst_tensor_desc,
											dst.grad_data_gpu,
											layer->conv_desc,
											layer->bwd_filter_algo,
											layer->conv_workspace_gpu,
											layer->workspace_size,
											&one,
											layer->filter_desc,
											layer->weight_diff_gpu));

    if (src.grad_data_gpu) {
        bcnn_cudnn_check(cudnnConvolutionBackwardData_v3(bcnn_cudnn_handle(),
											&one,
											layer->filter_desc,
											layer->weight_gpu,
											layer->dst_tensor_desc,
											dst.grad_data_gpu,
											layer->conv_desc,
											layer->bwd_data_algo,
											layer->conv_workspace_gpu,
											layer->workspace_size,
											&zero,
											layer->src_tensor_desc,
											src.grad_data_gpu));
    }
#else
	bcnn_backward_bias_gpu(layer->bias_diff_gpu, dst.grad_data_gpu, batch_size, layer->num, out_spatial_dim);
	for (i = 0; i < batch_size; ++i) {
		bcnn_im2col_gpu(src.data_gpu + i * sz,
			src.c, src.h, src.w,
			layer->size, layer->stride, layer->pad, layer->conv_workspace_gpu);
		bcnn_cuda_gemm(0, 1, layer->num, w_sz, out_spatial_dim, 1,
			dst.grad_data_gpu + i * layer->num * out_spatial_dim, out_spatial_dim, layer->conv_workspace_gpu, out_spatial_dim, 1,
			layer->weight_diff_gpu, w_sz);

		if (src.grad_data_gpu) {
			bcnn_cuda_gemm(1, 0, w_sz, out_spatial_dim, layer->num, 1,
			layer->weight_gpu, w_sz, dst.grad_data_gpu + i * out_spatial_dim * layer->num, out_spatial_dim, 0,
			layer->conv_workspace_gpu, out_spatial_dim);
			bcnn_col2im_gpu(layer->conv_workspace_gpu,
				src.c, src.h, src.w,
				layer->size, layer->stride, layer->pad, src.grad_data_gpu + i * sz);
		}
	}
#endif
	return BCNN_SUCCESS;
}


int bcnn_forward_deconv_layer_gpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int i, m, n, k, sz;
	int batch_size = dst.b;

	sz = batch_size * dst.w * dst.h * dst.c;
	bcnn_cuda_fill_f32(sz, 0, dst.data_gpu, 1);

	m = layer->num * layer->size * layer->size;
	k = src.c;
	n = src.w * src.h;
	sz = src.c * src.h * src.w;
	for (i = 0; i < batch_size; ++i){
		bcnn_cuda_gemm(1, 0, m, n, k, 1.0f, layer->weight_gpu, m, src.data_gpu + i * sz, n, 0.0f, layer->conv_workspace_gpu, n);
		bcnn_col2im_gpu(layer->conv_workspace_gpu, layer->num, dst.h, dst.w, layer->size,
			layer->stride, 0, dst.data_gpu + i * layer->num * dst.w * dst.h);
	}

	bcnn_forward_bias_gpu(dst.data_gpu, layer->bias_gpu, batch_size, layer->num, dst.w * dst.h);
	
	sz = dst.w * dst.h * dst.c * batch_size;
	bcnn_forward_activation_gpu(dst.data_gpu, sz, layer->activation);

	return BCNN_SUCCESS;
}

int bcnn_backward_deconv_layer_gpu(bcnn_connection *conn)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node src = conn->src_node;
	bcnn_node dst = conn->dst_node;
	int i, sz = src.w * src.h * src.c;
	int m = src.c;
	int n = layer->size * layer->size * dst.c;
	int k = src.w * src.h;
	int batch_size = src.b;
	float *a = NULL, *b = NULL, *c = NULL, *pdst = NULL;
	float alpha = 1.0f / batch_size;

	bcnn_backward_activation_gpu(dst.data_gpu, dst.grad_data_gpu,
		dst.w * dst.h * dst.c * batch_size,
		layer->activation);

	bcnn_backward_bias_gpu(layer->bias_diff_gpu, dst.grad_data_gpu, batch_size, layer->num,
		dst.h * dst.w);
	
	for (i = 0; i < batch_size; ++i) {
		a = src.data_gpu + i * src.c * layer->size * layer->size * layer->num;
		b = layer->conv_workspace_gpu;
		c = layer->weight_diff_gpu;

		pdst = dst.grad_data_gpu + i * layer->num * dst.w * dst.h;

		bcnn_im2col_gpu(pdst, dst.c, dst.h, dst.w,
			layer->size, layer->stride, 0, layer->conv_workspace_gpu);
		bcnn_cuda_gemm(0, 1, m, n, k, alpha, a, k, b, k, 1.0f, c, n);

		if (src.grad_data_gpu) {
			a = layer->weight_gpu;
			b = layer->conv_workspace_gpu;
			c = src.grad_data_gpu + i * sz;
			bcnn_cuda_gemm(0, 0, src.c, k, n, 1.0f, a, n, b, k, 0.0f, c, k);
		}
	}
	return 0;
}

#endif