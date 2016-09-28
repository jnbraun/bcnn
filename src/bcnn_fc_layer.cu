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

int bcnn_forward_fullc_layer_gpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i, input_size = layer->input_shape[2], output_size = layer->output_shape[2];
	int sz = output_size * wrk->batch_size;

	bcnn_cuda_fill_f32(output_size * wrk->batch_size, 0.0f, layer->output_gpu, 1);
	
	bcnn_cuda_gemm(0, 1, wrk->batch_size, output_size, input_size, 1,
		wrk->input_gpu, input_size, layer->weight_gpu, input_size, 1,
		 layer->output_gpu, output_size);

	for(i = 0; i < wrk->batch_size; ++i)
        bcnn_cuda_axpy(output_size, 1.0f, layer->bias_gpu, 1, layer->output_gpu + i * output_size, 1);

	bcnn_forward_activation_gpu(layer->output_gpu, sz, layer->activation);

	return BCNN_SUCCESS;
}


int bcnn_backward_fullc_layer_gpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i, input_size = layer->input_shape[2], output_size = layer->output_shape[2];
	int sz = output_size * wrk->batch_size;

	bcnn_backward_activation_gpu(layer->output_gpu, layer->diff_gpu, sz, layer->activation);

	for (i = 0; i < wrk->batch_size; ++i)
		bcnn_cuda_axpy(output_size, 1, layer->diff_gpu + i * output_size, 1, layer->bias_diff_gpu, 1);

	bcnn_cuda_gemm(1, 0, output_size, input_size, wrk->batch_size, 1,
		layer->diff_gpu, output_size, wrk->input_gpu, input_size, 1,
		layer->weight_diff_gpu, input_size);

	if (wrk->diff)
		bcnn_cuda_gemm(0, 0, wrk->batch_size, input_size, output_size, 1,
			layer->diff_gpu, output_size, layer->weight_gpu, input_size, 1, wrk->diff, input_size);

	return BCNN_SUCCESS;
}

#endif