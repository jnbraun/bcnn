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

int bcnn_forward_concat_layer_gpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i;
	int spatial_sz = layer->input_shape[0] * layer->input_shape[1];
	int sz = spatial_sz * layer->input_shape[2];
	int sz2 =  spatial_sz * (layer->output_shape[2] - layer->input_shape[2]);
	int dst_sz = sz + sz2;

	for (i = 0; i < wrk->batch_size; ++i) {
		bcnn_cuda_copy_f32(sz, wrk->input_gpu + i * sz, 1, layer->output_gpu + i * dst_sz, 1);
		bcnn_cuda_copy_f32(sz2, wrk->input2_gpu + i * sz2, 1, layer->output_gpu + i * dst_sz + sz, 1);
	}
	return BCNN_SUCCESS;
}


int bcnn_backward_concat_layer_gpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i;
	int spatial_sz = layer->input_shape[0] * layer->input_shape[1];
	int sz = spatial_sz * layer->input_shape[2];
	int sz2 =  spatial_sz * (layer->output_shape[2] - layer->input_shape[2]);
	int dst_sz = sz + sz2;

	for (i = 0; i < wrk->batch_size; ++i) {
		bcnn_cuda_copy_f32(sz, layer->diff_gpu + i * dst_sz, 1, wrk->diff + i * sz, 1);
		//bcnn_cuda_copy_f32(sz2, layer->diff_gpu + i * dst_sz + sz, 1, wrk->diff2 + i * sz2, 1);
	}

	return BCNN_SUCCESS;
}

#endif