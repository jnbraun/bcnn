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

__global__ void _bcnn_forward_softmax_layer_kernel(int n, int batch, float *input, float *output)
{
    int i;
    float sum = 0;
    float largest = -INFINITY;
	int b = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    
	if (b >= batch)
		return;
    
    for (i = 0; i < n; ++i) {
        int val = input[i+b*n];
        largest = (val>largest) ? val : largest;
    }

    for (i = 0; i < n; ++i) {
        sum += exp(input[i+b*n]-largest);
    }

    sum = (sum != 0) ? largest+log(sum) : largest-100;

    for (i = 0; i < n; ++i) {
        output[i+b*n] = exp(input[i+b*n]-sum);
    }
}

int bcnn_forward_softmax_layer_gpu(const bcnn_layer *layer, bcnn_workload *wrk)
{
    int input_size = layer->input_shape[0] * layer->input_shape[1] * layer->input_shape[2];
    _bcnn_forward_softmax_layer_kernel<<<bcnn_cuda_gridsize(wrk->batch_size), BCNN_CUDA_THREADS>>>(input_size,
	 wrk->batch_size, wrk->input_gpu, layer->output_gpu);
    bcnn_cuda_check(cudaPeekAtLastError());
	return BCNN_SUCCESS;
}

int bcnn_backward_softmax_layer_gpu(const bcnn_layer *layer, bcnn_workload *wrk)
{
    int input_size = layer->input_shape[0] * layer->input_shape[1] * layer->input_shape[2];
	bcnn_cuda_axpy(wrk->batch_size * input_size, 1, layer->diff_gpu, 1, wrk->diff, 1);
	return BCNN_SUCCESS;
}


#endif