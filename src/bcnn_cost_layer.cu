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

#include <bh/bh.h>

#include "bcnn/bcnn.h"

int bcnn_forward_cost_layer_gpu(bcnn_layer *layer, bcnn_workload *wrk)
{
	int i, j, offset, j_best, n, d;
	int input_size = layer->input_shape[0] * layer->input_shape[1] * layer->input_shape[2];
	int sz = wrk->batch_size * input_size;
	float p_max;
	float *input_cpu = NULL, *truth_cpu = NULL;
	// If no truth available, do nothing
	if (!wrk->truth)
		return 0;

	switch (layer->cost_type) {
	case COST_ERROR:
		*(layer->output) = 0.0f;
		input_cpu = (float *)calloc(sz, sizeof(float));
		truth_cpu = (float *)calloc(sz, sizeof(float));
		bcnn_cuda_memcpy_dev2host(wrk->input_gpu, input_cpu, sz);
		bcnn_cuda_memcpy_dev2host(wrk->truth_gpu, truth_cpu, sz);
		for (i = 0; i < wrk->batch_size; ++i) {
			offset = i * input_size;
			p_max = FLT_MIN;
			j_best = 0;
			for (j = 0; j < input_size; ++j) {
				if (input_cpu[offset + j] > p_max) {
					p_max = input_cpu[offset + j];
					j_best = j;
				}
			}
			if (truth_cpu[offset + j_best] == 0)
				*(layer->output) += 1.0f;
		}
		bcnn_cuda_copy_f32(sz, wrk->input_gpu, 1, layer->diff_gpu, 1);
		bcnn_cuda_axpy(sz, -1, wrk->truth_gpu, 1, layer->diff_gpu, 1);
		bh_free(input_cpu);
		bh_free(truth_cpu);
		break;
	case COST_SSE:
		bcnn_cuda_copy_f32(sz, wrk->input_gpu, 1, layer->diff_gpu, 1);
		bcnn_cuda_axpy(sz, -1, wrk->truth_gpu, 1, layer->diff_gpu, 1);
		bcnn_cuda_memcpy_dev2host(layer->diff_gpu, layer->diff, sz);
		*(layer->output) = bcnn_dot(sz, layer->diff, layer->diff);
		break;
	case COST_MSE:
		bcnn_cuda_copy_f32(sz, wrk->input_gpu, 1, layer->diff_gpu, 1);
		bcnn_cuda_axpy(sz, -1, wrk->truth_gpu, 1, layer->diff_gpu, 1);
		bcnn_cuda_memcpy_dev2host(layer->diff_gpu, layer->diff, sz);
		*(layer->output) = bcnn_dot(sz, layer->diff, layer->diff);
		*(layer->output) /= input_size;
		break;
	case COST_CRPS:
		*(layer->output) = 0.0f;
		input_cpu = (float *)calloc(sz, sizeof(float));
		truth_cpu = (float *)calloc(sz, sizeof(float));
		bcnn_cuda_memcpy_dev2host(wrk->input_gpu, input_cpu, sz);
		bcnn_cuda_memcpy_dev2host(wrk->truth_gpu, truth_cpu, sz);
		for (i = 0; i < wrk->batch_size; ++i) {
			offset = i * input_size;
			for (j = 1; j < input_size; ++j) {
				if (input_cpu[offset + j] < input_cpu[offset + j - 1]) {
					input_cpu[offset + j] = input_cpu[offset + j - 1];
				}
			}
		}
		bcnn_axpy(sz, -1, truth_cpu, input_cpu);
		*(layer->output) = bcnn_dot(sz, input_cpu, input_cpu);
		bcnn_cuda_copy_f32(sz, wrk->input_gpu, 1, layer->diff_gpu, 1);
		bcnn_cuda_axpy(sz, -1, wrk->truth_gpu, 1, layer->diff_gpu, 1);
		bh_free(input_cpu);
		bh_free(truth_cpu);
		break;
	case COST_LOGLOSS:
		*(layer->output) = 0.0f;
		input_cpu = (float *)calloc(sz, sizeof(float));
		truth_cpu = (float *)calloc(sz, sizeof(float));
		bcnn_cuda_memcpy_dev2host(wrk->input_gpu, input_cpu, sz);
		bcnn_cuda_memcpy_dev2host(wrk->truth_gpu, truth_cpu, sz);
		for (i = 0; i < wrk->batch_size; ++i) {
			offset = i * input_size;
			for (j = 0; j < input_size; ++j) {
				if (truth_cpu[offset + j] > 0.0f) {
					*(layer->output) += (float)-log(bh_clamp(input_cpu[offset + j], 1e-8f, 1.0f - 1e-8f));
				}
			}
		}
		bh_free(input_cpu);
		bh_free(truth_cpu);
		bcnn_cuda_copy_f32(sz, wrk->input_gpu, 1, layer->diff_gpu, 1);
		bcnn_cuda_axpy(sz, -1, wrk->truth_gpu, 1, layer->diff_gpu, 1);
		bcnn_cuda_memcpy_dev2host(layer->diff_gpu, layer->diff, sz);
		break;
	case COST_DICE:
		input_cpu = (float *)calloc(sz, sizeof(float));
		truth_cpu = (float *)calloc(sz, sizeof(float));
		bcnn_cuda_memcpy_dev2host(wrk->input_gpu, input_cpu, sz);
		bcnn_cuda_memcpy_dev2host(wrk->truth_gpu, truth_cpu, sz);
		*(layer->output) = 0.0f;
		for (i = 0; i < wrk->batch_size; ++i) {
			offset = i * input_size;
			n = 0;
			d = 0;
			for (j = 0; j < input_size; ++j) {
				n += truth_cpu[offset + j] * (input_cpu[offset + j] > 0.5f);
				d += truth_cpu[offset + j] + (input_cpu[offset + j] > 0.5f);
			}
			/*for (j = 0; j < input_size; ++j) {
				layer->diff[offset + j] = -(2.0f * truth_cpu[offset + j] * (d + 1.0f) - (2 * n + 1.0f)) / 
					((d + 1.0f) * (d + 1.0f));
				layer->diff[offset + j] = -(2.0f * truth_cpu[offset + j] * (truth_cpu[offset + j] + 1.0f) - 2.0f) /
					((truth_cpu[offset + j] + input_cpu[offset + j] + 1.0f) * (truth_cpu[offset + j] + input_cpu[offset + j] + 1.0f));
			}*/
			*(layer->output) += (float)(2.0f * n + 1.0f) / (d + 1.0f);
		}
		bh_free(input_cpu);
		bh_free(truth_cpu);
		bcnn_cuda_copy_f32(sz, wrk->input_gpu, 1, layer->diff_gpu, 1);
		bcnn_cuda_axpy(sz, -1, wrk->truth_gpu, 1, layer->diff_gpu, 1);
		bcnn_cuda_memcpy_dev2host(layer->diff_gpu, layer->diff, sz);
		//bcnn_cuda_memcpy_host2dev(layer->diff_gpu, layer->diff, sz);
		break;
	}

	return BCNN_SUCCESS;
}


int bcnn_backward_cost_layer_gpu(const bcnn_layer *layer, bcnn_workload *wrk)
{
	int input_size = layer->input_shape[0] * layer->input_shape[1] * layer->input_shape[2];
	int sz = wrk->batch_size * input_size;

	bcnn_cuda_axpy(sz, layer->scale, layer->diff_gpu, 1, wrk->diff, 1);

	return BCNN_SUCCESS;
}

#endif