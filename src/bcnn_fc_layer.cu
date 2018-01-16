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

int bcnn_forward_fullc_layer_gpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int i, batch_size = dst.n;
    int src_size = bcnn_tensor_get_size3d(&src);
    int dst_size = bcnn_tensor_get_size3d(&dst);
    int sz = bcnn_tensor_get_size(&dst);
    
    bcnn_cuda_fill_f32(dst_size * batch_size, 0.0f, dst.data_gpu, 1);

    bcnn_cuda_gemm(0, 1, batch_size, dst_size, src_size, 1,
        src.data_gpu, src_size, layer->weight_gpu, src_size, 1, dst.data_gpu, dst_size);

    for (i = 0; i < batch_size; ++i){
        bcnn_cuda_axpy(dst_size, 1, layer->bias_gpu, 1, dst.data_gpu + i * dst_size, 1);
    }
    bcnn_forward_activation_gpu(dst.data_gpu, sz, layer->activation);

    return BCNN_SUCCESS;
}


int bcnn_backward_fullc_layer_gpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int i, batch_size = dst.n;
    int src_size = bcnn_tensor_get_size3d(&src);
    int dst_size = bcnn_tensor_get_size3d(&dst);
    int sz = bcnn_tensor_get_size(&dst);

    bcnn_backward_activation_gpu(dst.data_gpu, dst.grad_data_gpu, sz, layer->activation);

    for (i = 0; i < batch_size; ++i) {
        bcnn_cuda_axpy(dst_size, 1, dst.grad_data_gpu + i * dst_size, 1, layer->bias_diff_gpu, 1);
    }

    bcnn_cuda_gemm(1, 0, dst_size, src_size, batch_size, 1,
        dst.grad_data_gpu, dst_size, src.data_gpu, src_size, 1,
        layer->weight_diff_gpu, src_size);
    if (src.grad_data_gpu) {
        bcnn_cuda_gemm(0, 0, batch_size, src_size, dst_size, 1,
            dst.grad_data_gpu, dst_size, layer->weight_gpu, src_size, 1, src.grad_data_gpu, src_size);
    }

    return BCNN_SUCCESS;
}

#endif