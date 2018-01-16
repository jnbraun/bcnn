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
#include "bcnn_mat.h"

int bcnn_forward_concat_layer_gpu(bcnn_node *src0_node, bcnn_node *src1_node,
    bcnn_node *dst_node)
{
    int j;
    bcnn_tensor src0 = src0_node->tensor;
    bcnn_tensor src1 = src1_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int src0_sz = bcnn_tensor_get_size3d(&src0);
    int src1_sz = bcnn_tensor_get_size3d(&src1);
    int dst_sz = bcnn_tensor_get_size3d(&dst);

    for (j = 0; j < src0.n; ++j) {
        bcnn_cuda_copy_f32(src0_sz, src0.data_gpu + j * src0_sz, 1, dst.data_gpu + j * dst_sz, 1);
    }
    for (j = 0; j < src0.n; ++j) {
        bcnn_cuda_copy_f32(src1_sz, src1.data_gpu + j * src1_sz, 1, dst.data_gpu + src0_sz + j * dst_sz, 1);
    }

    return BCNN_SUCCESS;
}

int bcnn_backward_concat_layer_gpu(bcnn_node *src0_node, bcnn_node *src1_node,
    bcnn_node *dst_node)
{
    int j;
    bcnn_tensor src0 = src0_node->tensor;
    bcnn_tensor src1 = src1_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int src0_sz = bcnn_tensor_get_size3d(&src0);
    int src1_sz = bcnn_tensor_get_size3d(&src1);
    int dst_sz = bcnn_tensor_get_size3d(&dst);

    for (j = 0; j < src0.n; ++j) {
        bcnn_cuda_axpy(src0_sz, 1.0f, dst.grad_data_gpu + j * dst_sz, 1, src0.grad_data_gpu + j * src0_sz, 1);
    }
    for (j = 0; j < src0.n; ++j) {
        bcnn_cuda_axpy(src1_sz, 1.0f, dst.grad_data_gpu + src0_sz + j * dst_sz, 1, src1.grad_data_gpu + j * src1_sz, 1);
    }

    return BCNN_SUCCESS;
}

#endif