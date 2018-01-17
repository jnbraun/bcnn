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

#include "bcnn/bcnn.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"

#ifdef BCNN_USE_BLAS
#include "cblas.h"
#endif

#include <bh/bh_mem.h>
#include <bh/bh_string.h>
#include <bh/bh_timer.h>

#include "bh_log.h"


int bcnn_add_convolutional_layer(bcnn_net *net, int n, int size, int stride, int pad,
    int batch_norm, bcnn_weights_init init, bcnn_activation activation, int quantize,
    char *src_id, char *dst_id)
{
    int i, sz, k, l;
    bcnn_connection conn = { 0 };
    float std_init = 0.0f;
    bcnn_gauss_gen g = { 0 };
#ifdef BCNN_USE_CUDNN
    size_t cudnn_wrk_sz = 0;
#endif
    bcnn_node dst_node = { 0 };

    if (net->nb_connections > 0) {
        int is_src_node_found = 0;
        for (i = net->num_nodes - 1; i >= 0 ; --i) {
            if (strcmp(net->nodes[i].id, src_id) == 0) {
                bcnn_connection_add_src_node(&conn, i);
                is_src_node_found = 1;
                break;
            }
        }
        bh_check(is_src_node_found, "Convolution layer: invalid input node name %s", src_id);
    }
    else {
        bh_check(bcnn_tensor_get_size(&net->nodes[0].tensor) > 0,
            "Invalid input size of the network. " 
            "Hint: you can use 'bcnn_net_set_input_shape' to set the network input size");
        bcnn_connection_add_src_node(&conn, 0);
    }

    // Setup layer
    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = CONVOLUTIONAL;
    conn.layer->num = n;
    conn.layer->stride = stride;
    conn.layer->size = size;
    conn.layer->pad = pad;
    conn.layer->quantize = quantize;
    conn.layer->bias_size = n;
    conn.layer->weights_size = net->nodes[conn.src[0]].tensor.c * n * size * size;
    conn.layer->weight = (float *)calloc(conn.layer->weights_size, sizeof(float));
    conn.layer->weight_diff = (float *)calloc(conn.layer->weights_size, sizeof(float));
    conn.layer->bias = (float *)calloc(conn.layer->bias_size, sizeof(float));
    conn.layer->bias_diff = (float *)calloc(conn.layer->bias_size, sizeof(float));
    switch (init) {
    case XAVIER:
        std_init = (float)sqrt(3.0f / (size * size * net->nodes[conn.src[0]].tensor.c));
        for (i = 0; i < conn.layer->weights_size; ++i) {
            conn.layer->weight[i] = std_init * (2 * ((float)rand() / RAND_MAX) - 1);
        }
        break;
    case MSRA:
        std_init = (float)sqrt(2.0f / (size * size * net->nodes[conn.src[0]].tensor.c));
        for (i = 0; i < conn.layer->weights_size; ++i) {
            conn.layer->weight[i] = std_init * bcnn_rng_gaussian(&g);
        }
        break;
    }
    if (net->learner.optimizer == ADAM) {
        conn.layer->adam_m = (float *)calloc(conn.layer->weights_size, sizeof(float));
        conn.layer->adam_v = (float *)calloc(conn.layer->weights_size, sizeof(float));
    }
    
    bh_strfill(&dst_node.id, dst_id);
    bcnn_tensor_set_shape(&dst_node.tensor,
        net->nodes[conn.src[0]].tensor.n,
        conn.layer->num,
        (net->nodes[conn.src[0]].tensor.h + 2 * conn.layer->pad - conn.layer->size) / conn.layer->stride + 1,
        (net->nodes[conn.src[0]].tensor.w + 2 * conn.layer->pad - conn.layer->size) / conn.layer->stride + 1,
        1);
    bcnn_tensor_allocate(&dst_node.tensor);
    // Add node to net
    bcnn_net_add_node(net, dst_node);
    // Add node pointer to connection
    bcnn_connection_add_dst_node(&conn, net->num_nodes - 1);
    sz = bcnn_tensor_get_size3d(&net->nodes[conn.dst[0]].tensor) * size * size;
    conn.layer->conv_workspace = (float *)calloc(sz, sizeof(float));

    if (conn.layer->quantize == 1) {
        bh_assert((net->nodes[conn.src[0]].tensor.c % BITS_IN_UINT32 == 0), "Number of channels in input must be a multiple of 32", BCNN_INVALID_PARAMETER);
        k = conn.layer->size * conn.layer->size * net->nodes[conn.src[0]].tensor.c;
        l = net->nodes[conn.dst[0]].tensor.w * net->nodes[conn.dst[0]].tensor.h;
        conn.layer->binary_workspace = (uint32_t *)calloc(l * k / (sizeof(float) * 8), sizeof(float));
        conn.layer->binary_weight = (uint32_t *)calloc(conn.layer->weights_size / BITS_IN_UINT32, sizeof(uint32_t));
    }

#ifdef BCNN_USE_CUDA
    conn.layer->weight_gpu = bcnn_cuda_memcpy_f32(conn.layer->weight, conn.layer->weights_size);
    conn.layer->weight_diff_gpu = bcnn_cuda_memcpy_f32(conn.layer->weight_diff, conn.layer->weights_size);
    conn.layer->bias_gpu = bcnn_cuda_memcpy_f32(conn.layer->bias, conn.layer->bias_size);
    conn.layer->bias_diff_gpu = bcnn_cuda_memcpy_f32(conn.layer->bias_diff, conn.layer->bias_size);
    
    if (net->learner.optimizer == ADAM) {
        conn.layer->adam_m_gpu = bcnn_cuda_memcpy_f32(conn.layer->adam_m, conn.layer->weights_size);
        conn.layer->adam_v_gpu = bcnn_cuda_memcpy_f32(conn.layer->adam_v, conn.layer->weights_size);
    }
#ifdef BCNN_USE_CUDNN

    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&conn.layer->src_tensor_desc));
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&conn.layer->dst_tensor_desc));
    bcnn_cudnn_check(cudnnCreateFilterDescriptor(&conn.layer->filter_desc));
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&conn.layer->bias_desc));
    bcnn_cudnn_check(cudnnCreateConvolutionDescriptor(&conn.layer->conv_desc));  
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(conn.layer->src_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        net->nodes[conn.dst[0]].tensor.n, net->nodes[conn.src[0]].tensor.c, net->nodes[conn.src[0]].tensor.h, net->nodes[conn.src[0]].tensor.w)); 
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(conn.layer->dst_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        net->nodes[conn.dst[0]].tensor.n, net->nodes[conn.dst[0]].tensor.c, net->nodes[conn.dst[0]].tensor.h, net->nodes[conn.dst[0]].tensor.w)); 
    bcnn_cudnn_check(cudnnSetFilter4dDescriptor(conn.layer->filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        conn.layer->num, net->nodes[conn.src[0]].tensor.c, conn.layer->size, conn.layer->size));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(conn.layer->bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        1, net->nodes[conn.dst[0]].tensor.c, 1, 1));
#if CUDNN_MAJOR >= 6
    bcnn_cudnn_check(cudnnSetConvolution2dDescriptor(conn.layer->conv_desc, conn.layer->pad, conn.layer->pad,
        conn.layer->stride, conn.layer->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
#else
    bcnn_cudnn_check(cudnnSetConvolution2dDescriptor(conn.layer->conv_desc, conn.layer->pad, conn.layer->pad,
        conn.layer->stride, conn.layer->stride, 1, 1, CUDNN_CROSS_CORRELATION));
#endif
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
            conn.layer->dst_tensor_desc,
            conn.layer->conv_desc,
            conn.layer->src_tensor_desc,
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
    net->workspace_size = bh_max(net->workspace_size, conn.layer->workspace_size);
    //conn.layer->conv_workspace_gpu = bcnn_cuda_malloc_f32(conn.layer->workspace_size);
#else
    conn.layer->workspace_size = net->nodes[conn.dst[0]].tensor.w * net->nodes[conn.dst[0]].tensor.h *
        net->nodes[conn.src[0]].tensor.c * size * size;
    net->workspace_size = bh_max(net->workspace_size, conn.layer->workspace_size);
    //conn.layer->conv_workspace_gpu = bcnn_cuda_memcpy_f32(conn.layer->conv_workspace, sz);
#endif
#endif
    conn.layer->activation = activation;
    bcnn_net_add_connection(net, conn);

    bh_log_info("[Convolutional] input_shape= %dx%dx%d nb_filters= %d kernel_size= %d stride= %d padding= %d output_shape= %dx%dx%d",
        net->nodes[conn.src[0]].tensor.w, net->nodes[conn.src[0]].tensor.h, net->nodes[conn.src[0]].tensor.c,
        n, size, stride, pad,
        net->nodes[conn.dst[0]].tensor.w, net->nodes[conn.dst[0]].tensor.h, net->nodes[conn.dst[0]].tensor.c);

    return 0;
}


int bcnn_forward_conv_layer_cpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    int i, j, m, n, k, sz;
    float *a = NULL, *b = NULL, *c = NULL;
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = src.n;
    
    sz = bcnn_tensor_get_size(&dst);

    memset(dst.data, 0, sz * sizeof(float));

    // Binarize weights
    if (layer->quantize) {
        for (i = 0; i < layer->weights_size; ++i) {
            layer->weight[i] = (layer->weight[i] > 0) ? 1.0f : -1.0f;
        }
    }

    m = layer->num;
    k = layer->size * layer->size * src.c;
    n = dst.w * dst.h;

    sz = src.c * src.h * src.w;

    a = layer->weight;
    b = layer->conv_workspace;
    c = dst.data;
    
    for (i = 0; i < batch_size; ++i) {
        if (layer->size == 1) {
            b = src.data;
        }
        else {
            bcnn_im2col(src.data, src.c, src.h, src.w,
                layer->size, layer->pad, layer->stride, b);
        }
        // Binarize inputs here (after padding)
        if (layer->quantize) {
            for (j = 0; j < k * n; ++j)
                b[j] = (b[j] > 0) ? 1.0f : -1.0f;
            if (layer->net_state == 0) { // inference phase
                //bh_timer_start(&t);
                // xnor / popcnt gemm
                get_binary_col_unrolled(b, layer->binary_workspace, k, n);
                get_binary_row(a, layer->binary_weight, m * k);
                bcnn_xnor_gemm(0, 0, m, n, k / BITS_IN_UINT32, 1.0f,
                    layer->binary_weight, k / BITS_IN_UINT32,
                    layer->binary_workspace, n,
                    1.0f,
                    c,  n);
                
                //bcnn_gemm(0, 0, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
                //bh_timer_stop(&t);
                //tconv += bh_timer_get_msec(&t);
                // Mapping to obtain similar range than xnor / popcnt gemm
                /*for (j = 0; j < n * m; ++j) {
                    c[j] = (c[j] + k) / 2;
                }*/
                
            }
            else { // training phase
                bcnn_gemm(0, 0, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
                // Mapping to obtain similar range output than xnor gemm
                for (j = 0; j < n * m; ++j) {
                    c[j] = (c[j] + k) / 2;
                }
            }
        }
        else {
#if BCNN_USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
#else
            bcnn_gemm(0, 0, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
#endif
        }

        c += n * m;
        src.data += sz;
    }

    bcnn_add_bias(dst.data, layer->bias, batch_size, layer->num, dst.w * dst.h);

    sz = dst.w * dst.h * dst.c * batch_size;
    bcnn_forward_activation_cpu(dst.data, sz, layer->activation);

    return BCNN_SUCCESS;
}




int bcnn_backward_conv_layer_cpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = src.n;
    int i, sz = src.w * src.h * src.c;
    int m = layer->num;
    int n = layer->size * layer->size * src.c;
    int k = dst.w * dst.h;
    float *a = NULL, *b = NULL, *c = NULL;
    
    bcnn_backward_activation_cpu(dst.data, dst.grad_data,
        dst.w * dst.h * dst.c * batch_size,
        layer->activation);

    bcnn_grad_bias(layer->bias_diff, dst.grad_data, batch_size, layer->num, k);

    for (i = 0; i < batch_size; ++i){
        a = dst.grad_data + i * m * k;
        b = layer->conv_workspace;
        c = layer->weight_diff;
        
        if (layer->size == 1) {
            b = src.data + i * sz;
        }
        else {
            bcnn_im2col(src.data + i * sz, src.c, src.h, src.w,
                layer->size, layer->pad, layer->stride, b);
        }
#if BCNN_USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a, k, b, k, 1.0f, c, n);
#else    
        bcnn_gemm(0, 1, m, n, k, 1.0f, a, k, b, k, 1.0f, c, n);
#endif

        if (src.grad_data) {
            a = layer->weight;
            b = dst.grad_data + i * m * k;
            c = layer->conv_workspace;
            
            if (layer->size == 1) {
#if BCNN_USE_BLAS
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, k, m, 1.0f,
                     a, n, b, k, 0.0f, src.grad_data + i * sz, k);
#else    
                bcnn_gemm(1, 0, n, k, m, 1.0f, a, n, b, k, 0.0f, src.grad_data + i * sz, k);
#endif
            }
            else {
#if BCNN_USE_BLAS
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, k, m, 1.0f, a, n, b, k, 0.0f, c, k);
#else
                bcnn_gemm(1, 0, n, k, m, 1.0f, a, n, b, k, 0.0f, c, k);
#endif
                bcnn_col2im(layer->conv_workspace, src.c, src.h, src.w,
                    layer->size, layer->pad, layer->stride, src.grad_data + i * sz);
            }
        }
    }

    if (layer->quantize && src.grad_data) {
        for (i = 0; i < batch_size * sz; ++i) {
            src.grad_data[i] = src.grad_data[i] * ((fabs(src.data[i]) <= 1.0f) ? 1.0f : 0.0f);
        }
    }
    
    return BCNN_SUCCESS;
}

#ifdef BCNN_USE_CUDA

int bcnn_forward_conv_layer_gpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = dst.n;
    int sz;
    /*bh_timer t = { 0 };
    bh_timer_start(&t);*/
    
#ifdef BCNN_USE_CUDNN
    float alpha = 1.0f, beta = 0.0f;
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
    bcnn_cudnn_check(cudnnAddTensor(bcnn_cudnn_handle(), &alpha,
              layer->bias_desc, layer->bias_gpu,
              &alpha,
              layer->dst_tensor_desc, dst.data_gpu));
#else
    int i, w_sz, out_sz, out_spatial_dim;

    out_sz = batch_size * dst.w * dst.h * dst.c;
    w_sz = layer->size * layer->size * src.c;
    out_spatial_dim = dst.w * dst.h;
    sz = src.c * src.h * src.w;

    bcnn_cuda_fill_f32(out_sz, 0, dst.data_gpu, 1);
    for (i = 0; i < batch_size; ++i) {
        if (layer->size == 1)
            layer->conv_workspace_gpu = src.data_gpu + i * sz;
        else {
            bcnn_cuda_im2col(src.data_gpu + i * sz,
                src.c, src.h, src.w,
                layer->size, layer->stride, layer->pad, layer->conv_workspace_gpu);
        }
        bcnn_cuda_gemm(0, 0, layer->num, out_spatial_dim, w_sz, 1.0f,
            layer->weight_gpu, w_sz, layer->conv_workspace_gpu, out_spatial_dim, 1.0f,
            dst.data_gpu + i * layer->num * out_spatial_dim, out_spatial_dim);
    }
    bcnn_cuda_add_bias(dst.data_gpu, layer->bias_gpu, batch_size, layer->num, out_spatial_dim);
#endif

    sz = dst.w * dst.h * dst.c * batch_size;
    bcnn_forward_activation_gpu(dst.data_gpu, sz, layer->activation);
    
    /*bh_timer_stop(&t);
    fprintf(stderr, "conv-forward-time %lf sec\n", bh_timer_get_msec(&t) / 1000);*/

    return BCNN_SUCCESS;
}


int bcnn_backward_conv_layer_gpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = dst.n;
#ifndef BCNN_USE_CUDNN
    int i, sz = src.w * src.h * src.c;
    int w_sz = layer->size * layer->size * src.c;
    int out_spatial_dim = dst.w * dst.h;
#else
    float one = 1.0f, zero = 0.0f;
#endif
    /*bh_timer t = { 0 };
    bh_timer_start(&t);*/

    bcnn_backward_activation_gpu(dst.data_gpu, dst.grad_data_gpu,
        dst.w * dst.h * dst.c * batch_size,
        layer->activation);

#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnConvolutionBackwardBias(bcnn_cudnn_handle(),
              &one,
              layer->dst_tensor_desc,  dst.grad_data_gpu,
              &one,
              layer->bias_desc, layer->bias_diff_gpu));
    bcnn_cudnn_check(cudnnConvolutionBackwardFilter(bcnn_cudnn_handle(),
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
        bcnn_cudnn_check(cudnnConvolutionBackwardData(bcnn_cudnn_handle(),
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
    bcnn_cuda_grad_bias(layer->bias_diff_gpu, dst.grad_data_gpu, batch_size, layer->num, out_spatial_dim);
    for (i = 0; i < batch_size; ++i) {
        if (layer->size == 1)
            layer->conv_workspace_gpu = src.data_gpu + i * sz;
        else {
            bcnn_cuda_im2col(src.data_gpu + i * sz,
                src.c, src.h, src.w,
                layer->size, layer->stride, layer->pad, layer->conv_workspace_gpu);
        }
        bcnn_cuda_gemm(0, 1, layer->num, w_sz, out_spatial_dim, 1,
            dst.grad_data_gpu + i * layer->num * out_spatial_dim, out_spatial_dim, layer->conv_workspace_gpu, out_spatial_dim, 1,
            layer->weight_diff_gpu, w_sz);

        if (src.grad_data_gpu) {
            if (layer->size == 1) {
                bcnn_cuda_gemm(1, 0, w_sz, out_spatial_dim, layer->num, 1,
                    layer->weight_gpu, w_sz, dst.grad_data_gpu + i * out_spatial_dim * layer->num, out_spatial_dim, 0,
                    src.grad_data_gpu + i * sz, out_spatial_dim);
            }
            else {
                bcnn_cuda_gemm(1, 0, w_sz, out_spatial_dim, layer->num, 1,
                    layer->weight_gpu, w_sz, dst.grad_data_gpu + i * out_spatial_dim * layer->num, out_spatial_dim, 0,
                    layer->conv_workspace_gpu, out_spatial_dim);
                bcnn_cuda_col2im(layer->conv_workspace_gpu,
                    src.c, src.h, src.w,
                    layer->size, layer->stride, layer->pad, src.grad_data_gpu + i * sz);
            }
        }
    }
#endif
    /*bh_timer_stop(&t);
    fprintf(stderr, "conv-backward-time %lf sec\n", bh_timer_get_msec(&t) / 1000);*/
    return BCNN_SUCCESS;
}

#endif



int bcnn_forward_conv_layer(bcnn_net *net, bcnn_connection *conn)
{
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_conv_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_forward_conv_layer_cpu(conn->layer, src, dst);
#endif
}

int bcnn_backward_conv_layer(bcnn_net *net, bcnn_connection *conn)
{
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_conv_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_backward_conv_layer_cpu(conn->layer, src, dst);
#endif
}