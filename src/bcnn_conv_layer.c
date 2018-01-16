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

#ifdef BCNN_USE_BLAS
#include "cblas.h"
#endif

#include <bh/bh_mem.h>
#include <bh/bh_string.h>
#include <bh/bh_timer.h>

#include "bh_log.h"

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
            bcnn_add_scalar(size, bias[i], output + i * size);
        }
        output += n * size;
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
            _bcnn_im2col(src.data, src.c, src.h, src.w,
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

    _bcnn_add_bias(dst.data, layer->bias, batch_size, layer->num, dst.w * dst.h);

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

    _bcnn_backward_bias(layer->bias_diff, dst.grad_data, batch_size, layer->num, k);

    for (i = 0; i < batch_size; ++i){
        a = dst.grad_data + i * m * k;
        b = layer->conv_workspace;
        c = layer->weight_diff;
        
        if (layer->size == 1) {
            b = src.data + i * sz;
        }
        else {
            _bcnn_im2col(src.data + i * sz, src.c, src.h, src.w,
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
                _bcnn_col2im(layer->conv_workspace, src.c, src.h, src.w,
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


/* Deconv layer */
int bcnn_add_deconvolutional_layer(bcnn_net *net, int n, int size, int stride, int pad,
    bcnn_weights_init init, bcnn_activation activation, char *src_id, char *dst_id)
{
    int i, sz;
    float std_init = 0.0f;
    bcnn_gauss_gen g = { 0 };
    bcnn_connection conn = { 0 };
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
        bh_check(is_src_node_found, "Deconvolution layer: invalid input node name %s", src_id);
    }
    else {
        bcnn_connection_add_src_node(&conn, 0);
    }

    // Create layer
    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = DECONVOLUTIONAL;
    conn.layer->num = n;
    conn.layer->stride = stride;
    conn.layer->size = size;
    conn.layer->pad = pad;
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
        conn.layer->stride * (net->nodes[conn.src[0]].tensor.h - 1) + conn.layer->size - 2 * conn.layer->pad,
        conn.layer->stride * (net->nodes[conn.src[0]].tensor.w - 1) + conn.layer->size - 2 * conn.layer->pad,
        1);
    bcnn_tensor_allocate(&dst_node.tensor);
    // Add node to net
    bcnn_net_add_node(net, dst_node);
    // Add node pointer to connection
    bcnn_connection_add_dst_node(&conn, net->num_nodes - 1);
    sz = net->nodes[conn.dst[0]].tensor.w * net->nodes[conn.dst[0]].tensor.h * net->nodes[conn.src[0]].tensor.c * size * size;
    conn.layer->conv_workspace = (float *)calloc(sz, sizeof(float));

#ifdef BCNN_USE_CUDA
    conn.layer->weight_gpu = bcnn_cuda_memcpy_f32(conn.layer->weight, conn.layer->weights_size);
    conn.layer->weight_diff_gpu = bcnn_cuda_memcpy_f32(conn.layer->weight_diff, conn.layer->weights_size);
    conn.layer->bias_gpu = bcnn_cuda_memcpy_f32(conn.layer->bias, conn.layer->bias_size);
    conn.layer->bias_diff_gpu = bcnn_cuda_memcpy_f32(conn.layer->bias_diff, conn.layer->bias_size);
    sz = net->nodes[conn.dst[0]].tensor.w * net->nodes[conn.dst[0]].tensor.h * net->nodes[conn.src[0]].tensor.c * size * size;
    conn.layer->conv_workspace_gpu = bcnn_cuda_memcpy_f32(conn.layer->conv_workspace, sz);
    if (net->learner.optimizer == ADAM) {
        conn.layer->adam_m_gpu = bcnn_cuda_memcpy_f32(conn.layer->adam_m, conn.layer->weights_size);
        conn.layer->adam_v_gpu = bcnn_cuda_memcpy_f32(conn.layer->adam_v, conn.layer->weights_size);
    }
#endif
    conn.layer->activation = activation;

    bcnn_net_add_connection(net, conn);

    bh_log_info("[Deconvolutional] input_shape= %dx%dx%d nb_filters= %d kernel_size= %d stride= %d output_shape= %dx%dx%d\n",
        net->nodes[conn.src[0]].tensor.w, net->nodes[conn.src[0]].tensor.h, net->nodes[conn.src[0]].tensor.c, n, size, stride,
        net->nodes[conn.dst[0]].tensor.w, net->nodes[conn.dst[0]].tensor.h, net->nodes[conn.dst[0]].tensor.c);

    return BCNN_SUCCESS;
}


int bcnn_forward_deconv_layer_cpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = src.n;
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


int bcnn_backward_deconv_layer_cpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = src.n;
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
        bcnn_gemm(0, 1, m, n, k, alpha, src.data + i * src.c * src.h * src.w,
            k, layer->conv_workspace, k, 1.0f, layer->weight_diff, n);

        if (src.grad_data) {
            bcnn_gemm(0, 0, src.c, k, n, 1.0f, layer->weight, n, layer->conv_workspace, k, 0.0f, src.grad_data + i * sz, k);
        }
    }
    return BCNN_SUCCESS;
}


int bcnn_forward_deconv_layer(bcnn_net *net, bcnn_connection *conn)
{
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_deconv_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_forward_deconv_layer_cpu(conn->layer, src, dst);
#endif
}

int bcnn_backward_deconv_layer(bcnn_net *net, bcnn_connection *conn)
{
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_deconv_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_backward_deconv_layer_cpu(conn->layer, src, dst);
#endif
}


/* Depthwise Separable convolution */

int bcnn_add_depthwise_sep_conv_layer(bcnn_net *net, int size, int stride, int pad,
    int batch_norm, bcnn_weights_init init, bcnn_activation activation, char *src_id, char *dst_id)
{
    int nb_connections = net->nb_connections + 1;
    int i, sz;
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
        bh_check(is_src_node_found, "Dephtwise convolution layer: invalid input node name %s", src_id);
    }
    else {
        bcnn_connection_add_src_node(&conn, 0);
    }

    // Create layer
    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = DEPTHWISE_CONV;
    conn.layer->num = net->nodes[conn.src[0]].tensor.c;
    conn.layer->stride = stride;
    conn.layer->size = size;
    conn.layer->pad = pad;
    conn.layer->bias_size = net->nodes[conn.src[0]].tensor.c;
    conn.layer->weights_size = net->nodes[conn.src[0]].tensor.c * size * size;
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
        net->nodes[conn.src[0]].tensor.c,
        (net->nodes[conn.src[0]].tensor.h + 2 * conn.layer->pad - conn.layer->size) / conn.layer->stride + 1,
        (net->nodes[conn.src[0]].tensor.w + 2 * conn.layer->pad - conn.layer->size) / conn.layer->stride + 1,
        1);
    bcnn_tensor_allocate(&dst_node.tensor);
    // Add node to net
    bcnn_net_add_node(net, dst_node);
    // Add node pointer to connection
    bcnn_connection_add_dst_node(&conn, net->num_nodes - 1);

    sz = net->nodes[conn.dst[0]].tensor.w * net->nodes[conn.dst[0]].tensor.h * net->nodes[conn.src[0]].tensor.c * size * size;
    conn.layer->conv_workspace = (float *)calloc(sz, sizeof(float));

#ifdef BCNN_USE_CUDA
    conn.layer->weight_gpu = bcnn_cuda_memcpy_f32(conn.layer->weight, conn.layer->weights_size);
    conn.layer->weight_diff_gpu = bcnn_cuda_memcpy_f32(conn.layer->weight_diff, conn.layer->weights_size);
    conn.layer->bias_gpu = bcnn_cuda_memcpy_f32(conn.layer->bias, conn.layer->bias_size);
    conn.layer->bias_diff_gpu = bcnn_cuda_memcpy_f32(conn.layer->bias_diff, conn.layer->bias_size);
    if (net->learner.optimizer == ADAM) {
        conn.layer->adam_m_gpu = bcnn_cuda_memcpy_f32(conn.layer->adam_m, conn.layer->weights_size);
        conn.layer->adam_v_gpu = bcnn_cuda_memcpy_f32(conn.layer->adam_v, conn.layer->weights_size);
    }
    sz = net->nodes[conn.dst[0]].tensor.w * net->nodes[conn.dst[0]].tensor.h * net->nodes[conn.src[0]].tensor.c * size * size;
    conn.layer->conv_workspace_gpu = bcnn_cuda_memcpy_f32(conn.layer->conv_workspace, sz);
#endif
    conn.layer->activation = activation;

    bcnn_net_add_connection(net, conn);

    bh_log_info("[DepthwiseConvolutional] input_shape= %dx%dx%d nb_filters= %d kernel_size= %d stride= %d padding= %d output_shape= %dx%dx%d\n",
        net->nodes[conn.src[0]].tensor.w, net->nodes[conn.src[0]].tensor.h, net->nodes[conn.src[0]].tensor.c,
        net->nodes[conn.src[0]].tensor.c, size, stride, pad,
        net->nodes[conn.dst[0]].tensor.w, net->nodes[conn.dst[0]].tensor.h, net->nodes[conn.dst[0]].tensor.c);

    return 0;
}

int bcnn_forward_depthwise_sep_conv_layer_cpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    int n, sz, c, h, w, kh, kw, h_in, w_in, offset;
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = src.n;
    float *dst_data = NULL;
    const float *bias_data = NULL;
    const float *weight_data = NULL;
    float val = 0;
    /*bh_timer t = { 0 };
    bh_timer_start(&t);*/

    sz = bcnn_tensor_get_size(&dst);

    dst_data = dst.data;
    memset(dst_data, 0, sz * sizeof(float));
    
    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < dst.c; ++c) {
            for (h = 0; h < dst.h; ++h) {
                if (h * layer->stride - layer->pad >= 0 && (h * layer->stride - layer->pad + layer->size) < src.h) {
                    for (w = 0; w < dst.w; ++w) {
                        weight_data = layer->weight + c * layer->size * layer->size;
                        val = 0;
                        if (w * layer->stride - layer->pad >= 0 && (w * layer->stride - layer->pad + layer->size) < src.w) {
                            for (kh = 0; kh < layer->size; ++kh) {
                                for (kw = 0; kw < layer->size; ++kw) {
                                    h_in = -layer->pad + h * layer->stride + kh;
                                    w_in = -layer->pad + w * layer->stride + kw;
                                    offset = ((n * dst.c + c) * src.h + h_in) * src.w + w_in;
                                    val += (*weight_data) * src.data[offset];
                                    ++weight_data;
                                }
                            }
                        }
                        else {
                            for (kh = 0; kh < layer->size; ++kh) {
                                for (kw = 0; kw < layer->size; ++kw) {
                                    h_in = -layer->pad + h * layer->stride + kh;
                                    w_in = -layer->pad + w * layer->stride + kw;
                                    if ((w_in >= 0) && (w_in < src.w)) {
                                        offset = ((n * dst.c + c) * src.h + h_in) * src.w + w_in;
                                        val += (*weight_data) * src.data[offset];
                                    }
                                    ++weight_data;
                                }
                            }
                        }
                        *dst_data++ = val;
                    }
                }
                else {
                    for (w = 0; w < dst.w; ++w) {
                        weight_data = layer->weight + c * layer->size * layer->size;
                        val = 0;
                        if (w * layer->stride - layer->pad >= 0 && (w * layer->stride - layer->pad + layer->size) < src.w) {
                            for (kh = 0; kh < layer->size; ++kh) {
                                for (kw = 0; kw < layer->size; ++kw) {
                                    h_in = -layer->pad + h * layer->stride + kh;
                                    w_in = -layer->pad + w * layer->stride + kw;
                                    if ((h_in >= 0) && (h_in < src.h)) {
                                        offset = ((n * dst.c + c) * src.h + h_in) * src.w + w_in;
                                        val += (*weight_data) * src.data[offset];
                                    }
                                    ++weight_data;
                                }
                            }
                        }
                        else {
                            for (kh = 0; kh < layer->size; ++kh) {
                                for (kw = 0; kw < layer->size; ++kw) {
                                    h_in = -layer->pad + h * layer->stride + kh;
                                    w_in = -layer->pad + w * layer->stride + kw;
                                    if ((h_in >= 0) && (h_in < src.h) && (w_in >= 0) && (w_in < src.w)) {
                                        offset = ((n * dst.c + c) * src.h + h_in) * src.w + w_in;
                                        val += (*weight_data) * src.data[offset];
                                    }
                                    ++weight_data;
                                }
                            }
                        }
                        *dst_data++ = val;
                    }
                }
            }
        }
    }
       
    _bcnn_add_bias(dst.data, layer->bias, batch_size, dst.c, dst.w * dst.h);

    sz = dst.w * dst.h * dst.c * batch_size;
    bcnn_forward_activation_cpu(dst.data, sz, layer->activation);

    /*bh_timer_stop(&t);
    fprintf(stderr, "sep-conv-forward-time %lf sec\n", bh_timer_get_msec(&t) / 1000);*/
    
    return BCNN_SUCCESS;
}


int bcnn_backward_depthwise_sep_conv_layer_cpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    int sz, n, c, h, w, kh, kw, w_in, h_in, offset;
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = src.n;
    float *dst_grad_data = NULL;
    float *weight_diff_base = NULL, *weight_diff = NULL;
    float *weight_data_base = NULL, *weight_data = NULL;
    float *bias_diff = NULL;
    /*bh_timer t = { 0 };
    bh_timer_start(&t);*/
    
    sz = bcnn_tensor_get_size(&dst);
    
    bcnn_backward_activation_cpu(dst.data, dst.grad_data,
        dst.w * dst.h * dst.c * batch_size,
        layer->activation);

    _bcnn_backward_bias(layer->bias_diff, dst.grad_data, batch_size, dst.c, dst.w * dst.h);
    
    if (src.grad_data) {
        dst_grad_data = dst.grad_data;
        weight_diff_base = layer->weight_diff;;
        for (n = 0; n < batch_size; ++n) {
            for (c = 0; c < dst.c; ++c) {
                for (h = 0; h < dst.h; ++h) {
                    if (h * layer->stride - layer->pad >= 0 && (h * layer->stride - layer->pad + layer->size) < src.h) {
                        for (w = 0; w < dst.w; ++w) {
                            weight_diff = weight_diff_base + c * layer->size * layer->size;
                            if (w * layer->stride - layer->pad >= 0 && (w * layer->stride - layer->pad + layer->size) < src.w) {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride + kh;
                                        w_in = -layer->pad + w * layer->stride + kw;
                                        offset = ((n * dst.c + c) * src.h + h_in) * src.w + w_in;
                                        *weight_diff += src.data[offset] * (*dst_grad_data);
                                        ++weight_diff;
                                    }
                                }
                            }
                            else {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride + kh;
                                        w_in = -layer->pad + w * layer->stride + kw;
                                        if ((w_in >= 0) && (w_in < src.w)) {
                                            offset = ((n * dst.c + c) * src.h + h_in) * src.w + w_in;
                                            *weight_diff += src.data[offset] * (*dst_grad_data);
                                        }
                                        ++weight_diff;
                                    }
                                }
                            }
                            ++dst_grad_data;
                        }
                    }
                    else {
                        for (w = 0; w < dst.w; ++w) {
                            weight_diff = weight_diff_base + c * layer->size * layer->size;
                            if (w * layer->stride - layer->pad >= 0 && (w * layer->stride - layer->pad + layer->size) < src.w) {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride + kh;
                                        w_in = -layer->pad + w * layer->stride + kw;
                                        if ((h_in >= 0) && (h_in < src.h)) {
                                            offset = ((n * dst.c + c) * src.h + h_in) * src.w + w_in;
                                            *weight_diff += src.data[offset] * (*dst_grad_data);
                                        }
                                        ++weight_diff;
                                    }
                                }
                            }
                            else {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride + kh;
                                        w_in = -layer->pad + w * layer->stride + kw;
                                        if ((h_in >= 0) && (h_in < src.h) && (w_in >= 0) && (w_in < src.w)) {
                                            offset = ((n * dst.c + c) * src.h + h_in) * src.w + w_in;
                                            *weight_diff += src.data[offset] * (*dst_grad_data);
                                        }
                                        ++weight_diff;
                                    }
                                }
                            }
                            ++dst_grad_data;
                        }
                    }
                }
            }
        }
    }
    if (src.grad_data) {
        dst_grad_data = dst.grad_data;
        weight_data_base = layer->weight;
        for (n = 0; n < batch_size; ++n) {
            for (c = 0; c < dst.c; ++c) {
                for (h = 0; h < dst.h; ++h) {
                    if (h * layer->stride - layer->pad >= 0 && (h * layer->stride - layer->pad + layer->size) < src.h) {
                        for (w = 0; w < dst.w; ++w) {
                            weight_data = weight_data_base + c * layer->size * layer->size;
                            if (w * layer->stride - layer->pad >= 0 && (w * layer->stride - layer->pad + layer->size) < src.w) {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride + kh;
                                        w_in = -layer->pad + w * layer->stride + kw;	
                                        offset = ((n * dst.c + c) * src.h + h_in) * src.w + w_in;
                                        src.grad_data[offset] += (*weight_data) * (*dst_grad_data);
                                        ++weight_data;
                                    }
                                }
                            }
                            else {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride + kh;
                                        w_in = -layer->pad + w * layer->stride + kw;
                                        if ((w_in >= 0) && (w_in < src.w)) {
                                            offset = ((n * dst.c + c) * src.h + h_in) * src.w + w_in;
                                            src.grad_data[offset] += (*weight_data) * (*dst_grad_data);
                                        }
                                        ++weight_data;
                                    }
                                }
                            }
                            ++dst_grad_data;
                        }
                    }
                    else {
                        for (w = 0; w < dst.w; ++w) {
                            weight_data = weight_data_base + c * layer->size * layer->size;
                            if (w * layer->stride - layer->pad >= 0 && (w * layer->stride - layer->pad + layer->size) < src.w) {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride + kh;
                                        w_in = -layer->pad + w * layer->stride + kw;
                                        if ((h_in >= 0) && (h_in < src.h)) {
                                            offset = ((n * dst.c + c) * src.h + h_in) * src.w + w_in;
                                            src.grad_data[offset] += (*weight_data) * (*dst_grad_data);
                                        }
                                        ++weight_data;
                                    }
                                }
                            }
                            else {
                                for (kh = 0; kh < layer->size; ++kh) {
                                    for (kw = 0; kw < layer->size; ++kw) {
                                        h_in = -layer->pad + h * layer->stride + kh;
                                        w_in = -layer->pad + w * layer->stride + kw;
                                        if ((h_in >= 0) && (h_in < src.h) && (w_in >= 0) && (w_in < src.w)) {
                                            offset = ((n * dst.c + c) * src.h + h_in) * src.w + w_in;
                                            src.grad_data[offset] += (*weight_data) * (*dst_grad_data);
                                        }
                                        ++weight_data;
                                    }
                                }
                            }
                            ++dst_grad_data;
                        }
                    }
                }
            }
        }
    }
    
    /*bh_timer_stop(&t);
    fprintf(stderr, "sep-conv-backward-time %lf sec\n", bh_timer_get_msec(&t) / 1000);*/
    
    return BCNN_SUCCESS;
}


int bcnn_forward_depthwise_sep_conv_layer(bcnn_net *net, bcnn_connection *conn)
{
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_depthwise_sep_conv_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_forward_depthwise_sep_conv_layer_cpu(conn->layer, src, dst);
#endif
}

int bcnn_backward_depthwise_sep_conv_layer(bcnn_net *net, bcnn_connection *conn)
{
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_depthwise_sep_conv_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_backward_depthwise_sep_conv_layer_cpu(conn->layer, src, dst);
#endif
}