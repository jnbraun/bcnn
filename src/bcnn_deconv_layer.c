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

#include "bcnn_deconv_layer.h"

#ifdef BCNN_USE_BLAS
#include "cblas.h"
#endif

#include <bh/bh_mem.h>
#include <bh/bh_string.h>
#include <bh/bh_timer.h>

#include "bcnn_activation_layer.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"
#include "bh_log.h"

/* Deconv layer */
int bcnn_add_deconvolutional_layer(bcnn_net *net, int n, int size, int stride,
                                   int pad, bcnn_filler_type init,
                                   bcnn_activation activation, char *src_id,
                                   char *dst_id) {
    int i, sz;
    float std_init = 0.0f;
    bcnn_gauss_gen g = {0};
    bcnn_connection conn = {0};
    bcnn_node dst_node = {0};

    if (net->nb_connections > 0) {
        int is_src_node_found = 0;
        for (i = net->num_nodes - 1; i >= 0; --i) {
            if (strcmp(net->nodes[i].id, src_id) == 0) {
                bcnn_connection_add_src_node(&conn, i);
                is_src_node_found = 1;
                break;
            }
        }
        bh_check(is_src_node_found,
                 "Deconvolution layer: invalid input node name %s", src_id);
    } else {
        bcnn_connection_add_src_node(&conn, 0);
    }

    // Create layer
    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = DECONVOLUTIONAL;
    conn.layer->num = n;
    conn.layer->stride = stride;
    conn.layer->size = size;
    conn.layer->pad = pad;

    // Setup layer weights
    bcnn_tensor_create(&conn.layer->weights, 1, 1, 1,
                       net->nodes[conn.src[0]].tensor.c * n * size * size, 1);
    bcnn_tensor_filler w_filler = {
        .range = (size * size * net->nodes[conn.src[0]].tensor.c),
        .type = init};
    bcnn_tensor_fill(&conn.layer->weights, w_filler);
    // Setup layer biases
    bcnn_tensor_create(&conn.layer->biases, 1, 1, 1, n, 1);

    if (net->learner.optimizer == ADAM) {
        int weights_size = bcnn_tensor_get_size(&conn.layer->weights);
        conn.layer->adam_m = (float *)calloc(weights_size, sizeof(float));
        conn.layer->adam_v = (float *)calloc(weights_size, sizeof(float));
    }

    bh_strfill(&dst_node.id, dst_id);
    bcnn_tensor_set_shape(
        &dst_node.tensor, net->nodes[conn.src[0]].tensor.n, conn.layer->num,
        conn.layer->stride * (net->nodes[conn.src[0]].tensor.h - 1) +
            conn.layer->size - 2 * conn.layer->pad,
        conn.layer->stride * (net->nodes[conn.src[0]].tensor.w - 1) +
            conn.layer->size - 2 * conn.layer->pad,
        1);
    bcnn_tensor_allocate(&dst_node.tensor);
    // Add node to net
    bcnn_net_add_node(net, dst_node);
    // Add node pointer to connection
    bcnn_connection_add_dst_node(&conn, net->num_nodes - 1);
    sz = net->nodes[conn.dst[0]].tensor.w * net->nodes[conn.dst[0]].tensor.h *
         net->nodes[conn.src[0]].tensor.c * size * size;
    conn.layer->conv_workspace = (float *)calloc(sz, sizeof(float));

#ifdef BCNN_USE_CUDA
    sz = net->nodes[conn.dst[0]].tensor.w * net->nodes[conn.dst[0]].tensor.h *
         net->nodes[conn.src[0]].tensor.c * size * size;
    conn.layer->conv_workspace_gpu =
        bcnn_cuda_memcpy_f32(conn.layer->conv_workspace, sz);
    if (net->learner.optimizer == ADAM) {
        int weights_size = bcnn_tensor_get_size(&conn.layer->weights);
        conn.layer->adam_m_gpu =
            bcnn_cuda_memcpy_f32(conn.layer->adam_m, weights_size);
        conn.layer->adam_v_gpu =
            bcnn_cuda_memcpy_f32(conn.layer->adam_v, weights_size);
    }
#endif
    conn.layer->activation = activation;

    bcnn_net_add_connection(net, conn);

    bh_log_info(
        "[Deconvolutional] input_shape= %dx%dx%d nb_filters= %d kernel_size= "
        "%d stride= %d output_shape= %dx%dx%d\n",
        net->nodes[conn.src[0]].tensor.w, net->nodes[conn.src[0]].tensor.h,
        net->nodes[conn.src[0]].tensor.c, n, size, stride,
        net->nodes[conn.dst[0]].tensor.w, net->nodes[conn.dst[0]].tensor.h,
        net->nodes[conn.dst[0]].tensor.c);

    return BCNN_SUCCESS;
}

int bcnn_forward_deconv_layer_cpu(bcnn_layer *layer, bcnn_node *src_node,
                                  bcnn_node *dst_node) {
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
        bcnn_gemm(1, 0, m, n, k, 1.0f, layer->weights.data, m,
                  src.data + i * sz, n, 0.0f, layer->conv_workspace, n);
        bcnn_col2im(layer->conv_workspace, layer->num, dst.h, dst.w,
                    layer->size, 0, layer->stride,
                    dst.data + i * layer->num * dst.w * dst.h);
    }

    bcnn_add_bias(dst.data, layer->biases.data, batch_size, layer->num,
                  dst.w * dst.h);

    sz = dst.w * dst.h * dst.c * batch_size;
    bcnn_forward_activation_cpu(dst.data, sz, layer->activation);

    return BCNN_SUCCESS;
}

int bcnn_backward_deconv_layer_cpu(bcnn_layer *layer, bcnn_node *src_node,
                                   bcnn_node *dst_node) {
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

    bcnn_grad_bias(layer->biases.grad_data, dst.grad_data, batch_size,
                   layer->num, dst.w * dst.h);

    for (i = 0; i < batch_size; ++i) {
        pdst = dst.grad_data + i * layer->num * dst.w * dst.h;
        bcnn_im2col(pdst, dst.c, dst.h, dst.w, layer->size, 0, layer->stride,
                    layer->conv_workspace);
        bcnn_gemm(0, 1, m, n, k, alpha, src.data + i * src.c * src.h * src.w, k,
                  layer->conv_workspace, k, 1.0f, layer->weights.grad_data, n);

        if (src.grad_data) {
            bcnn_gemm(0, 0, src.c, k, n, 1.0f, layer->weights.data, n,
                      layer->conv_workspace, k, 0.0f, src.grad_data + i * sz,
                      k);
        }
    }
    return BCNN_SUCCESS;
}

#ifdef BCNN_USE_CUDA

int bcnn_forward_deconv_layer_gpu(bcnn_layer *layer, bcnn_node *src_node,
                                  bcnn_node *dst_node) {
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int i, m, n, k, sz;
    int batch_size = dst.n;

    sz = batch_size * dst.w * dst.h * dst.c;
    bcnn_cuda_fill_f32(sz, 0, dst.data_gpu, 1);

    m = layer->num * layer->size * layer->size;
    k = src.c;
    n = src.w * src.h;
    sz = src.c * src.h * src.w;
    for (i = 0; i < batch_size; ++i) {
        bcnn_cuda_gemm(1, 0, m, n, k, 1.0f, layer->weights.data_gpu, m,
                       src.data_gpu + i * sz, n, 0.0f,
                       layer->conv_workspace_gpu, n);
        bcnn_cuda_col2im(layer->conv_workspace_gpu, layer->num, dst.h, dst.w,
                         layer->size, layer->stride, 0,
                         dst.data_gpu + i * layer->num * dst.w * dst.h);
    }

    bcnn_cuda_add_bias(dst.data_gpu, layer->biases.data_gpu, batch_size,
                       layer->num, dst.w * dst.h);

    sz = dst.w * dst.h * dst.c * batch_size;
    bcnn_forward_activation_gpu(dst.data_gpu, sz, layer->activation);

    return BCNN_SUCCESS;
}

int bcnn_backward_deconv_layer_gpu(bcnn_layer *layer, bcnn_node *src_node,
                                   bcnn_node *dst_node) {
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int i, sz = src.w * src.h * src.c;
    int m = src.c;
    int n = layer->size * layer->size * dst.c;
    int k = src.w * src.h;
    int batch_size = src.n;
    float *a = NULL, *b = NULL, *c = NULL, *pdst = NULL;
    float alpha = 1.0f / batch_size;

    bcnn_backward_activation_gpu(dst.data_gpu, dst.grad_data_gpu,
                                 dst.w * dst.h * dst.c * batch_size,
                                 layer->activation);

    bcnn_cuda_grad_bias(layer->biases.grad_data_gpu, dst.grad_data_gpu,
                        batch_size, layer->num, dst.h * dst.w);

    for (i = 0; i < batch_size; ++i) {
        a = src.data_gpu + i * src.c * src.w * src.h;
        b = layer->conv_workspace_gpu;
        c = layer->weights.grad_data_gpu;

        pdst = dst.grad_data_gpu + i * dst.c * dst.w * dst.h;

        bcnn_cuda_im2col(pdst, dst.c, dst.h, dst.w, layer->size, layer->stride,
                         0, layer->conv_workspace_gpu);
        bcnn_cuda_gemm(0, 1, m, n, k, alpha, a, k, b, k, 1.0f, c, n);

        if (src.grad_data_gpu) {
            a = layer->weights.data_gpu;
            b = layer->conv_workspace_gpu;
            c = src.grad_data_gpu + i * sz;
            bcnn_cuda_gemm(0, 0, src.c, k, n, 1.0f, a, n, b, k, 0.0f, c, k);
        }
    }
    return 0;
}

#endif

int bcnn_forward_deconv_layer(bcnn_net *net, bcnn_connection *conn) {
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_deconv_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_forward_deconv_layer_cpu(conn->layer, src, dst);
#endif
}

int bcnn_backward_deconv_layer(bcnn_net *net, bcnn_connection *conn) {
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_deconv_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_backward_deconv_layer_cpu(conn->layer, src, dst);
#endif
}