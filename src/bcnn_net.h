
/*
 * Copyright (c) 2016-present Jean-Noel Braun.
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

#ifndef BCNN_NET_H
#define BCNN_NET_H

#include <bcnn/bcnn.h>
#include "bcnn_data.h"
#include "bcnn_learner.h"
#include "bcnn_node.h"
#include "bcnn_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef BCNN_USE_CUDA
typedef struct bcnn_cuda_context {
    int workspace_size;
    float *workspace_gpu;
} bcnn_cuda_context;
#endif

/**
 * Net definition
 */
struct bcnn_net {
    int batch_size;
    int num_nodes;   /* Number of nodes hold in the network */
    int num_tensors; /* Number of tensors hold in the network */
    bcnn_mode mode;
    bcnn_log_context log_ctx; /* Logging stuff */
    bcnn_node *nodes;         /* Array of 'num_nodes' nodes */
    bcnn_tensor *tensors;     /* Array of 'num_tensors' tensors */
    bcnn_learner *learner;    /* Learner/optimizer parameters */
    bcnn_loader *data_loader; /* Handles the loading and iteration over training
                                 / testing datasets */
    bcnn_data_augmenter *data_aug; /* Handles the online data augmentation */
    void *gemm_ctx;
#ifdef BCNN_USE_CUDA
    void *cuda_ctx;
#endif
    int num_threads; /* Number of threads (CPU only) */
};

bcnn_status bcnn_net_create_gemm_context(bcnn_net *net);
#ifdef BCNN_USE_CUDA
bcnn_status bcnn_net_create_cuda_context(bcnn_net *net);
#endif
bcnn_status bcnn_net_add_node(bcnn_net *net, bcnn_node node);
bcnn_status bcnn_net_add_tensor(bcnn_net *net, bcnn_tensor tensor);
void bcnn_net_set_param(bcnn_net *net, const char *name, const char *val);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_NET_H