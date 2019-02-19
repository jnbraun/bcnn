
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

#ifdef __cplusplus
extern "C" {
#endif

#ifdef BCNN_USE_CUDA
typedef struct bcnn_cuda_context {
    int workspace_size;
    float *workspace_gpu;
} bcnn_cuda_context;
#endif

struct bcnn_loader {
    int n_samples;
    int input_width;
    int input_height;
    int input_depth;
    bcnn_loader_type type;
    uint8_t *input_uchar;
    uint8_t *input_net;
    FILE *f_input;
    FILE *f_label;
    FILE *f_test;
    FILE *f_test_extra;
    bcnn_data_augmenter *data_aug; /* Parameters for online data augmentation */
};

bcnn_status bcnn_create_gemm_context(bcnn_net *net);
#ifdef BCNN_USE_CUDA
bcnn_status bcnn_create_cuda_context(bcnn_net *net);
#endif
bcnn_status bcnn_net_add_node(bcnn_net *net, bcnn_node node);
bcnn_status bcnn_node_add_input(bcnn_net *net, bcnn_node *node, int index);
bcnn_status bcnn_node_add_output(bcnn_net *net, bcnn_node *node, int index);
bcnn_status bcnn_net_add_tensor(bcnn_net *net, bcnn_tensor tensor);
int bcnn_get_tensor_index_with_name(bcnn_net *net, const char *name);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_NET_H