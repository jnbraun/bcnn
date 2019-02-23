
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

#ifndef BCNN_NODE_H
#define BCNN_NODE_H

#include <bcnn/bcnn.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Node definition
 */
struct bcnn_node {
    int num_src;
    int num_dst;
    bcnn_layer_type type;
    size_t param_size;
    int *src; /* Array of input tensors indexes */
    int *dst; /* Array of output tensors indexes */
    void *param;
    void (*forward)(struct bcnn_net *net, struct bcnn_node *node);
    void (*backward)(struct bcnn_net *net, struct bcnn_node *node);
    void (*update)(struct bcnn_net *net, struct bcnn_node *node);
    void (*release_param)(struct bcnn_node *node);
};
typedef struct bcnn_node bcnn_node;

bcnn_status bcnn_node_add_input(bcnn_net *net, bcnn_node *node, int index);
bcnn_status bcnn_node_add_output(bcnn_net *net, bcnn_node *node, int index);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_NODE_H