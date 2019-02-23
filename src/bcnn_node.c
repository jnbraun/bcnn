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

#include "bcnn_node.h"

bcnn_status bcnn_node_add_output(bcnn_net *net, bcnn_node *node, int index) {
    int *p_dst = NULL;
    node->num_dst++;
    p_dst = (int *)realloc(node->dst, node->num_dst * sizeof(int));
    BCNN_CHECK_AND_LOG(net->log_ctx, (p_dst != NULL), BCNN_FAILED_ALLOC,
                       "Internal allocation error");
    node->dst = p_dst;
    node->dst[node->num_dst - 1] = index;
    return BCNN_SUCCESS;
}

bcnn_status bcnn_node_add_input(bcnn_net *net, bcnn_node *node, int index) {
    int *p_src = NULL;
    node->num_src++;
    p_src = (int *)realloc(node->src, node->num_src * sizeof(int));
    BCNN_CHECK_AND_LOG(net->log_ctx, (p_src != NULL), BCNN_FAILED_ALLOC,
                       "Internal allocation error");
    node->src = p_src;
    node->src[node->num_src - 1] = index;
    return BCNN_SUCCESS;
}