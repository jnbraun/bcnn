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

#include "bcnn_regression_loader.h"

#include <bh/bh_macros.h>
#include <bh/bh_string.h>
#include "bcnn_data.h"
#include "bcnn_utils.h"

bcnn_status bcnn_loader_list_reg_init(bcnn_loader *iter, bcnn_net *net,
                                      const char *path_input,
                                      const char *path_extra) {
    FILE *f_list = NULL;
    f_list = fopen(path_input, "rb");
    if (f_list == NULL) {
        fprintf(stderr, "[ERROR] Can not open file %s\n", path_input);
        return BCNN_INVALID_PARAMETER;
    }
    // Allocate img buffer
    iter->input_uchar = (unsigned char *)calloc(
        iter->input_width * iter->input_height * iter->input_depth,
        sizeof(unsigned char));

    rewind(f_list);
    iter->f_input = f_list;
    return BCNN_SUCCESS;
}

void bcnn_loader_list_reg_terminate(bcnn_loader *iter) {
    if (iter->f_input != NULL) {
        fclose(iter->f_input);
    }
    bh_free(iter->input_uchar);
}

bcnn_status bcnn_loader_list_reg_next(bcnn_loader *iter, bcnn_net *net,
                                      int idx) {
    char *line = bh_fgetline(iter->f_input);
    if (line == NULL) {
        rewind(iter->f_input);
        line = bh_fgetline(iter->f_input);
    }
    char **tok = NULL;
    int num_toks = bh_strsplit(line, ' ', &tok);
    // Load image, perform data augmentation if required and fill input tensor
    bcnn_fill_input_tensor(net, iter, tok[0], idx);
    // Fill label tensor
    if (net->mode != PREDICT) {
        int label_sz = bcnn_tensor_size3d(&net->tensors[1]);
        BCNN_CHECK_AND_LOG(net->log_ctx, (num_toks - 1 == label_sz),
                           BCNN_INVALID_DATA, "Unexpected label format");
        float *y = net->tensors[1].data + idx * label_sz;
        memset(y, 0, label_sz * sizeof(float));
        for (int i = 0; i < label_sz; ++i) {
            y[i] = (float)atof(tok[i + 1]);
        }
    }
    bh_free(line);
    for (int i = 0; i < num_toks; ++i) {
        bh_free(tok[i]);
    }
    bh_free(tok);

    return BCNN_SUCCESS;
}