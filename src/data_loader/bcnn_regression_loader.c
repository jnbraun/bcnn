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
#include "bcnn_tensor.h"
#include "bcnn_utils.h"

bcnn_status bcnn_loader_list_reg_init(bcnn_loader *iter, bcnn_net *net,
                                      const char *train_path,
                                      const char *train_path_extra,
                                      const char *test_path,
                                      const char *test_path_extra) {
    // Open the files handles according to each dataset path
    BCNN_CHECK_STATUS(bcnn_open_dataset(iter, net, train_path, train_path_extra,
                                        test_path, test_path_extra, false));
    // Allocate img buffer
    BCNN_CHECK_AND_LOG(
        net->log_ctx,
        net->tensors[0].w > 0 && net->tensors[0].h > 0 && net->tensors[0].c > 0,
        BCNN_INVALID_PARAMETER,
        "Input's width, height and channels must be > 0");
    iter->input_uchar = (uint8_t *)calloc(
        net->tensors[0].w * net->tensors[0].h * net->tensors[0].c,
        sizeof(uint8_t));

    return BCNN_SUCCESS;
}

void bcnn_loader_list_reg_terminate(bcnn_loader *iter) {
    if (iter->f_train != NULL) {
        fclose(iter->f_train);
    }
    if (iter->f_test != NULL) {
        fclose(iter->f_test);
    }
    bh_free(iter->input_uchar);
}

bcnn_status bcnn_loader_list_reg_next(bcnn_loader *iter, bcnn_net *net,
                                      int idx) {
    char *line = bh_fgetline(iter->f_current);
    if (line == NULL) {
        rewind(iter->f_current);
        line = bh_fgetline(iter->f_current);
    }
    char **tok = NULL;
    int num_toks = bh_strsplit(line, ' ', &tok);
    // Load image, perform data augmentation if required and fill input tensor
    bcnn_fill_input_tensor(net, iter, tok[0], idx);
    // Fill label tensor
    if (net->mode != BCNN_MODE_PREDICT) {
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