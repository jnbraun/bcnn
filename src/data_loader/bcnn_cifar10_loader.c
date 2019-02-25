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

#include "bcnn_cifar10_loader.h"

#include <bh/bh_macros.h>
#include <bh/bh_string.h>
/* include bip image processing lib */
#include <bip/bip.h>
#include "bcnn_tensor.h"
#include "bcnn_utils.h"

bcnn_status bcnn_loader_cifar10_init(bcnn_loader *iter, bcnn_net *net,
                                     const char *train_path,
                                     const char *train_path_extra,
                                     const char *test_path,
                                     const char *test_path_extra) {
    // Open the files handles according to each dataset path
    BCNN_CHECK_STATUS(bcnn_open_dataset(iter, net, train_path, train_path_extra,
                                        test_path, test_path_extra, false));
    iter->input_width = 32;
    iter->input_height = 32;
    iter->input_depth = 3;
    iter->input_uchar = (unsigned char *)calloc(
        iter->input_width * iter->input_height * iter->input_depth,
        sizeof(unsigned char));
    BCNN_CHECK_AND_LOG(
        net->log_ctx,
        net->tensors[0].w > 0 && net->tensors[0].h > 0 && net->tensors[0].c > 0,
        BCNN_INVALID_PARAMETER,
        "Input's width, height and channels must be > 0");
    iter->input_net = (uint8_t *)calloc(
        net->tensors[0].w * net->tensors[0].h * net->tensors[0].c,
        sizeof(uint8_t));

    return BCNN_SUCCESS;
}

void bcnn_loader_cifar10_terminate(bcnn_loader *iter) {
    if (iter->f_train != NULL) {
        fclose(iter->f_train);
    }
    if (iter->f_test != NULL) {
        fclose(iter->f_test);
    }
    bh_free(iter->input_uchar);
    bh_free(iter->input_net);
}

bcnn_status bcnn_loader_cifar10_next(bcnn_loader *iter, bcnn_net *net,
                                     int idx) {
    unsigned char l;
    if (net->mode == BCNN_MODE_TRAIN) {
        int rand_skip = /*(int)(10.0f * (float)rand() / RAND_MAX) + 1*/ 0;
        for (int i = 0; i < rand_skip; ++i) {
            if (fread((char *)&l, 1, sizeof(char), iter->f_current) == 0) {
                rewind(iter->f_current);
            } else {
                fseek(iter->f_current, -1, SEEK_CUR);
            }
            fseek(
                iter->f_current,
                iter->input_width * iter->input_height * iter->input_depth + 1,
                SEEK_CUR);
        }
    }
    if (fread((char *)&l, 1, sizeof(char), iter->f_current) == 0) {
        rewind(iter->f_current);
    } else {
        fseek(iter->f_current, -1, SEEK_CUR);
    }

    // Read label
    size_t n = fread((char *)&l, 1, sizeof(char), iter->f_current);
    int class_label = (int)l;
    // Read img
    char tmp[3072];
    n = fread(tmp, 1,
              iter->input_width * iter->input_height * iter->input_depth,
              iter->f_current);
    // Swap depth <-> spatial dim arrangement
    for (int k = 0; k < iter->input_depth; ++k) {
        for (int y = 0; y < iter->input_height; ++y) {
            for (int x = 0; x < iter->input_width; ++x) {
                iter->input_uchar[(x + iter->input_width * y) *
                                      iter->input_depth +
                                  k] =
                    tmp[iter->input_width * (iter->input_height * k + y) + x];
            }
        }
    }
    /*bip_write_image("test00.png", iter->input_uchar, iter->input_width,
       iter->input_height, iter->input_depth,
        iter->input_width * iter->input_depth);*/

    // Data augmentation
    if (net->mode == BCNN_MODE_TRAIN) {
        int use_buffer_img = (net->data_aug->range_shift_x != 0 ||
                              net->data_aug->range_shift_y != 0 ||
                              net->data_aug->rotation_range != 0 ||
                              net->data_aug->random_fliph != 0);
        unsigned char *img_tmp = NULL;
        if (use_buffer_img) {
            int sz_img =
                iter->input_width * iter->input_height * iter->input_depth;
            img_tmp = (unsigned char *)calloc(sz_img, sizeof(unsigned char));
        }
        bcnn_data_augmentation(iter->input_uchar, iter->input_width,
                               iter->input_height, iter->input_depth,
                               net->data_aug, img_tmp);
        bh_free(img_tmp);
    }
    // bip_write_image("test.png", iter->input_uchar, iter->input_width,
    // iter->input_height, iter->input_depth, iter->input_width *
    // iter->input_depth);
    // Fill input tensor
    int input_sz = bcnn_tensor_size3d(&net->tensors[0]);
    float *x = net->tensors[0].data + idx * input_sz;
    if (net->tensors[0].w < iter->input_width ||
        net->tensors[0].h < iter->input_height) {
        bip_crop_image(iter->input_uchar, iter->input_width, iter->input_height,
                       iter->input_width * iter->input_depth,
                       (iter->input_width - net->tensors[0].w) / 2,
                       (iter->input_height - net->tensors[0].h) / 2,
                       iter->input_net, net->tensors[0].w, net->tensors[0].h,
                       net->tensors[0].w * net->tensors[0].c,
                       net->tensors[0].c);
        // Map [0;255] uint8 values to [-1;1] float values
        bcnn_convert_img_to_float(iter->input_net, net->tensors[0].w,
                                  net->tensors[0].h, net->tensors[0].c,
                                  1 / 127.5f, net->data_aug->swap_to_bgr,
                                  127.5f, 127.5f, 127.5f, x);
    } else {
        // Map [0;255] uint8 values to [-1;1] float values
        bcnn_convert_img_to_float(iter->input_uchar, net->tensors[0].w,
                                  net->tensors[0].h, net->tensors[0].c,
                                  1 / 127.5f, net->data_aug->swap_to_bgr,
                                  127.5f, 127.5f, 127.5f, x);
    }
    // bip_write_image("test1.png", tmp_buf, net->tensors[0].w,
    // net->tensors[0].h, net->tensors[0].c, net->tensors[0].w *
    // net->tensors[0].c);
    if (net->mode != BCNN_MODE_PREDICT) {
        int label_sz = bcnn_tensor_size3d(&net->tensors[1]);
        float *y = net->tensors[1].data + idx * label_sz;
        memset(y, 0, label_sz * sizeof(float));
        // Load truth
        y[class_label] = 1;
    }

    return BCNN_SUCCESS;
}