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

#include "bcnn_mnist_loader.h"

#include <bh/bh_macros.h>
#include <bh/bh_string.h>
/* include bip image processing lib */
#include <bip/bip.h>
#include "bcnn_data.h"
#include "bcnn_utils.h"

static uint32_t read_uint32(char *v) {
    uint32_t ret = 0;
    for (int i = 0; i < 4; ++i) {
        ret <<= 8;
        ret |= (uint8_t)v[i];
    }
    return ret;
}

bcnn_status bcnn_loader_mnist_init(bcnn_loader *iter, bcnn_net *net,
                                   const char *path_img,
                                   const char *path_label) {
    FILE *f_img = NULL, *f_label = NULL;
    f_img = fopen(path_img, "rb");
    BCNN_CHECK_AND_LOG(net->log_ctx, f_img, BCNN_INVALID_PARAMETER,
                       "Cound not open file %s", path_img);
    f_label = fopen(path_label, "rb");
    BCNN_CHECK_AND_LOG(net->log_ctx, f_label, BCNN_INVALID_PARAMETER,
                       "Cound not open file %s", path_label);

    iter->f_input = f_img;
    iter->f_label = f_label;
    // Read header
    char tmp[16] = {0};
    size_t nr = fread(tmp, 1, 16, iter->f_input);
    uint32_t num_img = read_uint32(tmp + 4);
    iter->input_height = read_uint32(tmp + 8);
    iter->input_width = read_uint32(tmp + 12);
    iter->input_depth = 1;
    nr = fread(tmp, 1, 8, iter->f_label);
    uint32_t num_labels = read_uint32(tmp + 4);
    BCNN_CHECK_AND_LOG(net->log_ctx, num_img == num_labels, BCNN_INVALID_DATA,
                       "Inconsistent MNIST data: number of images and labels "
                       "must be the same");
    iter->input_uchar = (unsigned char *)calloc(
        iter->input_width * iter->input_height, sizeof(unsigned char));
    BCNN_CHECK_AND_LOG(
        net->log_ctx,
        net->tensors[0].w > 0 && net->tensors[0].h > 0 && net->tensors[0].c > 0,
        BCNN_INVALID_PARAMETER,
        "Input's width, height and channels must be > 0");
    iter->input_net = (uint8_t *)calloc(
        net->tensors[0].w * net->tensors[0].h * net->tensors[0].c,
        sizeof(uint8_t));
    rewind(iter->f_input);
    rewind(iter->f_label);

    return BCNN_SUCCESS;
}

void bcnn_loader_mnist_terminate(bcnn_loader *iter) {
    if (iter->f_input != NULL) {
        fclose(iter->f_input);
    }
    if (iter->f_label != NULL) {
        fclose(iter->f_label);
    }
    bh_free(iter->input_uchar);
    bh_free(iter->input_net);
}

bcnn_status bcnn_loader_mnist_next(bcnn_loader *iter, bcnn_net *net, int idx) {
    unsigned char l;
    if (fread((char *)&l, 1, sizeof(char), iter->f_input) == 0) {
        rewind(iter->f_input);
    } else {
        fseek(iter->f_input, -1, SEEK_CUR);
    }
    if (fread((char *)&l, 1, sizeof(char), iter->f_label) == 0) {
        rewind(iter->f_label);
    } else {
        fseek(iter->f_label, -1, SEEK_CUR);
    }

    if (ftell(iter->f_input) == 0 && ftell(iter->f_label) == 0) {
        char tmp[16];
        size_t n = fread(tmp, 1, 16, iter->f_input);
        unsigned int num_img = read_uint32(tmp + 4);
        iter->input_height = read_uint32(tmp + 8);
        iter->input_width = read_uint32(tmp + 12);
        n = fread(tmp, 1, 8, iter->f_label);
        unsigned int num_labels = read_uint32(tmp + 4);
        BCNN_CHECK_AND_LOG(net->log_ctx, (num_img == num_labels),
                           BCNN_INVALID_DATA,
                           "MNIST data: number of images and labels must be "
                           "the same. Found %d images and %d labels",
                           num_img, num_labels);
        BCNN_CHECK_AND_LOG(net->log_ctx,
                           (net->tensors[0].h == iter->input_height &&
                            net->tensors[0].w == iter->input_width),
                           BCNN_INVALID_DATA,
                           "MNIST data: incoherent image width and height");
        iter->n_samples = num_img;
    }

    // Read label
    size_t n = fread((char *)&l, 1, sizeof(char), iter->f_label);
    int class_label = (int)l;
    // Read img
    n = fread(iter->input_uchar, 1, iter->input_width * iter->input_height,
              iter->f_input);

    // Data augmentation
    if (net->mode == BCNN_MODE_TRAIN) {
        int use_buffer_img = (net->data_aug.range_shift_x != 0 ||
                              net->data_aug.range_shift_y != 0 ||
                              net->data_aug.rotation_range != 0 ||
                              net->data_aug.random_fliph != 0);
        unsigned char *img_tmp = NULL;
        if (use_buffer_img) {
            int sz_img =
                iter->input_width * iter->input_height * iter->input_depth;
            img_tmp = (unsigned char *)calloc(sz_img, sizeof(unsigned char));
        }
        bcnn_data_augmentation(iter->input_uchar, iter->input_width,
                               iter->input_height, iter->input_depth,
                               &net->data_aug, img_tmp);
        if (use_buffer_img) {
            bh_free(img_tmp);
        }
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
                                  1 / 127.5f, net->data_aug.swap_to_bgr, 127.5f,
                                  127.5f, 127.5f, x);
    } else {
        // Map [0;255] uint8 values to [-1;1] float values
        bcnn_convert_img_to_float(iter->input_uchar, net->tensors[0].w,
                                  net->tensors[0].h, net->tensors[0].c,
                                  1 / 127.5f, net->data_aug.swap_to_bgr, 127.5f,
                                  127.5f, 127.5f, x);
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