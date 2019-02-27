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

#include "bcnn_detection_loader.h"

#include <bh/bh_macros.h>
#include <bh/bh_string.h>
/* include bip image processing lib */
#include <bip/bip.h>
#include "bcnn_tensor.h"
#include "bcnn_utils.h"

bcnn_status bcnn_loader_list_detection_init(bcnn_loader *iter, bcnn_net *net,
                                            const char *train_path,
                                            const char *train_path_extra,
                                            const char *test_path,
                                            const char *test_path_extra) {
    // Open the files handles according to each dataset path
    BCNN_CHECK_STATUS(bcnn_open_dataset(iter, net, train_path, train_path_extra,
                                        test_path, test_path_extra, false));
    // Allocate img buffer
    iter->input_uchar = (unsigned char *)calloc(
        net->tensors[0].w * net->tensors[0].h * net->tensors[0].c,
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

void bcnn_loader_list_detection_terminate(bcnn_loader *iter) {
    if (iter->f_train != NULL) {
        fclose(iter->f_train);
    }
    if (iter->f_test != NULL) {
        fclose(iter->f_test);
    }
    bh_free(iter->input_uchar);
    bh_free(iter->input_net);
}

bcnn_status bcnn_loader_list_detection_next(bcnn_loader *iter, bcnn_net *net,
                                            int idx) {
    char *line = NULL;
    char **tok = NULL;
    line = bh_fgetline(iter->f_current);
    if (line == NULL) {
        rewind(iter->f_current);
        line = bh_fgetline(iter->f_current);
    }
    int num_toks = bh_strsplit(line, ' ', &tok);
    if (((num_toks - 1) % 5 != 0)) {
        bcnn_log(net->log_ctx, BCNN_LOG_WARNING,
                 "Wrong data format for detection %s. Found %d labels but "
                 "expected multiple of 5",
                 line);
        BCNN_PARSE_CLEANUP(line, tok, num_toks);
        return BCNN_INVALID_DATA;
    }
    int w_img = 0, h_img = 0, c_img = 0, x_ul = 0, y_ul = 0;
    unsigned char *pimg = NULL;
    bip_load_image(tok[0], &pimg, &w_img, &h_img, &c_img);
    if (!(w_img > 0 && h_img > 0 && pimg)) {
        bcnn_log(net->log_ctx, BCNN_LOG_WARNING, "Skip invalid image %s",
                 tok[0]);
        bh_free(pimg);
        BCNN_PARSE_CLEANUP(line, tok, num_toks);
        return BCNN_INVALID_DATA;
    }
    if (net->tensors[0].c != c_img) {
        bcnn_log(net->log_ctx, BCNN_LOG_WARNING,
                 "Skip image %s: unexpected number of channels");
        bh_free(pimg);
        BCNN_PARSE_CLEANUP(line, tok, num_toks);
        return BCNN_INVALID_DATA;
    }

    float wh_ratio = w_img / h_img;
    int nh, nw;
    if (wh_ratio < 1) {
        nh = net->tensors[0].h;
        nw = nh * wh_ratio;
    } else {
        nw = net->tensors[0].w;
        nh = nw / wh_ratio;
    }
    unsigned char *buf =
        (unsigned char *)calloc(nw * nh * c_img, sizeof(unsigned char));
    bip_resize_bilinear(pimg, w_img, h_img, w_img * c_img, buf, nw, nh,
                        nw * c_img, c_img);
    int dx, dy;  // Canvas offsets
    if (net->mode == BCNN_MODE_TRAIN) {
        dx = rand_between(0, (net->tensors[0].w - nw));
        dy = rand_between(0, (net->tensors[0].h - nh));
    } else {
        dx = (net->tensors[0].w - nw) / 2;
        dy = (net->tensors[0].h - nh) / 2;
    }
    memset(iter->input_net, 128,
           net->tensors[0].c * net->tensors[0].w * net->tensors[0].h);
    bip_crop_image(buf, nw, nh, nw * c_img, -dx, -dy, iter->input_net,
                   net->tensors[0].w, net->tensors[0].h,
                   net->tensors[0].w * net->tensors[0].c, net->tensors[0].c);
    bh_free(buf);
    if (net->mode == BCNN_MODE_TRAIN) {
        // TODO: only brightness / contrast / flip is currently supported for
        // detection
        net->data_aug->apply_fliph = 0;
        if (net->data_aug->random_fliph) {
            net->data_aug->apply_fliph = ((float)rand() / RAND_MAX > 0.5f);
        }
        bcnn_data_augmentation(iter->input_net, net->tensors[0].w,
                               net->tensors[0].h, net->tensors[0].c,
                               net->data_aug, iter->input_uchar);
    }
    memcpy(iter->input_uchar, iter->input_net,
           net->tensors[0].c * net->tensors[0].w * net->tensors[0].h);
    // Fill input tensor
    int input_sz = bcnn_tensor_size3d(&net->tensors[0]);
    float *x = net->tensors[0].data + idx * input_sz;
    // Map [0;255] uint8 values to [-1;1] float values
    bcnn_convert_img_to_float(iter->input_uchar, net->tensors[0].w,
                              net->tensors[0].h, net->tensors[0].c, 1 / 127.5f,
                              net->data_aug->swap_to_bgr, 127.5f, 127.5f,
                              127.5f, x);
    if (net->mode != BCNN_MODE_PREDICT) {
        // Fill labels
        int label_sz = bcnn_tensor_size3d(&net->tensors[1]);
        float *y = net->tensors[1].data + idx * label_sz;
        memset(y, 0, label_sz * sizeof(float));
        int num_boxes = (num_toks - 1) / 5;
        if (num_boxes > BCNN_DETECTION_MAX_BOXES) {
            num_boxes = BCNN_DETECTION_MAX_BOXES;
        }
        int offset = 1;  // Offset the image path
        float scale_x = (float)nw / (float)net->tensors[0].w;
        float scale_y = (float)nh / (float)net->tensors[0].h;
        float scale_dx = (float)dx / (float)net->tensors[0].w;
        float scale_dy = (float)dy / (float)net->tensors[0].h;
        for (int i = 0; i < num_boxes; ++i) {
            // We encode box center (x, y) and w, h
            y[i * 5 + 0] =
                (float)atof(tok[5 * i + 1 + offset]) * scale_x + scale_dx;
            y[i * 5 + 1] =
                (float)atof(tok[5 * i + 2 + offset]) * scale_y + scale_dy;
            y[i * 5 + 2] = (float)atof(tok[5 * i + 3 + offset]) * scale_x;
            y[i * 5 + 3] = (float)atof(tok[5 * i + 4 + offset]) * scale_y;
            if (net->data_aug->apply_fliph) {
                y[i * 5 + 0] = 1.0f - y[i * 5 + 0];
            }
            // Class
            y[i * 5 + 4] = (float)atoi(tok[5 * i + offset]);
        }
    }
    bh_free(pimg);
    BCNN_PARSE_CLEANUP(line, tok, num_toks);
    return BCNN_SUCCESS;
}