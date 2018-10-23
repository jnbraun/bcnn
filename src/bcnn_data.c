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
#include <bh/bh_log.h>
#include <bh/bh_macros.h>
#include <bh/bh_string.h>
#include "bcnn/bcnn.h"

/* include bip image processing lib */
#include <bip/bip.h>

int bcnn_convert_img_to_float(unsigned char *src, int w, int h, int c,
                              int no_input_norm, int swap_to_bgr, float mean_r,
                              float mean_g, float mean_b, float *dst) {
    int x, y, k;
    float m = 0.0f;
    float sn = 1.0f, sd = 1.0f;

    if (!no_input_norm) {
        sn = 2.0f;
        sd = 1 / 255.0f;
    }
    if (swap_to_bgr) {
        for (k = 0; k < c; ++k) {
            switch (k) {
                case 0:
                    m = mean_r;
                    break;
                case 1:
                    m = mean_g;
                    break;
                case 2:
                    m = mean_b;
                    break;
            }
            for (y = 0; y < h; ++y) {
                for (x = 0; x < w; ++x) {
                    dst[w * (h * (2 - k) + y) + x] =
                        ((float)src[c * (x + w * y) + k] * sd - m) * sn;
                }
            }
        }
    } else {
        for (k = 0; k < c; ++k) {
            for (y = 0; y < h; ++y) {
                for (x = 0; x < w; ++x) {
                    dst[w * (h * k + y) + x] =
                        ((float)src[c * (x + w * y) + k] * sd - 0.5f) * sn;
                }
            }
        }
    }
    return 0;
}

void bcnn_convert_img_to_float2(unsigned char *src, int w, int h, int c,
                                float norm_coeff, int swap_to_bgr, float mean_r,
                                float mean_g, float mean_b, float *dst) {
    float m[3] = {mean_r, mean_g, mean_b};
    if (swap_to_bgr) {
        if (c != 3) {
            bh_log(BCNN_LOG_ERROR,
                   "bcnn_convert_img_to_float2: number of channels %d is "
                   "inconsistent. Expected 3",
                   c);
            return;
        }
        for (int k = 0; k < c; ++k) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    dst[w * (h * (2 - k) + y) + x] =
                        ((float)src[c * (x + w * y) + k] - m[k]) * norm_coeff;
                }
            }
        }
    } else {
        for (int k = 0; k < c; ++k) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    dst[w * (h * k + y) + x] =
                        ((float)src[c * (x + w * y) + k] - m[k]) * norm_coeff;
                }
            }
        }
    }
}

/* IO */
int bcnn_load_image_from_csv(bcnn_net *net, char *str, int w, int h, int c,
                             unsigned char **img) {
    int i, n_tok, sz = w * h * c;
    char **tok = NULL;
    unsigned char *ptr_img = NULL;

    n_tok = bh_strsplit(str, ',', &tok);

    BCNN_CHECK_AND_LOG(net->log_ctx, (n_tok == sz), BCNN_INVALID_DATA,
                       "Incorrect data size in csv");

    ptr_img = (unsigned char *)calloc(sz, sizeof(unsigned char));
    for (i = 0; i < n_tok; ++i) {
        ptr_img[i] = (unsigned char)atoi(tok[i]);
    }
    *img = ptr_img;

    for (i = 0; i < n_tok; ++i) bh_free(tok[i]);
    bh_free(tok);

    return BCNN_SUCCESS;
}

/* Load image from disk, performs crop to fit the required size if needed and
 * copy in pre-allocated memory */
int bcnn_load_image_from_path(bcnn_net *net, char *path, int w, int h, int c,
                              unsigned char *img, int state, int *x_shift,
                              int *y_shift) {
    int w_img, h_img, c_img, x_ul = 0, y_ul = 0;
    unsigned char *buf = NULL, *pimg = NULL;

    bip_load_image(path, &buf, &w_img, &h_img, &c_img);
    BCNN_CHECK_AND_LOG(net->log_ctx, (w_img > 0 && h_img > 0 && buf),
                       BCNN_INVALID_DATA, "Invalid image %s", path);
    if (c != c_img) {
        bcnn_log(net->log_ctx, BCNN_LOG_ERROR,
                 "Unexpected number of channels of image %s\n", path);
        bh_free(buf);
        return BCNN_INVALID_DATA;
    }

    if (w_img != w || h_img != h) {
        if (state == 0) {  // state predict, always center crop
            x_ul = (w_img - w) / 2;
            y_ul = (h_img - h) / 2;
        } else {  // state train, random crop
            x_ul = (int)((float)rand() / RAND_MAX * (w_img - w));
            y_ul = (int)((float)rand() / RAND_MAX * (h_img - h));
        }
        pimg = (unsigned char *)calloc(w * h * c, sizeof(unsigned char));
        bip_crop_image(buf, w_img, h_img, w_img * c_img, x_ul, y_ul, pimg, w, h,
                       w * c, c);
        memcpy(img, pimg, w * h * c);
        bh_free(pimg);
    } else {
        memcpy(img, buf, w * h * c);
    }
    bh_free(buf);
    *x_shift = x_ul;
    *y_shift = y_ul;

    return BCNN_SUCCESS;
}

int bcnn_load_image_from_memory(bcnn_net *net, unsigned char *buffer,
                                int buffer_size, int w, int h, int c,
                                unsigned char **img, int state, int *x_shift,
                                int *y_shift) {
    int w_img, h_img, c_img, x_ul = 0, y_ul = 0;
    unsigned char *tmp = NULL, *pimg = NULL;

    BIP_CHECK_STATUS(bip_load_image_from_memory(buffer, buffer_size, &tmp,
                                                &w_img, &h_img, &c_img));
    BCNN_CHECK_AND_LOG(net->log_ctx, (w_img > 0 && h_img > 0 && buffer),
                       BCNN_INVALID_DATA, "Invalid image");
    if (c != c_img) {
        bcnn_log(net->log_ctx, BCNN_LOG_ERROR,
                 "Unexpected number of channels of image\n");
        bh_free(tmp);
        return BCNN_INVALID_DATA;
    }

    if (w_img != w || h_img != h) {
        if (state == 0) {  // state predict, always center crop
            x_ul = (w_img - w) / 2;
            y_ul = (h_img - h) / 2;
        } else {  // state train, random crop
            x_ul = (int)((float)rand() / RAND_MAX * (w_img - w));
            y_ul = (int)((float)rand() / RAND_MAX * (h_img - h));
        }
        pimg = (unsigned char *)calloc(w * h * c, sizeof(unsigned char));
        bip_crop_image(tmp, w_img, h_img, w_img * c_img, x_ul, y_ul, pimg, w, h,
                       w * c, c);
        memcpy(*img, pimg, w * h * c);
        bh_free(pimg);
    } else {
        memcpy(*img, tmp, w * h * c);
    }
    bh_free(tmp);
    *x_shift = x_ul;
    *y_shift = y_ul;

    return BCNN_SUCCESS;
}

/* Mnist iter */
static unsigned int _read_int(char *v) {
    int i;
    unsigned int ret = 0;

    for (i = 0; i < 4; ++i) {
        ret <<= 8;
        ret |= (unsigned char)v[i];
    }

    return ret;
}

static int bcnn_mnist_next_iter(bcnn_net *net, bcnn_iterator *iter) {
    char tmp[16];
    unsigned char l;
    unsigned int n_img = 0, n_labels = 0;
    size_t n = 0;

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
        n = fread(tmp, 1, 16, iter->f_input);
        n_img = _read_int(tmp + 4);
        iter->input_height = _read_int(tmp + 8);
        iter->input_width = _read_int(tmp + 12);
        n = fread(tmp, 1, 8, iter->f_label);
        n_labels = _read_int(tmp + 4);
        BCNN_CHECK_AND_LOG(net->log_ctx, (n_img == n_labels), BCNN_INVALID_DATA,
                           "MNIST data: number of images and labels must be "
                           "the same. Found %d images and %d labels",
                           n_img, n_labels);
        BCNN_CHECK_AND_LOG(net->log_ctx,
                           (net->input_height == iter->input_height &&
                            net->input_width == iter->input_width),
                           BCNN_INVALID_DATA,
                           "MNIST data: incoherent image width and height");
        iter->n_samples = n_img;
    }

    // Read label
    n = fread((char *)&l, 1, sizeof(char), iter->f_label);
    iter->label_int[0] = (int)l;
    // Read img
    n = fread(iter->input_uchar, 1, iter->input_width * iter->input_height,
              iter->f_input);

    return BCNN_SUCCESS;
}

static int bcnn_init_bin_iterator(bcnn_net *net, bcnn_iterator *iter,
                                  char *path_input) {
    FILE *f_bin = NULL, *f_lst = NULL;
    char *line = NULL;
    bcnn_label_type type;
    int nr = 0;

    iter->type = ITER_BIN;

    f_lst = fopen(path_input, "rt");
    BCNN_CHECK_AND_LOG(net->log_ctx, (f_lst != NULL), BCNN_INVALID_PARAMETER,
                       "Can not open file %s", path_input);
    // Open first binary file
    line = bh_fgetline(f_lst);
    BCNN_CHECK_AND_LOG(net->log_ctx, (line != NULL), BCNN_INVALID_DATA,
                       "Empty data list");

    // bh_strstrip(line);
    f_bin = fopen(line, "rb");
    BCNN_CHECK_AND_LOG(net->log_ctx, (f_bin != NULL), BCNN_INVALID_PARAMETER,
                       "Can not open file %s", line);

    nr = fread(&iter->n_samples, 1, sizeof(int), f_bin);
    nr = fread(&iter->label_width, 1, sizeof(int), f_bin);
    nr = fread(&type, 1, sizeof(int), f_bin);
    iter->input_width = net->input_width;
    iter->input_height = net->input_height;
    iter->input_depth = net->input_channels;
    iter->input_uchar = (unsigned char *)calloc(
        iter->input_width * iter->input_height * iter->input_depth,
        sizeof(unsigned char));
    iter->label_float = (float *)calloc(iter->label_width, sizeof(float));

    iter->f_input = f_bin;
    iter->f_list = f_lst;

    bh_free(line);

    return BCNN_SUCCESS;
}

static int bcnn_bin_iter(bcnn_net *net, bcnn_iterator *iter) {
    unsigned char l;
    size_t n = 0, nr = 0;
    int i, buf_sz = 0, label_width, type;
    float lf;
    unsigned char *buf = NULL;
    char *line = NULL;

    if (fread((char *)&l, 1, sizeof(char), iter->f_input) == 0) {
        // Jump to next binary part file
        fclose(iter->f_input);
        line = bh_fgetline(iter->f_list);
        if (line == NULL) {
            rewind(iter->f_list);
            line = bh_fgetline(iter->f_list);
        }
        iter->f_input = fopen(line, "rb");
        if (iter->f_input == NULL) {
            fprintf(stderr, "[ERROR] Can not open file %s\n", line);
            return BCNN_INVALID_PARAMETER;
        }
        bh_free(line);
    } else {
        fseek(iter->f_input, -1, SEEK_CUR);
    }

    if (ftell(iter->f_input) == 0) {
        nr = fread(&n, 1, sizeof(int), iter->f_input);
        nr = fread(&label_width, 1, sizeof(int), iter->f_input);
        nr = fread(&type, 1, sizeof(int), iter->f_input);
    }

    // Read image
    nr = fread(&buf_sz, 1, sizeof(int), iter->f_input);
    buf = (unsigned char *)calloc(buf_sz, sizeof(unsigned char));
    nr = fread(buf, 1, buf_sz, iter->f_input);
    bcnn_load_image_from_memory(net, buf, buf_sz, net->input_width,
                                net->input_height, net->input_channels,
                                &iter->input_uchar, net->state,
                                &net->data_aug.shift_x, &net->data_aug.shift_y);
    bh_free(buf);

    // Read label
    for (i = 0; i < iter->label_width; ++i) {
        nr = fread(&lf, 1, sizeof(float), iter->f_input);
        iter->label_float[i] = lf;
    }

    return BCNN_SUCCESS;
}

/* Handles cifar10 binary format */
static int bcnn_init_cifar10_iterator(bcnn_net *net, bcnn_iterator *iter,
                                      char *path_input) {
    FILE *f_bin = NULL;

    iter->type = ITER_CIFAR10;

    f_bin = fopen(path_input, "rb");
    if (f_bin == NULL) {
        fprintf(stderr, "[ERROR] Can not open file %s\n", path_input);
        return BCNN_INVALID_PARAMETER;
    }

    iter->n_samples = 0;  // not used
    iter->label_width = 1;

    iter->label_int = (int *)calloc(1, sizeof(int));
    iter->input_width = 32;
    iter->input_height = 32;
    iter->input_depth = 3;
    iter->input_uchar = (unsigned char *)calloc(
        iter->input_width * iter->input_height * iter->input_depth,
        sizeof(unsigned char));
    iter->f_input = f_bin;

    return BCNN_SUCCESS;
}

static int bcnn_cifar10_iter(bcnn_net *net, bcnn_iterator *iter) {
    unsigned char l;
    unsigned int n_img = 0, n_labels = 0;
    size_t n = 0;
    int x, y, k, i,
        rand_skip = /*(int)(10.0f * (float)rand() / RAND_MAX) + 1*/ 0;
    char tmp[3072];

    if (net->state == TRAIN) {
        for (i = 0; i < rand_skip; ++i) {
            if (fread((char *)&l, 1, sizeof(char), iter->f_input) == 0) {
                rewind(iter->f_input);
            } else {
                fseek(iter->f_input, -1, SEEK_CUR);
            }
            fseek(
                iter->f_input,
                iter->input_width * iter->input_height * iter->input_depth + 1,
                SEEK_CUR);
        }
    }
    if (fread((char *)&l, 1, sizeof(char), iter->f_input) == 0) {
        rewind(iter->f_input);
    } else {
        fseek(iter->f_input, -1, SEEK_CUR);
    }

    // Read label
    n = fread((char *)&l, 1, sizeof(char), iter->f_input);
    iter->label_int[0] = (int)l;
    // Read img
    n = fread(tmp, 1,
              iter->input_width * iter->input_height * iter->input_depth,
              iter->f_input);
    // Swap depth <-> spatial dim arrangement
    for (k = 0; k < iter->input_depth; ++k) {
        for (y = 0; y < iter->input_height; ++y) {
            for (x = 0; x < iter->input_width; ++x) {
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

    return BCNN_SUCCESS;
}

static int bcnn_init_list_iterator(bcnn_net *net, bcnn_iterator *iter,
                                   char *path_input) {
    int i;
    FILE *f_list = NULL;
    char *line = NULL;
    char **tok = NULL;
    int n_tok = 0;
    unsigned char *img = NULL;
    int out_w = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].w;
    int out_h = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].h;
    int out_c = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].c;

    iter->type = ITER_LIST;

    f_list = fopen(path_input, "rb");
    if (f_list == NULL) {
        fprintf(stderr, "[ERROR] Can not open file %s\n", path_input);
        return BCNN_INVALID_PARAMETER;
    }

    iter->input_width = net->input_width;
    iter->input_height = net->input_width;
    iter->input_depth = net->input_channels;
    iter->input_uchar = (unsigned char *)calloc(
        iter->input_width * iter->input_height * iter->input_depth,
        sizeof(unsigned char));
    line = bh_fgetline(f_list);
    n_tok = bh_strsplit(line, ' ', &tok);
    if (net->prediction_type == CLASSIFICATION ||
        net->prediction_type == REGRESSION ||
        net->prediction_type == HEATMAP_REGRESSION) {
        iter->label_width = n_tok - 1;
    } else if (net->prediction_type == DETECTION) {
        iter->label_width =
            5 * BCNN_DETECTION_MAX_BOXES;  // 4 coords + class id
    } else if (net->prediction_type == SEGMENTATION) {
        iter->label_width = out_w * out_h * out_c;
    }
    iter->label_float = (float *)calloc(iter->label_width, sizeof(float));

    rewind(f_list);
    iter->f_input = f_list;
    bh_free(line);
    bh_free(img);
    for (i = 0; i < n_tok; ++i) bh_free(tok[i]);
    bh_free(tok);

    return BCNN_SUCCESS;
}

static float bcnn_rand_between(float min, float max) {
    return ((float)rand() / RAND_MAX * (max - min)) + min;
}

bcnn_status bcnn_data_iter_detection(bcnn_net *net, bcnn_iterator *iter) {
    char *line = NULL;
    char **tok = NULL;
    line = bh_fgetline(iter->f_input);
    if (line == NULL) {
        rewind(iter->f_input);
        line = bh_fgetline(iter->f_input);
    }
    int n_tok = bh_strsplit(line, ' ', &tok);
    if (((n_tok - 1) % 5 != 0)) {
        bcnn_log(net->log_ctx, BCNN_LOG_WARNING,
                 "Wrong data format for detection %s. Found %d labels but "
                 "expected multiple of 5",
                 line);
        return BCNN_INVALID_DATA;
    }
    int w_img = 0, h_img = 0, c_img = 0, x_ul = 0, y_ul = 0;
    unsigned char *pimg = NULL;
    bip_load_image(tok[0], &pimg, &w_img, &h_img, &c_img);
    if (!(w_img > 0 && h_img > 0 && pimg)) {
        bcnn_log(net->log_ctx, BCNN_LOG_WARNING, "Skip invalid image %s",
                 tok[0]);
        bh_free(pimg);
        bh_free(line);
        for (int i = 0; i < n_tok; ++i) {
            bh_free(tok[i]);
        }
        bh_free(tok);
        return BCNN_INVALID_DATA;
    }
    if (net->input_channels != c_img) {
        bcnn_log(net->log_ctx, BCNN_LOG_WARNING,
                 "Skip image %s: unexpected number of channels");
        bh_free(pimg);
        bh_free(line);
        for (int i = 0; i < n_tok; ++i) {
            bh_free(tok[i]);
        }
        bh_free(tok);
        return BCNN_INVALID_DATA;
    }

    float wh_ratio = w_img / h_img;
    int nh, nw;
    if (wh_ratio < 1) {
        nh = net->input_height;
        nw = nh * wh_ratio;
    } else {
        nw = net->input_width;
        nh = nw / wh_ratio;
    }
    unsigned char *buf =
        (unsigned char *)calloc(nw * nh * c_img, sizeof(unsigned char));
    bip_resize_bilinear(pimg, w_img, h_img, w_img * c_img, buf, nw, nh,
                        nw * c_img, c_img);
    int dx, dy;  // Canvas offsets
    if (net->task == TRAIN && net->state) {
        dx = (int)bcnn_rand_between(0.f, (float)(net->input_width - nw));
        dy = (int)bcnn_rand_between(0.f, (float)(net->input_height - nh));
    } else {
        dx = (net->input_width - nw) / 2;
        dy = (net->input_height - nh) / 2;
    }
    memset(net->input_buffer, 128,
           net->input_channels * net->input_width * net->input_height);
    bip_crop_image(buf, nw, nh, nw * c_img, -dx, -dy, net->input_buffer,
                   net->input_width, net->input_height,
                   net->input_width * net->input_channels, net->input_channels);
    bh_free(buf);
    if (net->task == TRAIN && net->state) {
        // TODO: only brightness / contrast / flip is currently supported for
        // detection
        net->data_aug.apply_fliph = 0;
        if (net->data_aug.random_fliph) {
            net->data_aug.apply_fliph = ((float)rand() / RAND_MAX > 0.5f);
        }
        bcnn_data_augmentation(net->input_buffer, net->input_width,
                               net->input_height, net->input_channels,
                               &net->data_aug, iter->input_uchar);
    }
    if (net->task != PREDICT) {
        // Fill labels
        memset(iter->label_float, 0, iter->label_width * sizeof(float));
        int num_boxes = (n_tok - 1) / 5;
        if (num_boxes > BCNN_DETECTION_MAX_BOXES) {
            num_boxes = BCNN_DETECTION_MAX_BOXES;
        }
        int offset = 1;  // Offset the image path
        float scale_x = (float)nw / (float)net->input_width;
        float scale_y = (float)nh / (float)net->input_height;
        float scale_dx = (float)dx / (float)net->input_width;
        float scale_dy = (float)dy / (float)net->input_height;
        int skip = 0;
        for (int i = 0; i < num_boxes; ++i) {
            // We encode box center (x, y) and w, h
            iter->label_float[(i - skip) * 5 + 4] =
                (float)atoi(tok[5 * i + offset]);
            iter->label_float[(i - skip) * 5 + 0] =
                (float)atof(tok[5 * i + 1 + offset]) * scale_x + scale_dx;
            iter->label_float[(i - skip) * 5 + 1] =
                (float)atof(tok[5 * i + 2 + offset]) * scale_y + scale_dy;
            iter->label_float[(i - skip) * 5 + 2] =
                (float)atof(tok[5 * i + 3 + offset]) * scale_x;
            iter->label_float[(i - skip) * 5 + 3] =
                (float)atof(tok[5 * i + 4 + offset]) * scale_y;
            if (net->data_aug.apply_fliph) {
                iter->label_float[(i - skip) * 5 + 0] =
                    1.0f - iter->label_float[(i - skip) * 5 + 0];
            }
        }
    }
    bh_free(pimg);
    bh_free(line);
    for (int i = 0; i < n_tok; ++i) {
        bh_free(tok[i]);
    }
    bh_free(tok);

    return BCNN_SUCCESS;
}

static int bcnn_list_iter(bcnn_net *net, bcnn_iterator *iter) {
    char *line = NULL;
    char **tok = NULL;
    int i, n_tok = 0, tmp_x, tmp_y;
    int out_w = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].w;
    int out_h = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].h;
    int out_c = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].c;
    unsigned char *img = NULL;
    // nb_lines_skipped = (int)((float)rand() / RAND_MAX * net->batch_size);
    // bh_fskipline(f, nb_lines_skipped);
    line = bh_fgetline(iter->f_input);
    if (line == NULL) {
        rewind(iter->f_input);
        line = bh_fgetline(iter->f_input);
    }
    n_tok = bh_strsplit(line, ' ', &tok);
    if (net->task != PREDICT && net->prediction_type == CLASSIFICATION) {
        BCNN_CHECK_AND_LOG(net->log_ctx, n_tok == 2, BCNN_INVALID_DATA,
                           "Wrong data format for classification");
    }
    if (iter->type == ITER_LIST) {
        bcnn_load_image_from_path(
            net, tok[0], net->input_width, net->input_height,
            net->input_channels, iter->input_uchar, net->state,
            &net->data_aug.shift_x, &net->data_aug.shift_y);
    } else {
        bcnn_load_image_from_csv(net, tok[0], net->input_width,
                                 net->input_height, net->input_channels,
                                 &iter->input_uchar);
    }

    // Label
    if (net->prediction_type != SEGMENTATION) {
        BCNN_CHECK_AND_LOG(net->log_ctx, (n_tok == iter->label_width + 1),
                           BCNN_INVALID_DATA, "Unexpected label format");
        for (i = 0; i < iter->label_width; ++i) {
            iter->label_float[i] = (float)atof(tok[i + 1]);
        }
    } else {
        for (i = 0; i < iter->label_width; ++i) {
            if (iter->type == ITER_LIST) {
                bcnn_load_image_from_path(net, tok[i], out_w, out_h, out_c, img,
                                          net->state, &tmp_x, &tmp_y);
            } else {
                bcnn_load_image_from_csv(net, tok[i], out_w, out_h, out_c,
                                         &img);
            }
            bcnn_convert_img_to_float(img, out_w, out_h, out_c, 0, 0, 0, 0, 0,
                                      iter->label_float);
            bh_free(img);
        }
    }

    bh_free(line);
    for (i = 0; i < n_tok; ++i) {
        bh_free(tok[i]);
    }
    bh_free(tok);

    return BCNN_SUCCESS;
}

/* Data augmentation */
int bcnn_data_augmentation(unsigned char *img, int width, int height, int depth,
                           bcnn_data_augment *param, unsigned char *buffer) {
    int sz = width * height * depth;
    unsigned char *img_scale = NULL;
    int x_ul = 0, y_ul = 0, w_scale, h_scale;
    float scale = 1.0f, theta = 0.0f, contrast = 1.0f, kx, ky, distortion;
    int brightness = 0;

    if (param->random_fliph) {
        if (param->apply_fliph) {
            bip_fliph_image(img, width, height, depth, width * depth, buffer,
                            width * depth);
            memcpy(img, buffer, sz * sizeof(unsigned char));
        }
    }
    if (param->range_shift_x || param->range_shift_y) {
        memset(buffer, 128, sz);
        if (param->use_precomputed) {
            x_ul = param->shift_x;
            y_ul = param->shift_y;
        } else {
            x_ul = (int)((float)(rand() - RAND_MAX / 2) / RAND_MAX *
                         param->range_shift_x);
            y_ul = (int)((float)(rand() - RAND_MAX / 2) / RAND_MAX *
                         param->range_shift_y);
            param->shift_x = x_ul;
            param->shift_y = y_ul;
        }
        bip_crop_image(img, width, height, width * depth, x_ul, y_ul, buffer,
                       width, height, width * depth, depth);
        memcpy(img, buffer, sz * sizeof(unsigned char));
    }
    if (param->max_scale > 0.0f || param->min_scale > 0.0f) {
        if (param->use_precomputed) {
            scale = param->scale;
        } else {
            scale = (((float)rand() / RAND_MAX) *
                         (param->max_scale - param->min_scale) +
                     param->min_scale);
            param->scale = scale;
        }
        w_scale = (int)(width * scale);
        h_scale = (int)(height * scale);
        img_scale = (unsigned char *)calloc(w_scale * h_scale * depth,
                                            sizeof(unsigned char));
        bip_resize_bilinear(img, width, height, width * depth, img_scale,
                            w_scale, h_scale, w_scale * depth, depth);
        bip_crop_image(img_scale, w_scale, h_scale, w_scale * depth, x_ul, y_ul,
                       img, width, height, width * depth, depth);
        bh_free(img_scale);
    }
    if (param->rotation_range > 0.0f) {
        if (param->use_precomputed) {
            theta = param->rotation;
        } else {
            theta = bip_deg2rad((float)(rand() - RAND_MAX / 2) / RAND_MAX *
                                param->rotation_range);
            param->rotation = theta;
        }
        memset(buffer, 128, sz);
        bip_rotate_image(img, width, height, width * depth, buffer, width,
                         height, width * depth, depth, theta, width / 2,
                         height / 2, BILINEAR);
        memcpy(img, buffer, width * height * depth * sizeof(unsigned char));
    }
    if (param->min_contrast > 0.0f || param->max_contrast > 0.0f) {
        if (param->use_precomputed) {
            contrast = param->contrast;
        } else {
            contrast = (((float)rand() / RAND_MAX) *
                            (param->max_contrast - param->min_contrast) +
                        param->min_contrast);
            param->contrast = contrast;
        }
        bip_contrast_stretch(img, width * depth, width, height, depth, img,
                             width * depth, contrast);
    }
    if (param->min_brightness != 0 || param->max_brightness != 0) {
        if (param->use_precomputed) {
            brightness = param->brightness;
        } else {
            brightness =
                (int)(((float)rand() / RAND_MAX) *
                          (param->max_brightness - param->min_brightness) +
                      param->min_brightness);
            param->brightness = brightness;
        }
        bip_image_brightness(img, width * depth, width, height, depth, img,
                             width * depth, brightness);
    }
    if (param->max_distortion > 0.0f) {
        if (param->use_precomputed) {
            kx = param->distortion_kx;
            ky = param->distortion_ky;
            distortion = param->distortion;
        } else {
            kx = (((float)rand() - RAND_MAX / 2) / RAND_MAX);
            ky = (((float)rand() - RAND_MAX / 2) / RAND_MAX);
            distortion = ((float)rand() / RAND_MAX) * (param->max_distortion);
            param->distortion_kx = kx;
            param->distortion_ky = ky;
            param->distortion = distortion;
        }
        bip_image_perlin_distortion(img, width * depth, width, height, depth,
                                    buffer, width * depth, param->distortion,
                                    kx, ky);
        memcpy(img, buffer, width * height * depth * sizeof(unsigned char));
    }

    return BCNN_SUCCESS;
}

static int bcnn_init_mnist_iterator(bcnn_net *net, bcnn_iterator *iter,
                                    char *path_img, char *path_label) {
    FILE *f_img = NULL, *f_label = NULL;
    char tmp[16] = {0};
    int n_img = 0, n_lab = 0, nr = 0;

    iter->type = ITER_MNIST;
    f_img = fopen(path_img, "rb");
    BCNN_CHECK_AND_LOG(net->log_ctx, f_img, BCNN_INVALID_PARAMETER,
                       "Cound not open file %s", path_img);
    f_label = fopen(path_label, "rb");
    BCNN_CHECK_AND_LOG(net->log_ctx, f_label, BCNN_INVALID_PARAMETER,
                       "Cound not open file %s", path_label);

    iter->f_input = f_img;
    iter->f_label = f_label;
    iter->n_iter = 0;
    // Read header
    nr = fread(tmp, 1, 16, iter->f_input);
    n_img = _read_int(tmp + 4);
    iter->input_height = _read_int(tmp + 8);
    iter->input_width = _read_int(tmp + 12);
    iter->input_depth = 1;
    nr = fread(tmp, 1, 8, iter->f_label);
    n_lab = _read_int(tmp + 4);
    BCNN_CHECK_AND_LOG(net->log_ctx, n_img == n_lab, BCNN_INVALID_DATA,
                       "Inconsistent MNIST data: number of images and labels "
                       "must be the same");

    iter->input_uchar = (unsigned char *)calloc(
        iter->input_width * iter->input_height, sizeof(unsigned char));
    iter->label_int = (int *)calloc(1, sizeof(int));
    rewind(iter->f_input);
    rewind(iter->f_label);

    return BCNN_SUCCESS;
}

static int bcnn_init_multi_iterator(bcnn_net *net, bcnn_iterator *iter,
                                    char *path_input) {
    int i;
    FILE *f_list = NULL;
    char *line = NULL;
    char **tok = NULL;
    int n_tok = 0;
    unsigned char *img = NULL;
    int out_w = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].w;
    int out_h = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].h;
    int out_c = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].c;

    iter->type = ITER_MULTI;

    f_list = fopen(path_input, "rb");
    if (f_list == NULL) {
        fprintf(stderr, "[ERROR] Can not open file %s\n", path_input);
        return BCNN_INVALID_PARAMETER;
    }

    iter->input_width = net->input_width;
    iter->input_height = net->input_width;
    iter->input_depth = net->input_channels;
    iter->input_uchar = (unsigned char *)calloc(
        iter->input_width * iter->input_height * iter->input_depth,
        sizeof(unsigned char));
    iter->input_uchar2 = (unsigned char *)calloc(
        bcnn_tensor_size3d(&net->tensors[2]), sizeof(unsigned char));
    iter->input_uchar3 = (unsigned char *)calloc(
        bcnn_tensor_size3d(&net->tensors[3]), sizeof(unsigned char));
    iter->input_uchar4 = (unsigned char *)calloc(
        bcnn_tensor_size3d(&net->tensors[4]), sizeof(unsigned char));

    line = bh_fgetline(f_list);
    n_tok = bh_strsplit(line, ' ', &tok);

    iter->label_width = out_w * out_h * out_c;
    if (net->prediction_type == HEATMAP_REGRESSION) {
        iter->label_uchar =
            (unsigned char *)calloc(iter->label_width, sizeof(unsigned char));
    } else {
        iter->label_float = (float *)calloc(iter->label_width, sizeof(float));
    }

    rewind(f_list);
    iter->f_input = f_list;
    bh_free(line);
    bh_free(img);
    for (i = 0; i < n_tok; ++i) bh_free(tok[i]);
    bh_free(tok);
    return BCNN_SUCCESS;
}

static int bcnn_multi_iter(bcnn_net *net, bcnn_iterator *iter) {
    char *line = NULL;
    char **tok = NULL;
    int i, n_tok = 0, tmp_x, tmp_y;
    int out_w = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].w;
    int out_h = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].h;
    int out_c = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].c;
    // nb_lines_skipped = (int)((float)rand() / RAND_MAX * net->batch_size);
    // bh_fskipline(f, nb_lines_skipped);
    line = bh_fgetline(iter->f_input);
    if (line == NULL) {
        rewind(iter->f_input);
        line = bh_fgetline(iter->f_input);
    }
    n_tok = bh_strsplit(line, ' ', &tok);
    // fprintf(stderr, "ntok %d iter->label_width %d\n", n_tok,
    // iter->label_width);
    /*if (net->task != PREDICT && net->prediction_type == CLASSIFICATION) {
        BCNN_CHECK_AND_LOG(net->log_ctx, n_tok == 2, BCNN_INVALID_DATA,
                           "Wrong data format for classification");
    }*/

    bcnn_load_image_from_path(net, tok[0], net->input_width, net->input_height,
                              net->input_channels, iter->input_uchar,
                              net->state, &net->data_aug.shift_x,
                              &net->data_aug.shift_y);
    bcnn_load_image_from_path(net, tok[1], net->tensors[2].w, net->tensors[2].h,
                              net->tensors[2].c, iter->input_uchar2, net->state,
                              &net->data_aug.shift_x, &net->data_aug.shift_y);
    bcnn_load_image_from_path(net, tok[2], net->tensors[3].w, net->tensors[3].h,
                              net->tensors[3].c, iter->input_uchar3, net->state,
                              &net->data_aug.shift_x, &net->data_aug.shift_y);
#if defined(USE_GRID)
    iter->input_float[0] = atof(tok[3]);
    iter->input_float[1] = atof(tok[4]);
    iter->input_float[2] = atof(tok[5]);
#elif defined(USE_FACECROP)
    bcnn_load_image_from_path(net, tok[3], net->tensors[4].w, net->tensors[4].h,
                              net->tensors[4].c, iter->input_uchar4, net->state,
                              &net->data_aug.shift_x, &net->data_aug.shift_y);
    iter->input_float[0] = atof(tok[4]);
    iter->input_float[1] = atof(tok[5]);
    iter->input_float[2] = atof(tok[6]);
#elif defined(USE_MASKHP)
    bcnn_load_image_from_path(net, tok[3], net->tensors[4].w, net->tensors[4].h,
                              net->tensors[4].c, iter->input_uchar4, net->state,
                              &net->data_aug.shift_x, &net->data_aug.shift_y);
#endif

    // Label
    if (net->prediction_type != HEATMAP_REGRESSION) {
#if defined(USE_GRID)
        BCNN_CHECK_AND_LOG(net->log_ctx, (n_tok == iter->label_width + 6),
                           BCNN_INVALID_DATA, "Unexpected label format");
        for (i = 0; i < iter->label_width; ++i) {
            iter->label_float[i] = (float)atof(tok[i + 6]);
        }
#elif defined(USE_FACECROP)
        BCNN_CHECK_AND_LOG(net->log_ctx, (n_tok == iter->label_width + 7),
                           BCNN_INVALID_DATA, "Unexpected label format");
        for (i = 0; i < iter->label_width; ++i) {
            iter->label_float[i] = (float)atof(tok[i + 7]);
        }
#elif defined(USE_MASKHP)
        BCNN_CHECK_AND_LOG(net->log_ctx, (n_tok == iter->label_width + 4),
                           BCNN_INVALID_DATA, "Unexpected label format");
        for (i = 0; i < iter->label_width; ++i) {
            iter->label_float[i] = (float)atof(tok[i + 4]);
        }
#endif
    } else {
        BCNN_CHECK_AND_LOG(net->log_ctx, (n_tok == 1 + 4), BCNN_INVALID_DATA,
                           "Unexpected label format");
        bcnn_load_image_from_path(
            net, tok[4], net->tensors[1].w, net->tensors[1].h,
            net->tensors[1].c, iter->label_uchar, net->state,
            &net->data_aug.shift_x, &net->data_aug.shift_y);
    }

    bh_free(line);
    for (i = 0; i < n_tok; ++i) {
        bh_free(tok[i]);
    }
    bh_free(tok);

    return BCNN_SUCCESS;
}

int bcnn_iterator_initialize(bcnn_net *net, bcnn_iterator *iter,
                             char *path_input, char *path_label, char *type) {
    if (strcmp(type, "mnist") == 0) {
        return bcnn_init_mnist_iterator(net, iter, path_input, path_label);
    } else if (strcmp(type, "bin") == 0) {
        return bcnn_init_bin_iterator(net, iter, path_input);
    } else if (strcmp(type, "list") == 0) {
        return bcnn_init_list_iterator(net, iter, path_input);
    } else if (strcmp(type, "cifar10") == 0) {
        return bcnn_init_cifar10_iterator(net, iter, path_input);
    } else if (strcmp(type, "multi") == 0) {
        return bcnn_init_multi_iterator(net, iter, path_input);
    } else {
        BCNN_ERROR(net->log_ctx, BCNN_INVALID_PARAMETER,
                   "Unknown data_format. Available are 'mnist' 'bin' 'list' "
                   "'cifar10'");
    }
    return BCNN_SUCCESS;
}

int bcnn_iterator_next(bcnn_net *net, bcnn_iterator *iter) {
    switch (iter->type) {
        case ITER_MNIST:
            bcnn_mnist_next_iter(net, iter);
            break;
        case ITER_CIFAR10:
            bcnn_cifar10_iter(net, iter);
            break;
        case ITER_BIN:
            bcnn_bin_iter(net, iter);
            break;
        case ITER_LIST:
            bcnn_list_iter(net, iter);
            break;
        case ITER_MULTI:
            bcnn_multi_iter(net, iter);
            break;
        default:
            break;
    }
    return 0;
}

int bcnn_iterator_terminate(bcnn_iterator *iter) {
    if (iter->f_input != NULL) {
        fclose(iter->f_input);
    }
    if (iter->f_label != NULL) {
        fclose(iter->f_label);
    }
    if (iter->f_list != NULL) {
        fclose(iter->f_list);
    }
    bh_free(iter->input_uchar);
    bh_free(iter->input_uchar2);
    bh_free(iter->input_uchar3);
    bh_free(iter->input_uchar4);
    // bh_free(iter->input_float);
    bh_free(iter->label_float);
    bh_free(iter->label_uchar);
    bh_free(iter->label_int);

    return BCNN_SUCCESS;
}