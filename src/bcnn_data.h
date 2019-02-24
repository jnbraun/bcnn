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

#ifndef BCNN_DATA_H
#define BCNN_DATA_H

#include <stdbool.h>
#include "bcnn/bcnn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int input_width;
    int input_height;
    int input_depth;
    bool has_extra_data;
    bcnn_loader_type type;
    uint8_t *input_uchar;
    uint8_t *input_net;
    FILE *f_train;       /* Handle to primary train file */
    FILE *f_train_extra; /* Handle to secondary train file if needed */
    FILE *f_test;        /* Handle to primary test file */
    FILE *f_test_extra;  /* Handle to secondary test file if needed */
    FILE *f_current;
    FILE *f_current_extra;
} bcnn_loader;

/**
 *  Structure for online data augmentation parameters.
 */
typedef struct {
    int range_shift_x;  /* X-shift allowed range (chosen between
                            [-range_shift_x / 2; range_shift_x / 2]). */
    int range_shift_y;  /* Y-shift allowed range (chosen between
                           [-range_shift_y / 2; range_shift_y / 2]). */
    int random_fliph;   /* If !=0, randomly (with probability of 0.5) apply
                           horizontal flip to image. */
    int min_brightness; /* Minimum brightness factor allowed (additive factor,
                           range [-255;255]). */
    int max_brightness; /* Maximum brightness factor allowed (additive factor,
                           range [-255;255]). */
    int swap_to_bgr; /* Swap 1st and 3rd channel. Bcnn default image IO has RGB
                        layout */
    int no_input_norm;    /* If set to 1, Input data range is *not* normalized
                             between [-1;1] */
    int max_random_spots; /* Add a random number between [0;max_random_spots]
                             of saturated blobs. */
    float min_scale;      /* Minimum scale factor allowed. */
    float max_scale;      /* Maximum scale factor allowed. */
    float rotation_range; /* Rotation angle allowed range (chosen between
                             [-rotation_range / 2; rotation_range / 2]).
                             Expressed in degree. */
    float min_contrast;   /* Minimum contrast allowed (mult factor). */
    float max_contrast;   /* Maximum contrast allowed (mult factor). */
    float max_distortion; /* Maximum distortion factor allowed. */
    float mean_r;         /* 1st channel mean value to substract */
    float mean_g;         /* 2nd channel mean value to substract */
    float mean_b;         /* 3rd channel mean value to substract */
    /** Internal values */
    int use_precomputed; /* Flag set to 1 if the parameters to be applied are
                            those already set. */
    int brightness;      /* Current brightness factor. */
    int apply_fliph;     /* Current flip flag. */
    int shift_x;         /* Current x-shift. */
    int shift_y;         /* Current y-shift. */
    float rotation;      /* Current rotation angle. */
    float scale;         /* Current scale factor. */
    float contrast;      /* Current contrast factor. */
    float distortion;    /* Current distortion factor. */
    float distortion_kx; /* Current distortion x kernel. */
    float distortion_ky; /* Current distortion y kernel. */
} bcnn_data_augmenter;

typedef bcnn_status (*bcnn_loader_init_func)(bcnn_loader *iter, bcnn_net *net,
                                             const char *train_path,
                                             const char *train_path_extra,
                                             const char *test_path,
                                             const char *test_path_extra);

typedef bcnn_status (*bcnn_loader_next_func)(bcnn_loader *iter, bcnn_net *net,
                                             int idx);

typedef void (*bcnn_loader_terminate_func)(bcnn_loader *iter);

bcnn_status bcnn_loader_next(bcnn_net *net);

bcnn_status bcnn_data_augmentation(unsigned char *img, int width, int height,
                                   int depth, bcnn_data_augmenter *param,
                                   unsigned char *buffer);

bcnn_status bcnn_open_dataset(bcnn_loader *iter, bcnn_net *net,
                              const char *train_path,
                              const char *train_path_extra,
                              const char *test_path,
                              const char *test_path_extra, bool has_extra);

bcnn_status bcnn_switch_data_handles(bcnn_net *net, bcnn_loader *iter);

void bcnn_fill_input_tensor(bcnn_net *net, bcnn_loader *iter, char *path_img,
                            int idx);

void bcnn_destroy_data_loader(bcnn_net *net);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_DATA_H