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

#include "bcnn/bcnn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef bcnn_status (*bcnn_loader_init_func)(bcnn_loader *iter, bcnn_net *net,
                                             const char *path_data,
                                             const char *path_extra);

typedef bcnn_status (*bcnn_loader_next_func)(bcnn_loader *iter, bcnn_net *net,
                                             int idx);

typedef void (*bcnn_loader_terminate_func)(bcnn_loader *iter);

bcnn_status bcnn_data_augmentation(unsigned char *img, int width, int height,
                                   int depth, bcnn_data_augmenter *param,
                                   unsigned char *buffer);

void bcnn_fill_input_tensor(bcnn_net *net, bcnn_loader *iter, char *path_img,
                            int idx);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_DATA_H