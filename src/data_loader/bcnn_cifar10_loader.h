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

#ifndef BCNN_CIFAR10_LOADER_H
#define BCNN_CIFAR10_LOADER_H

#include "bcnn/bcnn.h"

#ifdef __cplusplus
extern "C" {
#endif

bcnn_status bcnn_loader_cifar10_init(bcnn_loader *iter, bcnn_net *net,
                                     const char *train_path,
                                     const char *train_path_extra,
                                     const char *test_path,
                                     const char *test_path_extra);
void bcnn_loader_cifar10_terminate(bcnn_loader *iter);
bcnn_status bcnn_loader_cifar10_next(bcnn_loader *iter, bcnn_net *net, int idx);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_CIFAR10_LOADER_H