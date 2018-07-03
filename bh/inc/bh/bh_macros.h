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

#ifndef BH_MACROS_H
#define BH_MACROS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

#define bh_clamp(x, a, b) (((x) < (a)) ? (a) : (((x) > (b)) ? (b) : (x)))
#define bh_min(a, b) (((a) < (b)) ? (a) : (b))
#define bh_max(a, b) (((a) > (b)) ? (a) : (b))
#define bh_swap(a, b, type) \
    {                       \
        type t = (a);       \
        (a) = (b);          \
        (b) = t;            \
    }
#define bh_abs(a) (((a) < (0)) ? -(a) : (a))
#define bh_free(buf)      \
    {                     \
        if (buf) {        \
            free(buf);    \
            (buf) = NULL; \
        }                 \
    }

#ifdef __cplusplus
}
#endif

#endif  // BH_MACROS_H