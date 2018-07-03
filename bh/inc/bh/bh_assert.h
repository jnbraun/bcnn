/*
 * Copyright (c) 2016 Jean-No�l Braun.
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

#ifndef BH_ASSERT_H
#define BH_ASSERT_H

#ifdef __cplusplus
extern "C" {
#endif
#include <stdarg.h> /* va_list, va_start, va_arg, va_end */
#include <stdio.h>  /* printf */

static inline void bh_assert(int exp, int err, const char *fmt, ...) {
    if (!exp) {
#if (BH_LOG_ENABLED)
        char msg[2048];
        va_list args;
        va_start(args, fmt);
        vsnprintf(msg, sizeof(msg), fmt, args);
        va_end(args);
        fprintf(stderr, "\33[31;1m[FATAL] Assert failed: %s\33[0m\n", msg);
#endif
        exit(err);
    }
}

#ifdef __cplusplus
}
#endif

#endif  // BH_ASSERT_H