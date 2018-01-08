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

#ifndef BH_LOG_H
#define BH_LOG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <bh/bh.h>

// Error log handle
typedef enum {
    INFO = 0,
    WARNING = 1,
    ERROR = 2,
    SILENT = 3
} log_level;

#ifndef BCNN_LOG_LEVEL
#define BCNN_LOG_LEVEL 0 // INFO
#endif

static bh_inline void bh_check(int exp, const char *fmt, ...)
{
    if (!exp) {
#if (BCNN_LOG_LEVEL <= ERROR)
        char msg[2048];
        va_list args;
        va_start(args, fmt);
        vsnprintf(msg, sizeof(msg), fmt, args);
        va_end(args);
        fprintf(stderr, "\33[31;1m[ERROR] %s\33[0m\n", msg);
#endif
        exit(-1);
    }
}

static bh_inline void bh_log_error(const char *fmt, ...)
{
#if (BCNN_LOG_LEVEL <= ERROR)
    char msg[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
    fprintf(stderr, "\33[31;1m[ERROR] %s\33[0m\n", msg);
#endif
    exit(-1);
}

static bh_inline void bh_log_warning(const char *fmt, ...)
{
#if (BCNN_LOG_LEVEL <= WARNING)
    char msg[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
    fprintf(stderr, "\33[35;1m[WARNING] %s\33[0m\n", msg);
#endif
}

static bh_inline void bh_log_info(const char *fmt, ...)
{
#if (BCNN_LOG_LEVEL <= INFO)
    char msg[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
    fprintf(stderr, "[INFO] %s\n", msg);
#endif
}

#ifdef __cplusplus
}
#endif

#endif // BH_LOG_H

