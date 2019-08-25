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

#ifndef BH_LOG_H
#define BH_LOG_H

#include <stdarg.h> /* va_list, va_start, va_arg, va_end */
#include <stdio.h>  /* printf */

#ifdef __cplusplus
extern "C" {
#endif

#define BH_MAX_LENGTH_MSG 2048

#define BH_LOG_RESET "\033[0m"
#define BH_LOG_BLACK "\033[30m"
#define BH_LOG_RED "\033[31m"
#define BH_LOG_GREEN "\033[32m"
#define BH_LOG_YELLOW "\033[33m"
#define BH_LOG_BLUE "\033[34m"
#define BH_LOG_MAGENTA "\033[35m"
#define BH_LOG_CYAN "\033[36m"
#define BH_LOG_WHITE "\033[37m"
#define BH_LOG_BOLDBLACK "\033[1m\033[30m"
#define BH_LOG_BOLDRED "\033[1m\033[31m"
#define BH_LOG_BOLDGREEN "\033[1m\033[32m"
#define BH_LOG_BOLDYELLOW "\033[1m\033[33m"
#define BH_LOG_BOLDBLUE "\033[1m\033[34m"
#define BH_LOG_BOLDMAGENTA "\033[1m\033[35m"
#define BH_LOG_BOLDCYAN "\033[1m\033[36m"
#define BH_LOG_BOLDWHITE "\033[1m\033[37m"

typedef enum {
    BH_LOG_INFO = 0,
    BH_LOG_WARNING = 1,
    BH_LOG_ERROR = 2,
    BH_LOG_SILENT = 3
} bh_log_level;

static inline void bh_log(bh_log_level level, const char *fmt, ...) {
    char msg[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
    switch (level) {
        case BH_LOG_INFO: {
            fprintf(stderr, "\e[0;32m[INFO]\e[0m %s", msg);
            break;
        }
        case BH_LOG_WARNING: {
            fprintf(stderr, "\e[1;35m[WARNING] %s\e[0m", msg);
            break;
        }
        case BH_LOG_ERROR: {
            fprintf(stderr, "\e[1;31m[ERROR] %s\e[0m", msg);
            break;
        }
        case BH_LOG_SILENT: {
            break;
        }
    }
    return;
}

#ifdef __cplusplus
}
#endif

#endif  // BH_LOG_H
