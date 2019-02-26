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

#ifndef BH_MEM_H
#define BH_MEM_H

#ifdef __cplusplus
extern "C" {
#endif
#include <stdlib.h>
#include <string.h>  // memset

#ifndef bh_free
#define bh_free(buf)      \
    {                     \
        if (buf) {        \
            free(buf);    \
            (buf) = NULL; \
        }                 \
    }
#endif

#define bh_is_aligned8(x) (!(((uintptr_t)(x)) & 7))
#define bh_is_aligned16(x) (!(((uintptr_t)(x)) & 15))
#define bh_is_aligned32(x) (!(((uintptr_t)(x)) & 31))
#define bh_is_aligned64(x) (!(((uintptr_t)(x)) & 63))

static inline void *bh_align_malloc(size_t size, size_t align) {
    void *buf = NULL;
#if defined(_MSC_VER)
    buf = _aligned_malloc(size, align);
#elif defined(__MINGW32__) || defined(__MINGW64__)
    buf = __mingw_aligned_malloc(size, align);
#elif defined(__GNUC__)
    void *ptr = NULL;
    int res;
    align = (align + sizeof(void *) - 1) & ~(sizeof(void *) - 1);
    size = (size + align - 1) & ~(align - 1);
    res = posix_memalign(&ptr, align, size);
    buf = res ? NULL : ptr;
#else
    buf = malloc(size);
#endif
    return buf;
}

static inline void *bh_align_calloc(size_t size, size_t align) {
    void *buf = bh_align_malloc(size, align);
    if (buf != NULL) memset(buf, 0, size);
    return buf;
}

static inline void bh_align_free(void *buf) {
    if (buf != NULL) {
#if defined(_MSC_VER)
        _aligned_free(buf);
#elif defined(__MINGW32__) || defined(__MINGW64__)
        return __mingw_aligned_free(buf);
#else
        bh_free(buf);
#endif
    }
    buf = NULL;
}

#ifdef __cplusplus
}
#endif

#endif  // BH_MEM_H