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

#include "bip/bip.h"

#include <float.h>
#include <limits.h>

#ifdef BIP_USE_STB_IMAGE
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image_write.h>
#endif

#include <bh/bh_macros.h>

/* Convenient macros */
#define BIP_CHECK_STATUS(err)                                        \
    do {                                                             \
        if ((err) != BIP_SUCCESS) {                                  \
            fprintf(stderr, "[ERROR] %s\n", bip_status_string(err)); \
            return err;                                              \
        }                                                            \
    } while (0)

#define BIP_CHECK_PTR(p)                     \
    do {                                     \
        if (((void *)p) == ((void *)NULL)) { \
            return BIP_INVALID_PTR;          \
        }                                    \
    } while (0)

#define BIP_CHECK_SIZE(size)         \
    do {                             \
        if (size <= 0) {             \
            return BIP_INVALID_SIZE; \
        }                            \
    } while (0)

/* Color space conversions */
bip_status bip_rgb2gray(uint8_t *src, size_t width, size_t height,
                        size_t src_stride, uint8_t *dst, size_t dst_stride) {
    size_t x, y;
    int32_t w = (int32_t)(0.333f * (1 << 12) + 0.5);

    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            dst[x] = (w * (int32_t)src[3 * x] + w * (int32_t)src[3 * x + 1] +
                      w * (int32_t)src[3 * x + 2] + (1 << 11)) >>
                     12;
        }
        src += src_stride;
        dst += dst_stride;
    }

    return BIP_SUCCESS;
}

/* Contrast / Color corrections */
bip_status bip_contrast_stretch(uint8_t *src, size_t src_stride, size_t width,
                                size_t height, size_t depth, uint8_t *dst,
                                size_t dst_stride, float contrast) {
    size_t x, y, d;
    uint32_t *mean = NULL;
    int32_t c = (int32_t)(contrast * (1 << 12) + 0.5);
    int32_t pix, round = (1 << 11);

    BIP_CHECK_SIZE(width);
    BIP_CHECK_SIZE(height);
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);

    mean = (uint32_t *)calloc(depth, sizeof(uint32_t));
    // Compute image mean
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (d = 0; d < depth; ++d) {
                mean[d] += src[depth * x + d];
            }
        }
        src += src_stride;
    }
    for (d = 0; d < depth; ++d) {
        mean[d] /= (width * height);
    }
    src -= src_stride * height;

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (d = 0; d < depth; ++d) {
                pix = ((((int32_t)src[depth * x + d] - (int32_t)mean[d]) * c +
                        round) >>
                       12) +
                      mean[d];
                dst[depth * x + d] = (uint8_t)bh_clamp(pix, 0, 255);
            }
        }
        src += src_stride;
        dst += dst_stride;
    }

    bh_free(mean);
    return BIP_SUCCESS;
}

bip_status bip_image_brightness(uint8_t *src, size_t src_stride, size_t width,
                                size_t height, size_t depth, uint8_t *dst,
                                size_t dst_stride, int32_t brightness) {
    size_t x, y;
    int32_t pix;

    BIP_CHECK_SIZE(width);
    BIP_CHECK_SIZE(height);
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width * depth; ++x) {
            pix = (int32_t)(src[x] + brightness);
            dst[x] = (uint8_t)bh_clamp(pix, 0, 255);
        }
        src += src_stride;
        dst += dst_stride;
    }

    return BIP_SUCCESS;
}

static float _bip_noise2d(int x, int y, int octave, int seed) {
    int i = x * 1619 + y * 31337 + octave * 3463 + seed * 13397;
    int n = (i << 13) ^ i;
    return (1.0f -
            ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) /
                1073741824.0f);
}

static float _bip_interpolate(float v1, float v2, float x) {
    float t = (1.0f - (float)cos(x * BIP_PI)) * 0.5f;
    return v1 * (1.0f - t) + v2 * t;
}

static float _bip_smooth2d(float x, float y, int octave, int seed) {
    int intx = (int)floor(x);
    float fracx = x - intx;
    int inty = (int)floor(y);
    float fracy = y - inty;

    float v1 = _bip_noise2d(intx, inty, octave, seed);
    float v2 = _bip_noise2d(intx + 1, inty, octave, seed);
    float v3 = _bip_noise2d(intx, inty + 1, octave, seed);
    float v4 = _bip_noise2d(intx + 1, inty + 1, octave, seed);

    float i1 = _bip_interpolate(v1, v2, fracx);
    float i2 = _bip_interpolate(v3, v4, fracx);

    return _bip_interpolate(i1, i2, fracy);
}

static float _bip_perlin_noise2d(float x, float y, float persistence,
                                 int octaves, int seed) {
    float total = 0.0;
    float frequency = 1.0;
    float amplitude = 1.0;
    int32_t i = 0;

    for (i = 0; i < octaves; i++) {
        total +=
            _bip_smooth2d(x * frequency, y * frequency, i, seed) * amplitude;
        frequency *= 0.5f;
        amplitude *= persistence;
    }

    return total;
}

static inline int is_positive_and_inferior_to(int a, int b) {
    return (unsigned int)a < (unsigned int)b;
}

bip_status bip_image_perlin_distortion(uint8_t *src, size_t src_stride,
                                       size_t width, size_t height,
                                       size_t depth, uint8_t *dst,
                                       size_t dst_stride, float distortion,
                                       float kx, float ky) {
    size_t x, y, k;
    int32_t index, x_map, y_map;
    int32_t seed = rand();
    float persistence, x_norm, y_norm, px, py, x_diff, y_diff;

    BIP_CHECK_SIZE(width);
    BIP_CHECK_SIZE(height);
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);

    persistence = 0.0f;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            x_norm = (float)x / width;
            y_norm = (float)y / height;
            px = (x_norm +
                  _bip_perlin_noise2d(x_norm + kx, y_norm + ky, persistence, 1,
                                      seed) *
                      distortion) *
                 width;
            py = (y_norm +
                  _bip_perlin_noise2d(x_norm + kx, y_norm + ky, persistence, 1,
                                      seed) *
                      distortion) *
                 height;
            x_map = (int)(px);
            y_map = (int)(py);
            x_diff = (px - (float)floor(px));
            y_diff = (py - (float)floor(py));
            index = (y_map * width + x_map);
            // if (x_map >= 0 && x_map < ((int32_t)width - 1) && y_map >= 0 &&
            // y_map < ((int32_t)height - 1)) {
            if (is_positive_and_inferior_to(x_map, (int32_t)width - 1) &&
                is_positive_and_inferior_to(y_map, (int32_t)height - 1)) {
                for (k = 0; k < depth; ++k) {
                    uint8_t level =
                        (uint8_t)((float)src[index * depth + k] * (1 - x_diff) *
                                      (1 - y_diff) +
                                  (float)src[(index + 1) * depth + k] *
                                      (x_diff) * (1 - y_diff) +
                                  (float)src[(index + width) * depth + k] *
                                      (1 - x_diff) * (y_diff) +
                                  (float)src[(index + width + 1) * depth + k] *
                                      (x_diff) * (y_diff));
                    dst[x * depth + k] = (uint8_t)level;
                }
            } else {
                for (k = 0; k < depth; ++k) {
                    dst[x * depth + k] = (uint8_t)0;
                }
            }
        }
        dst += depth * width;
    }

    return BIP_SUCCESS;
}

static int rand_between(int min, int max) {
    if (min > max) {
        return 0.f;
    }
    return (int)(((float)rand() / RAND_MAX * (max - min)) + min + 0.5);
}

static float frand_between(float min, float max) {
    if (min > max) {
        return 0.f;
    }
    return ((float)rand() / RAND_MAX * (max - min)) + min + 0.5;
}

bip_status bip_add_random_spotlights(
    uint8_t *src, size_t src_stride, size_t width, size_t height, size_t depth,
    uint8_t *dst, size_t dst_stride, uint32_t num_spots, float min_spot_width,
    float max_spot_width, float min_spot_height, float max_spot_height) {
    BIP_CHECK_SIZE(width);
    BIP_CHECK_SIZE(height);
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);
    if (dst != src) {
        memcpy(dst, src, width * height * depth);
    }
    uint8_t *dst0 = dst;
    for (uint32_t i = 0; i < num_spots; ++i) {
        int mu_x = rand_between(0, width - 1);
        int mu_y = rand_between(0, height - 1);
        float sigma_x = frand_between(min_spot_width, max_spot_width);
        float sigma_y = frand_between(min_spot_height, max_spot_height);
        float inv_sigma2_x = 1 / (sigma_x * sigma_x);
        float inv_sigma2_y = 1 / (sigma_y * sigma_y);
        dst = dst0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float val =
                    exp(-0.5f * (inv_sigma2_x * (x - (mu_x)) * (x - (mu_x)) +
                                 inv_sigma2_y * (y - (mu_y)) * (y - (mu_y))));
                for (int c = 0; c < depth; ++c) {
                    dst[depth * x + c] = (unsigned char)bh_clamp(
                        255.0f * val + dst[depth * x + c], 0, 255);
                }
            }
            dst += dst_stride;
        }
    }
    return BIP_SUCCESS;
}

/* Basic image manipulations */
bip_status bip_crop_image(uint8_t *src, size_t src_width, size_t src_height,
                          size_t src_stride, int32_t x_ul, int32_t y_ul,
                          uint8_t *dst, size_t dst_width, size_t dst_height,
                          size_t dst_stride, size_t depth) {
    size_t y;
    uint32_t off_x = bh_max(0, 0 - x_ul);
    uint32_t off_y = bh_max(0, 0 - y_ul);
    uint32_t off_src_x = bh_max(0, x_ul);
    uint32_t off_src_y = bh_max(0, y_ul);
    uint32_t row_cp_sz =
        bh_clamp(src_stride - off_src_x * depth, 0, dst_stride - off_x * depth);
    uint32_t maxy = bh_min(dst_height - off_y, src_height - off_src_y);

    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);

    if (off_y > dst_height || off_x > dst_width || off_src_x > src_width ||
        off_src_y > src_height)
        return 0;
    else
        dst += off_y * dst_stride + off_x * depth;

    src += off_src_y * src_stride;
    for (y = 0; y < maxy; ++y) {
        memcpy(dst, src + off_src_x * depth, row_cp_sz);
        src += src_stride;
        dst += dst_stride;
    }

    return BIP_SUCCESS;
}

/* Primitives for building pyramid */
bip_status bip_pyramid_down(uint8_t *src, size_t src_width, size_t src_height,
                            size_t src_stride, uint8_t *dst, size_t dst_width,
                            size_t dst_height, size_t dst_stride) {
    uint8_t *s0 = NULL, *s1 = NULL, *end = NULL;
    uint8_t *d = NULL;
    size_t y, even_width;

    BIP_CHECK_SIZE(src_width);
    BIP_CHECK_SIZE(src_height);
    BIP_CHECK_SIZE(dst_width);
    BIP_CHECK_SIZE(dst_height);
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);
    if ((src_width + 1) / 2 != dst_width || (src_height + 1) / 2 != dst_height)
        return BIP_INVALID_SIZE;

    even_width = src_width & ~1;
    for (y = 0; y < src_height; y += 2) {
        s0 = src;
        s1 = (y == src_height - 1 ? src : src + src_stride);
        end = src + even_width;
        d = dst;
        for (; s0 < end; s0 += 2, s1 += 2, d += 1) {
            d[0] = (s0[0] + s0[1] + s1[0] + s1[1] + 2) >> 2;
        }
        if (even_width != src_width) {
            d[0] = (s0[0] + s1[0] + 1) >> 1;
        }
        src += 2 * src_stride;
        dst += dst_stride;
    }
    return BIP_SUCCESS;
}

bip_status bip_pyramid_up(uint8_t *src, size_t src_width, size_t src_height,
                          size_t src_stride, uint8_t *dst, size_t dst_width,
                          size_t dst_height, size_t dst_stride) {
    uint8_t *dst0 = NULL, *dst1 = NULL;
    size_t x, y;
    uint8_t val = 0;

    BIP_CHECK_SIZE(src_width);
    BIP_CHECK_SIZE(src_height);
    BIP_CHECK_SIZE(dst_width);
    BIP_CHECK_SIZE(dst_height);
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);
    if (src_width * 2 != dst_width || src_height * 2 != dst_height)
        return BIP_INVALID_SIZE;

    for (y = 0; y < src_height; ++y) {
        dst0 = dst;
        dst1 = dst + dst_stride;
        for (x = 0; x < src_width; x++, dst0 += 2, dst1 += 2) {
            val = src[x];
            dst0[0] = val;
            dst0[1] = val;
            dst1[0] = val;
            dst1[1] = val;
        }
        src += src_stride;
        dst += 2 * dst_stride;
    }
    return BIP_SUCCESS;
}

bip_status bip_mirror_borders_8u(uint8_t *src, int32_t src_width,
                                 int32_t src_height, int32_t src_depth,
                                 int32_t src_stride, uint8_t *dst,
                                 int32_t dst_width, int32_t dst_height,
                                 int32_t dst_depth, int32_t dst_stride,
                                 int32_t top, int32_t bottom, int32_t left,
                                 int32_t right) {
    int32_t x, y, d;
    int32_t width2 = src_width + left + right;
    int32_t height2 = src_height + top + bottom;

    BIP_CHECK_SIZE(src_width);
    BIP_CHECK_SIZE(src_height);
    BIP_CHECK_SIZE(src_depth);
    BIP_CHECK_SIZE(dst_width);
    BIP_CHECK_SIZE(dst_height);
    BIP_CHECK_SIZE(dst_depth);
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);

    dst += top * dst_stride;
    for (y = 0; y < src_height; ++y) {
        for (x = 0; x < left; ++x) {
            for (d = 0; d < src_depth; ++d) {
                dst[x * src_depth + d] = src[(left - x - 1) * src_depth + d];
            }
        }
        for (x = 0; x < right; ++x) {
            for (d = 0; d < src_depth; ++d) {
                dst[(x + src_width + left) * src_depth + d] =
                    src[(src_width - x - 1) * src_depth + d];
            }
        }
        memcpy(dst + left * src_depth, src,
               src_width * src_depth * sizeof(uint8_t));
        src += src_stride;
        dst += dst_stride;
    }
    dst -= (src_height + top) * dst_stride;
    for (y = 0; y < top; ++y) {
        memcpy(dst + (top - y - 1) * width2 * src_depth,
               dst + (y + top) * width2 * src_depth,
               width2 * src_depth * sizeof(uint8_t));
    }
    for (y = 0; y < bottom; ++y) {
        memcpy(dst + (src_height + top + y) * width2 * src_depth,
               dst + (height2 - bottom - y - 1) * width2 * src_depth,
               width2 * src_depth * sizeof(uint8_t));
    }

    return BIP_SUCCESS;
}

bip_status bip_mirror_borders_32f(float *src, int32_t src_width,
                                  int32_t src_height, int32_t src_depth,
                                  int32_t src_stride, float *dst,
                                  int32_t dst_width, int32_t dst_height,
                                  int32_t dst_depth, int32_t dst_stride,
                                  int32_t top, int32_t bottom, int32_t left,
                                  int32_t right) {
    int32_t x, y, d;
    int32_t width2 = src_width + left + right;
    int32_t height2 = src_height + top + bottom;

    BIP_CHECK_SIZE(src_width);
    BIP_CHECK_SIZE(src_height);
    BIP_CHECK_SIZE(src_depth);
    BIP_CHECK_SIZE(dst_width);
    BIP_CHECK_SIZE(dst_height);
    BIP_CHECK_SIZE(dst_depth);
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);

    src_stride /= sizeof(float);
    dst_stride /= sizeof(float);

    dst += top * dst_stride;
    for (y = 0; y < src_height; ++y) {
        for (x = 0; x < left; ++x) {
            for (d = 0; d < src_depth; ++d) {
                dst[x * src_depth + d] = src[(left - x - 1) * src_depth + d];
            }
        }
        for (x = 0; x < right; ++x) {
            for (d = 0; d < src_depth; ++d) {
                dst[(x + src_width + left) * src_depth + d] =
                    src[(src_width - x - 1) * src_depth + d];
            }
        }
        memcpy(dst + left * src_depth, src,
               src_width * src_depth * sizeof(float));
        src += src_stride;
        dst += dst_stride;
    }
    dst -= (src_height + top) * dst_stride;
    for (y = 0; y < top; ++y) {
        memcpy(dst + (top - y - 1) * width2 * src_depth,
               dst + (y + top) * width2 * src_depth,
               width2 * src_depth * sizeof(float));
    }
    for (y = 0; y < bottom; ++y) {
        memcpy(dst + (src_height + top + y) * width2 * src_depth,
               dst + (height2 - bottom - y - 1) * width2 * src_depth,
               width2 * src_depth * sizeof(float));
    }

    return BIP_SUCCESS;
}

/* Image statistics primitives */
bip_status bip_image_integral(uint8_t *src, size_t width, size_t height,
                              size_t src_stride, uint32_t *dst,
                              size_t dst_stride) {
    size_t x, y;
    uint32_t val = 0;

    BIP_CHECK_SIZE(width);
    BIP_CHECK_SIZE(height);
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);

    dst_stride /= sizeof(uint32_t);

    memset(dst, 0, (width + 1) * sizeof(uint32_t));
    dst += dst_stride + 1;

    for (y = 0; y < height; ++y) {
        val = 0;
        dst[-1] = 0;
        for (x = 0; x < width; ++x) {
            val += (uint32_t)src[x];
            dst[x] = val + dst[x - dst_stride];
        }
        src += src_stride;
        dst += dst_stride;
    }

    return BIP_SUCCESS;
}

bip_status bip_image_square_integral(uint8_t *src, size_t width, size_t height,
                                     size_t src_stride, uint32_t *dst,
                                     size_t dst_stride, double *dst_square,
                                     size_t dst_square_stride) {
    size_t x, y;
    uint32_t val = 0;
    double sq_val = 0;

    BIP_CHECK_SIZE(width);
    BIP_CHECK_SIZE(height);
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);
    BIP_CHECK_PTR(dst_square);

    dst_stride /= sizeof(uint32_t);
    dst_square_stride /= sizeof(double);

    memset(dst, 0, (width + 1) * sizeof(uint32_t));
    dst += dst_stride + 1;

    memset(dst_square, 0, (width + 1) * sizeof(double));
    dst_square += dst_square_stride + 1;

    for (y = 0; y < height; ++y) {
        val = 0;
        sq_val = 0;
        dst[-1] = 0;
        dst_square[-1] = 0;
        for (x = 0; x < width; ++x) {
            val += (uint32_t)src[x];
            sq_val += (double)src[x] * (double)src[x];
            dst[x] = val + dst[x - dst_stride];
            dst_square[x] = sq_val + dst_square[x - dst_square_stride];
        }
        src += src_stride;
        dst += dst_stride;
        dst_square += dst_square_stride;
    }

    return BIP_SUCCESS;
}

bip_status bip_image_sliding_mean(uint8_t *src, size_t width, size_t height,
                                  size_t src_stride, uint8_t *dst,
                                  size_t dst_stride, size_t kernel_width,
                                  size_t kernel_height) {
    size_t x, y, w2 = kernel_width / 2, h2 = kernel_height / 2,
                 sz = (2 * w2 + 1) * (2 * h2 + 1);
    int32_t min_x, min_y, max_x, max_y, val;
    uint32_t *sum = NULL;
    bip_status err;

    BIP_CHECK_SIZE(width);
    BIP_CHECK_SIZE(height);
    BIP_CHECK_SIZE(kernel_width);
    BIP_CHECK_SIZE(kernel_height);
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);

    sum = (uint32_t *)calloc((height + 1) * (width + 1), sizeof(uint32_t));

    if ((err = bip_image_integral(src, width, height, src_stride, sum,
                                  (width + 1) * sizeof(uint32_t))) !=
        BIP_SUCCESS)
        return err;

    for (y = 0; y <= h2; ++y) {
        for (x = 0; x <= w2; ++x) {
            min_x = 0;
            min_y = 0;
            max_x = x + w2;
            max_y = y + h2;
            val = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum[(min_y) * (width + 1) + (max_x + 1)] -
                   sum[(max_y + 1) * (width + 1) + (min_x)] +
                   sum[(min_y) * (width + 1) + (min_x)]) /
                  ((max_x - min_x + 1) * (max_y - min_y + 1));
            dst[x] = (uint8_t)bh_clamp(val, 0, 255);
        }
        for (; x < width - w2; ++x) {
            min_x = x - w2;
            min_y = 0;
            max_x = x + w2;
            max_y = y + h2;
            val = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum[(min_y) * (width + 1) + (max_x + 1)] -
                   sum[(max_y + 1) * (width + 1) + (min_x)] +
                   sum[(min_y) * (width + 1) + (min_x)]) /
                  ((max_x - min_x + 1) * (max_y - min_y + 1));
            dst[x] = (uint8_t)bh_clamp(val, 0, 255);
        }
        for (; x < width; ++x) {
            min_x = x - w2;
            min_y = 0;
            max_x = width - 1;
            max_y = y + h2;
            val = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum[(min_y) * (width + 1) + (max_x + 1)] -
                   sum[(max_y + 1) * (width + 1) + (min_x)] +
                   sum[(min_y) * (width + 1) + (min_x)]) /
                  ((max_x - min_x + 1) * (max_y - min_y + 1));
            dst[x] = (uint8_t)bh_clamp(val, 0, 255);
        }
        dst += dst_stride;
    }
    for (; y < height - h2; ++y) {
        for (x = 0; x <= w2; ++x) {
            min_x = 0;
            min_y = y - h2;
            max_x = x + w2;
            max_y = y + h2;
            val = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum[(min_y) * (width + 1) + (max_x + 1)] -
                   sum[(max_y + 1) * (width + 1) + (min_x)] +
                   sum[(min_y) * (width + 1) + (min_x)]) /
                  ((max_x - min_x + 1) * (max_y - min_y + 1));
            dst[x] = (uint8_t)bh_clamp(val, 0, 255);
        }
        for (; x < width - w2; ++x) {
            min_x = x - w2;
            min_y = y - h2;
            max_x = x + w2;
            max_y = y + h2;
            val = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum[(min_y) * (width + 1) + (max_x + 1)] -
                   sum[(max_y + 1) * (width + 1) + (min_x)] +
                   sum[(min_y) * (width + 1) + (min_x)]) /
                  (sz);
            dst[x] = (uint8_t)bh_clamp(val, 0, 255);
        }
        for (; x < width; ++x) {
            min_x = x - w2;
            min_y = y - h2;
            max_x = width - 1;
            max_y = y + h2;
            val = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum[(min_y) * (width + 1) + (max_x + 1)] -
                   sum[(max_y + 1) * (width + 1) + (min_x)] +
                   sum[(min_y) * (width + 1) + (min_x)]) /
                  ((max_x - min_x + 1) * (max_y - min_y + 1));
            dst[x] = (uint8_t)bh_clamp(val, 0, 255);
        }
        dst += dst_stride;
    }
    for (; y < height; ++y) {
        for (x = 0; x <= w2; ++x) {
            min_x = 0;
            min_y = y - h2;
            max_x = x + w2;
            max_y = height - 1;
            val = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum[(min_y) * (width + 1) + (max_x + 1)] -
                   sum[(max_y + 1) * (width + 1) + (min_x)] +
                   sum[(min_y) * (width + 1) + (min_x)]) /
                  ((max_x - min_x + 1) * (max_y - min_y + 1));
            dst[x] = (uint8_t)bh_clamp(val, 0, 255);
        }
        for (; x < width - w2; ++x) {
            min_x = x - w2;
            min_y = y - h2;
            max_x = x + w2;
            max_y = height - 1;
            val = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum[(min_y) * (width + 1) + (max_x + 1)] -
                   sum[(max_y + 1) * (width + 1) + (min_x)] +
                   sum[(min_y) * (width + 1) + (min_x)]) /
                  ((max_x - min_x + 1) * (max_y - min_y + 1));
            dst[x] = (uint8_t)bh_clamp(val, 0, 255);
        }
        for (; x < width; ++x) {
            min_x = x - w2;
            min_y = y - h2;
            max_x = width - 1;
            max_y = height - 1;
            val = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum[(min_y) * (width + 1) + (max_x + 1)] -
                   sum[(max_y + 1) * (width + 1) + (min_x)] +
                   sum[(min_y) * (width + 1) + (min_x)]) /
                  ((max_x - min_x + 1) * (max_y - min_y + 1));
            dst[x] = (uint8_t)bh_clamp(val, 0, 255);
        }
        dst += dst_stride;
    }

    bh_free(sum);
    return BIP_SUCCESS;
}

bip_status bip_image_sliding_mean_variance(
    uint8_t *src, size_t width, size_t height, size_t src_stride, uint8_t *dst,
    size_t dst_stride, double *dst_variance, size_t dst_variance_stride,
    size_t kernel_width, size_t kernel_height) {
    size_t x, y, w2 = kernel_width / 2, h2 = kernel_height / 2,
                 sz = (2 * w2 + 1) * (2 * h2 + 1), sz_part = 0;
    int32_t min_x, min_y, max_x, max_y, mean;
    double var;
    uint32_t *sum = NULL;
    double *sum_sqr = NULL;
    bip_status err;

    BIP_CHECK_SIZE(width);
    BIP_CHECK_SIZE(height);
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);
    BIP_CHECK_PTR(dst_variance);
    BIP_CHECK_SIZE(kernel_width);
    BIP_CHECK_SIZE(kernel_height);

    sum = (uint32_t *)calloc((height + 1) * (width + 1), sizeof(uint32_t));
    sum_sqr = (double *)calloc((height + 1) * (width + 1), sizeof(double));

    if ((err = bip_image_square_integral(
             src, width, height, src_stride, sum,
             (width + 1) * sizeof(uint32_t), sum_sqr,
             (width + 1) * sizeof(double))) != BIP_SUCCESS)
        return err;

    dst_variance_stride /= sizeof(double);
    for (y = 0; y <= h2; ++y) {
        for (x = 0; x <= w2; ++x) {
            min_x = 0;
            min_y = 0;
            max_x = x + w2;
            max_y = y + h2;
            sz_part = ((max_x - min_x + 1) * (max_y - min_y + 1));
            mean = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                    sum[(min_y) * (width + 1) + (max_x + 1)] -
                    sum[(max_y + 1) * (width + 1) + (min_x)] +
                    sum[(min_y) * (width + 1) + (min_x)]) /
                   sz_part;
            dst[x] = (uint8_t)bh_clamp(mean, 0, 255);

            var = (sum_sqr[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(min_y) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(max_y + 1) * (width + 1) + (min_x)] +
                   sum_sqr[(min_y) * (width + 1) + (min_x)]) /
                  sz_part;
            var = var - (double)mean * (double)mean;
            dst_variance[x] = var;
        }
        for (; x < width - w2; ++x) {
            min_x = x - w2;
            min_y = 0;
            max_x = x + w2;
            max_y = y + h2;
            sz_part = ((max_x - min_x + 1) * (max_y - min_y + 1));
            mean = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                    sum[(min_y) * (width + 1) + (max_x + 1)] -
                    sum[(max_y + 1) * (width + 1) + (min_x)] +
                    sum[(min_y) * (width + 1) + (min_x)]) /
                   sz_part;
            dst[x] = (uint8_t)bh_clamp(mean, 0, 255);

            var = (sum_sqr[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(min_y) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(max_y + 1) * (width + 1) + (min_x)] +
                   sum_sqr[(min_y) * (width + 1) + (min_x)]) /
                  sz_part;
            var = var - (double)mean * (double)mean;
            dst_variance[x] = var;
        }
        for (; x < width; ++x) {
            min_x = x - w2;
            min_y = 0;
            max_x = width - 1;
            max_y = y + h2;
            sz_part = ((max_x - min_x + 1) * (max_y - min_y + 1));
            mean = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                    sum[(min_y) * (width + 1) + (max_x + 1)] -
                    sum[(max_y + 1) * (width + 1) + (min_x)] +
                    sum[(min_y) * (width + 1) + (min_x)]) /
                   sz_part;
            dst[x] = (uint8_t)bh_clamp(mean, 0, 255);

            var = (sum_sqr[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(min_y) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(max_y + 1) * (width + 1) + (min_x)] +
                   sum_sqr[(min_y) * (width + 1) + (min_x)]) /
                  sz_part;
            var = var - (double)mean * (double)mean;
            dst_variance[x] = var;
        }
        dst += dst_stride;
        dst_variance += dst_variance_stride;
    }
    for (; y < height - h2; ++y) {
        for (x = 0; x <= w2; ++x) {
            min_x = 0;
            min_y = y - h2;
            max_x = x + w2;
            max_y = y + h2;
            sz_part = ((max_x - min_x + 1) * (max_y - min_y + 1));
            mean = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                    sum[(min_y) * (width + 1) + (max_x + 1)] -
                    sum[(max_y + 1) * (width + 1) + (min_x)] +
                    sum[(min_y) * (width + 1) + (min_x)]) /
                   sz_part;
            dst[x] = (uint8_t)bh_clamp(mean, 0, 255);

            var = (sum_sqr[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(min_y) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(max_y + 1) * (width + 1) + (min_x)] +
                   sum_sqr[(min_y) * (width + 1) + (min_x)]) /
                  sz_part;
            var = var - (double)mean * (double)mean;
            dst_variance[x] = var;
        }
        for (; x < width - w2; ++x) {
            min_x = x - w2;
            min_y = y - h2;
            max_x = x + w2;
            max_y = y + h2;
            mean = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                    sum[(min_y) * (width + 1) + (max_x + 1)] -
                    sum[(max_y + 1) * (width + 1) + (min_x)] +
                    sum[(min_y) * (width + 1) + (min_x)]) /
                   sz;
            dst[x] = (uint8_t)bh_clamp(mean, 0, 255);

            var = (sum_sqr[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(min_y) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(max_y + 1) * (width + 1) + (min_x)] +
                   sum_sqr[(min_y) * (width + 1) + (min_x)]) /
                  sz;
            var = var - (double)mean * (double)mean;
            dst_variance[x] = var;
        }
        for (; x < width; ++x) {
            min_x = x - w2;
            min_y = y - h2;
            max_x = width - 1;
            max_y = y + h2;
            sz_part = ((max_x - min_x + 1) * (max_y - min_y + 1));
            mean = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                    sum[(min_y) * (width + 1) + (max_x + 1)] -
                    sum[(max_y + 1) * (width + 1) + (min_x)] +
                    sum[(min_y) * (width + 1) + (min_x)]) /
                   sz_part;
            dst[x] = (uint8_t)bh_clamp(mean, 0, 255);

            var = (sum_sqr[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(min_y) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(max_y + 1) * (width + 1) + (min_x)] +
                   sum_sqr[(min_y) * (width + 1) + (min_x)]) /
                  sz_part;
            var = var - (double)mean * (double)mean;
            dst_variance[x] = var;
        }
        dst += dst_stride;
        dst_variance += dst_variance_stride;
    }
    for (; y < height; ++y) {
        for (x = 0; x <= w2; ++x) {
            min_x = 0;
            min_y = y - h2;
            max_x = x + w2;
            max_y = height - 1;
            sz_part = ((max_x - min_x + 1) * (max_y - min_y + 1));
            mean = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                    sum[(min_y) * (width + 1) + (max_x + 1)] -
                    sum[(max_y + 1) * (width + 1) + (min_x)] +
                    sum[(min_y) * (width + 1) + (min_x)]) /
                   sz_part;
            dst[x] = (uint8_t)bh_clamp(mean, 0, 255);

            var = (sum_sqr[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(min_y) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(max_y + 1) * (width + 1) + (min_x)] +
                   sum_sqr[(min_y) * (width + 1) + (min_x)]) /
                  sz_part;
            var = var - (double)mean * (double)mean;
            dst_variance[x] = var;
        }
        for (; x < width - w2; ++x) {
            min_x = x - w2;
            min_y = y - h2;
            max_x = x + w2;
            max_y = height - 1;
            sz_part = ((max_x - min_x + 1) * (max_y - min_y + 1));
            mean = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                    sum[(min_y) * (width + 1) + (max_x + 1)] -
                    sum[(max_y + 1) * (width + 1) + (min_x)] +
                    sum[(min_y) * (width + 1) + (min_x)]) /
                   sz_part;
            dst[x] = (uint8_t)bh_clamp(mean, 0, 255);

            var = (sum_sqr[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(min_y) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(max_y + 1) * (width + 1) + (min_x)] +
                   sum_sqr[(min_y) * (width + 1) + (min_x)]) /
                  sz_part;
            var = var - (double)mean * (double)mean;
            dst_variance[x] = var;
        }
        for (; x < width; ++x) {
            min_x = x - w2;
            min_y = y - h2;
            max_x = width - 1;
            max_y = height - 1;
            sz_part = ((max_x - min_x + 1) * (max_y - min_y + 1));
            mean = (sum[(max_y + 1) * (width + 1) + (max_x + 1)] -
                    sum[(min_y) * (width + 1) + (max_x + 1)] -
                    sum[(max_y + 1) * (width + 1) + (min_x)] +
                    sum[(min_y) * (width + 1) + (min_x)]) /
                   sz_part;
            dst[x] = (uint8_t)bh_clamp(mean, 0, 255);

            var = (sum_sqr[(max_y + 1) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(min_y) * (width + 1) + (max_x + 1)] -
                   sum_sqr[(max_y + 1) * (width + 1) + (min_x)] +
                   sum_sqr[(min_y) * (width + 1) + (min_x)]) /
                  sz_part;
            var = var - (double)mean * (double)mean;
            dst_variance[x] = var;
        }
        dst += dst_stride;
        dst_variance += dst_variance_stride;
    }

    bh_free(sum);
    bh_free(sum_sqr);
    return BIP_SUCCESS;
}

bip_status bip_image_histogram(uint8_t *src, size_t src_width,
                               size_t src_height, size_t src_stride,
                               uint32_t *histogram) {
    uint32_t h_part[4][256];
    size_t i, x, y, w_align = 0;

    BIP_CHECK_SIZE(src_width);
    BIP_CHECK_SIZE(src_height);
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(histogram);

    memset(h_part, 0, sizeof(uint32_t) * 256 * 4);
    w_align = src_width & ~(3);
    for (y = 0; y < src_height; ++y) {
        x = 0;
        for (; x < w_align; x += 4) {
            ++h_part[0][src[x]];
            ++h_part[1][src[x + 1]];
            ++h_part[2][src[x + 2]];
            ++h_part[3][src[x + 3]];
        }
        for (; x < src_width; ++x) ++h_part[0][src[x]];
        src += src_stride;
    }
    for (i = 0; i < 256; ++i)
        histogram[i] =
            h_part[0][i] + h_part[1][i] + h_part[2][i] + h_part[3][i];

    return BIP_SUCCESS;
}

bip_status bip_image_entropy(uint8_t *src, size_t src_width, size_t src_height,
                             size_t src_stride, float *entropy) {
    int i;
    unsigned int histo[256] = {0};
    float sum = 0.0f, norm = 1.0f / (src_width * src_height);

    BIP_CHECK_SIZE(src_width);
    BIP_CHECK_SIZE(src_height);
    BIP_CHECK_PTR(src);

    BIP_CHECK_STATUS(
        bip_image_histogram(src, src_width, src_height, src_stride, histo));

    for (i = 0; i < 256; ++i) {
        if (histo[i] > 0) sum -= norm * histo[i] * logf((float)histo[i] * norm);
    }
    *entropy = sum * BIP_LOG2;

    return BIP_SUCCESS;
}

bip_status bip_otsu(uint8_t *src, size_t src_width, size_t src_height,
                    size_t src_stride, float *thresh, float *var) {
    int i;
    unsigned int histo[256] = {0};
    float mu = 0.0f, mu1 = 0.0f, scale = 1.0f / (src_width * src_height);
    float q1 = 0.0f;
    float max_var = 0.0f, th = 0.0f;
    float p_i, q2, mu2, v;

    BIP_CHECK_SIZE(src_width);
    BIP_CHECK_SIZE(src_height);
    BIP_CHECK_PTR(src);

    BIP_CHECK_STATUS(
        bip_image_histogram(src, src_width, src_height, src_stride, histo));

    for (i = 0; i < 256; i++) mu += i * histo[i];

    mu *= scale;

    for (i = 0; i < 256; ++i) {
        p_i = histo[i] * scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1.0f - q1;

        if (bh_min(q1, q2) < FLT_EPSILON || bh_max(q1, q2) > 1.0f - FLT_EPSILON)
            continue;

        mu1 = (mu1 + i * p_i) / q1;
        mu2 = (mu - q1 * mu1) / q2;
        v = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
        if (v > max_var) {
            max_var = v;
            th = (float)i;
        }
    }

    (*thresh) = th;
    (*var) = max_var;
    return BIP_SUCCESS;
}

bip_status bip_resize_bilinear(uint8_t *src, size_t src_width,
                               size_t src_height, size_t src_stride,
                               uint8_t *dst, size_t dst_width,
                               size_t dst_height, size_t dst_stride,
                               size_t depth) {
    size_t i, x, y, c, dst_row_sz = dst_width * depth, k;
    float x_scale = (float)src_width / dst_width;
    float y_scale = (float)src_height / dst_height;
    float alpha = 0.0f;
    int32_t *table =
        (int32_t *)calloc(2 * (2 * dst_row_sz + dst_height), sizeof(int32_t));
    int32_t *table_ix = table;
    int32_t *table_ax = table_ix + dst_row_sz;
    int32_t *table_iy = table_ax + dst_row_sz;
    int32_t *table_ay = table_iy + dst_height;
    int32_t *table_p[2];
    ptrdiff_t previous = -2, index;
    int32_t *pb = NULL;
    uint8_t *ps = NULL;
    const int LINEAR_SHIFT = 4;
    const int LINEAR_ROUND_TERM = 1 << (LINEAR_SHIFT - 1);
    const int BILINEAR_SHIFT = LINEAR_SHIFT * 2;
    const int BILINEAR_ROUND_TERM = 1 << (BILINEAR_SHIFT - 1);
    const int FRACTION_RANGE = 1 << LINEAR_SHIFT;
    const double FRACTION_ROUND_TERM = 0.5 / FRACTION_RANGE;

    BIP_CHECK_SIZE(src_width);
    BIP_CHECK_SIZE(src_height);
    BIP_CHECK_PTR(src);
    BIP_CHECK_SIZE(dst_width);
    BIP_CHECK_SIZE(dst_height);
    BIP_CHECK_PTR(dst);
    if (depth < 1 || depth > 4) {
        fprintf(stderr,
                "resize_bilinear: 'depth' value must be >= 1 and <= 4\n");
        return BIP_INVALID_PARAMETER;
    }

    table_p[0] = (int32_t *)(table_ay + dst_height);
    table_p[1] = table_p[0] + dst_row_sz;

    // Y direction
    for (i = 0; i < dst_height; ++i) {
        alpha = (float)((i + 0.5) * y_scale - 0.5);
        index = (ptrdiff_t)floor(alpha);
        alpha -= index;
        if (index < 0) {
            index = 0;
            alpha = 0;
        }

        if (index > (ptrdiff_t)src_height - 2) {
            index = src_height - 2;
            alpha = 1;
        }

        table_iy[i] = (int32_t)(index);
        table_ay[i] = (int32_t)(alpha * FRACTION_RANGE + 0.5);
    }

    // X direction
    for (i = 0; i < dst_width; ++i) {
        alpha = (float)((i + 0.5) * x_scale - 0.5);
        index = (ptrdiff_t)floor(alpha);
        alpha -= index;
        if (index < 0) {
            index = 0;
            alpha = 0;
        }

        if (index > (ptrdiff_t)src_width - 2) {
            index = src_width - 2;
            alpha = 1;
        }

        for (c = 0; c < depth; ++c) {
            table_ix[i * depth + c] = (int32_t)(depth * index + c);
            table_ax[i * depth + c] = (int32_t)(alpha * FRACTION_RANGE + 0.5);
        }
    }

    for (y = 0; y < dst_height; ++y) {
        k = 0;
        if ((ptrdiff_t)table_iy[y] == previous)
            k = 2;
        else if ((ptrdiff_t)table_iy[y] == previous + 1) {
            bh_swap(table_p[0], table_p[1], int32_t *);
            k = 1;
        }
        previous = (ptrdiff_t)table_iy[y];
        for (; k < 2; k++) {
            pb = table_p[k];
            ps = src + ((ptrdiff_t)table_iy[y] + k) * src_stride;
            for (x = 0; x < dst_row_sz; ++x) {
                pb[x] =
                    (ps[table_ix[x]] << LINEAR_SHIFT) +
                    (ps[table_ix[x] + depth] - ps[table_ix[x]]) * table_ax[x];
            }
        }

        if (table_ay[y] == 0) {
            for (x = 0; x < dst_row_sz; ++x)
                dst[x] =
                    ((table_p[0][x] << LINEAR_SHIFT) + BILINEAR_ROUND_TERM) >>
                    BILINEAR_SHIFT;
        } else if (table_ay[y] == FRACTION_RANGE) {
            for (x = 0; x < dst_row_sz; ++x)
                dst[x] =
                    ((table_p[1][x] << LINEAR_SHIFT) + BILINEAR_ROUND_TERM) >>
                    BILINEAR_SHIFT;
        } else {
            for (x = 0; x < dst_row_sz; ++x) {
                dst[x] = ((table_p[0][x] << LINEAR_SHIFT) +
                          (table_p[1][x] - table_p[0][x]) * table_ay[y] +
                          BILINEAR_ROUND_TERM) >>
                         BILINEAR_SHIFT;
            }
        }
        dst += dst_stride;
    }

    bh_free(table);
    return BIP_SUCCESS;
}

bip_status bip_rotate_image(uint8_t *src, size_t src_width, size_t src_height,
                            size_t src_stride, uint8_t *dst, size_t dst_width,
                            size_t dst_height, size_t dst_stride, size_t depth,
                            float angle, int32_t center_x, int32_t center_y,
                            bip_interpolation interpolation) {
    size_t x, y, k;
    int32_t cosa, sina, tmp[4], x_map, y_map, index, rx, ry;
    float x_diff, y_diff;

    BIP_CHECK_SIZE(src_width);
    BIP_CHECK_SIZE(src_height);
    BIP_CHECK_PTR(src);
    BIP_CHECK_SIZE(dst_width);
    BIP_CHECK_SIZE(dst_height);
    BIP_CHECK_PTR(dst);

    cosa = (int32_t)(cos(angle) * 65536);
    sina = (int32_t)(sin(angle) * 65536);
    tmp[0] = center_x << 16;
    tmp[1] = center_y << 16;

    switch (interpolation) {
        case NEAREST_NEIGHBOR:
            for (y = 0; y < dst_height; ++y) {
                tmp[2] = (y - center_y);
                for (x = 0; x < dst_width; ++x) {
                    tmp[3] = (x - center_x);
                    x_map =
                        (cosa * tmp[3] - sina * tmp[2] + tmp[0] + 32768) >> 16;
                    y_map =
                        (sina * tmp[3] + cosa * tmp[2] + tmp[1] + 32768) >> 16;
                    if (x_map >= 0 && x_map < (src_width - 1) && y_map >= 0 &&
                        y_map < (src_height - 1)) {
                        for (k = 0; k < depth; ++k) {
                            dst[x * depth + k] =
                                src[(y_map * src_width + x_map) * depth + k];
                        }
                    } else {
                        for (k = 0; k < depth; ++k) {
                            dst[x * depth + k] = 0;
                        }
                    }
                }
                dst += dst_stride;
            }
            break;
        case BILINEAR:
            for (y = 0; y < dst_height; ++y) {
                tmp[2] = (y - center_y);
                for (x = 0; x < dst_width; ++x) {
                    tmp[3] = (x - center_x);
                    rx = (cosa * tmp[3] - sina * tmp[2] + tmp[0]);
                    ry = (sina * tmp[3] + cosa * tmp[2] + tmp[1]);
                    x_map = rx >> 16;
                    y_map = ry >> 16;
                    x_diff = (float)(rx - (x_map << 16)) / 65536;
                    y_diff = (float)(ry - (y_map << 16)) / 65536;
                    index = (y_map * src_width + x_map);
                    if (x_map >= 0 && x_map < (src_width - 1) && y_map >= 0 &&
                        y_map < (src_height - 1)) {
                        for (k = 0; k < depth; ++k) {
                            uint8_t level = (uint8_t)(
                                (float)src[index * depth + k] * (1 - x_diff) *
                                    (1 - y_diff) +
                                (float)src[(index + 1) * depth + k] * (x_diff) *
                                    (1 - y_diff) +
                                (float)src[(index + src_width) * depth + k] *
                                    (1 - x_diff) * (y_diff) +
                                (float)src[(index + src_width + 1) * depth +
                                           k] *
                                    (x_diff) * (y_diff));
                            dst[x * depth + k] = (uint8_t)level;
                        }
                    } else {
                        for (k = 0; k < depth; ++k) {
                            dst[x * depth + k] = (uint8_t)0;
                        }
                    }
                }
                dst += dst_stride;
            }
            break;
        default:
            fprintf(stderr,
                    "[ERROR] lcv_mat_rotate: interpolation type unknown\n");
            break;
    }

    return BIP_SUCCESS;
}

bip_status bip_invert_image(uint8_t *src, size_t width, size_t height,
                            size_t depth, size_t src_stride, uint8_t *dst,
                            size_t dst_stride) {
    size_t x, y;

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width * depth; ++x) {
            dst[x] = ~src[x];
        }
        src += src_stride;
        dst += dst_stride;
    }

    return BIP_SUCCESS;
}

bip_status bip_fliph_image(uint8_t *src, size_t width, size_t height,
                           size_t depth, size_t src_stride, uint8_t *dst,
                           size_t dst_stride) {
    size_t x, y, c;

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (c = 0; c < depth; ++c) {
                dst[x * depth + c] = src[(width - 1 - x) * depth + c];
            }
        }
        src += src_stride;
        dst += dst_stride;
    }

    return BIP_SUCCESS;
}

bip_status bip_convert_u8_to_f32(uint8_t *src, size_t width, size_t height,
                                 size_t depth, size_t src_stride, float *dst) {
    size_t x, y;
    float u8_to_f32unit[256] = {
        0.000000f, 0.003922f, 0.007843f, 0.011765f, 0.015686f, 0.019608f,
        0.023529f, 0.027451f, 0.031373f, 0.035294f, 0.039216f, 0.043137f,
        0.047059f, 0.050980f, 0.054902f, 0.058824f, 0.062745f, 0.066667f,
        0.070588f, 0.074510f, 0.078431f, 0.082353f, 0.086275f, 0.090196f,
        0.094118f, 0.098039f, 0.101961f, 0.105882f, 0.109804f, 0.113725f,
        0.117647f, 0.121569f, 0.125490f, 0.129412f, 0.133333f, 0.137255f,
        0.141176f, 0.145098f, 0.149020f, 0.152941f, 0.156863f, 0.160784f,
        0.164706f, 0.168627f, 0.172549f, 0.176471f, 0.180392f, 0.184314f,
        0.188235f, 0.192157f, 0.196078f, 0.200000f, 0.203922f, 0.207843f,
        0.211765f, 0.215686f, 0.219608f, 0.223529f, 0.227451f, 0.231373f,
        0.235294f, 0.239216f, 0.243137f, 0.247059f, 0.250980f, 0.254902f,
        0.258824f, 0.262745f, 0.266667f, 0.270588f, 0.274510f, 0.278431f,
        0.282353f, 0.286275f, 0.290196f, 0.294118f, 0.298039f, 0.301961f,
        0.305882f, 0.309804f, 0.313725f, 0.317647f, 0.321569f, 0.325490f,
        0.329412f, 0.333333f, 0.337255f, 0.341176f, 0.345098f, 0.349020f,
        0.352941f, 0.356863f, 0.360784f, 0.364706f, 0.368627f, 0.372549f,
        0.376471f, 0.380392f, 0.384314f, 0.388235f, 0.392157f, 0.396078f,
        0.400000f, 0.403922f, 0.407843f, 0.411765f, 0.415686f, 0.419608f,
        0.423529f, 0.427451f, 0.431373f, 0.435294f, 0.439216f, 0.443137f,
        0.447059f, 0.450980f, 0.454902f, 0.458824f, 0.462745f, 0.466667f,
        0.470588f, 0.474510f, 0.478431f, 0.482353f, 0.486275f, 0.490196f,
        0.494118f, 0.498039f, 0.501961f, 0.505882f, 0.509804f, 0.513725f,
        0.517647f, 0.521569f, 0.525490f, 0.529412f, 0.533333f, 0.537255f,
        0.541176f, 0.545098f, 0.549020f, 0.552941f, 0.556863f, 0.560784f,
        0.564706f, 0.568627f, 0.572549f, 0.576471f, 0.580392f, 0.584314f,
        0.588235f, 0.592157f, 0.596078f, 0.600000f, 0.603922f, 0.607843f,
        0.611765f, 0.615686f, 0.619608f, 0.623529f, 0.627451f, 0.631373f,
        0.635294f, 0.639216f, 0.643137f, 0.647059f, 0.650980f, 0.654902f,
        0.658824f, 0.662745f, 0.666667f, 0.670588f, 0.674510f, 0.678431f,
        0.682353f, 0.686275f, 0.690196f, 0.694118f, 0.698039f, 0.701961f,
        0.705882f, 0.709804f, 0.713725f, 0.717647f, 0.721569f, 0.725490f,
        0.729412f, 0.733333f, 0.737255f, 0.741176f, 0.745098f, 0.749020f,
        0.752941f, 0.756863f, 0.760784f, 0.764706f, 0.768627f, 0.772549f,
        0.776471f, 0.780392f, 0.784314f, 0.788235f, 0.792157f, 0.796078f,
        0.800000f, 0.803922f, 0.807843f, 0.811765f, 0.815686f, 0.819608f,
        0.823529f, 0.827451f, 0.831373f, 0.835294f, 0.839216f, 0.843137f,
        0.847059f, 0.850980f, 0.854902f, 0.858824f, 0.862745f, 0.866667f,
        0.870588f, 0.874510f, 0.878431f, 0.882353f, 0.886275f, 0.890196f,
        0.894118f, 0.898039f, 0.901961f, 0.905882f, 0.909804f, 0.913725f,
        0.917647f, 0.921569f, 0.925490f, 0.929412f, 0.933333f, 0.937255f,
        0.941176f, 0.945098f, 0.949020f, 0.952941f, 0.956863f, 0.960784f,
        0.964706f, 0.968627f, 0.972549f, 0.976471f, 0.980392f, 0.984314f,
        0.988235f, 0.992157f, 0.996078f, 1.000000f};

    BIP_CHECK_SIZE(width);
    BIP_CHECK_SIZE(height);
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width * depth; ++x) {
            dst[x] = u8_to_f32unit[src[x]];
        }
        src += src_stride;
        dst += width * depth;
    }

    return BIP_SUCCESS;
}

/* Local Binary Patterns */
bip_status bip_lbp_estimate(uint8_t *src, size_t width, size_t height,
                            size_t src_stride, uint8_t *dst,
                            size_t dst_stride) {
    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);
    BIP_CHECK_SIZE(width);
    BIP_CHECK_SIZE(height);

    memset(dst, 0, width);

    src += src_stride;
    dst += dst_stride;

    for (size_t y = 1; y < height - 1; ++y) {
        dst[0] = 0;
        for (size_t x = 1; x < width - 1; ++x) {
            dst[x] = (src[x - src_stride - 1] >= src[x]) << 0 |
                     (src[x - src_stride] >= src[x]) << 1 |
                     (src[x - src_stride + 1] >= src[x]) << 2 |
                     (src[x + 1] >= src[x]) << 3 |
                     (src[x + src_stride + 1] >= src[x]) << 4 |
                     (src[x + src_stride] >= src[x]) << 5 |
                     (src[x + src_stride - 1] >= src[x]) << 6 |
                     (src[x - 1] >= src[x]) << 7;
        }
        dst[width - 1] = 0;
        src += src_stride;
        dst += dst_stride;
    }

    memset(dst, 0, width);

    return BIP_SUCCESS;
}

bip_status bip_lbp_histogram_features(uint8_t *src, size_t src_width,
                                      size_t src_height, size_t src_stride,
                                      float *feat, int32_t norm,
                                      bip_lbp_mapping map) {
    size_t x, y, i, bsz;
    uint32_t histo[256] = {0};
    uint8_t lut[256] = {
        0,  1,  2,  3,  4,  58, 5,  6,  7,  58, 58, 58, 8,  58, 9,  10, 11, 58,
        58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15, 16, 58, 58, 58,
        58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 17, 58, 58, 58, 58, 58,
        58, 58, 18, 58, 58, 58, 19, 58, 20, 21, 22, 58, 58, 58, 58, 58, 58, 58,
        58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
        58, 58, 58, 58, 58, 58, 23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
        58, 58, 58, 58, 24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58,
        27, 28, 29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33,
        58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34, 58, 58,
        58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
        58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35, 36, 37, 58, 38, 58, 58,
        58, 39, 58, 58, 58, 58, 58, 58, 58, 40, 58, 58, 58, 58, 58, 58, 58, 58,
        58, 58, 58, 58, 58, 58, 58, 41, 42, 43, 58, 44, 58, 58, 58, 45, 58, 58,
        58, 58, 58, 58, 58, 46, 47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53,
        54, 55, 56, 57};

    BIP_CHECK_PTR(src);
    BIP_CHECK_SIZE(src_width);
    BIP_CHECK_SIZE(src_height);

    bsz = src_width * src_height;
    switch (map) {
        case BIP_LBP_MAP_NONE:
            memset(histo, 0, sizeof(uint32_t) * 256);
            for (y = 0; y < src_height; ++y) {
                for (x = 0; x < src_width; ++x) {
                    histo[src[x]]++;
                }
                src += src_stride;
            }
            if (!norm) {
                for (i = 0; i < 256; ++i) feat[i] = (float)histo[i];
            } else {
                for (i = 0; i < 256; ++i) feat[i] = (float)histo[i] / bsz;
            }
            break;
        case BIP_LBP_MAP_UNIFORM:
            memset(histo, 0, sizeof(uint32_t) * 59);
            for (y = 0; y < src_height; ++y) {
                for (x = 0; x < src_width; ++x) {
                    histo[lut[src[x]]]++;
                }
                src += src_stride;
            }
            if (!norm) {
                for (i = 0; i < 59; ++i) feat[i] = (float)histo[i];
            } else {
                for (i = 0; i < 59; ++i) feat[i] = (float)histo[i] / bsz;
            }
            break;
        default:
            return BIP_INVALID_PARAMETER;
    }

    return BIP_SUCCESS;
}

/* Sobel */
bip_status bip_sobel(uint8_t *src, size_t width, size_t height, size_t depth,
                     size_t src_stride, float *dst, size_t dst_stride) {
    size_t x, y, row_sz;
    uint8_t pix = 0;
    uint8_t *srcm = NULL, *srcp = NULL;
    int32_t dx, dy;

    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);
    BIP_CHECK_SIZE(width);
    BIP_CHECK_SIZE(height);

    dst_stride /= sizeof(float);
    row_sz = depth * width;
    // 1st row
    srcp = src + src_stride;
    for (x = 0; x < depth; ++x) {
        dx = 2 * src[x + depth] + src[x + depth] + srcp[x + depth] -
             2 * src[x] - src[x] - srcp[x];
        dy = 2 * src[x] + src[x + depth] + src[x] - 2 * srcp[x] -
             srcp[x + depth] - srcp[x];
        dst[x] = (float)sqrt((float)(dx * dx + dy * dy));
    }
    for (; x < row_sz - depth; ++x) {
        dx = 2 * src[x + depth] + src[x + depth] + srcp[x + depth] -
             2 * src[x - depth] - src[x - depth] - srcp[x - depth];
        dy = 2 * src[x] + src[x + depth] + src[x - depth] - 2 * srcp[x] -
             srcp[x + depth] - srcp[x - depth];
        dst[x] = (float)sqrt((float)(dx * dx + dy * dy));
    }
    for (; x < row_sz; ++x) {
        dx = 2 * src[x] + src[x] + srcp[x] - 2 * src[x - depth] -
             src[x - depth] - srcp[x - depth];
        dy = 2 * src[x] + src[x] + src[x - depth] - 2 * srcp[x] - srcp[x] -
             srcp[x - depth];
        dst[x] = (float)sqrt((float)(dx * dx + dy * dy));
    }
    src += src_stride;
    dst += dst_stride;
    // Middle rows
    for (y = 1; y < height - 1; ++y) {
        srcm = src - src_stride;
        srcp = src + src_stride;
        for (x = 0; x < depth; ++x) {
            dx = 2 * src[x + depth] + srcm[x + depth] + srcp[x + depth] -
                 2 * src[x] - srcm[x] - srcp[x];
            dy = 2 * srcm[x] + srcm[x + depth] + srcm[x] - 2 * srcp[x] -
                 srcp[x + depth] - srcp[x];
            dst[x] = (float)sqrt((float)(dx * dx + dy * dy));
        }
        for (; x < row_sz - depth; ++x) {
            dx = 2 * src[x + depth] + srcm[x + depth] + srcp[x + depth] -
                 2 * src[x - depth] - srcm[x - depth] - srcp[x - depth];
            dy = 2 * srcm[x] + srcm[x + depth] + srcm[x - depth] - 2 * srcp[x] -
                 srcp[x + depth] - srcp[x - depth];
            dst[x] = (float)sqrt((float)(dx * dx + dy * dy));
        }
        for (; x < row_sz; ++x) {
            dx = 2 * src[x] + srcm[x] + srcp[x] - 2 * src[x - depth] -
                 srcm[x - depth] - srcp[x - depth];
            dy = 2 * srcm[x] + srcm[x] + srcm[x - depth] - 2 * srcp[x] -
                 srcp[x] - srcp[x - depth];
            dst[x] = (float)sqrt((float)(dx * dx + dy * dy));
        }
        dst += dst_stride;
        src += src_stride;
    }
    // Last row
    srcm = src - src_stride;
    for (x = 0; x < depth; ++x) {
        dx = 2 * src[x + depth] + srcm[x + depth] + src[x + depth] -
             2 * src[x] - srcm[x] - src[x];
        dy = 2 * srcm[x] + srcm[x + depth] + srcm[x] - 2 * src[x] -
             src[x + depth] - src[x];
        dst[x] = (float)sqrt((float)(dx * dx + dy * dy));
    }
    for (; x < row_sz - depth; ++x) {
        dx = 2 * src[x + depth] + srcm[x + depth] + src[x + depth] -
             2 * src[x - depth] - srcm[x - depth] - src[x - depth];
        dy = 2 * srcm[x] + srcm[x + depth] + srcm[x - depth] - 2 * src[x] -
             src[x + depth] - src[x - depth];
        dst[x] = (float)sqrt((float)(dx * dx + dy * dy));
    }
    for (; x < row_sz; ++x) {
        dx = 2 * src[x] + srcm[x] + srcp[x] - 2 * src[x - depth] -
             srcm[x - depth] - srcp[x - depth];
        dy = 2 * srcm[x] + srcm[x] + srcm[x - depth] - 2 * src[x] - src[x] -
             src[x - depth];
        dst[x] = (float)sqrt((float)(dx * dx + dy * dy));
    }

    return BIP_SUCCESS;
}

/* Median 3x3 */
bip_status bip_median_3x3(uint8_t *src, size_t width, size_t height,
                          size_t src_stride, uint8_t *dst, size_t dst_stride) {
    size_t x, y;
    uint8_t *src0 = NULL, *srcm = NULL, *srcp = NULL;
    uint8_t tmp[9] = {0}, t;
#define _sort(a, b)      \
    {                    \
        if ((a) > (b)) { \
            t = (a);     \
            (a) = (b);   \
            (b) = t;     \
        }                \
    }

    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);
    BIP_CHECK_SIZE(width);
    BIP_CHECK_SIZE(height);

    // 1st row
    src0 = src;
    srcp = src + src_stride;
    *dst = *src;
    src0++;
    srcp++;
    for (x = 1; x < width - 1; ++x) {
        tmp[3] = (*--src0);
        tmp[4] = (*++src0);
        tmp[5] = (*++src0);
        tmp[6] = (*--srcp);
        tmp[7] = (*++srcp);
        tmp[8] = (*++srcp);
        tmp[0] = tmp[3];
        tmp[1] = tmp[4];
        tmp[2] = tmp[5];
        _sort(tmp[1], tmp[2]);
        _sort(tmp[4], tmp[5]);
        _sort(tmp[7], tmp[8]);
        _sort(tmp[0], tmp[1]);
        _sort(tmp[3], tmp[4]);
        _sort(tmp[6], tmp[7]);
        _sort(tmp[1], tmp[2]);
        _sort(tmp[4], tmp[5]);
        _sort(tmp[7], tmp[8]);
        _sort(tmp[0], tmp[3]);
        _sort(tmp[5], tmp[8]);
        _sort(tmp[4], tmp[7]);
        _sort(tmp[3], tmp[6]);
        _sort(tmp[1], tmp[4]);
        _sort(tmp[2], tmp[5]);
        _sort(tmp[4], tmp[7]);
        _sort(tmp[4], tmp[2]);
        _sort(tmp[6], tmp[4]);
        _sort(tmp[4], tmp[2]);
        dst[x] = tmp[4];
    }
    dst[width - 1] = *src0;
    src += src_stride;
    dst += dst_stride;
    // Middle rows
    for (y = 1; y < height - 1; ++y) {
        src0 = src;
        srcm = src - src_stride;
        srcp = src + src_stride;
        *dst = *src;
        src0++;
        srcm++;
        srcp++;
        for (x = 1; x < width - 1; ++x) {
            tmp[0] = (*--srcm);
            tmp[1] = (*++srcm);
            tmp[2] = (*++srcm);
            tmp[3] = (*--src0);
            tmp[4] = (*++src0);
            tmp[5] = (*++src0);
            tmp[6] = (*--srcp);
            tmp[7] = (*++srcp);
            tmp[8] = (*++srcp);
            _sort(tmp[1], tmp[2]);
            _sort(tmp[4], tmp[5]);
            _sort(tmp[7], tmp[8]);
            _sort(tmp[0], tmp[1]);
            _sort(tmp[3], tmp[4]);
            _sort(tmp[6], tmp[7]);
            _sort(tmp[1], tmp[2]);
            _sort(tmp[4], tmp[5]);
            _sort(tmp[7], tmp[8]);
            _sort(tmp[0], tmp[3]);
            _sort(tmp[5], tmp[8]);
            _sort(tmp[4], tmp[7]);
            _sort(tmp[3], tmp[6]);
            _sort(tmp[1], tmp[4]);
            _sort(tmp[2], tmp[5]);
            _sort(tmp[4], tmp[7]);
            _sort(tmp[4], tmp[2]);
            _sort(tmp[6], tmp[4]);
            _sort(tmp[4], tmp[2]);
            dst[x] = tmp[4];
        }
        dst[width - 1] = *src0;
        src += src_stride;
        dst += dst_stride;
    }
    // Last row
    src0 = src;
    srcm = src - src_stride;
    *dst = *src;
    src0++;
    srcm++;
    for (x = 1; x < width - 1; ++x) {
        tmp[0] = (*--srcm);
        tmp[1] = (*++srcm);
        tmp[2] = (*++srcm);
        tmp[3] = (*--src0);
        tmp[4] = (*++src0);
        tmp[5] = (*++src0);
        tmp[6] = tmp[3];
        tmp[7] = tmp[4];
        tmp[8] = tmp[5];
        _sort(tmp[1], tmp[2]);
        _sort(tmp[4], tmp[5]);
        _sort(tmp[7], tmp[8]);
        _sort(tmp[0], tmp[1]);
        _sort(tmp[3], tmp[4]);
        _sort(tmp[6], tmp[7]);
        _sort(tmp[1], tmp[2]);
        _sort(tmp[4], tmp[5]);
        _sort(tmp[7], tmp[8]);
        _sort(tmp[0], tmp[3]);
        _sort(tmp[5], tmp[8]);
        _sort(tmp[4], tmp[7]);
        _sort(tmp[3], tmp[6]);
        _sort(tmp[1], tmp[4]);
        _sort(tmp[2], tmp[5]);
        _sort(tmp[4], tmp[7]);
        _sort(tmp[4], tmp[2]);
        _sort(tmp[6], tmp[4]);
        _sort(tmp[4], tmp[2]);
        dst[x] = tmp[4];
    }
    dst[width - 1] = *src0;

    return BIP_SUCCESS;
}

/* Gaussian Blur 3x3 */
bip_status bip_gaussian_blur_3x3(uint8_t *src, size_t width, size_t height,
                                 size_t depth, size_t src_stride, uint8_t *dst,
                                 size_t dst_stride) {
    size_t x, y, row_sz;
    uint8_t pix = 0;
    uint8_t *srcm = NULL, *srcp = NULL;

    BIP_CHECK_PTR(src);
    BIP_CHECK_PTR(dst);
    BIP_CHECK_SIZE(width);
    BIP_CHECK_SIZE(height);

    row_sz = depth * width;
    // 1st row
    srcp = src + src_stride;
    for (x = 0; x < depth; ++x)
        dst[x] = ((src[x] + 2 * src[x] + src[x + depth] +
                   2 * (src[x] + 2 * src[x] + src[x + depth]) + srcp[x] +
                   2 * srcp[x] + srcp[x + depth]) +
                  8) >>
                 4;
    for (; x < row_sz - depth; ++x)
        dst[x] = ((src[x - depth] + 2 * src[x] + src[x + depth] +
                   2 * (src[x - depth] + 2 * src[x] + src[x + depth]) +
                   srcp[x - depth] + 2 * srcp[x] + srcp[x + depth]) +
                  8) >>
                 4;
    for (; x < row_sz; ++x)
        dst[x] = ((src[x - depth] + 2 * src[x] + src[x] +
                   2 * (src[x - depth] + 2 * src[x] + src[x]) +
                   srcp[x - depth] + 2 * srcp[x] + srcp[x]) +
                  8) >>
                 4;
    src += src_stride;
    dst += dst_stride;
    // Middle rows
    for (y = 1; y < height - 1; ++y) {
        srcm = src - src_stride;
        srcp = src + src_stride;
        for (x = 0; x < depth; ++x)
            dst[x] = ((srcm[x] + 2 * srcm[x] + srcm[x + depth] +
                       2 * (src[x] + 2 * src[x] + src[x + depth]) + srcp[x] +
                       2 * srcp[x] + srcp[x + depth]) +
                      8) >>
                     4;
        for (; x < row_sz - depth; ++x)
            dst[x] = ((srcm[x - depth] + 2 * srcm[x] + srcm[x + depth] +
                       2 * (src[x - depth] + 2 * src[x] + src[x + depth]) +
                       srcp[x - depth] + 2 * srcp[x] + srcp[x + depth]) +
                      8) >>
                     4;
        for (; x < row_sz; ++x)
            dst[x] = ((srcm[x - depth] + 2 * srcm[x] + srcm[x] +
                       2 * (src[x - depth] + 2 * src[x] + src[x]) +
                       srcp[x - depth] + 2 * srcp[x] + srcp[x]) +
                      8) >>
                     4;

        dst += dst_stride;
        src += src_stride;
    }
    // Last row
    srcm = src - src_stride;
    for (x = 0; x < depth; ++x)
        dst[x] = ((srcm[x] + 2 * srcm[x] + srcm[x + depth] +
                   2 * (src[x] + 2 * src[x] + src[x + depth]) + src[x] +
                   2 * src[x] + src[x + depth]) +
                  8) >>
                 4;
    for (; x < row_sz - depth; ++x)
        dst[x] = ((srcm[x - depth] + 2 * srcm[x] + srcm[x + depth] +
                   2 * (src[x - depth] + 2 * src[x] + src[x + depth]) +
                   src[x - depth] + 2 * src[x] + src[x + depth]) +
                  8) >>
                 4;
    for (; x < row_sz; ++x)
        dst[x] = ((srcm[x - depth] + 2 * srcm[x] + srcm[x] +
                   2 * (src[x - depth] + 2 * src[x] + src[x]) + src[x - depth] +
                   2 * src[x] + src[x]) +
                  8) >>
                 4;

    return BIP_SUCCESS;
}

const char *bip_status_string(bip_status status) {
    switch (status) {
        case BIP_SUCCESS:
            return "Success";
        case BIP_INVALID_PTR:
            return "Invalid pointer";
        case BIP_INVALID_SIZE:
            return "Invalid parameter size";
        case BIP_INVALID_PARAMETER:
            return "Invalid parameter";
        case BIP_UNKNOWN_ERROR:
            return "Unknown error";
        default:
            return "Unknown error";
    }
}

/* helpers for loading/writing images with stb_image */
#ifdef BIP_USE_STB_IMAGE
bip_status bip_load_image(char *filename, uint8_t **src, int32_t *src_width,
                          int32_t *src_height, int32_t *src_depth) {
    int w, h, c;
    unsigned char *val = stbi_load(filename, &w, &h, &c, 0);
    if (!val) {
        fprintf(stderr, "[ERROR] Cannot load file image %s\nSTB error: %s\n",
                filename, stbi_failure_reason());
        return BIP_UNKNOWN_ERROR;
    }
    *src = val;
    *src_width = w;
    *src_height = h;
    *src_depth = c;
    return BIP_SUCCESS;
}

bip_status bip_load_image_from_memory(unsigned char *buffer, int buffer_size,
                                      uint8_t **src, int32_t *src_width,
                                      int32_t *src_height, int32_t *src_depth) {
    int w, h, c;
    unsigned char *val =
        stbi_load_from_memory(buffer, buffer_size, &w, &h, &c, 0);
    if (!val) {
        fprintf(stderr,
                "[ERROR] Cannot load image from buffer\nSTB error: %s\n",
                stbi_failure_reason());
        return BIP_UNKNOWN_ERROR;
    }
    *src = val;
    *src_width = w;
    *src_height = h;
    *src_depth = c;
    return BIP_SUCCESS;
}

bip_status bip_write_image(char *filename, uint8_t *src, int32_t src_width,
                           int32_t src_height, int32_t src_depth,
                           int32_t src_stride) {
    BIP_CHECK_PTR(src);
    stbi_write_png(filename, src_width, src_height, src_depth, src, src_stride);
    return BIP_SUCCESS;
}

bip_status bip_write_image_to_memory(unsigned char **buffer,
                                     int32_t *buffer_size, uint8_t *src,
                                     int32_t src_width, int32_t src_height,
                                     int32_t src_depth, int32_t src_stride) {
    unsigned char *p_buf = NULL;
    int sz = 0;

    BIP_CHECK_PTR(src);

    p_buf = stbi_write_png_to_mem(src, src_stride, src_width, src_height,
                                  src_depth, &sz);
    *buffer_size = sz;
    *buffer = p_buf;

    return BIP_SUCCESS;
}

bip_status bip_write_float_image(char *filename, float *src, int32_t src_width,
                                 int32_t src_height, int32_t src_depth,
                                 int32_t src_stride) {
    int32_t x, y;
    uint8_t *buffer =
        (uint8_t *)calloc(src_width * src_height * src_depth, sizeof(uint8_t));

    BIP_CHECK_PTR(src);

    src_stride /= sizeof(float);
    for (y = 0; y < src_height; ++y) {
        for (x = 0; x < src_width * src_depth; ++x)
            buffer[x] = (uint8_t)bh_clamp(255.0f * src[x], 0, 255);
        buffer += src_width * src_depth;
        src += src_stride;
    }
    buffer -= src_width * src_height * src_depth;
    if (stbi_write_png(filename, src_width, src_height, src_depth, buffer,
                       src_stride) != 1) {
        fprintf(stderr, "[ERROR] Failed to write image %s\n", filename);
        return -1;
    }
    bh_free(buffer);
    return BIP_SUCCESS;
}

bip_status bip_write_float_image_norm(char *filename, float *src,
                                      int32_t src_width, int32_t src_height,
                                      int32_t src_depth, int32_t src_stride) {
    int32_t x, y;
    uint8_t *buffer =
        (uint8_t *)calloc(src_width * src_height * src_depth, sizeof(uint8_t));
    float min_src, max_src, norm;

    BIP_CHECK_PTR(src);
    min_src = src[0];
    max_src = src[0];

    src_stride /= sizeof(float);
    for (y = 0; y < src_height; ++y) {
        for (x = 0; x < src_width; ++x) {
            min_src = bh_min(min_src, src[x]);
            max_src = bh_max(max_src, src[x]);
        }
        src += src_stride;
    }
    src -= src_stride * src_height;
    if (max_src - min_src > 0)
        norm = 255.0f / (max_src - min_src);
    else
        norm = 0.0f;
    for (y = 0; y < src_height; ++y) {
        for (x = 0; x < src_width * src_depth; ++x)
            buffer[x] = (uint8_t)bh_clamp(norm * (src[x] - min_src), 0, 255);
        buffer += src_width * src_depth;
        src += src_stride;
    }
    buffer -= src_width * src_height * src_depth;
    if (stbi_write_png(filename, src_width, src_height, src_depth, buffer,
                       src_stride) != 1) {
        fprintf(stderr, "[ERROR] Failed to write image %s\n", filename);
        return -1;
    }
    bh_free(buffer);
    return BIP_SUCCESS;
}

bip_status bip_write_double_image(char *filename, double *src,
                                  int32_t src_width, int32_t src_height,
                                  int32_t src_depth, int32_t src_stride) {
    int32_t x, y;
    uint8_t *buffer =
        (uint8_t *)calloc(src_width * src_height * src_depth, sizeof(uint8_t));

    BIP_CHECK_PTR(src);

    src_stride /= sizeof(double);
    for (y = 0; y < src_height; ++y) {
        for (x = 0; x < src_width; ++x)
            buffer[x] = (uint8_t)bh_clamp(255.0f * src[x], 0, 255);
        buffer += src_width * src_depth;
        src += src_stride;
    }
    buffer -= src_width * src_height * src_depth;
    if (stbi_write_png(filename, src_width, src_height, src_depth, buffer,
                       src_stride) != 1) {
        fprintf(stderr, "[ERROR] Failed to write image %s\n", filename);
        return -1;
    }
    bh_free(buffer);
    return BIP_SUCCESS;
}
#endif
