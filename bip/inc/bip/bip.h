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

#ifndef BIP_H
#define BIP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <stdint.h>
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#if (defined(_WIN32) || defined(_WIN64))
#include <windows.h>
#elif defined(__linux__)
#include <sys/time.h>
#include <time.h>
#endif

#if defined(_MSC_VER) && (_MSC_VER > 1000)
#pragma once
#endif

#ifdef BIP_USE_SSE2
#include <emmintrin.h>  // SSE2
#endif

#if defined(__GNUC__) || (defined(_MSC_VER) && (_MSC_VER >= 1600))
#include <stdint.h>
#else
#if (_MSC_VER < 1300)
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#else
typedef signed __int8 int8_t;
typedef signed __int16 int16_t;
typedef signed __int32 int32_t;
typedef unsigned __int8 uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
#endif
typedef signed __int64 int64_t;
typedef unsigned __int64 uint64_t;
#endif

#define BIP_PI 3.14159265358979f
#define BIP_2PI 6.28318530717958f
#define BIP_HALF_PI 1.57079632679490f
#define BIP_SQRT2 1.41421356237310f
#define BIP_LOG2 1.4426950408889f

#define bip_deg2rad(x) (x) * 0.01745329252f

/**
 * \brief   Enum function returned status.
 */
typedef enum {
    BIP_SUCCESS,
    BIP_INVALID_PTR,
    BIP_INVALID_SIZE,
    BIP_INVALID_PARAMETER,
    BIP_UNKNOWN_ERROR
} bip_status;

/**
 * \brief   Enum interpolation methods.
 */
typedef enum { NEAREST_NEIGHBOR, BILINEAR } bip_interpolation;

const char *bip_status_string(bip_status status);

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

/**
 * \brief   Convert a rgb image into a gray image.
 *
 * \param   src             Pointer to input image.
 * \param   width           Input image width.
 * \param   height          Input image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image (may be the input
 * image).
 * \param   dst_stride      Output image row size in bytes.
 */
bip_status bip_rgb2gray(uint8_t *src, size_t width, size_t height,
                        size_t src_stride, uint8_t *dst, size_t dst_stride);

/**
 * \brief   Apply linear stretch to the contrast of an image.
 *  For each pixel, apply a linear stretch to the difference of the current
 * pixel intensity and the image mean intensity:
 *      dst[i] = contrast * (src[i] - mean) + mean;
 *
 * \param   src             Pointer to input image.
 * \param   src_stride      Input image row size in bytes.
 * \param   width           Input image width.
 * \param   height          Input image height.
 * \param   depth           Input image depth.
 * \param   dst             Pointer to output image (may be the input image).
 * \param   dst_stride      Output image row size in bytes.
 * \param   contrast        Stretch coefficient to be applied.
 */
bip_status bip_contrast_stretch(uint8_t *src, size_t src_stride, size_t width,
                                size_t height, size_t depth, uint8_t *dst,
                                size_t dst_stride, float contrast);

/**
 * \brief   Apply additive factor to the brightness of an image.
 *  For each pixel, apply a linear stretch to the difference of the current
 * pixel intensity and the image mean intensity:
 *      dst[i] = src[i] + bightness;
 *
 * \param   src             Pointer to input image.
 * \param   src_stride      Input image row size in bytes.
 * \param   width           Input image width.
 * \param   height          Input image height.
 * \param   depth           Input image depth.
 * \param   dst             Pointer to output image (may be the input image).
 * \param   dst_stride      Output image row size in bytes.
 * \param   contrast        Stretch coefficient to be applied.
 */
bip_status bip_image_brightness(uint8_t *src, size_t src_stride, size_t width,
                                size_t height, size_t depth, uint8_t *dst,
                                size_t dst_stride, int32_t brightness);

/**
 * \brief   Distort an image with random distortion map generated by Perlin
 * noise.
 *
 * \param   src             Pointer to input image.
 * \param   src_stride      Input image row size in bytes.
 * \param   width           Input image width.
 * \param   height          Input image height.
 * \param   depth           Input image depth.
 * \param   dst             Pointer to output image (may be the input image).
 * \param   dst_stride      Output image row size in bytes.
 * \param   distortion      Distortion coefficient.
 * \param   kx              Seed x-coordinate.
 * \param   ky              Seed y-coordinate.
 */
bip_status bip_image_perlin_distortion(uint8_t *src, size_t src_stride,
                                       size_t width, size_t height,
                                       size_t depth, uint8_t *dst,
                                       size_t dst_stride, float distortion,
                                       float kx, float ky);

/**
 * \brief   Mirror the borders of an image given an canvas.
 *
 * \param   src             Pointer to input image.
 * \param   src_width       Input image width.
 * \param   src_height      Input image height.
 * \param   src_depth       Input image depth.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image.
 * \param   dst_width       Output cropped image width.
 * \param   dst_height      Output cropped image height.
 * \param   dst_depth       Output cropped image depth.
 * \param   top             Number of top rows to be mirrored.
 * \param   bottom          Number of bottom rows to be mirrored.
 * \param   left            Number of left columns to be mirrored.
 * \param   right           Number of right columns to be mirrored.
 */
bip_status bip_mirror_borders_8u(uint8_t *src, int32_t src_width,
                                 int32_t src_height, int32_t src_depth,
                                 int32_t src_stride, uint8_t *dst,
                                 int32_t dst_width, int32_t dst_height,
                                 int32_t dst_depth, int32_t dst_stride,
                                 int32_t top, int32_t bottom, int32_t left,
                                 int32_t right);

/**
 * \brief   Mirror the borders of a float image given an canvas.
 *
 * \param   src             Pointer to input image.
 * \param   src_width       Input image width.
 * \param   src_height      Input image height.
 * \param   src_depth       Input image depth.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image.
 * \param   dst_width       Output cropped image width.
 * \param   dst_height      Output cropped image height.
 * \param   dst_depth       Output cropped image depth.
 * \param   dst_stride      Output cropped image row size in bytes.
 * \param   top             Number of top rows to be mirrored.
 * \param   bottom          Number of bottom rows to be mirrored.
 * \param   left            Number of left columns to be mirrored.
 * \param   right           Number of right columns to be mirrored.
 */
bip_status bip_mirror_borders_32f(float *src, int32_t src_width,
                                  int32_t src_height, int32_t src_depth,
                                  int32_t src_stride, float *dst,
                                  int32_t dst_width, int32_t dst_height,
                                  int32_t dst_depth, int32_t dst_stride,
                                  int32_t top, int32_t bottom, int32_t left,
                                  int32_t right);

/**
 * \brief   Crop an image.
 *
 * \param   src             Pointer to input image.
 * \param   src_width       Input image width.
 * \param   src_height      Input image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   x_ul            Upper-left corner x-coordinate of the crop.
 * \param   y_ul            Upper-left corner y-coordinate of the crop.
 * \param   dst             Pointer to output cropped image.
 * \param   dst_width       Output cropped image width.
 * \param   dst_height      Output cropped image height.
 * \param   dst_stride      Output cropped image row size in bytes.
 * \param   dst_depth       Output cropped image depth.
 */
bip_status bip_crop_image(uint8_t *src, size_t src_width, size_t src_height,
                          size_t src_stride, int32_t x_ul, int32_t y_ul,
                          uint8_t *dst, size_t dst_width, size_t dst_height,
                          size_t dst_stride, size_t depth);

/**
 * \brief   Computes the sum integral of an image.
 *
 * \param   src             Pointer to input image.
 * \param   width           Image width.
 * \param   height          Image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image.
 * \param   dst_stride      Output resized image row size in bytes.
 */
bip_status bip_image_integral(uint8_t *src, size_t width, size_t height,
                              size_t src_stride, uint32_t *dst,
                              size_t dst_stride);

/**
 * \brief   Computes the squared sum integral of an image.
 *
 * \param   src             Pointer to input image.
 * \param   width           Image width.
 * \param   height          Image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image.
 * \param   dst_stride      Output resized image row size in bytes.
 */
bip_status bip_image_square_integral(uint8_t *src, size_t width, size_t height,
                                     size_t src_stride, uint32_t *dst,
                                     size_t dst_stride, double *dst_square,
                                     size_t dst_square_stride);

/**
 * \brief   Computes a mean image using a sliding window.
 *
 * \param   src             Pointer to input image.
 * \param   width           Image width.
 * \param   height          Image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image.
 * \param   dst_stride      Output resized image row size in bytes.
 * \param   kernel_width    Sliding window width.
 * \param   kernel_height   Sliding window height.
 */
bip_status bip_image_sliding_mean(uint8_t *src, size_t width, size_t height,
                                  size_t src_stride, uint8_t *dst,
                                  size_t dst_stride, size_t kernel_width,
                                  size_t kernel_height);

/**
 * \brief   Computes a mean image and a variance image using a sliding window.
 *
 * \param   src             Pointer to input image.
 * \param   width           Image width.
 * \param   height          Image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image.
 * \param   dst_stride      Output resized image row size in bytes.
 * \param   kernel_width    Sliding window width.
 * \param   kernel_height   Sliding window height.
 */
bip_status bip_image_sliding_mean_variance(
    uint8_t *src, size_t width, size_t height, size_t src_stride, uint8_t *dst,
    size_t dst_stride, double *dst_variance, size_t dst_variance_stride,
    size_t kernel_width, size_t kernel_height);

/**
 * \brief   Compute histogram of an image.
 *
 * \param   src             Pointer to input image.
 * \param   src_width       Input image width.
 * \param   src_height      Input image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   histogram       Output computed histogram (allocated by user).
 */
bip_status bip_image_histogram(uint8_t *src, size_t src_width,
                               size_t src_height, size_t src_stride,
                               uint32_t *histogram);

/**
 * \brief   Compute the Shannon entropy of an image.
 *
 * \param   src             Pointer to input image.
 * \param   src_width       Input image width.
 * \param   src_height      Input image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   entropy         Output entropy.
 */
bip_status bip_image_entropy(uint8_t *src, size_t src_width, size_t src_height,
                             size_t src_stride, float *entropy);

/**
 * \brief   Compute automatic threshold according to Otsu's method.
 *
 * \param   src             Pointer to input image.
 * \param   src_width       Input image width.
 * \param   src_height      Input image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   thresh          Output otsu threshold.
 * \param   var             Output inter-class variance.
 */
bip_status bip_otsu(uint8_t *src, size_t src_width, size_t src_height,
                    size_t src_stride, float *thresh, float *var);

/**
 * \brief   Resize an image with bilinear interpolation.
 *
 * \param   src             Pointer to input image.
 * \param   src_width       Input image width.
 * \param   src_height      Input image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image.
 * \param   dst_width       Output resized image width.
 * \param   dst_height      Output resized image height.
 * \param   dst_stride      Output resized image row size in bytes.
 * \param   depth           Image depth.
 */
bip_status bip_resize_bilinear(uint8_t *src, size_t src_width,
                               size_t src_height, size_t src_stride,
                               uint8_t *dst, size_t dst_width,
                               size_t dst_height, size_t dst_stride,
                               size_t depth);

/**
 * \brief   Apply given rotation to an image.
 *
 * \param   src             Pointer to input image.
 * \param   src_width       Input image width.
 * \param   src_height      Input image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image.
 * \param   dst_width       Output resized image width.
 * \param   dst_height      Output resized image height.
 * \param   dst_stride      Output resized image row size in bytes.
 * \param   depth           Image depth.
 * \param   angle           Rotation angle.
 * \param   center_x        X-coord of the rotation center.
 * \param   center_y        Y-coord of the rotation center.
 * \param   interpolation   Interpolation method.
 */
bip_status bip_rotate_image(uint8_t *src, size_t src_width, size_t src_height,
                            size_t src_stride, uint8_t *dst, size_t dst_width,
                            size_t dst_height, size_t dst_stride, size_t depth,
                            float angle, int32_t center_x, int32_t center_y,
                            bip_interpolation interpolation);

/**
 * \brief   Apply gaussian blur using a 3x3 kernel.
 *
 * \param   src             Pointer to input image.
 * \param   width           Input image width.
 * \param   height          Input image height.
 * \param   depth           Image depth.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image.
 * \param   dst_stride      Output image row size in bytes.
 */
bip_status bip_gaussian_blur_3x3(uint8_t *src, size_t width, size_t height,
                                 size_t depth, size_t src_stride, uint8_t *dst,
                                 size_t dst_stride);

/**
 * \brief   Apply sobel filter and returns image of intensities of the
 * gradients.
 *
 * \param   src             Pointer to input image.
 * \param   width           Input image width.
 * \param   height          Input image height.
 * \param   depth           Image depth.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image (must be float).
 * \param   dst_stride      Output image row size in bytes.
 */
bip_status bip_sobel(uint8_t *src, size_t width, size_t height, size_t depth,
                     size_t src_stride, float *dst, size_t dst_stride);

/**
 * \brief   Apply 3x3 median filter to image. Image must be grayscale.
 *
 * \param   src             Pointer to input image.
 * \param   width           Input image width.
 * \param   height          Input image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image.
 * \param   dst_stride      Output image row size in bytes.
 */
bip_status bip_median_3x3(uint8_t *src, size_t width, size_t height,
                          size_t src_stride, uint8_t *dst, size_t dst_stride);

/**
 * \brief   Apply downsampling of a factor 2 then blurring with a 3x3 gaussian
 * kernel to an image.
 *
 * \param   src             Pointer to input image.
 * \param   src_width       Input image width.
 * \param   src_height      Input image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image.
 * \param   dst_width       Output image width.
 * \param   dst_height      Output image height.
 * \param   dst_stride      Output image row size in bytes.
 */
bip_status bip_pyramid_down(uint8_t *src, size_t src_width, size_t src_height,
                            size_t src_stride, uint8_t *dst, size_t dst_width,
                            size_t dst_height, size_t dst_stride);

/**
 * \brief   Apply upsampling of a factor 2 to an image.
 *
 * \param   src             Pointer to input image.
 * \param   src_width       Input image width.
 * \param   src_height      Input image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image.
 * \param   dst_width       Output image width.
 * \param   dst_height      Output image height.
 * \param   dst_stride      Output image row size in bytes.
 */
bip_status bip_pyramid_up(uint8_t *src, size_t src_width, size_t src_height,
                          size_t src_stride, uint8_t *dst, size_t dst_width,
                          size_t dst_height, size_t dst_stride);

/**
* \brief    Invert image (negative).
*
* \param    src             Pointer to input image.
* \param    width           Image width.
* \param    height          Image height.
* \param    depth           Image depth.
* \param    src_stride      Input image row size in bytes.
* \param    dst             Pointer to output image.
*/
bip_status bip_invert_image(uint8_t *src, size_t width, size_t height,
                            size_t depth, size_t src_stride, uint8_t *dst,
                            size_t dst_stride);

/**
* \brief    Horizontal flip image.
*
* \param    src             Pointer to input image.
* \param    width           Image width.
* \param    height          Image height.
* \param    depth           Image depth.
* \param    src_stride      Input image row size in bytes.
* \param    dst             Pointer to output image.
*/
bip_status bip_fliph_image(uint8_t *src, size_t width, size_t height,
                           size_t depth, size_t src_stride, uint8_t *dst,
                           size_t dst_stride);

/**
 * \brief   Convert from unsigned char to float image.
 *
 * \param   src             Pointer to input unsigned char image.
 * \param   width           Image width.
 * \param   height          Image height.
 * \param   depth           Image depth.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output image.
 */
bip_status bip_convert_u8_to_f32(uint8_t *src, size_t width, size_t height,
                                 size_t depth, size_t src_stride, float *dst);

/**
 * \brief   Enum of mapping types for LBP.
 */
typedef enum { BIP_LBP_MAP_NONE, BIP_LBP_MAP_UNIFORM } bip_lbp_mapping;

/**
 * \brief   Computes Local Binary Pattern (LBP) for an image.
 *
 * \param   src             Pointer to input unsigned char image.
 * \param   width           Image width.
 * \param   height          Image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   dst             Pointer to output LBP image.
 * \param   dst_stride      Ouput LBP image row size in bytes.
 */
bip_status bip_lbp_estimate(uint8_t *src, size_t width, size_t height,
                            size_t src_stride, uint8_t *dst, size_t dst_stride);

/**
 * \brief   Computes the histogram feature vector from a LBP image using a
 * specified mapping.
 *
 * \param   src             Pointer to input LBP image.
 * \param   src_width       Input image width.
 * \param   src_height      Input image height.
 * \param   src_stride      Input image row size in bytes.
 * \param   feat            Pointer to output feature vector.
 * \param   norm            Flag wether to normalize or not the feature vector.
 * If set to 0, no normalization.
 * \param   map             Type of mapping applied.
 */
bip_status bip_lbp_histogram_features(uint8_t *src, size_t src_width,
                                      size_t src_height, size_t src_stride,
                                      float *feat, int32_t norm,
                                      bip_lbp_mapping map);

#ifdef BIP_USE_STB_IMAGE
/**
* \brief    Load image from file.
*
* \param    filename            Image file path.
* \param    src                 Pointer to loaded image.
* \param    src_width           Image width.
* \param    src_height          Image height.
* \param    src_depth           Image depth.
*/
bip_status bip_load_image(char *filename, uint8_t **src, int32_t *src_width,
                          int32_t *src_height, int32_t *src_depth);

/**
* \brief    Load image from memory.

* \param    buffer              Pointer to image buffer.
* \param    buffer_size         Image buffer size.
* \param    src                 Pointer to loaded image.
* \param    src_width           Image width.
* \param    src_height          Image height.
* \param    src_depth           Image depth.
*/
bip_status bip_load_image_from_memory(unsigned char *buffer, int buffer_size,
                                      uint8_t **src, int32_t *src_width,
                                      int32_t *src_height, int32_t *src_depth);

/**
* \brief    Write image on disk.
*
* \param    filename            Image file path.
* \param    src                 Pointer to image.
* \param    src_width           Image width.
* \param    src_height          Image height.
* \param    src_depth           Image depth.
* \param    src_stride          Image row size in bytes.
*/
bip_status bip_write_image(char *filename, uint8_t *src, int32_t src_width,
                           int32_t src_height, int32_t src_depth,
                           int32_t src_stride);

/**
* \brief    Write image to memory buffer.
*
* \param    buffer              Pointer to output buffer.
* \param    buffer_size         Size of output buffer.
* \param    src                 Pointer to image.
* \param    src_width           Image width.
* \param    src_height          Image height.
* \param    src_depth           Image depth.
* \param    src_stride          Image row size in bytes.
*/
bip_status bip_write_image_to_memory(unsigned char **buffer,
                                     int32_t *buffer_size, uint8_t *src,
                                     int32_t src_width, int32_t src_height,
                                     int32_t src_depth, int32_t src_stride);

/**
* \brief Write image (as a float array) on disk.
*
* \param    filename            Image file path.
* \param    src                 Pointer to image.
* \param    src_width           Image width.
* \param    src_height          Image height.
* \param    src_depth           Image depth.
* \param    src_stride          Image row size in bytes.
*/
bip_status bip_write_float_image(char *filename, float *src, int32_t src_width,
                                 int32_t src_height, int32_t src_depth,
                                 int32_t src_stride);

bip_status bip_write_float_image_norm(char *filename, float *src,
                                      int32_t src_width, int32_t src_height,
                                      int32_t src_depth, int32_t src_stride);

/**
* \brief    Write image (as a double array) on disk.
*
* \param    filename            Image file path.
* \param    src                 Pointer to image.
* \param    src_width           Image width.
* \param    src_height          Image height.
* \param    src_depth           Image depth.
* \param    src_stride          Image row size in bytes.
*/
bip_status bip_write_double_image(char *filename, double *src,
                                  int32_t src_width, int32_t src_height,
                                  int32_t src_depth, int32_t src_stride);
#endif

#ifdef __cplusplus
}
#endif

#endif
