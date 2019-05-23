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

#ifndef BCNN_MAT_H
#define BCNN_MAT_H

#include <stdio.h>

/* OpenMP */
#ifdef BCNN_USE_OPENMP
#include <omp.h>
#endif
/* Cuda include */
#ifdef BCNN_USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#ifdef BCNN_USE_CUDNN
#include <cudnn.h>
#endif
#endif
/* ARM Neon */
#ifdef BCNN_USE_NEON
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#else
#undef BCNN_USE_NEON
#endif
#endif
/* x86 SSE/AVX */
#ifdef BCNN_USE_AVX
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if (defined(__aarch64__))
#define GEMM_ALIGN 0x03fffUL
#define MC 128
#define KC 240
#define NC 12288
#define MR 4
#define NR 4
#else
#ifdef BCNN_USE_NEON
#define MC 384
#define KC 384
#define NC 4096
#if (defined(__aarch64__))  // legacy
#define MR 8
#define NR 8
#else
#define MR 4
#define NR 4
#endif  // __aarch64__
#else
#define MC 128
#define KC 384
#define NC 4096
#define MR 8
#define NR 8
#endif  // BCNN_USE_NEON
#endif  // __aarch64__

#if (defined(__aarch64__))
#define CONV_TILED 16  // 16
#else
#define CONV_TILED 8
#endif  // __aarch64__

#define CONV3x3_SRC_BLOCK 64
#define CONV3x3_WEIGHT_BLOCK 256
#define CONV3x3_SRC_BLOCK_VEC 16
#define CONV3x3_SRC_BLOCK_UNIT 3
#define CONV3x3_BLOCK_UNIT 4

typedef struct bcnn_gemm_context {
#if !defined(BCNN_USE_BLAS) && !defined(BCNN_USE_CUDA)
#if (defined(__aarch64__))  // use sgemm_openblas
    float buffer_a[(((MC + NC) * KC + GEMM_ALIGN) & ~(GEMM_ALIGN))]
        __attribute__((aligned(32)));
    float *buffer_b;
    float *buffer_c;
    float *buffer_ab;
#else
    float buffer_a[MC * KC] __attribute__((aligned(32)));
    float buffer_b[KC * NC] __attribute__((aligned(32)));
    float buffer_c[MR * NR] __attribute__((aligned(32)));
    float buffer_ab[MR * NR] __attribute__((aligned(32)));
#endif
#else
    float *buffer_a;
    float *buffer_b;
    float *buffer_c;
    float *buffer_ab;
#endif
} bcnn_gemm_context;

typedef struct bv_float4_t {
#if defined(BCNN_USE_AVX)
    __m128 val;
#elif defined(BCNN_USE_NEON)
    float32x4_t val;
#else
    float val[4];
#endif
} bv_float4;

/* Matrix computation routines */
int bcnn_fill_f32(int n, float a, float *x);
int bcnn_copy_f32(int n, float *x, float *y);
int bcnn_axpy(int n, float a, float *x, float *y);
void bcnn_axpy_strided(int num_batches, float a, float *x, float *y,
                       int stride[2], int x_dim[3], int y_dim[3],
                       int min_dim[3]);
int bcnn_scal(int n, float a, float *x);
int bcnn_add_scalar(int n, float a, float *x);
int bcnn_pow(int n, float *x, float a, float *y);
float bcnn_dot(int n, float *x, float *y);
int bcnn_vsum(int n, float *x, float *sum);
int bcnn_vadd(int n, float *a, float *b, float *y);
int bcnn_vsub(int n, float *a, float *b, float *y);
int bcnn_vdiv(int n, float *a, float *b, float *y);
int bcnn_vmul(int n, float *a, float *b, float *y);
int bcnn_axpby(int n, float a, float *x, float b, float *y);
int bcnn_gemv(int trans_a, int m, int n, float alpha, float *a, float *x,
              float beta, float *y);
int bcnn_gemm(bcnn_gemm_context *ctx, int trans_a, int trans_b, int M, int N,
              int K, float ALPHA, float *A, int lda, float *B, int ldb,
              float BETA, float *C, int ldc);
float bcnn_l2_distance(float *x, float *y, int n);
float bcnn_sqrdiff_vs(float *x, float a, int n);
float bcnn_shiftdot(int n, float *x, float a, float *y, float b);
int bcnn_varnorm(int n, float *a, float c, float *y);
int bcnn_varmean(int n, float *m, float a, float *var);
void bcnn_add_bias(float *output, float *bias, int batch_size, int num_channels,
                   int spatial_size);
void bcnn_grad_bias(float *grad_bias, float *grad_data, int batch_size,
                    int num_channels, int spatial_size);
void bcnn_scales(float *output, float *scales, int batch_size, int num_channels,
                 int spatial_size);
void bcnn_grad_scales(float *x_norm, float *delta, int batch, int n, int size,
                      float *scale_updates);
void bcnn_im2col(const float *data_im, const int channels, const int height,
                 const int width, const int kernel_size, const int pad,
                 const int stride, float *data_col);
void bcnn_col2im(const float *data_col, const int channels, const int height,
                 const int width, const int kernel, const int pad,
                 const int stride, float *data_im);
void bcnn_im2col_mt(const float *data_im, const int channels, const int height,
                    const int width, const int kernel_size, const int pad,
                    const int stride, float *data_col);
void bcnn_conv3x3_convert_weights(const float *src, float *dst,
                                  int src_channels, int dst_channels);
void bcnn_conv3x3_convert_dst(const float *src, float *dst, size_t step);
void bcnn_conv3x3_convert_src(const float *src, float *dst, size_t step);
void bcnn_conv3x3s1_kernel(float *src, int src_w, int src_h, int src_c,
                           float *dst, int dst_w, int dst_h, int dst_c,
                           int batch_size, int pad, float *weights,
                           float *scales, float *biases, float *workspace,
                           int workspace_sz, int num_threads);
void bcnn_nchw_to_nc4hw4(float *dst, const float *src, size_t area,
                         size_t depth);
void bcnn_nc4hw4_to_nchw(float *dst, const float *src, size_t area,
                         size_t depth);
void bcnn_add_bias_with_relu(float *dst, const float *bias, size_t planeNumber,
                             size_t biasNumber);
void bcnn_scale_and_add_bias_with_lrelu(float *dst, const float *src,
                                        const float *bias, const float *alpha,
                                        size_t planeNumber, size_t biasNumber);

static inline bv_float4 bv_float4_load(const float *x) {
    bv_float4 v;
#if defined(BCNN_USE_AVX)
    v.val = _mm_load_ps(x);
#elif defined(BCNN_USE_NEON)
    v.val = vld1q_f32(x);
#else
    v.val[0] = x[0];
    v.val[1] = x[1];
    v.val[2] = x[2];
    v.val[3] = x[3];
#endif
    return v;
}
static inline void bv_float4_store(bv_float4 v, float *x) {
#if defined(BCNN_USE_AVX)
    _mm_store_ps(x, v.val);
#elif defined(BCNN_USE_NEON)
    vst1q_f32(x, v.val);
#else
    x[0] = v.val[0];
    x[1] = v.val[1];
    x[0] = v.val[2];
    x[1] = v.val[3];
#endif
}
static inline bv_float4 bv_float4_add(bv_float4 va, bv_float4 vb) {
    bv_float4 v;
#if defined(BCNN_USE_AVX)
    v.val = _mm_add_ps(va.val, vb.val);
#elif defined(BCNN_USE_NEON)
    v.val = vaddq_f32(va.val, vb.val);
#else
    v.val[0] = va.val[0] + vb.val[0];
    v.val[1] = va.val[1] + vb.val[1];
    v.val[2] = va.val[2] + vb.val[2];
    v.val[3] = va.val[3] + vb.val[3];
#endif
    return v;
}
static inline bv_float4 bv_float4_sub(bv_float4 va, bv_float4 vb) {
    bv_float4 v;
#if defined(BCNN_USE_AVX)
    v.val = _mm_sub_ps(va.val, vb.val);
#elif defined(BCNN_USE_NEON)
    v.val = vsubq_f32(va.val, vb.val);
#else
    v.val[0] = va.val[0] - vb.val[0];
    v.val[1] = va.val[1] - vb.val[1];
    v.val[2] = va.val[2] - vb.val[2];
    v.val[3] = va.val[3] - vb.val[3];
#endif
    return v;
}

/* Cuda kernels routines */
#ifdef BCNN_USE_CUDA

/* Math routines */
void bcnn_cuda_gemm(int trans_a, int trans_b, int m, int n, int k, float alpha,
                    float *a, int lda, float *b, int ldb, float beta, float *c,
                    int ldc);
void bcnn_cuda_gemv(int trans_a, const int m, const int n, const float alpha,
                    const float *a, const float *x, const float beta, float *y);
void bcnn_cuda_fill_f32(int n, float alpha, float *x, int incx);
void bcnn_cuda_copy_f32(int n, float *x, int incx, float *y, int incy);
void bcnn_cuda_axpy(int n, float alpha, float *x, int incx, float *y, int incy);
void bcnn_cuda_scal(int n, float alpha, float *x, int incx);
void bcnn_cuda_pow(int n, float *x, float a, float *y);
void bcnn_cuda_axpby(int n, float a, float *x, float b, float *y);
void bcnn_cuda_axpy_strided(int num_batches, float a, float *x, float *y,
                            int stride[2], int x_dim[3], int y_dim[3],
                            int min_dim[3]);
void bcnn_cuda_add_scalar(int n, float a, float *y);
void bcnn_cuda_vadd(int n, float *a, float *b, float *y);
void bcnn_cuda_vsub(int n, float *a, float *b, float *y);
void bcnn_cuda_vmul(int n, float *a, float *b, float *y);
void bcnn_cuda_vdiv(int n, float *a, float *b, float *y);

void bcnn_cuda_mean_variance_forward(float *x, int b, int c, int wxh,
                                     float *mean, float *var);
void bcnn_cuda_norm_forward(float *x, float *mean, float *variance, int b,
                            int c, int wxh);
void bcnn_cuda_mean_variance_backward(float *x, float *grad, float *mean,
                                      float *var, int b, int c, int wxh,
                                      float *mean_diff, float *var_diff);
void bcnn_cuda_norm_backward(float *x, float *mean, float *var,
                             float *mean_diff, float *var_diff, int b, int c,
                             int wxh, float *grad);

void bcnn_op_cuda_tanh(int n, float *x, float *y);
void bcnn_op_cuda_tanh_grad(int n, float *x, float *dx);
void bcnn_op_cuda_relu(int n, float *x, float *y);
void bcnn_op_cuda_relu_grad(int n, float *x, float *dx);
void bcnn_op_cuda_clamp(int n, float *x, float *y);
void bcnn_op_cuda_clamp_grad(int n, float *x, float *dx);
void bcnn_op_cuda_ramp(int n, float *x, float *y);
void bcnn_op_cuda_ramp_grad(int n, float *x, float *dx);

void bcnn_cuda_add_bias(float *output, float *bias, int batch_size,
                        int num_channels, int spatial_size);
void bcnn_cuda_grad_bias(float *grad_bias, float *grad_data, int batch_size,
                         int num_channels, int spatial_size);
void bcnn_scales_gpu(float *output, float *biases, int batch, int n, int size);
void bcnn_grad_scales_gpu(float *x_norm, float *delta, int batch, int n,
                          int size, float *scale_updates);

void bcnn_cuda_im2col(float *im, int channels, int height, int width, int ksize,
                      int stride, int pad, float *data_col);
void bcnn_cuda_col2im(float *data_col, int channels, int height, int width,
                      int ksize, int stride, int pad, float *data_im);
#endif

#ifdef __cplusplus
}
#endif

#endif  // BCNN_MAT_H
