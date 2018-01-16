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

#include "bcnn_mat.h"

#include <bh/bh.h>
#include <bh/bh_error.h>
#include <bh/bh_mem.h>

#include "bcnn/bcnn.h"


int bcnn_fill_f32(int n, float a, float *x)
{
    int i;
    for (i = 0; i < n; ++i) {
        x[i] = a;
    }
    return 0;
}


int bcnn_copy_f32(int n, float *x, float *y)
{
    memcpy(y, x, n * sizeof(float));
    return 0;
}

int bcnn_axpy(int n, float a, float *x, float *y)
{
#ifndef BCNN_USE_SSE2
    int i;
    for (i = 0; i < n; ++i)
        y[i] += a * x[i];
#else
    int i, nd, nm;
    __m128 sum0;
    __m128 sum1;
    __m128 reg0, reg1, reg2, reg3;
    __m128 areg = _mm_set1_ps(a);
    __m128 prod;
    int data_is_aligned = bh_is_aligned32(x) & bh_is_aligned32(y);

    nd = n / 8 * 8;
    nm = n % 8;
    if (data_is_aligned) {
        for (i = 0; i < nd; i += 8) {
            reg0 = _mm_load_ps(x + 0);
            reg1 = _mm_load_ps(x + 4);
            reg2 = _mm_load_ps(y + 0);
            reg3 = _mm_load_ps(y + 4);
            prod = _mm_mul_ps(reg0, areg);
            sum0 = _mm_add_ps(prod, reg2);
            prod = _mm_mul_ps(reg1, areg);
            sum1 = _mm_add_ps(prod, reg3);
            _mm_store_ps(y + 0, sum0);
            _mm_store_ps(y + 4, sum1);
            x += 8;
            y += 8;
        }
    }
    else {
        for (i = 0; i < nd; i += 8) {
            reg0 = _mm_loadu_ps(x + 0);
            reg1 = _mm_loadu_ps(x + 4);
            reg2 = _mm_loadu_ps(y + 0);
            reg3 = _mm_loadu_ps(y + 4);
            prod = _mm_mul_ps(reg0, areg);
            sum0 = _mm_add_ps(prod, reg2);
            prod = _mm_mul_ps(reg1, areg);
            sum1 = _mm_add_ps(prod, reg3);
            _mm_storeu_ps(y + 0, sum0);
            _mm_storeu_ps(y + 4, sum1);
            x += 8;
            y += 8;
        }
    }
    for (i = 0; i < nm; ++i)
        y[i] += a * x[i];
#endif
    return 0;
}


int bcnn_axpby(int n, float a, float *x, float b, float *y)
{
#ifndef BCNN_USE_SSE2
    int i;
    for (i = 0; i < n; ++i)
        y[i] = a * x[i] + b * y[i];
#else
    int i, nd, nm;
    __m128 sum0;
    __m128 sum1;
    __m128 reg0, reg1, reg2, reg3;
    __m128 areg = _mm_set1_ps(a);
    __m128 breg = _mm_set1_ps(b);
    __m128 prod0, prod1;
    int data_is_aligned = bh_is_aligned32(x) & bh_is_aligned32(y);

    nd = n / 8 * 8;
    nm = n % 8;
    if (data_is_aligned) {
        for (i = 0; i < nd; i += 8) {
            reg0 = _mm_load_ps(x + 0);
            reg1 = _mm_load_ps(x + 4);
            reg2 = _mm_load_ps(y + 0);
            reg3 = _mm_load_ps(y + 4);
            prod0 = _mm_mul_ps(reg0, areg);
            prod1 = _mm_mul_ps(reg2, breg);
            sum0 = _mm_add_ps(prod0, prod1);
            prod0 = _mm_mul_ps(reg1, areg);
            prod1 = _mm_mul_ps(reg3, breg);
            sum1 = _mm_add_ps(prod0, prod1);
            _mm_store_ps(y + 0, sum0);
            _mm_store_ps(y + 4, sum1);
            x += 8;
            y += 8;
        }
    }
    else {
        for (i = 0; i < nd; i += 8) {
            reg0 = _mm_loadu_ps(x + 0);
            reg1 = _mm_loadu_ps(x + 4);
            reg2 = _mm_loadu_ps(y + 0);
            reg3 = _mm_loadu_ps(y + 4);
            prod0 = _mm_mul_ps(reg0, areg);
            prod1 = _mm_mul_ps(reg2, breg);
            sum0 = _mm_add_ps(prod0, prod1);
            prod0 = _mm_mul_ps(reg1, areg);
            prod1 = _mm_mul_ps(reg3, breg);
            sum1 = _mm_add_ps(prod0, prod1);
            _mm_storeu_ps(y + 0, sum0);
            _mm_storeu_ps(y + 4, sum1);
            x += 8;
            y += 8;
        }
    }
    for (i = 0; i < nm; ++i)
        y[i] = a * x[i] + b * y[i];
#endif
    return 0;
}

int bcnn_pow(int n, float *x, float a, float *y)
{
    int i;
    for (i = 0; i < n; ++i) {
        y[i] = powf(x[i], a);
    }
    return 0;
}


int bcnn_vadd(int n, float *a, float *b, float *y)
{
#ifndef BCNN_USE_SSE2
    int i;
    for (i = 0; i < n; ++i) {
        y[i] = a[i] + b[i];
    }
#else
    int i, nd, nm;
    __m128 r0, r1, r2, r3;

    nd = n / 8 * 8;
    nm = n % 8;
    for (i = 0; i < nd; i += 8) {
        r0 = _mm_loadu_ps(a);
        r1 = _mm_loadu_ps(a + 4);
        r2 = _mm_loadu_ps(b);
        r3 = _mm_loadu_ps(b + 4);
        r0 = _mm_add_ps(r0, r2);
        r1 = _mm_add_ps(r1, r3);
        _mm_storeu_ps(y, r0);
        _mm_storeu_ps(y + 4, r1);
        a += 8;
        b += 8;
        y += 8;
    }
    for (i = 0; i < nm; ++i) {
        y[i] = a[i] + b[i];
    }
#endif
    return 0;
}

int bcnn_vsub(int n, float *a, float *b, float *y)
{
#ifndef BCNN_USE_SSE2
    int i;
    for (i = 0; i < n; ++i) {
        y[i] = a[i] - b[i];
    }
#else
    int i, nd, nm;
    __m128 r0, r1, r2, r3;

    nd = n / 8 * 8;
    nm = n % 8;
    for (i = 0; i < nd; i += 8) {
        r0 = _mm_loadu_ps(a);
        r1 = _mm_loadu_ps(a + 4);
        r2 = _mm_loadu_ps(b);
        r3 = _mm_loadu_ps(b + 4);
        r0 = _mm_sub_ps(r0, r2);
        r1 = _mm_sub_ps(r1, r3);
        _mm_storeu_ps(y, r0);
        _mm_storeu_ps(y + 4, r1);
        a += 8;
        b += 8;
        y += 8;
    }
    for (i = 0; i < nm; ++i) {
        y[i] = a[i] - b[i];
    }
#endif
    return 0;
}

int bcnn_vmul(int n, float *a, float *b, float *y)
{	
#ifndef BCNN_USE_SSE2
    int i;
    for (i = 0; i < n; ++i) {
        y[i] = a[i] * b[i];
    }
#else
    int i, nd, nm;
    __m128 r0, r1, r2, r3;

    nd = n / 8 * 8;
    nm = n % 8;
    for (i = 0; i < nd; i += 8) {
        r0 = _mm_loadu_ps(a);
        r1 = _mm_loadu_ps(a + 4);
        r2 = _mm_loadu_ps(b);
        r3 = _mm_loadu_ps(b + 4);
        r0 = _mm_mul_ps(r0, r2);
        r1 = _mm_mul_ps(r1, r3);
        _mm_storeu_ps(y, r0);
        _mm_storeu_ps(y + 4, r1);
        a += 8;
        b += 8;
        y += 8;
    }
    for (i = 0; i < nm; ++i) {
        y[i] = a[i] * b[i];
    }
#endif
    return 0;
}

int bcnn_vdiv(int n, float *a, float *b, float *y)
{
#ifndef BCNN_USE_SSE2
    int i;
    for (i = 0; i < n; ++i) {
        if (bh_abs(b[i]) >  0.00001f)
            y[i] = a[i] / b[i];
        else
            y[i] = 0.0f;
    }
#else
    int i, nd, nm;
    __m128 r0, r1, r2, r3;

    nd = n / 8 * 8;
    nm = n % 8;
    for (i = 0; i < nd; i += 8) {
        r0 = _mm_loadu_ps(a);
        r1 = _mm_loadu_ps(a + 4);
        r2 = _mm_loadu_ps(b);
        r3 = _mm_loadu_ps(b + 4);
        r0 = _mm_div_ps(r0, r2);
        r1 = _mm_div_ps(r1, r3);
        _mm_storeu_ps(y, r0);
        _mm_storeu_ps(y + 4, r1);
        a += 8;
        b += 8;
        y += 8;
    }
    for (i = 0; i < nm; ++i) {
        if (bh_abs(b[i]) >  0.00001f)
            y[i] = a[i] / b[i];
        else
            y[i] = 0.0f;
    }
#endif
    return 0;
}

int bcnn_scal(int n, float a, float *x)
{
#ifndef BCNN_USE_SSE2
    int i;
    if (a == 0.0f) {
        memset(x, 0, n * sizeof(float));
    }
    else if (a != 1.0f) {
        for (i = 0; i < n; ++i) 
            x[i] *= a;
    }
#else
    int i, nd, nm;
    __m128 reg0, reg1;
    __m128 areg = _mm_set1_ps(a);
    __m128 prod;
    int data_is_aligned = bh_is_aligned32(x);

    if (a == 0.0f) {
        memset(x, 0, n * sizeof(float));
    }
    else if (a != 1.0f) {
        nd = n / 8 * 8;
        nm = n % 8;
        if (data_is_aligned) {
            for (i = 0; i < nd; i += 8) {
                reg0 = _mm_load_ps(x + 0);
                reg1 = _mm_load_ps(x + 4);
                prod = _mm_mul_ps(reg0, areg);
                _mm_store_ps(x + 0, prod);
                prod = _mm_mul_ps(reg1, areg);
                _mm_store_ps(x + 4, prod);
                x += 8;
            }
        }
        else {
            for (i = 0; i < nd; i += 8) {
                reg0 = _mm_loadu_ps(x + 0);
                reg1 = _mm_loadu_ps(x + 4);
                prod = _mm_mul_ps(reg0, areg);
                _mm_storeu_ps(x + 0, prod);
                prod = _mm_mul_ps(reg1, areg);
                _mm_storeu_ps(x + 4, prod);
                x += 8;
            }
        }
        for (i = 0; i < nm; ++i)
            x[i] *= a;
    }
#endif
    return 0;
}


int bcnn_add_scalar(int n, float a, float *x)
{
#ifndef BCNN_USE_SSE2
    int i;
    for (i = 0; i < n; ++i) {
        x[i] += a;
    }
#else
    int i, nd, nm;
    __m128 reg0, reg1;
    __m128 areg = _mm_set1_ps(a);
    __m128 prod;
    int data_is_aligned = bh_is_aligned32(x);

    if (a == 0.0f) {
        return 0;
    }
    else if (a != 1.0f) {
        nd = n / 8 * 8;
        nm = n % 8;
        if (data_is_aligned) {
            for (i = 0; i < nd; i += 8) {
                reg0 = _mm_load_ps(x + 0);
                reg1 = _mm_load_ps(x + 4);
                prod = _mm_add_ps(reg0, areg);
                _mm_store_ps(x + 0, prod);
                prod = _mm_add_ps(reg1, areg);
                _mm_store_ps(x + 4, prod);
                x += 8;
            }
        }
        else {
            for (i = 0; i < nd; i += 8) {
                reg0 = _mm_loadu_ps(x + 0);
                reg1 = _mm_loadu_ps(x + 4);
                prod = _mm_add_ps(reg0, areg);
                _mm_storeu_ps(x + 0, prod);
                prod = _mm_add_ps(reg1, areg);
                _mm_storeu_ps(x + 4, prod);
                x += 8;
            }
        }
        for (i = 0; i < nm; ++i)
            x[i] += a;
    }
#endif
    return 0;
}

float bcnn_dot(int n, float *x, float *y)
{
#ifndef BCNN_USE_SSE2
    int i;
    float dot = 0;
    for (i = 0; i < n; ++i) 
        dot += x[i] * y[i];
    return dot;
#else
    int i, nd, nm;
    float sum = 0;
    float sum_res[4];
    __m128 sum_r = _mm_setzero_ps();
    __m128 r0, r1, r2, r3;

    nd = n / 8 * 8;
    nm = n % 8;
    for (i = 0; i < nd; i += 8) {
        r0 = _mm_loadu_ps(x);
        r1 = _mm_loadu_ps(y);
        r2 = _mm_loadu_ps(x + 4);
        r3 = _mm_loadu_ps(y + 4);
        r0 = _mm_mul_ps(r0, r1);
        r2 = _mm_mul_ps(r2, r3);
        sum_r = _mm_add_ps(sum_r, r0);
        sum_r = _mm_add_ps(sum_r, r2);
        x += 8;
        y += 8;
    }
    _mm_storeu_ps(sum_res, sum_r);
    sum += sum_res[0] + sum_res[1] + sum_res[2] + sum_res[3];
    for (i = 0; i < nm; ++i)
        sum += x[i] * y[i];
    return sum;
#endif
}

int bcnn_vsum(int n, float *x, float *sum)
{
#ifndef BCNN_USE_SSE2
    int i;
    float s = 0.0f;
    for (i = 0; i < n; ++i) 
        s += x[i];
    *(sum) = s;
#else
    int i, nd, nm;
    float s = 0.0f;
    float sum_res[4];
    __m128 sum_r = _mm_setzero_ps();
    __m128 r0, r1;

    nd = n / 8 * 8;
    nm = n % 8;
    for (i = 0; i < nd; i += 8) {
        r0 = _mm_loadu_ps(x);
        r1 = _mm_loadu_ps(x + 4);
        sum_r = _mm_add_ps(sum_r, r0);
        sum_r = _mm_add_ps(sum_r, r1);
        x += 8;
    }
    _mm_storeu_ps(sum_res, sum_r);
    s += sum_res[0] + sum_res[1] + sum_res[2] + sum_res[3];
    for (i = 0; i < nm; ++i)
        s += x[i];
    *(sum) = s;
#endif
    return 0;
}


int bcnn_gemv(int trans_a, int m, int n, float alpha, float *a, float *x,
    float beta, float *y)
{
    int i, j;
#ifdef BCNN_USE_SSE2
    int nd, md;
    __m128 apart, mula, mul0, areg, xreg, yreg;
    float sum[4] = { 0 };
#endif

    if (!trans_a) {
        if (beta != 1.0f) {
            for (i = 0; i < m; ++i) {
                y[i] *= beta;
            }
        }
#ifndef BCNN_USE_SSE2	
        for (i = 0; i < m; ++i) {
            for (j = 0; j < n; ++j) {
                y[i] += alpha * a[i * n + j] * x[j];
            }
        }
#else
        nd = n / 4 * 4;
        apart = _mm_set1_ps(alpha);
        for (i = 0; i < m; ++i) {
            memset(sum, 0, 4 * sizeof(float));
            yreg = _mm_setzero_ps();
            for (j = 0; j < nd; j += 4) {
                areg = _mm_loadu_ps(&a[i * n + j]);
                xreg = _mm_loadu_ps(&x[j]);
                mula = _mm_mul_ps(apart, areg);
                mul0 = _mm_mul_ps(xreg, mula);
                yreg = _mm_add_ps(yreg, mul0);
            }
            _mm_storeu_ps(sum, yreg);
            y[i] += sum[0] + sum[1] + sum[2] + sum[3];
            for (; j < n; ++j)
                y[i] += alpha * a[i * n + j] * x[j];
        }
#endif
    }
    else {
        if (beta != 1.0f) {
            for (i = 0; i < n; ++i) {
                y[i] *= beta;
            }
        }
#ifndef BCNN_USE_SSE2	
        for (i = 0; i < n; ++i) {
            for (j = 0; j < m; ++j) {
                y[i] += alpha * a[i * m + j] * x[j];
            }
        }
#else
        md = m / 4 * 4;
        apart = _mm_set1_ps(alpha);
        for (i = 0; i < n; ++i) {
            memset(sum, 0, 4 * sizeof(float));
            yreg = _mm_setzero_ps();
            for (j = 0; j < md; j += 4) {
                areg = _mm_loadu_ps(&a[i * m + j]);
                xreg = _mm_loadu_ps(&x[j]);
                mula = _mm_mul_ps(apart, areg);
                mul0 = _mm_mul_ps(xreg, mula);
                yreg = _mm_add_ps(yreg, mul0);
            }
            _mm_storeu_ps(sum, yreg);
            y[i] += sum[0] + sum[1] + sum[2] + sum[3];
            for (; j < m; ++j)
                y[i] += alpha * a[i * m + j] * x[j];
        }
#endif
    }
    return 0;
}


// General Matrix-Matrix multiplication
//			   ldb n
//			_________
//			|		|
//			|   B	| k
//			|		|
//	________|_______|
//  |		|		|
// m|		|		| m
//	|	A	|	C	|
//	|_______|_______|
//	lda k	  ldc n
//
int bcnn_gemm(int trans_a, int trans_b, int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc)
{
    int i, j, k;
#ifdef BCNN_USE_SSE2
    int nd, kd;
    __m128 apart, bpart, cpart, mul0, areg, mula;
    float sum[4] = { 0 };
#endif

    if (BETA != 1.0f) {
        for (i = 0; i < M; ++i){
            for (j = 0; j < N; ++j){
                C[i * ldc + j] *= BETA;
            }
        }
    }

    if (!trans_a && !trans_b) {
#ifndef BCNN_USE_SSE2
        for (i = 0; i < M; ++i){
            for (k = 0; k < K; ++k){
                register float tmp = ALPHA * A[i * lda + k];
                for (j = 0; j < N; ++j) {
                    C[i * ldc + j] += tmp * B[k * ldb + j];
                }
            }
        }
#else
        apart = _mm_setzero_ps();
        cpart = _mm_setzero_ps();
        nd = N / 4 * 4;

        for (i = 0; i < M; ++i) {
            for (k = 0; k < K; ++k){
                register float tmp = ALPHA * A[i * lda + k];
                apart = _mm_set1_ps(tmp);
                for (j = 0; j < nd; j += 4) {
                    cpart = _mm_loadu_ps(&C[i * ldc + j]);
                    bpart = _mm_loadu_ps(&B[k * ldb + j]);
                    mul0 = _mm_mul_ps(bpart, apart);
                    cpart = _mm_add_ps(cpart, mul0);
                    _mm_storeu_ps(&C[i * ldc + j], cpart);
                }
                for (; j < N; ++j)
                    C[i * ldc + j] += tmp * B[k * ldb + j];
            }
        }
#endif
    }
    else if (trans_a && !trans_b) {
#ifndef BCNN_USE_SSE2
        for (i = 0; i < M; ++i){
            for (k = 0; k < K; ++k){
                register float tmp = ALPHA * A[k * lda + i];
                for (j = 0; j < N; ++j) {
                    C[i * ldc + j] += tmp * B[k * ldb + j];
                }
            }
        }
#else
        cpart = _mm_setzero_ps();
        nd = N / 4 * 4;

        for (i = 0; i < M; ++i) {
            for (k = 0; k < K; ++k){
                register float tmp = ALPHA * A[k*lda + i];
                apart = _mm_set1_ps(tmp);
                for (j = 0; j < nd; j += 4) {
                    cpart = _mm_loadu_ps(&C[i * ldc + j]);
                    bpart = _mm_loadu_ps(&B[k * ldb + j]);
                    mul0 = _mm_mul_ps(bpart, apart);
                    cpart = _mm_add_ps(cpart, mul0);
                    _mm_storeu_ps(&C[i * ldc + j], cpart);
                }
                for (; j < N; ++j)
                    C[i * ldc + j] += tmp * B[k * ldb + j];
            }
        }
#endif
    }
    else if (!trans_a && trans_b) {
#ifndef BCNN_USE_SSE2
        float sum = 0;
        for (i = 0; i < M; ++i){
            for (j = 0; j < N; ++j){
                sum = 0;
                for (k = 0; k < K; ++k){
                    sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
                }
                C[i * ldc + j] += sum;
            }
        }
#else
        areg = _mm_set1_ps(ALPHA);
        kd = K / 4 * 4;

        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                memset(sum, 0, 4 * sizeof(float));
                cpart = _mm_setzero_ps();
                for (k = 0; k < kd; k += 4) {
                    apart = _mm_loadu_ps(&A[i * lda + k]);
                    bpart = _mm_loadu_ps(&B[j * ldb + k]);
                    mula = _mm_mul_ps(apart, areg);
                    mul0 = _mm_mul_ps(bpart, mula);
                    cpart = _mm_add_ps(cpart, mul0);
                }
                _mm_storeu_ps(sum, cpart);
                C[i * ldc + j] += sum[0] + sum[1] + sum[2] + sum[3];
                for (; k < K; ++k)
                    C[i * ldc + j] += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }
        }
#endif
    }
    else {
        for (i = 0; i < M; ++i){
            for (k = 0; k < K; ++k){
                register float tmp = ALPHA * A[i * lda + k];
                for (j = 0; j < N; ++j) {
                    C[i * ldc + j] += tmp * B[k * ldb + j];
                }
            }
        }
    }
    return 0;
}


// From https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetNaive
static uint32_t portable_popcnt(uint32_t x)
{
    x = x - ((x >> 1) & (uint32_t)~(uint32_t)0/3);                          
    x = (x & (uint32_t)~(uint32_t)0/15*3) + ((x >> 2) & (uint32_t)~(uint32_t)0/15*3);      
    x = (x + (x >> 4)) & (uint32_t)~(uint32_t)0/255*15;                      
    return (uint32_t)(x * ((uint32_t)~(uint32_t)0/255)) >> (sizeof(uint32_t) - 1) * 8; 
}

#ifdef _MSC_VER
#  include <intrin.h>
#  include <nmmintrin.h>
#  define __builtin_popcount _mm_popcnt_u32
#else
# define __builtin_popcount portable_popcnt
#endif

int bcnn_xnor_gemm(int trans_a, int trans_b, int M, int N, int K, float ALPHA,
    uint32_t *A, int lda,
    uint32_t *B, int ldb,
    float BETA,
    float *C, int ldc)
{
    int m,k,n;
  
    for (m = 0; m < M; ++m) {
        for (k = 0; k < K; k++) {
            uint32_t A_PART = A[m * lda + k];
            for (n = 0; n < N; ++n) {
#ifdef _MSC_VER
                C[m*ldc+n] += _mm_popcnt_u32(~(A_PART ^ B[k*ldb+n]));
#else
                C[m * ldc + n] += portable_popcnt(~(A_PART ^ B[k*ldb+n]));
#endif
            }
        }
    }
    
    return 0;
}


float bcnn_l2_distance(float *x, float *y, int n)
{
    float dist = 0.0f;
    int i;
#ifdef BCNN_USE_SSE2
    int data_is_aligned = bh_is_aligned32(x) & bh_is_aligned32(y);
    __m128 vx0, vy0, vx1, vy1, vdiff0, vdiff1;
    __m128 vdist = _mm_set1_ps(0.0f);
    float dist4f[4] = { 0.0f };
    int nd, nm;

    nd = n / 8 * 8;
    nm = n % 8;
    if (data_is_aligned) {
        for (i = 0; i < nd; i += 8) {
            vx0 = _mm_load_ps(x);
            vy0 = _mm_load_ps(y);
            vx1 = _mm_load_ps(x + 4);
            vy1 = _mm_load_ps(y + 4);
            vdiff0 = _mm_sub_ps(vx0, vy0);
            vdiff0 = _mm_mul_ps(vdiff0, vdiff0);
            vdiff1 = _mm_sub_ps(vx1, vy1);
            vdiff1 = _mm_mul_ps(vdiff1, vdiff1);
            vdist = _mm_add_ps(vdist, vdiff0);
            vdist = _mm_add_ps(vdist, vdiff1);
            x += 8;
            y += 8;
        }
        _mm_store_ps(dist4f, vdist);
    }
    else {
        for (i = 0; i < nd; i += 8) {
            vx0 = _mm_loadu_ps(x);
            vy0 = _mm_loadu_ps(y);
            vx1 = _mm_loadu_ps(x + 4);
            vy1 = _mm_loadu_ps(y + 4);
            vdiff0 = _mm_sub_ps(vx0, vy0);
            vdiff0 = _mm_mul_ps(vdiff0, vdiff0);
            vdiff1 = _mm_sub_ps(vx1, vy1);
            vdiff1 = _mm_mul_ps(vdiff1, vdiff1);
            vdist = _mm_add_ps(vdist, vdiff0);
            vdist = _mm_add_ps(vdist, vdiff1);
            x += 8;
            y += 8;
        }
        _mm_storeu_ps(dist4f, vdist);
    }
    dist += dist4f[0] + dist4f[1] + dist4f[2] + dist4f[3];
    for (i = 0; i < nm; ++i)
        dist += (x[i] - y[i]) * (x[i] - y[i]);
#else
    for (i = 0; i < n; ++i)
        dist += (x[i] - y[i]) * (x[i] - y[i]);
#endif
    return dist; 
}

float bcnn_sqrdiff_vs(float *x, float a, int n)
{
    float dist = 0.0f;
    int i;
#ifndef BCNN_USE_SSE2
    for (i = 0; i < n; ++i)
        dist += (x[i] - a) * (x[i] - a);
#else
    int data_is_aligned = bh_is_aligned32(x);
    __m128 vx0, vx1, vdiff0, vdiff1;
    __m128 vdist = _mm_set1_ps(0.0f);
    __m128 areg = _mm_set1_ps(a);
    float dist4f[4] = { 0.0f };
    int nd, nm;

    nd = n / 8 * 8;
    nm = n % 8;
    if (data_is_aligned) {
        for (i = 0; i < nd; i += 8) {
            vx0 = _mm_load_ps(x);
            vx1 = _mm_load_ps(x + 4);
            vdiff0 = _mm_sub_ps(vx0, areg);
            vdiff0 = _mm_mul_ps(vdiff0, vdiff0);
            vdiff1 = _mm_sub_ps(vx1, areg);
            vdiff1 = _mm_mul_ps(vdiff1, vdiff1);
            vdist = _mm_add_ps(vdist, vdiff0);
            vdist = _mm_add_ps(vdist, vdiff1);
            x += 8;
        }
        _mm_store_ps(dist4f, vdist);
    }
    else {
        for (i = 0; i < nd; i += 8) {
            vx0 = _mm_loadu_ps(x);
            vx1 = _mm_loadu_ps(x + 4);
            vdiff0 = _mm_sub_ps(vx0, areg);
            vdiff0 = _mm_mul_ps(vdiff0, vdiff0);
            vdiff1 = _mm_sub_ps(vx1, areg);
            vdiff1 = _mm_mul_ps(vdiff1, vdiff1);
            vdist = _mm_add_ps(vdist, vdiff0);
            vdist = _mm_add_ps(vdist, vdiff1);
            x += 8;
        }
        _mm_storeu_ps(dist4f, vdist);
    }
    dist += dist4f[0] + dist4f[1] + dist4f[2] + dist4f[3];
    for (i = 0; i < nm; ++i)
        dist += (x[i] - a) * (x[i] - a);
#endif

    return dist; 
}

float bcnn_shiftdot(int n, float *x, float a, float *y, float b)
{
#ifndef BCNN_USE_SSE2
    int i;
    float dot = 0;
    for (i = 0; i < n; ++i) 
        dot += (x[i] - a) * (y[i] - b);
    return dot;
#else
    int i, nd, nm;
    float sum = 0;
    float sum_res[4];
    __m128 sum_r = _mm_setzero_ps();
    __m128 r0, r1, r2, r3;
    __m128 areg = _mm_set1_ps(a);
    __m128 breg = _mm_set1_ps(b);

    nd = n / 8 * 8;
    nm = n % 8;
    for (i = 0; i < nd; i += 8) {
        r0 = _mm_loadu_ps(x);
        r1 = _mm_loadu_ps(y);
        r2 = _mm_loadu_ps(x + 4);
        r3 = _mm_loadu_ps(y + 4);
        r0 = _mm_sub_ps(r0, areg);
        r1 = _mm_sub_ps(r1, breg);
        r2 = _mm_sub_ps(r2, areg);
        r3 = _mm_sub_ps(r3, breg);
        r0 = _mm_mul_ps(r0, r1);
        r2 = _mm_mul_ps(r2, r3);
        sum_r = _mm_add_ps(sum_r, r0);
        sum_r = _mm_add_ps(sum_r, r2);
        x += 8;
        y += 8;
    }
    _mm_storeu_ps(sum_res, sum_r);
    sum += sum_res[0] + sum_res[1] + sum_res[2] + sum_res[3];
    for (i = 0; i < nm; ++i)
        sum += (x[i] - a) * (y[i] - b);
    return sum;
#endif
}


int bcnn_varnorm(int n, float *a, float c, float *y)
{	
#ifndef BCNN_USE_SSE2
    int i;
    for (i = 0; i < n; ++i) {
        y[i] *= c / (a[i] * sqrtf(a[i]) + 0.00001f);
    }
#else
    int i, nd, nm;
    __m128 r0, r1, reg0, reg1;
    __m128 creg = _mm_set1_ps(c);
    __m128 epsreg = _mm_set1_ps(0.00001f);

    nd = n / 8 * 8;
    nm = n % 8;
    for (i = 0; i < nd; i += 8) {
        reg0 = _mm_loadu_ps(y);
        reg1 = _mm_loadu_ps(y + 4);
        r0 = _mm_loadu_ps(a);
        r1 = _mm_loadu_ps(a + 4);
        r0 = _mm_mul_ps(reg0, _mm_div_ps(creg, _mm_add_ps(_mm_mul_ps(r0, _mm_sqrt_ps(r0)), epsreg)));
        r1 = _mm_mul_ps(reg1, _mm_div_ps(creg, _mm_add_ps(_mm_mul_ps(r1, _mm_sqrt_ps(r1)), epsreg)));
        _mm_storeu_ps(y, r0);
        _mm_storeu_ps(y + 4, r1);
        a += 8;
        y += 8;
    }
    for (i = 0; i < nm; ++i) {
        y[i] *= c / (a[i] * sqrtf(a[i]) + 0.00001f);
    }
#endif
    return 0;
}


int bcnn_varmean(int n, float *m, float a, float *var)
{	
#ifndef BCNN_USE_SSE2
    int i;
    for (i = 0; i < n; ++i) {
        var[i] = var[i] * a - m[i] * m[i];
    }
#else
    int i, nd, nm;
    __m128 r0, r1, reg0, reg1;
    __m128 areg = _mm_set1_ps(a);
    

    nd = n / 8 * 8;
    nm = n % 8;
    for (i = 0; i < nd; i += 8) {
        reg0 = _mm_loadu_ps(var);
        reg1 = _mm_loadu_ps(var + 4);
        r0 = _mm_loadu_ps(m);
        r1 = _mm_loadu_ps(m + 4);
        r0 = _mm_sub_ps(_mm_mul_ps(reg0, areg), _mm_mul_ps(r0, r0));
        r1 = _mm_sub_ps(_mm_mul_ps(reg1, areg), _mm_mul_ps(r1, r1));
        _mm_storeu_ps(var, r0);
        _mm_storeu_ps(var + 4, r1);
        m += 8;
        var += 8;
    }
    for (i = 0; i < nm; ++i) {
        var[i] = var[i] * a - m[i] * m[i];
    }
#endif
    return 0;
}







