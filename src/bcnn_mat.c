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

#include <bh/bh_log.h>
#include <bh/bh_macros.h>
#include <bh/bh_mem.h>

#include "bcnn/bcnn.h"

int bcnn_fill_f32(int n, float a, float *x) {
    int i;
    for (i = 0; i < n; ++i) {
        x[i] = a;
    }
    return 0;
}

int bcnn_copy_f32(int n, float *x, float *y) {
    memcpy(y, x, n * sizeof(float));
    return 0;
}

int bcnn_axpy(int n, float a, float *x, float *y) {
#ifndef BCNN_USE_AVX
    int i;
    for (i = 0; i < n; ++i) y[i] += a * x[i];
#else
    int i, nd, nm;
    __m256 sum0;
    __m256 sum1;
    __m256 reg0, reg1, reg2, reg3;
    __m256 areg = _mm256_set1_ps(a);
    __m256 prod;
    int data_is_aligned = bh_is_aligned32(x) & bh_is_aligned32(y);

    nd = n / 16 * 16;
    nm = n % 16;
    if (data_is_aligned) {
        for (i = 0; i < nd; i += 16) {
            reg0 = _mm256_load_ps(x + 0);
            reg1 = _mm256_load_ps(x + 8);
            reg2 = _mm256_load_ps(y + 0);
            reg3 = _mm256_load_ps(y + 8);
            prod = _mm256_mul_ps(reg0, areg);
            sum0 = _mm256_add_ps(prod, reg2);
            prod = _mm256_mul_ps(reg1, areg);
            sum1 = _mm256_add_ps(prod, reg3);
            _mm256_store_ps(y + 0, sum0);
            _mm256_store_ps(y + 8, sum1);
            x += 16;
            y += 16;
        }
    } else {
        for (i = 0; i < nd; i += 16) {
            reg0 = _mm256_loadu_ps(x + 0);
            reg1 = _mm256_loadu_ps(x + 8);
            reg2 = _mm256_loadu_ps(y + 0);
            reg3 = _mm256_loadu_ps(y + 8);
            prod = _mm256_mul_ps(reg0, areg);
            sum0 = _mm256_add_ps(prod, reg2);
            prod = _mm256_mul_ps(reg1, areg);
            sum1 = _mm256_add_ps(prod, reg3);
            _mm256_storeu_ps(y + 0, sum0);
            _mm256_storeu_ps(y + 8, sum1);
            x += 16;
            y += 16;
        }
    }
    for (i = 0; i < nm; ++i) y[i] += a * x[i];
#endif
    return 0;
}

int bcnn_axpby(int n, float a, float *x, float b, float *y) {
#ifndef BCNN_USE_AVX
    int i;
    for (i = 0; i < n; ++i) y[i] = a * x[i] + b * y[i];
#else
    int i, nd, nm;
    __m256 sum0;
    __m256 sum1;
    __m256 reg0, reg1, reg2, reg3;
    __m256 areg = _mm256_set1_ps(a);
    __m256 breg = _mm256_set1_ps(b);
    __m256 prod0, prod1;
    int data_is_aligned = bh_is_aligned32(x) & bh_is_aligned32(y);

    nd = n / 16 * 16;
    nm = n % 16;
    if (data_is_aligned) {
        for (i = 0; i < nd; i += 16) {
            reg0 = _mm256_load_ps(x + 0);
            reg1 = _mm256_load_ps(x + 8);
            reg2 = _mm256_load_ps(y + 0);
            reg3 = _mm256_load_ps(y + 8);
            prod0 = _mm256_mul_ps(reg0, areg);
            prod1 = _mm256_mul_ps(reg2, breg);
            sum0 = _mm256_add_ps(prod0, prod1);
            prod0 = _mm256_mul_ps(reg1, areg);
            prod1 = _mm256_mul_ps(reg3, breg);
            sum1 = _mm256_add_ps(prod0, prod1);
            _mm256_store_ps(y + 0, sum0);
            _mm256_store_ps(y + 8, sum1);
            x += 16;
            y += 16;
        }
    } else {
        for (i = 0; i < nd; i += 16) {
            reg0 = _mm256_loadu_ps(x + 0);
            reg1 = _mm256_loadu_ps(x + 8);
            reg2 = _mm256_loadu_ps(y + 0);
            reg3 = _mm256_loadu_ps(y + 8);
            prod0 = _mm256_mul_ps(reg0, areg);
            prod1 = _mm256_mul_ps(reg2, breg);
            sum0 = _mm256_add_ps(prod0, prod1);
            prod0 = _mm256_mul_ps(reg1, areg);
            prod1 = _mm256_mul_ps(reg3, breg);
            sum1 = _mm256_add_ps(prod0, prod1);
            _mm256_storeu_ps(y + 0, sum0);
            _mm256_storeu_ps(y + 8, sum1);
            x += 16;
            y += 16;
        }
    }
    for (i = 0; i < nm; ++i) y[i] = a * x[i] + b * y[i];
#endif
    return 0;
}

int bcnn_pow(int n, float *x, float a, float *y) {
    int i;
    for (i = 0; i < n; ++i) {
        y[i] = powf(x[i], a);
    }
    return 0;
}

int bcnn_vadd(int n, float *a, float *b, float *y) {
#ifndef BCNN_USE_AVX
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

int bcnn_vsub(int n, float *a, float *b, float *y) {
#ifndef BCNN_USE_AVX
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

int bcnn_vmul(int n, float *a, float *b, float *y) {
#ifndef BCNN_USE_AVX
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

int bcnn_vdiv(int n, float *a, float *b, float *y) {
#ifndef BCNN_USE_AVX
    int i;
    for (i = 0; i < n; ++i) {
        if (bh_abs(b[i]) > 0.00001f)
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
        if (bh_abs(b[i]) > 0.00001f)
            y[i] = a[i] / b[i];
        else
            y[i] = 0.0f;
    }
#endif
    return 0;
}

int bcnn_scal(int n, float a, float *x) {
#ifndef BCNN_USE_AVX
    int i;
    if (a == 0.0f) {
        memset(x, 0, n * sizeof(float));
    } else if (a != 1.0f) {
        for (i = 0; i < n; ++i) x[i] *= a;
    }
#else
    int i, nd, nm;
    __m128 reg0, reg1;
    __m128 areg = _mm_set1_ps(a);
    __m128 prod;
    int data_is_aligned = bh_is_aligned32(x);

    if (a == 0.0f) {
        memset(x, 0, n * sizeof(float));
    } else if (a != 1.0f) {
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
        } else {
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
        for (i = 0; i < nm; ++i) x[i] *= a;
    }
#endif
    return 0;
}

int bcnn_add_scalar(int n, float a, float *x) {
#ifndef BCNN_USE_AVX
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
    } else if (a != 1.0f) {
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
        } else {
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
        for (i = 0; i < nm; ++i) x[i] += a;
    }
#endif
    return 0;
}

float bcnn_dot(int n, float *x, float *y) {
#ifndef BCNN_USE_AVX
    int i;
    float dot = 0;
    for (i = 0; i < n; ++i) dot += x[i] * y[i];
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
    for (i = 0; i < nm; ++i) sum += x[i] * y[i];
    return sum;
#endif
}

int bcnn_vsum(int n, float *x, float *sum) {
#ifndef BCNN_USE_AVX
    int i;
    float s = 0.0f;
    for (i = 0; i < n; ++i) s += x[i];
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
    for (i = 0; i < nm; ++i) s += x[i];
    *(sum) = s;
#endif
    return 0;
}

int bcnn_gemv(int trans_a, int m, int n, float alpha, float *a, float *x,
              float beta, float *y) {
    int i, j;
#ifdef BCNN_USE_AVX
    int nd, md;
    __m128 apart, mula, mul0, areg, xreg, yreg;
    float sum[4] = {0};
#endif

    if (!trans_a) {
        if (beta != 1.0f) {
            for (i = 0; i < m; ++i) {
                y[i] *= beta;
            }
        }
#ifndef BCNN_USE_AVX
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
            for (; j < n; ++j) y[i] += alpha * a[i * n + j] * x[j];
        }
#endif
    } else {
        if (beta != 1.0f) {
            for (i = 0; i < n; ++i) {
                y[i] *= beta;
            }
        }
#ifndef BCNN_USE_AVX
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
            for (; j < m; ++j) y[i] += alpha * a[i * m + j] * x[j];
        }
#endif
    }
    return 0;
}

float bcnn_l2_distance(float *x, float *y, int n) {
    float dist = 0.0f;
    int i;
#ifdef BCNN_USE_AVX
    int data_is_aligned = bh_is_aligned32(x) & bh_is_aligned32(y);
    __m128 vx0, vy0, vx1, vy1, vdiff0, vdiff1;
    __m128 vdist = _mm_set1_ps(0.0f);
    float dist4f[4] = {0.0f};
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
    } else {
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
    for (i = 0; i < nm; ++i) dist += (x[i] - y[i]) * (x[i] - y[i]);
#else
    for (i = 0; i < n; ++i) dist += (x[i] - y[i]) * (x[i] - y[i]);
#endif
    return dist;
}

float bcnn_sqrdiff_vs(float *x, float a, int n) {
    float dist = 0.0f;
    int i;
#ifndef BCNN_USE_AVX
    for (i = 0; i < n; ++i) dist += (x[i] - a) * (x[i] - a);
#else
    int data_is_aligned = bh_is_aligned32(x);
    __m128 vx0, vx1, vdiff0, vdiff1;
    __m128 vdist = _mm_set1_ps(0.0f);
    __m128 areg = _mm_set1_ps(a);
    float dist4f[4] = {0.0f};
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
    } else {
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
    for (i = 0; i < nm; ++i) dist += (x[i] - a) * (x[i] - a);
#endif

    return dist;
}

float bcnn_shiftdot(int n, float *x, float a, float *y, float b) {
#ifndef BCNN_USE_AVX
    int i;
    float dot = 0;
    for (i = 0; i < n; ++i) dot += (x[i] - a) * (y[i] - b);
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
    for (i = 0; i < nm; ++i) sum += (x[i] - a) * (y[i] - b);
    return sum;
#endif
}

int bcnn_varnorm(int n, float *a, float c, float *y) {
#ifndef BCNN_USE_AVX
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
        r0 = _mm_mul_ps(
            reg0, _mm_div_ps(creg, _mm_add_ps(_mm_mul_ps(r0, _mm_sqrt_ps(r0)),
                                              epsreg)));
        r1 = _mm_mul_ps(
            reg1, _mm_div_ps(creg, _mm_add_ps(_mm_mul_ps(r1, _mm_sqrt_ps(r1)),
                                              epsreg)));
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

int bcnn_varmean(int n, float *m, float a, float *var) {
#ifndef BCNN_USE_AVX
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

void bcnn_add_bias(float *output, float *bias, int batch_size, int num_channels,
                   int spatial_size) {
    int i, j, b;
    for (b = 0; b < batch_size; ++b) {
        for (i = 0; i < num_channels; ++i) {
            bcnn_add_scalar(spatial_size, bias[i], output + i * spatial_size);
        }
        output += num_channels * spatial_size;
    }
}

void bcnn_scales(float *output, float *scales, int batch_size, int num_channels,
                 int spatial_size) {
    int i, j, b;
    for (b = 0; b < batch_size; ++b) {
        for (i = 0; i < num_channels; ++i) {
            bcnn_scal(spatial_size, scales[i], output + i * spatial_size);
        }
        output += num_channels * spatial_size;
    }
}

void bcnn_grad_scales(float *x_norm, float *delta, int batch, int n, int size,
                      float *scale_updates) {
    int i, b, f;
    for (f = 0; f < n; ++f) {
        float sum = 0;
        for (b = 0; b < batch; ++b) {
            for (i = 0; i < size; ++i) {
                int index = i + size * (f + n * b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

void bcnn_grad_bias(float *grad_bias, float *grad_data, int batch_size,
                    int num_channels, int spatial_size) {
    int i, j, b;
    float *p = NULL;

    for (b = 0; b < batch_size; ++b) {
        for (i = 0; i < num_channels; ++i) {
            p = grad_data + spatial_size * (i + b * num_channels);
            for (j = 0; j < spatial_size; ++j) {
                grad_bias[i] += p[j];
            }
        }
    }
}

static inline int is_a_positive_and_inferior_to_b(int a, int b) {
    return (unsigned int)a < (unsigned int)b;
}

void bcnn_im2col(const float *data_im, const int channels, const int height,
                 const int width, const int kernel_size, const int pad,
                 const int stride, float *data_col) {
    int channel, kernel_row, kernel_col, output_rows, output_cols, input_col,
        input_row, output_col;
    const int output_h = (height + 2 * pad - kernel_size) / stride + 1;
    const int output_w = (width + 2 * pad - kernel_size) / stride + 1;
    const int channel_size = height * width;

    for (channel = channels; channel--; data_im += channel_size) {
        for (kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
            for (kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                input_row = -pad + kernel_row;
                for (output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_positive_and_inferior_to_b(input_row, height)) {
                        for (output_cols = output_w; output_cols;
                             output_cols--) {
                            *(data_col++) = 0;
                        }
                    } else {
                        input_col = -pad + kernel_col;
                        for (output_col = output_w; output_col; output_col--) {
                            if (is_a_positive_and_inferior_to_b(input_col,
                                                                width)) {
                                *(data_col++) =
                                    data_im[input_row * width + input_col];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col += stride;
                        }
                    }
                    input_row += stride;
                }
            }
        }
    }
}

void bcnn_col2im(const float *data_col, const int channels, const int height,
                 const int width, const int kernel, const int pad,
                 const int stride, float *data_im) {
    int channel, kernel_row, kernel_col, output_rows, input_col, input_row,
        output_col;
    const int output_h = (height + 2 * pad - kernel) / stride + 1;
    const int output_w = (width + 2 * pad - kernel) / stride + 1;
    const int channel_size = height * width;

    bcnn_fill_f32(height * width * channels, 0.0f, data_im);

    for (channel = channels; channel--; data_im += channel_size) {
        for (kernel_row = 0; kernel_row < kernel; kernel_row++) {
            for (kernel_col = 0; kernel_col < kernel; kernel_col++) {
                input_row = -pad + kernel_row;
                for (output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_positive_and_inferior_to_b(input_row, height)) {
                        data_col += output_w;
                    } else {
                        input_col = -pad + kernel_col;
                        for (output_col = output_w; output_col; output_col--) {
                            if (is_a_positive_and_inferior_to_b(input_col,
                                                                width)) {
                                data_im[input_row * width + input_col] +=
                                    *data_col;
                            }
                            data_col++;
                            input_col += stride;
                        }
                    }
                    input_row += stride;
                }
            }
        }
    }
}

// General Matrix-Matrix multiplication
//             ldb n
//          _________
//          |       |
//          |   B   | k
//          |       |
//  ________|______ |
//  |       |       |
// m|       |       | m
//  |   A   |   C   |
//  |_______|_______|
//  lda k     ldc n
//

// This implementation follows the Blis micro-kernel algorithm
// Reference: BLIS: A Framework for Rapidly Instantiating BLAS Functionality
#ifdef BCNN_USE_NEON
#define MC 384
#define KC 384
#define NC 4096
#ifdef PACK_4x4
#define MR 4
#define NR 4
#else
#define MR 8
#define NR 8
#endif
#else
#define MC 128
#define KC 384
#define NC 4096
#define MR 8
#define NR 8
#endif  // BCNN_USE_NEON

float A_[MC * KC] __attribute__((aligned(32)));
float B_[KC * NC] __attribute__((aligned(32)));
float C_[MR * NR] __attribute__((aligned(32)));
float AB_[MR * NR] __attribute__((aligned(32)));

static int equal(float a, float b) {
    const float EPSILON = 1e-5;
    if (fabsf(a - b) < EPSILON) {
        return 1;
    }
    return 0;
}

static void sgemm_nn_pack_MRxk8(int k, const float *A, int inc_row_A,
                                int inc_col_A, float *buffer) {
    int j, a2 = inc_row_A, a3 = 2 * inc_row_A, a4 = 3 * inc_row_A;
    int a5 = 4 * inc_row_A;
    int a6 = 5 * inc_row_A;
    int a7 = 6 * inc_row_A;
    int a8 = 7 * inc_row_A;
    for (j = 0; j < k; ++j) {
        buffer[0] = A[0];
        buffer[1] = A[a2];
        buffer[2] = A[a3];
        buffer[3] = A[a4];
        buffer[4] = A[a5];
        buffer[5] = A[a6];
        buffer[6] = A[a7];
        buffer[7] = A[a8];
        A += 1;
        buffer += MR;
    }
}

static void sgemm_nn_pack_MRxk4(int k, const float *A, int inc_row_A,
                                int inc_col_A, float *buffer) {
    int j, a2 = inc_row_A, a3 = 2 * inc_row_A, a4 = 3 * inc_row_A;
    for (j = 0; j < k; ++j) {
        buffer[0] = A[0];
        buffer[1] = A[a2];
        buffer[2] = A[a3];
        buffer[3] = A[a4];
        A += 1;
        buffer += MR;
    }
}

static void sgemm_nn_pack_A(int mc, int kc, const float *A, int inc_row_A,
                            int inc_col_A, float *buffer) {
    int mp = mc / MR;
    int _mr = mc % MR;
    int tmp1 = kc * MR;
    int tmp2 = MR * inc_row_A;
    int i, j;

    for (i = 0; i < mp; ++i) {
#ifdef BCNN_USE_NEON
#ifdef PACK_4x4
        sgemm_nn_pack_MRxk4(kc, A, inc_row_A, inc_col_A, buffer);
#else
        sgemm_nn_pack_MRxk8(kc, A, inc_row_A, inc_col_A, buffer);
#endif
#else
        sgemm_nn_pack_MRxk8(kc, A, inc_row_A, inc_col_A, buffer);
#endif
        buffer += tmp1;
        A += tmp2;
    }
    if (_mr > 0) {
        for (j = 0; j < kc; ++j) {
            for (i = 0; i < _mr; ++i) {
                buffer[i] = A[i * inc_row_A];
            }
            for (i = _mr; i < MR; ++i) {
                buffer[i] = 0.0;
            }
            A += 1;
            buffer += MR;
        }
    }
}

static void sgemm_pack_A(int mc, int kc, const float *A, int inc_row_A,
                         int inc_col_A, float *p) {
    int j, l, i0, i, nu;
    int mp = (mc + MR - 1) / MR;

    for (j = 0; j < kc; ++j) {
        for (l = 0; l < mp; ++l) {
            for (i0 = 0; i0 < MR; ++i0) {
                i = l * MR + i0;
                nu = l * MR * kc + j * MR + i0;
                p[nu] = (i < mc) ? A[i * inc_row_A + j * inc_col_A] : 0;
            }
        }
    }
}

static void sgemm_pack_B(int kc, int nc, const float *B, int inc_row_B,
                         int inc_col_B, float *p) {
    int i, l, j0;
    const int np = (nc + NR - 1) / NR;

    for (l = 0; l < np; ++l) {
        for (i = 0; i < kc; ++i) {
            for (j0 = 0; j0 < NR; ++j0) {
                int j = l * NR + j0;
                int nu = l * NR * kc + i * NR + j0;
                p[nu] = (j < nc) ? B[i * inc_row_B + j * inc_col_B] : 0;
            }
        }
    }
}

static void sgemm_nn_pack_kxNR(int k, const float *B, int inc_row_B,
                               int inc_col_B, float *buffer) {
    int i, j;
    for (i = 0; i < k; ++i) {
        for (j = 0; j < NR; ++j) {
            buffer[j] = B[j];
        }
        B += inc_row_B;
        buffer += NR;
    }
}

static void sgemm_nn_pack_B(int kc, int nc, const float *B, int inc_row_B,
                            int inc_col_B, float *buffer) {
    int np = nc / NR;
    int _nr = nc % NR;
    int tmp1 = kc * NR;
    int i, j;

    for (j = 0; j < np; ++j) {
        sgemm_nn_pack_kxNR(kc, B, inc_row_B, inc_col_B, buffer);
        B += NR;
        buffer += tmp1;
    }
    if (_nr > 0) {
        for (i = 0; i < kc; ++i) {
            for (j = 0; j < _nr; ++j) {
                buffer[j] = B[j];
            }
            for (j = _nr; j < NR; ++j) {
                buffer[j] = 0.0;
            }
            buffer += NR;
            B += inc_row_B;
        }
    }
}

static void sgemm_ukernel(int kc, float alpha, const float *A, const float *B,
                          float beta, float *C, int inc_row_C, int inc_col_C) {
#if (defined(BCNN_USE_AVX))
    __m256 abv0 = _mm256_setzero_ps();
    __m256 abv1 = _mm256_setzero_ps();
    __m256 abv2 = _mm256_setzero_ps();
    __m256 abv3 = _mm256_setzero_ps();

    __m256 abv4 = _mm256_setzero_ps();
    __m256 abv5 = _mm256_setzero_ps();
    __m256 abv6 = _mm256_setzero_ps();
    __m256 abv7 = _mm256_setzero_ps();

    __m256 av;

    for (int l = 0; l < kc; ++l) {
        av = _mm256_load_ps(A);
        abv0 = _mm256_add_ps(abv0, _mm256_mul_ps(_mm256_broadcast_ss(B), av));
        abv1 =
            _mm256_add_ps(abv1, _mm256_mul_ps(_mm256_broadcast_ss(B + 1), av));
        abv2 =
            _mm256_add_ps(abv2, _mm256_mul_ps(_mm256_broadcast_ss(B + 2), av));
        abv3 =
            _mm256_add_ps(abv3, _mm256_mul_ps(_mm256_broadcast_ss(B + 3), av));
        abv4 =
            _mm256_add_ps(abv4, _mm256_mul_ps(_mm256_broadcast_ss(B + 4), av));
        abv5 =
            _mm256_add_ps(abv5, _mm256_mul_ps(_mm256_broadcast_ss(B + 5), av));
        abv6 =
            _mm256_add_ps(abv6, _mm256_mul_ps(_mm256_broadcast_ss(B + 6), av));
        abv7 =
            _mm256_add_ps(abv7, _mm256_mul_ps(_mm256_broadcast_ss(B + 7), av));

        A += MR;
        B += NR;
    }
    _mm256_store_ps(AB_ + 0, abv0);
    _mm256_store_ps(AB_ + 8, abv1);
    _mm256_store_ps(AB_ + 16, abv2);
    _mm256_store_ps(AB_ + 24, abv3);
    _mm256_store_ps(AB_ + 32, abv4);
    _mm256_store_ps(AB_ + 40, abv5);
    _mm256_store_ps(AB_ + 48, abv6);
    _mm256_store_ps(AB_ + 56, abv7);
#elif (defined(BCNN_USE_NEON))
#if PACK_4x4
    float32x4_t cv0 = vdupq_n_f32(0.0);
    float32x4_t cv1 = vdupq_n_f32(0.0);
    float32x4_t cv2 = vdupq_n_f32(0.0);
    float32x4_t cv3 = vdupq_n_f32(0.0);
    float32x4_t av;
    float32x4_t bv;
    float32x2_t bv01;
    float32x2_t bv23;
    for (int p = 0; p < kc; ++p) {
        av = vld1q_f32(A);
        bv = vld1q_f32(B);
        bv01 = vget_low_f32(bv);
        cv0 = vmlaq_lane_f32(cv0, av, bv01, 0);
        cv1 = vmlaq_lane_f32(cv1, av, bv01, 1);
        bv23 = vget_high_f32(bv);
        cv2 = vmlaq_lane_f32(cv2, av, bv23, 0);
        cv3 = vmlaq_lane_f32(cv3, av, bv23, 1);
        A += MR;
        B += NR;
    }
    vst1q_f32(AB_ + 0, cv0);
    vst1q_f32(AB_ + 4, cv1);
    vst1q_f32(AB_ + 8, cv2);
    vst1q_f32(AB_ + 12, cv3);
#else
    float32x4_t va0, va1, vb0, vb1;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11,
        vc12, vc13, vc14, vc15;
    vc0 = vdupq_n_f32(0.0);
    vc1 = vdupq_n_f32(0.0);
    vc2 = vdupq_n_f32(0.0);
    vc3 = vdupq_n_f32(0.0);
    vc4 = vdupq_n_f32(0.0);
    vc5 = vdupq_n_f32(0.0);
    vc6 = vdupq_n_f32(0.0);
    vc7 = vdupq_n_f32(0.0);
    vc8 = vdupq_n_f32(0.0);
    vc9 = vdupq_n_f32(0.0);
    vc10 = vdupq_n_f32(0.0);
    vc11 = vdupq_n_f32(0.0);
    vc12 = vdupq_n_f32(0.0);
    vc13 = vdupq_n_f32(0.0);
    vc14 = vdupq_n_f32(0.0);
    vc15 = vdupq_n_f32(0.0);
    for (int p = 0; p < kc; ++p) {
        va0 = vld1q_f32(A);
        va1 = vld1q_f32(A + 4);
        vb0 = vld1q_f32(B);
        vb1 = vld1q_f32(B + 4);
        vc0 = vfmaq_laneq_f32(vc0, va0, vb0, 0);
        vc1 = vfmaq_laneq_f32(vc1, va1, vb0, 0);
        vc2 = vfmaq_laneq_f32(vc2, va0, vb0, 1);
        vc3 = vfmaq_laneq_f32(vc3, va1, vb0, 1);
        vc4 = vfmaq_laneq_f32(vc4, va0, vb0, 2);
        vc5 = vfmaq_laneq_f32(vc5, va1, vb0, 2);
        vc6 = vfmaq_laneq_f32(vc6, va0, vb0, 3);
        vc7 = vfmaq_laneq_f32(vc7, va1, vb0, 3);
        vc8 = vfmaq_laneq_f32(vc8, va0, vb1, 0);
        vc9 = vfmaq_laneq_f32(vc9, va1, vb1, 0);
        vc10 = vfmaq_laneq_f32(vc10, va0, vb1, 1);
        vc11 = vfmaq_laneq_f32(vc11, va1, vb1, 1);
        vc12 = vfmaq_laneq_f32(vc12, va0, vb1, 2);
        vc13 = vfmaq_laneq_f32(vc13, va1, vb1, 2);
        vc14 = vfmaq_laneq_f32(vc14, va0, vb1, 3);
        vc15 = vfmaq_laneq_f32(vc15, va1, vb1, 3);
        B += NR;
        A += MR;
    }
    vst1q_f32(AB_, vc0);
    vst1q_f32(AB_ + 4, vc1);
    vst1q_f32(AB_ + 8, vc2);
    vst1q_f32(AB_ + 12, vc3);
    vst1q_f32(AB_ + 16, vc4);
    vst1q_f32(AB_ + 20, vc5);
    vst1q_f32(AB_ + 24, vc6);
    vst1q_f32(AB_ + 28, vc7);
    vst1q_f32(AB_ + 32, vc8);
    vst1q_f32(AB_ + 36, vc9);
    vst1q_f32(AB_ + 40, vc10);
    vst1q_f32(AB_ + 44, vc11);
    vst1q_f32(AB_ + 48, vc12);
    vst1q_f32(AB_ + 52, vc13);
    vst1q_f32(AB_ + 56, vc14);
    vst1q_f32(AB_ + 60, vc15);
#endif  // PACK_4x4
#else
    for (int i = 0; i < MR * NR; ++i) {
        AB_[i] = 0.0f;
    }
    for (int l = 0; l < kc; ++l) {
        for (int j = 0; j < NR; ++j) {
            for (int i = 0; i < MR; ++i) {
                AB_[i + j * MR] += A[i] * B[j];
            }
        }
        A += MR;
        B += NR;
    }
#endif
    if (equal(beta, 0.0)) {
        for (int j = 0; j < NR; ++j) {
            for (int i = 0; i < MR; ++i) {
                C[i * inc_row_C + j * inc_col_C] = 0.0;
            }
        }
    } else if (!equal(beta, 1.0)) {
        for (int j = 0; j < NR; ++j) {
            for (int i = 0; i < MR; ++i) {
                C[i * inc_row_C + j * inc_col_C] *= beta;
            }
        }
    }
    if (!equal(alpha, 1.0)) {
        for (int j = 0; j < NR; ++j) {
            for (int i = 0; i < MR; ++i) {
                C[i * inc_row_C + j * inc_col_C] += alpha * AB_[i + j * MR];
            }
        }
    } else {
        for (int j = 0; j < NR; ++j) {
            for (int i = 0; i < MR; ++i) {
                C[i * inc_row_C + j * inc_col_C] += AB_[i + j * MR];
            }
        }
    }
}

static void sgemm_axpy(int m, int n, float alpha, const float *X, int incRowX,
                       int incColX, float *Y, int incRowY, int incColY) {
    int i, j;
    if (!equal(alpha, 1.0)) {
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m; ++i) {
                Y[i * incRowY + j] += alpha * X[i + j * incColX];
            }
        }
    } else {
        for (j = 0; j < n; ++j) {
            for (i = 0; i < m; ++i) {
                Y[i * incRowY + j] += X[i + j * incColX];
            }
        }
    }
}

static void sgemm_scal(int m, int n, float alpha, float *X, int incRowX,
                       int incColX) {
    int i, j;
    if (!equal(alpha, 0.0)) {
        for (i = 0; i < m; ++i) {
            for (j = 0; j < n; ++j) {
                X[i * incRowX + j] *= alpha;
            }
        }
    } else {
        for (i = 0; i < m; ++i) {
            for (j = 0; j < n; ++j) {
                X[i * incRowX + j] = 0.0;
            }
        }
    }
}

static void sgemm_mkernel(int mc, int nc, int kc, float alpha, float beta,
                          float *C, int inc_row_C, int inc_col_C) {
    int mp = (mc + MR - 1) / MR;
    int np = (nc + NR - 1) / NR;

    int _mr = mc % MR;
    int _nr = nc % NR;

    int i, j;

    for (j = 0; j < np; ++j) {
        int nr = (j != np - 1 || _nr == 0) ? NR : _nr;

        for (i = 0; i < mp; ++i) {
            int mr = (i != mp - 1 || _mr == 0) ? MR : _mr;

            if (mr == MR && nr == NR) {
                sgemm_ukernel(kc, alpha, &A_[i * kc * MR], &B_[j * kc * NR],
                              beta, &C[i * MR * inc_row_C + j * NR], inc_row_C,
                              inc_col_C);
            } else {
                sgemm_ukernel(kc, alpha, &A_[i * kc * MR], &B_[j * kc * NR],
                              0.0, C_, 1, MR);
                sgemm_scal(mr, nr, beta, &C[i * MR * inc_row_C + j * NR],
                           inc_row_C, inc_col_C);
                sgemm_axpy(mr, nr, 1.0, C_, 1, MR,
                           &C[i * MR * inc_row_C + j * NR], inc_row_C,
                           inc_col_C);
            }
        }
    }
}

static void sgemm_nn(int m, int n, int k, float alpha, const float *A,
                     int inc_row_A, int inc_col_A, const float *B,
                     int inc_row_B, int inc_col_B, float beta, float *C,
                     int inc_row_C, int inc_col_C) {
    int mb = (m + MC - 1) / MC;
    int nb = (n + NC - 1) / NC;
    int kb = (k + KC - 1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;

    float _beta;

    if (equal(alpha, 0.0) || k == 0) {
        sgemm_scal(m, n, beta, C, inc_row_C, inc_col_C);
        return;
    }

    for (j = 0; j < nb; ++j) {
        nc = (j != nb - 1 || _nc == 0) ? NC : _nc;

        for (l = 0; l < kb; ++l) {
            kc = (l != kb - 1 || _kc == 0) ? KC : _kc;
            _beta = (l == 0) ? beta : 1.0f;

            sgemm_nn_pack_B(kc, nc, &B[l * KC * inc_row_B + j * NC], inc_row_B,
                            inc_col_B, B_);
            for (i = 0; i < mb; ++i) {
                mc = (i != mb - 1 || _mc == 0) ? MC : _mc;
                sgemm_nn_pack_A(mc, kc, &A[i * MC * inc_row_A + l * KC],
                                inc_row_A, inc_col_A, A_);
                sgemm_mkernel(mc, nc, kc, alpha, _beta,
                              &C[i * MC * inc_row_C + j * NC], inc_row_C,
                              inc_col_C);
            }
        }
    }
}

static void sgemm(int m, int n, int k, float alpha, const float *A,
                  int inc_row_A, int inc_col_A, const float *B, int inc_row_B,
                  int inc_col_B, float beta, float *C, int inc_row_C,
                  int inc_col_C) {
    int mb = (m + MC - 1) / MC;
    int nb = (n + NC - 1) / NC;
    int kb = (k + KC - 1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;

    float _beta;

    if (equal(alpha, 0.0) || k == 0) {
        sgemm_scal(m, n, beta, C, inc_row_C, inc_col_C);
        return;
    }

    for (j = 0; j < nb; ++j) {
        nc = (j != nb - 1 || _nc == 0) ? NC : _nc;

        for (l = 0; l < kb; ++l) {
            kc = (l != kb - 1 || _kc == 0) ? KC : _kc;
            _beta = (l == 0) ? beta : 1.0f;
            sgemm_pack_B(kc, nc, &B[l * KC * inc_row_B + j * NC], inc_row_B,
                         inc_col_B, B_);
            for (i = 0; i < mb; ++i) {
                mc = (i != mb - 1 || _mc == 0) ? MC : _mc;
                sgemm_pack_A(mc, kc, &A[i * MC * inc_row_A + l * KC], inc_row_A,
                             inc_col_A, A_);
                sgemm_mkernel(mc, nc, kc, alpha, _beta,
                              &C[i * MC * inc_row_C + j * NC], inc_row_C,
                              inc_col_C);
            }
        }
    }
}

int bcnn_gemm(int trans_a, int trans_b, int m, int n, int k, float alpha,
              float *A, int lda, float *B, int ldb, float beta, float *C,
              int ldc) {
    int inc_row_A = (!trans_a) ? lda : 1;
    int inc_col_A = (!trans_a) ? 1 : lda;

    int inc_row_B = (!trans_b) ? ldb : 1;
    int inc_col_B = (!trans_b) ? 1 : ldb;

    if (!trans_a && !trans_b) {
        sgemm_nn(m, n, k, alpha, A, inc_row_A, inc_col_A, B, inc_row_B,
                 inc_col_B, beta, C, ldc, 1);
    } else {
        sgemm(m, n, k, alpha, A, inc_row_A, inc_col_A, B, inc_row_B, inc_col_B,
              beta, C, ldc, 1);
    }

    return 0;
}