/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

#include "openblas_sgemm.h"
#include "openblas_internal.h"

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

#define sgemm_itcopy sgemm_tcopy_4
#define sgemm_incopy sgemm_ncopy_4
#define sgemm_oncopy sgemm_ncopy_4
#define sgemm_otcopy sgemm_tcopy_4

#define ICOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER) \
    sgemm_itcopy(M, N, (float *)(A) + ((Y) + (X) * (LDA)), LDA, BUFFER)
#define ICOPYT_OPERATION(M, N, A, LDA, X, Y, BUFFER) \
    sgemm_incopy(M, N, (float *)(A) + ((X) + (Y) * (LDA)), LDA, BUFFER)

#define OCOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER) \
    sgemm_oncopy(M, N, (float *)(A) + ((X) + (Y) * (LDA)), LDA, BUFFER)
#define OCOPYT_OPERATION(M, N, A, LDA, X, Y, BUFFER) \
    sgemm_otcopy(M, N, (float *)(A) + ((Y) + (X) * (LDA)), LDA, BUFFER)
#define KERNEL_OPERATION(M, N, K, ALPHA, SA, SB, C, LDC, X, Y) \
    sgemm_kernel(M, N, K, ALPHA, SA, SB, (float *)(C) + ((X) + (Y)*LDC), LDC)
#define BETA_OPERATION(M_FROM, M_TO, N_FROM, N_TO, BETA, C, LDC)              \
    sgemm_beta((M_TO) - (M_FROM), (N_TO - N_FROM), 0, BETA, NULL, 0, NULL, 0, \
               (float *)(C) + (M_FROM) + (N_FROM) * (LDC), LDC)

static int sgemm_beta(long m, long n, long dummy1, float beta, float *dummy2,
                      long dummy3, float *dummy4, long dummy5, float *c,
                      long ldc) {
    long i, j;
    float *c_offset1, *c_offset;
    float ctemp1, ctemp2, ctemp3, ctemp4;
    float ctemp5, ctemp6, ctemp7, ctemp8;

    c_offset = c;

    if (beta == ZERO) {
        j = n;
        do {
            c_offset1 = c_offset;
            c_offset += ldc;

            i = (m >> 3);
            if (i > 0) {
                do {
                    *(c_offset1 + 0) = ZERO;
                    *(c_offset1 + 1) = ZERO;
                    *(c_offset1 + 2) = ZERO;
                    *(c_offset1 + 3) = ZERO;
                    *(c_offset1 + 4) = ZERO;
                    *(c_offset1 + 5) = ZERO;
                    *(c_offset1 + 6) = ZERO;
                    *(c_offset1 + 7) = ZERO;
                    c_offset1 += 8;
                    i--;
                } while (i > 0);
            }

            i = (m & 7);
            if (i > 0) {
                do {
                    *c_offset1 = ZERO;
                    c_offset1++;
                    i--;
                } while (i > 0);
            }
            j--;
        } while (j > 0);

    } else {
        j = n;
        do {
            c_offset1 = c_offset;
            c_offset += ldc;

            i = (m >> 3);
            if (i > 0) {
                do {
                    ctemp1 = *(c_offset1 + 0);
                    ctemp2 = *(c_offset1 + 1);
                    ctemp3 = *(c_offset1 + 2);
                    ctemp4 = *(c_offset1 + 3);
                    ctemp5 = *(c_offset1 + 4);
                    ctemp6 = *(c_offset1 + 5);
                    ctemp7 = *(c_offset1 + 6);
                    ctemp8 = *(c_offset1 + 7);

                    ctemp1 *= beta;
                    ctemp2 *= beta;
                    ctemp3 *= beta;
                    ctemp4 *= beta;
                    ctemp5 *= beta;
                    ctemp6 *= beta;
                    ctemp7 *= beta;
                    ctemp8 *= beta;

                    *(c_offset1 + 0) = ctemp1;
                    *(c_offset1 + 1) = ctemp2;
                    *(c_offset1 + 2) = ctemp3;
                    *(c_offset1 + 3) = ctemp4;
                    *(c_offset1 + 4) = ctemp5;
                    *(c_offset1 + 5) = ctemp6;
                    *(c_offset1 + 6) = ctemp7;
                    *(c_offset1 + 7) = ctemp8;
                    c_offset1 += 8;
                    i--;
                } while (i > 0);
            }

            i = (m & 7);
            if (i > 0) {
                do {
                    ctemp1 = *c_offset1;
                    ctemp1 *= beta;
                    *c_offset1 = ctemp1;
                    c_offset1++;
                    i--;
                } while (i > 0);
            }
            j--;
        } while (j > 0);
    }
    return 0;
}

int openblas_sgemm(bcnn_gemm_context *ctx, int transa, int transb, long m,
                   long n, long k, float alpha, float *a, long lda, float *b,
                   long ldb, float beta, float *c, long ldc) {
    long m_from, m_to, n_from, n_to;

    long ls, is, js;
    long min_l, min_i, min_j;
    long jjs, min_jj;
    float *sa = ctx->buffer_a;
    float *sb =
        (float *)((char *)ctx->buffer_a +
                  ((MC * KC * sizeof(float) + GEMM_ALIGN) & ~GEMM_ALIGN));
    long l1stride, gemm_p, l2size;

    m_from = 0;
    m_to = m;
    n_from = 0;
    n_to = n;
    if (beta != 1) {
        BETA_OPERATION(m_from, m_to, n_from, n_to, beta, c, ldc);
    }

    if ((k == 0) || (alpha == 0)) {
        return 0;
    }
    l2size = MC * KC;

    for (js = n_from; js < n_to; js += NC) {
        min_j = n_to - js;
        if (min_j > NC) {
            min_j = NC;
        }
        for (ls = 0; ls < k; ls += min_l) {
            min_l = k - ls;
            if (min_l >= KC * 2) {
                gemm_p = MC;
                min_l = KC;
            } else {
                if (min_l > KC) {
                    min_l = (min_l / 2 + MR - 1) & ~(MR - 1);
                }
                gemm_p = ((l2size / min_l + MR - 1) & ~(MR - 1));
                while (gemm_p * min_l > l2size) {
                    gemm_p -= MR;
                }
            }
            /* First, we have to move data A to L2 cache */
            min_i = m_to - m_from;
            l1stride = 1;
            if (min_i >= MC * 2) {
                min_i = MC;
            } else if (min_i > MC) {
                min_i = (min_i / 2 + MR - 1) & ~(MR - 1);
            } else {
                l1stride = 0;
            }

            if (transa) {
                ICOPYT_OPERATION(min_l, min_i, a, lda, ls, m_from, sa);
            } else {
                ICOPY_OPERATION(min_l, min_i, a, lda, ls, m_from, sa);
            }

            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = min_j + js - jjs;
                if (min_jj >= 3 * NR) {
                    min_jj = 3 * NR;
                } else if (min_jj > NR) {
                    min_jj = NR;
                }
                if (transb) {
                    OCOPYT_OPERATION(min_l, min_jj, b, ldb, ls, jjs,
                                     sb + min_l * (jjs - js) * l1stride);
                } else {
                    OCOPY_OPERATION(min_l, min_jj, b, ldb, ls, jjs,
                                    sb + min_l * (jjs - js) * l1stride);
                }
                KERNEL_OPERATION(min_i, min_jj, min_l, alpha, sa,
                                 sb + min_l * (jjs - js) * l1stride, c, ldc,
                                 m_from, jjs);
            }
            for (is = m_from + min_i; is < m_to; is += min_i) {
                min_i = m_to - is;
                if (min_i >= MC * 2) {
                    min_i = MC;
                } else if (min_i > MC) {
                    min_i = (min_i / 2 + MR - 1) & ~(MR - 1);
                }
                if (transa) {
                    ICOPYT_OPERATION(min_l, min_i, a, lda, ls, is, sa);
                } else {
                    ICOPY_OPERATION(min_l, min_i, a, lda, ls, is, sa);
                }
                KERNEL_OPERATION(min_i, min_j, min_l, alpha, sa, sb, c, ldc, is,
                                 js);
            } /* end of is */
        }     /* end of js */
    }         /* end of ls */

    return 0;
}
