#ifndef OPENBLAS_SGEMM_H
#define OPENBLAS_SGEMM_H

#include "bcnn_mat.h"

int openblas_sgemm(bcnn_gemm_context *ctx, int transa, int transb, long m,
                   long n, long k, float alpha, float *a, long lda, float *b,
                   long ldb, float beta, float *c, long ldc);

#endif  // OPENBLAS_SGEMM_H
