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


#ifdef BCNN_USE_CUDA

#include "bcnn/bcnn.h"

void bcnn_cuda_gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = bcnn_cublas_handle();
    int ldaa = (TA == 0) ? K : M;
    int ldbb = (TB == 0) ? N : K;
    cublasStatus_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
                        (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldbb, A_gpu, ldaa, &BETA, C_gpu, N);
    bcnn_cublas_check(status);
}


void bcnn_cuda_gemv(int TA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) 
{
    cublasHandle_t handle = bcnn_cublas_handle();
    cublasOperation_t cuTA = (TA ? CUBLAS_OP_T : CUBLAS_OP_N);
    cublasStatus_t status = cublasSgemv(handle, cuTA, N, M, &alpha,
        A, N, x, 1, &beta, y, 1);
    bcnn_cublas_check(status);
}


__global__ void _bcnn_cuda_fill_f32_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N)
        X[i*INCX] = ALPHA;
}

void bcnn_cuda_fill_f32(int n, float alpha, float *x, int incx)
{
    _bcnn_cuda_fill_f32_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, alpha, x, incx);
    bcnn_cuda_check(cudaPeekAtLastError());
}


void bcnn_cuda_copy_f32(int n, float *x, int incx, float *y, int incy)
{
    cublasHandle_t handle = bcnn_cublas_handle();
    cublasStatus_t status = cublasScopy(handle, n, x, incx, y, incy);
    bcnn_cublas_check(status);
}

void bcnn_cuda_axpy(int n, float alpha, float *x, int incx, float *y, int incy)
{
    cublasHandle_t handle = bcnn_cublas_handle();
    cublasStatus_t status = cublasSaxpy(handle, n, &alpha, x, incx, y, incy);
    bcnn_cublas_check(status);
}

void bcnn_cuda_scal(int n, float alpha, float *x, int incx)
{
    cublasHandle_t handle = bcnn_cublas_handle();
    cublasStatus_t status = cublasSscal(handle, n, &alpha, x, incx);
    bcnn_cublas_check(status);
}

__global__ void _bcnn_vadd_kernel(int n, float *a, float *b, float *y)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a[i] + b[i];
}

void bcnn_cuda_vadd(int n, float *a, float *b, float *y)
{
    _bcnn_vadd_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, a, b, y);
}

__global__ void _bcnn_vsub_kernel(int n, float *a, float *b, float *y)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a[i] - b[i];
}

void bcnn_cuda_vsub(int n, float *a, float *b, float *y)
{
    _bcnn_vsub_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, a, b, y);
}


__global__ void _bcnn_vmul_kernel(int n, float *a, float *b, float *y)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a[i] * b[i];
}

void bcnn_cuda_vmul(int n, float *a, float *b, float *y)
{
    _bcnn_vmul_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, a, b, y);
}

__global__ void _bcnn_vdiv_kernel(int n, float *a, float *b, float *y)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a[i] / b[i];
}

void bcnn_cuda_vdiv(int n, float *a, float *b, float *y)
{
    _bcnn_vdiv_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, a, b, y);
}

__global__ void _bcnn_pow_kernel(int n, float *x, float a, float *y)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = pow(x[i], a);
}

void bcnn_cuda_pow(int n, float *x, float a, float *y)
{
    _bcnn_pow_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, x, a, y);
}


void bcnn_cuda_axpby(int n, float a, float *x, float b, float *y)
{
    bcnn_cuda_scal(n, b, y, 1);
    bcnn_cuda_axpy(n, a, x, 1, y, 1);
}

__global__ void _bcnn_add_scalar_kernel(int n, float a, float *y)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += a;
}


void bcnn_cuda_add_scalar(int n, float a, float* y)
{
    _bcnn_add_scalar_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, a, y);
}


__global__ void _bcnn_vsum_kernel(int n, float *x, float *sum)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
        *sum += x[i];
}

void bcnn_cuda_vsum(int n, float *x, float *sum)
{
    _bcnn_vsum_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, x, sum);
}


__global__ void _mean_variance_forward_kernel(float *x, int b, int c, int wxh, float *mean, float *var)
{
    float scale = 1.0f / (b * wxh);
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x, j, k, ind;
    if (i >= c)
        return;

    mean[i] = 0;
    for (j = 0; j < b; ++j){
        for (k = 0; k < wxh; ++k){
            ind = j *c * wxh + i * wxh + k;
            mean[i] += x[ind];
            var[i] += x[ind] * x[ind];
        }
    }
    mean[i] *= scale;
    var[i] = var[i] * scale - mean[i] * mean[i];
}


void bcnn_cuda_mean_variance_forward(float *x, int b, int c, int wxh, float *mean, float *var)
{
    _mean_variance_forward_kernel<<<bcnn_cuda_gridsize(c), BCNN_CUDA_THREADS>>>(x, b, c, wxh, mean, var);
}


__global__ void _norm_forward_kernel(float *x, float *mean, float *variance, int b, int c, int wxh)
{
    int ind = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int j = (ind / wxh) % c;

    if (ind >= b * c * wxh)
        return;   
    
    x[ind] = (x[ind] - mean[j]) / (sqrt(variance[j] + 0.000001f));
}

void bcnn_cuda_norm_forward(float *x, float *mean, float *variance, int b, int c, int wxh)
{
    _norm_forward_kernel<<<bcnn_cuda_gridsize(b * c * wxh), BCNN_CUDA_THREADS>>>(x, mean, variance, b, c, wxh);
}


__global__ void _norm_backward_kernel(float *x, float *mean, float *var, float *mean_diff, float *var_diff, int b, int c, int wxh, float *grad)
{
    int ind = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int j = (ind / wxh) % c;

    if (ind >= b * c * wxh)
        return;   
    
    grad[ind] = grad[ind] * 1.0f / (sqrtf(var[j] + 0.00001f)) + var_diff[j] * 2.0f * (x[ind] - mean[j]) / (wxh * b) + mean_diff[j] / (wxh * b);
}

void bcnn_cuda_norm_backward(float *x, float *mean, float *var, float *mean_diff, float *var_diff, int b, int c, int wxh, float *grad)
{
    _norm_backward_kernel<<<bcnn_cuda_gridsize(b * c * wxh), BCNN_CUDA_THREADS>>>(x, mean, var, mean_diff, var_diff, b, c, wxh, grad);
}


__global__ void _mean_variance_backward_kernel(float *x, float *grad, float *mean, float *var, int b, int c, int wxh, float *mean_diff, float *var_diff)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x, j, k, ind;
    
    if (i >= c)
        return;

    mean_diff[i] = 0;
    var_diff[i] = 0;
    for (j = 0; j < b; ++j) {
        for (k = 0; k < wxh; ++k) {
            ind = j * c * wxh + i * wxh + k;
            mean_diff[i] += grad[ind];
            var_diff[i] += grad[ind] * (x[ind] - mean[i]);
        }
    }
    mean_diff[i] *= (-1.0f / sqrt (var[i] + 0.00001f));
    var_diff[i] *= -0.5f / (var[i] * sqrtf(var[i]) + 0.00001f);
}

void bcnn_cuda_mean_variance_backward(float *x, float *grad, float *mean, float *var, int b, int c, int wxh, float *mean_diff, float *var_diff)
{
    _mean_variance_backward_kernel<<<bcnn_cuda_gridsize(c), BCNN_CUDA_THREADS>>>(x, grad, mean, var, b, c, wxh, mean_diff, var_diff);
}


#endif