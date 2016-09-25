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

__global__ void _bcnn_add_scalar_kernel(int n, float a, float* y)
{
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
		y[i] += a;
}


void bcnn_cuda_add_scalar(int n, float a, float* y)
{
	_bcnn_add_scalar_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, a, y);
}

#endif