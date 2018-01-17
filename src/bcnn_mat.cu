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

#include "bcnn_mat.h"
#include "bcnn/bcnn.h"
#include "bcnn_utils.h"

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

__global__ void bcnn_op_cuda_tanh_kernel(int n, float *x, float *y)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = (exp(2 * x[i]) - 1) / (exp(2 * x[i]) + 1);
    }
    return;
}

void bcnn_op_cuda_tanh(int n, float *x, float *y)
{
    bcnn_op_cuda_tanh_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, x, y);
}

__global__ void bcnn_op_cuda_tanh_grad_kernel(int n, float *x, float *dx)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        dx[i] *= (1 - x[i] * x[i]);
    }
    return;
}

void bcnn_op_cuda_tanh_grad(int n, float *x, float *dx)
{
    bcnn_op_cuda_tanh_grad_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, x, dx);
}

__global__ void bcnn_op_cuda_relu_kernel(int n, float *x, float *y)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] * (x[i] > 0);
    }
    return;
}

void bcnn_op_cuda_relu(int n, float *x, float *y)
{
    bcnn_op_cuda_relu_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, x, y);
}

__global__ void bcnn_op_cuda_relu_grad_kernel(int n, float *x, float *dx)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        dx[i] *= ((float)(x[i] > 0));
    }
    return;
}

void bcnn_op_cuda_relu_grad(int n, float *x, float *dx)
{
    bcnn_op_cuda_relu_grad_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, x, dx);
}

__global__ void bcnn_op_cuda_ramp_kernel(int n, float *x, float *y)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] * (x[i] > 0) + 0.1 * x[i];
    }
    return;
}

void bcnn_op_cuda_ramp(int n, float *x, float *y)
{
    bcnn_op_cuda_ramp_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, x, y);
}

__global__ void bcnn_op_cuda_ramp_grad_kernel(int n, float *x, float *dx)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        dx[i] *= ((float)(x[i] > 0) + 0.1f);
    }
    return;
}

void bcnn_op_cuda_ramp_grad(int n, float *x, float *dx)
{
    bcnn_op_cuda_ramp_grad_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, x, dx);
}

__global__ void bcnn_op_cuda_clamp_kernel(int n, float *x, float *y)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = bh_clamp(x[i], 0, 1);
    }
    return;
}

void bcnn_op_cuda_clamp(int n, float *x, float *y)
{
    bcnn_op_cuda_clamp_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, x, y);
}

__global__ void bcnn_op_cuda_clamp_grad_kernel(int n, float *x, float *dx)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        dx[i] *= (float)(x[i] > 0.0f && (x[i] < 1.0f));
    }
    return;
}

void bcnn_op_cuda_clamp_grad(int n, float *x, float *dx)
{
    bcnn_op_cuda_clamp_grad_kernel<<<bcnn_cuda_gridsize(n), BCNN_CUDA_THREADS>>>(n, x, dx);
}

__global__ void bcnn_cuda_add_bias_kernel(float *output, float *bias, int num_channels, int spatial_size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    int batch_size = blockIdx.z;

    if (offset < spatial_size)
        output[(batch_size * num_channels + channel) * spatial_size + offset] += bias[channel];
}

void bcnn_cuda_add_bias(float *output, float *bias, int batch_size, int num_channels, int spatial_size)
{
    dim3 dimGrid((spatial_size - 1) / BCNN_CUDA_THREADS + 1, num_channels, batch_size);
    dim3 dimBlock(BCNN_CUDA_THREADS, 1, 1);

    bcnn_cuda_add_bias_kernel<<<dimGrid, dimBlock>>>(output, bias, num_channels, spatial_size);
    bcnn_cuda_check(cudaPeekAtLastError());
}

__global__ void bcnn_cuda_grad_bias_kernel(float *grad_bias, float *grad_data, int num_channels, int spatial_size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    int batch_size = blockIdx.z;

    if (offset < spatial_size)
        grad_bias[channel] += grad_data[(batch_size * num_channels + channel) * spatial_size + offset];
}

void bcnn_cuda_grad_bias(float *grad_bias, float *grad_data, int batch_size, int num_channels, int spatial_size)
{
    dim3 dimGrid((spatial_size - 1) / BCNN_CUDA_THREADS + 1, num_channels, batch_size);
    dim3 dimBlock(BCNN_CUDA_THREADS, 1, 1);

    bcnn_cuda_grad_bias_kernel<<<dimGrid, dimBlock>>>(grad_bias, grad_data, num_channels, spatial_size);
    bcnn_cuda_check(cudaPeekAtLastError());
}


// im2col and col2im functions from caffe
// Reference https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu

__global__ void bcnn_cuda_im2col_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) 
{
    int i, j, w, h, w_out, h_index, h_out, channel_in, channel_out;
    int h_in, w_in;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float *data_col_ptr = NULL;
    const float *data_im_ptr = NULL;

    for(; index < n; index += blockDim.x * gridDim.x) {
        w_out = index % width_col;
        h_index = index / width_col;
        h_out = h_index % height_col;
        channel_in = h_index / height_col;
        channel_out = channel_in * ksize * ksize;
        h_in = h_out * stride - pad;
        w_in = w_out * stride - pad;
        data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (i = 0; i < ksize; ++i) {
            for (j = 0; j < ksize; ++j) {
                h = h_in + i;
                w = w_in + j;
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void bcnn_cuda_im2col(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col)
{
    pad = pad ? ksize/2 : 0;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    bcnn_cuda_im2col_kernel<<<(num_kernels + BCNN_CUDA_THREADS - 1) / BCNN_CUDA_THREADS, BCNN_CUDA_THREADS>>>(
                            num_kernels, im, height, width, ksize, pad,
                            stride, height_col,
                            width_col, data_col);
}


__global__ void bcnn_cuda_col2im_kernel(const int n, const float* data_col,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_im)
{
    int w, h, c, w_col_start, w_col_end, h_col_start, h_col_end;
    int offset, coeff_h_col, coeff_w_col, h_col, w_col;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    float val;

    for (; index < n; index += blockDim.x * gridDim.x) {
        val = 0;
        w = index % width + pad;
        h = (index / width) % height + pad;
        c = index / (width * height);

        // compute the start and end of the output
        w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        w_col_end = bh_min(w / stride + 1, width_col);
        h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        h_col_end = bh_min(h / stride + 1, height_col);

        // equivalent implementation
        offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
        coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        coeff_w_col = (1 - stride * height_col * width_col);
        for (h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        data_im[index] += val;
    }
}

void bcnn_cuda_col2im(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im)
{
    int height_col, width_col, num_kernels;

    pad = pad ? ksize/2 : 0;
    height_col = (height + 2 * pad - ksize) / stride + 1;
    width_col = (width + 2 * pad - ksize) / stride + 1;
    num_kernels = channels * height * width;

    bcnn_cuda_col2im_kernel<<<(num_kernels + BCNN_CUDA_THREADS - 1) / BCNN_CUDA_THREADS,
                             BCNN_CUDA_THREADS>>>(
                            num_kernels, data_col, height, width, ksize, pad,
                            stride, height_col,
                            width_col, data_im);
}


#endif