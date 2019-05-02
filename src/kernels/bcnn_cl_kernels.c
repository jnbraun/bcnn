#ifdef BCNN_USE_OPENCL

#include <stdlib.h>
#include <string.h>

#include <bh/bh_mem.h>
#include "bcnn_mat.h"
#include "bcnn_net.h"
#include "bcnn_ocl_utils.h"

#include "bcnn_cl_kernels.cl"

cl_program opencl_im2col_kernels_program;
cl_kernel opencl_im2col_gpu_kernel;

int bcnn_opencl_im2col_kernel_create(bcnn_net* net) {
    cl_int rc;
    // const char* kernel_source = "bcnn_cl_kernel_im2col_source";
    const char* kernel_name = "bcnn_im2col_opencl_kernel";
    const size_t size = strlen(bcnn_cl_kernel_im2col_source);
    bcnn_opencl_context* opencl_ctx = (bcnn_opencl_context*)net->opencl_ctx;
    opencl_im2col_kernels_program = clCreateProgramWithSource(
        opencl_ctx->ctx, 1, (const char**)&bcnn_cl_kernel_im2col_source, &size,
        &rc);
    BCNN_OPENCL_CHECK(rc);
    fprintf(stderr, "buildprog 0\n");
    rc = clBuildProgram(opencl_im2col_kernels_program, 1, &(opencl_ctx->device),
                        /*option=*/NULL, NULL, NULL);
    if (rc != CL_SUCCESS) {
        char* log;
        size_t log_size;
        BCNN_OPENCL_CHECK(clGetProgramBuildInfo(
            opencl_im2col_kernels_program, opencl_ctx->device,
            CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
        log = (char*)calloc(log_size + 1, sizeof(char));
        BCNN_OPENCL_CHECK(clGetProgramBuildInfo(
            opencl_im2col_kernels_program, opencl_ctx->device,
            CL_PROGRAM_BUILD_LOG, log_size, log, NULL));
        log[log_size] = '\0';
        fprintf(stderr, "Opencl kernel create: %s\n", log);
        bh_free(log);
    }
    fprintf(stderr, "buildprog\n");

    opencl_im2col_gpu_kernel =
        clCreateKernel(opencl_im2col_kernels_program, kernel_name, &rc);
    if (rc != CL_SUCCESS) {
        fprintf(stderr, "[ERROR] Could not create kernel %s. %s\n", kernel_name,
                clGetErrorString(rc));
        BCNN_OPENCL_CHECK(rc);
    }
    return 0;
}

void bcnn_opencl_im2col_kernel_release() {
    clReleaseKernel(opencl_im2col_gpu_kernel);
    clReleaseProgram(opencl_im2col_kernels_program);
}

/* Adapted from cuda kernel */
void bcnn_opencl_im2col(bcnn_net* net, cl_mem im, int offset, int channels,
                        int height, int width, int ksize, int stride, int pad,
                        cl_mem data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;

    dim2 dimGrid;

    const int num_threads = 512;
    dimGrid =
        dim2_create((num_kernels + num_threads - 1) / num_threads, num_threads);
    // dimGrid = dim2_create(num_kernels, 1);

    int zero = 0;
    fprintf(stderr, "k0\n");
    bcnn_opencl_run_kernel(
        net, opencl_im2col_gpu_kernel, dimGrid, 24, &num_kernels,
        sizeof(cl_int), &im, sizeof(cl_mem), &height, sizeof(cl_int), &width,
        sizeof(cl_int), &ksize, sizeof(cl_int), &pad, sizeof(cl_int), &stride,
        sizeof(cl_int), &height_col, sizeof(cl_int), &width_col, sizeof(cl_int),
        &data_col, sizeof(cl_mem), &zero, sizeof(cl_int), &offset,
        sizeof(cl_int));
    fprintf(stderr, "k1\n");
}

#endif  // BCNN_USE_OPENCL