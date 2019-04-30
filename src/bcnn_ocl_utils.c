#ifdef BCNN_USE_OPENCL

#include "bcnn_ocl_utils.h"

dim2 dim2_create(const int x, const int y) {
    dim2 ret;

    ret.x = x;
    ret.y = y;

    return ret;
}

int bcnn_opencl_run_kernel(bcnn_net *net, cl_kernel kernel,
                           const dim2 globalItemSize, const int argc, ...) {
    cl_int clErr;
    fprintf(stderr, "run kernel\n");
    va_list vl;
    va_start(vl, argc);

    size_t argSize = 0;
    void *argValue = NULL;

    int i, j;
    for (i = 0, j = 0; i < argc; i += 2, ++j) {
        argValue = va_arg(vl, void *);
        argSize = va_arg(vl, size_t);
        fprintf(stderr, "set kernel arg %d ...", i);
        clErr = clSetKernelArg(kernel, j, argSize, argValue);
        if (clErr != CL_SUCCESS) {
            fprintf(stderr, "failed %s\n", clGetErrorString(clErr));
            const size_t bufferSize = 2048;
            char *kernelName = (char *)calloc(bufferSize, sizeof(char));
            BCNN_OPENCL_CHECK(clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME,
                                              bufferSize, kernelName, NULL));
            free(kernelName);
        }
        fprintf(stderr, "success\n");
    }

    va_end(vl);

    size_t global_work_offset[2], global_work_size[2];
    global_work_offset[0] = 0;
    global_work_offset[1] = 0;
    global_work_size[0] = globalItemSize.x;
    global_work_size[1] = globalItemSize.y;

    bcnn_opencl_context *context = net->opencl_ctx;

    cl_event e;
    clErr = clEnqueueNDRangeKernel(context->cmd_queue, kernel, 2,
                                   /*global_work_offset*/ 0, global_work_size,
                                   NULL, 0, NULL, &e);
    if (clErr != CL_SUCCESS) {
        const size_t bufferSize = 2048;
        char *kernelName = (char *)calloc(bufferSize, sizeof(char));

        BCNN_OPENCL_CHECK(clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME,
                                          bufferSize, kernelName, NULL));

        free(kernelName);
    } else {
        fprintf(stderr, "enqueue kernel success\n");
    }
    cl_int rc = clWaitForEvents(1, &e);
    fprintf(stderr, "kernel0 %d\n", rc);
    BCNN_OPENCL_CHECK(clReleaseEvent(e));

    return 0;
}

#endif