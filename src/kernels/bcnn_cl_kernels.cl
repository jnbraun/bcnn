#ifndef BCNN_CL_KERNELS
#define BCNN_CL_KERNELS

static const char* const bcnn_cl_kernel_im2col_source =
    BCNN_CL_KERNEL_TO_STRING(__kernel void bcnn_im2col_opencl_kernel(
        int n, __global float* data_im, int height, int width, int ksize,
        int pad, int stride, int height_col, int width_col,
        __global float* data_col, int col_offset, int im_offset) {
        data_col = data_col + col_offset;
        data_im = data_im + im_offset;
        int index = get_global_id(1) * get_global_size(0) + get_global_id(0);
        for (; index < n; index += get_global_size(1) * get_global_size(0)) {
            int w_out = index % width_col;
            int h_index = index / width_col;
            int h_out = h_index % height_col;
            int channel_in = h_index / height_col;
            int channel_out = channel_in * ksize * ksize;
            int h_in = h_out * stride - pad;
            int w_in = w_out * stride - pad;

            int data_col_offset =
                (channel_out * height_col + h_out) * width_col + w_out;
            int data_im_offset = (channel_in * height + h_in) * width + w_in;

            for (int i = 0; i < ksize; ++i) {
                for (int j = 0; j < ksize; ++j) {
                    int h = h_in + i;
                    int w = w_in + j;

                    data_col[data_col_offset] =
                        (h >= 0 && w >= 0 && h < height && w < width)
                            ? data_im[data_im_offset + i * width + j]
                            : 0;

                    // data_col[data_col_offset] = data_im[data_im_offset + i *
                    // width + j];

                    data_col_offset += height_col * width_col;
                }
            }
        }
    });

static const char* const bcnn_cl_kernel_forward_maxpool_source =
    BCNN_CL_KERNEL_TO_STRING(__kernel void forward_maxpool_layer_kernel(
        int n, int in_h, int in_w, int in_c, int stride, int size, int pad,
        __global float* input, __global float* output, __global int* indexes) {
        int h = (in_h - 1) / stride + 1;
        int w = (in_w - 1) / stride + 1;
        int c = in_c;

        int id = (get_group_id(0) + get_group_id(1) * get_num_groups(0)) *
                     get_local_size(0) +
                 get_local_id(0);
        if (id >= n) {
            return;
        }

        int j = id % w;
        id /= w;
        int i = id % h;
        id /= h;
        int k = id % c;
        id /= c;
        int b = id;

        int out_index = j + w * (i + h * (k + c * b));
        float max = -FLT_MAX;
        int max_i = -1;
        for (int l = 0; l < size; ++l) {
            for (int m = 0; m < size; ++m) {
                int cur_h = i * stride + l;
                int cur_w = j * stride + m;
                int index = cur_w + in_w * (cur_h + in_h * (k + b * in_c));
                int valid =
                    (cur_h >= 0 && cur_h < in_h && cur_w >= 0 && cur_w < in_w);
                float val = (valid != 0) ? input[index] : -FLT_MAX;
                max_i = (val > max) ? index : max_i;
                max = (val > max) ? val : max;
            }
        }
        output[out_index] = max;
        indexes[out_index] = max_i;
    });

#endif  // BCNN_CL_KERNELS