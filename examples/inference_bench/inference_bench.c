#include <float.h>
#include <math.h>
#include <stdio.h>

#include <bcnn/bcnn.h>
#include <bh/bh_macros.h>
#include <bh/bh_timer.h>
#include <bip/bip.h>

#include "bcnn_utils.h"
#include "kernels/bcnn_mat.h"

void show_usage(int argc, char **argv) {
    fprintf(stderr,
            "Usage: ./%s <input> <config> <model> <runs> <num_threads> [mean] "
            "[scale]\n",
            argv[0]);
    fprintf(stderr,
            "\tRequired:\n"
            "\t\t<input>: path to input image.\n"
            "\t\t<config>: path to configuration file.\n"
            "\t\t<model>: path to model weights file.\n"
            "\t\t<runs>: number of inferences to be run.\n"
            "\t\t<num_threads>: number of threads used.\n"
            "\tOptional:\n"
            "\t\t[mean]: value between [0;255] to be substracted to input "
            "pixel values. Default: 127.5f\n"
            "\t\t[scale]: scale value to be applied to pixel values. Default: "
            "1 / 127.5\n");
}

/* This demonstrates how to run a network inference given an input image and
 * benchmark the inference speed */
int main(int argc, char **argv) {
    if (argc < 6) {
        show_usage(argc, argv);
        return 1;
    }
    /* Create net */
    bcnn_net *net = NULL;
    bcnn_init_net(&net, BCNN_MODE_PREDICT);
    bcnn_set_num_threads(net, atoi(argv[5]));
    fprintf(stderr, "Number of threads used: %d\n", bcnn_get_num_threads(net));
    /* Load net config and weights */
    if (bcnn_load_net(net, argv[2], argv[3]) != BCNN_SUCCESS) {
        bcnn_end_net(&net);
        return -1;
    }
    /* Compile net */
    if (bcnn_compile_net(net) != BCNN_SUCCESS) {
        bcnn_end_net(&net);
        return -1;
    }
    /* Load test image */
    unsigned char *img = NULL;
    int w, h, c;
    int ret = bip_load_image(argv[1], &img, &w, &h, &c);
    if (ret != BIP_SUCCESS) {
        fprintf(stderr, "[ERROR] Failed to open image %s\n", argv[1]);
        return -1;
    }
    /* Get a pointer to the input tensor */
    bcnn_tensor *input_tensor = bcnn_get_tensor_by_name(net, "input");
    /* Check if input image depth is consistent */
    if (c != input_tensor->c) {
        fprintf(stderr,
                "[ERROR] Image depth is not supported: expected %d, found %d\n",
                input_tensor->c, c);
        return -1;
    }
    /* Resize image if needed */
    if (input_tensor->w != w || input_tensor->h != h) {
        uint8_t *img_rz = (uint8_t *)calloc(
            input_tensor->w * input_tensor->h * c, sizeof(uint8_t));
        bip_resize_bilinear(img, w, h, w * c, img_rz, input_tensor->w,
                            input_tensor->h, input_tensor->w * c, c);
        free(img);
        img = img_rz;
    }

    /* Fill the input tensor with the current image data */
    float mean = 127.5f;
    float scale = 1 / 127.5f;
    if (argc == 8) {
        mean = atof(argv[6]);
        scale = atof(argv[7]);
    }
    bcnn_fill_tensor_with_image(net, img, input_tensor->w, input_tensor->h, c,
                                scale, 0, mean, mean, mean,
                                /*tensor_index=*/0, /*batch_index=*/0);

    int sz =
        input_tensor->w * input_tensor->h * input_tensor->c * input_tensor->n;
    /* Setup timer */
    bh_timer t = {0};
    double elapsed_min = DBL_MAX;
    double elapsed_max = -DBL_MAX;
    double elapsed_avg = 0;
    int num_runs = atoi(argv[4]) > 0 ? atoi(argv[4]) : 1;
    /* Inference runs */
    for (int i = 0; i < num_runs; ++i) {
        bh_timer_start(&t);
        /* Run the forward pass */
        bcnn_forward(net);
        bh_timer_stop(&t);
        double elapsed = bh_timer_get_msec(&t);
        elapsed_avg += elapsed;
        elapsed_min = bh_min(elapsed_min, elapsed);
        elapsed_max = bh_max(elapsed_max, elapsed);
    }
    elapsed_avg /= num_runs;
    fprintf(
        stderr,
        "model %s %s img %s : min= %lf msecs max= %lf msecs avg= %lf msecs\n",
        argv[2], argv[3], argv[1], elapsed_min, elapsed_max, elapsed_avg);
    /* Get the output tensor pointer */
    /* Note: output tensor is expected to be named 'out' */
    bcnn_tensor *out = bcnn_get_tensor_by_name(net, "out");
    if (out != NULL) {
        float max_p = -1.f;
        int max_class = -1;
#ifdef BCNN_USE_CUDA
        bcnn_cuda_memcpy_dev2host(out->data_gpu, out->data, out->c);
#endif
        for (int i = 0; i < out->c; ++i) {
            if (out->data[i] > max_p) {
                max_p = out->data[i];
                max_class = i;
            }
            if (out->data[i] > 0.2) {
                fprintf(stderr, "candidate class: %d (%f)\n", i, out->data[i]);
            }
        }
        fprintf(stderr, "best class predicted: %d (%f)\n", max_class, max_p);
    }
    /* Cleanup */
    bcnn_end_net(&net);
    free(img);
    return 0;
}