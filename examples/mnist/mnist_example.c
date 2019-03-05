/*
 * Copyright (c) 2016-present Jean-Noel Braun.
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

#include <stdio.h>
#include <string.h>

#include "bcnn/bcnn.h"

#include <bh/bh_timer.h>  // For timing

int create_network(bcnn_net *net) {
    bcnn_set_input_shape(net, 28, 28, 1, 16);

    bcnn_add_convolutional_layer(net, 32, 3, 1, 1, 1, 0, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "input", "conv1");
    bcnn_add_batchnorm_layer(net, "conv1", "bn1");
    bcnn_add_maxpool_layer(net, 2, 2, BCNN_PADDING_SAME, "bn1", "pool1");

    bcnn_add_convolutional_layer(net, 32, 3, 1, 1, 1, 0, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "pool1", "conv2");
    bcnn_add_batchnorm_layer(net, "conv2", "bn2");
    bcnn_add_maxpool_layer(net, 2, 2, BCNN_PADDING_SAME, "bn2", "pool2");

    bcnn_add_fullc_layer(net, 256, BCNN_FILLER_XAVIER, BCNN_ACT_RELU, 0,
                         "pool2", "fc1");
    bcnn_add_batchnorm_layer(net, "fc1", "bn3");

    bcnn_add_fullc_layer(net, 10, BCNN_FILLER_XAVIER, BCNN_ACT_RELU, 0, "bn3",
                         "fc2");

    bcnn_add_softmax_layer(net, "fc2", "softmax");
    bcnn_add_cost_layer(net, BCNN_LOSS_EUCLIDEAN, BCNN_METRIC_ERROR_RATE, 1.0f,
                        "softmax", "label", "cost");

    return 0;
}

int predict_mnist(bcnn_net *net, const char *test_img, const char *test_label,
                  int num_pred, float *avg_loss, const char *pred_out) {
    bcnn_set_mode(net, BCNN_MODE_VALID);

    FILE *f = fopen(pred_out, "wt");
    if (f == NULL) {
        fprintf(stderr, "[ERROR] Could not open file %s", pred_out);
        return -1;
    }

    int batch_size = bcnn_get_batch_size(net);
    int n = num_pred / batch_size;
    float loss = 0.0f;
    for (int i = 0; i < n; ++i) {
        bcnn_tensor *out = NULL;
        loss += bcnn_predict_on_batch(net, &out);
        // Save predictions
        int out_sz = out->w * out->h * out->c;
        for (int j = 0; j < batch_size; ++j) {
            for (int k = 0; k < out_sz; ++k)
                fprintf(f, "%f ", out->data[j * out_sz + k]);
        }
    }
    *avg_loss = loss / num_pred;

    if (f != NULL) {
        fclose(f);
    }
    return 0;
}

int train_mnist(bcnn_net *net, const char *train_img, const char *train_label,
                const char *test_img, const char *test_label, int num_iter,
                int eval_period, float *error) {
    float error_batch = 0.0f, sum_error = 0.0f, error_valid = 0.0f;
    bh_timer t = {0}, tp = {0};

    bcnn_set_mode(net, BCNN_MODE_TRAIN);

    bh_timer_start(&t);
    for (int i = 0; i < num_iter; ++i) {
        sum_error += bcnn_train_on_batch(net);

        if (i % eval_period == 0 && i > 0) {
            bh_timer_stop(&t);
            bh_timer_start(&tp);
            predict_mnist(net, test_img, test_label, 10000, &error_valid,
                          "pred_mnist.txt");
            bh_timer_stop(&tp);
            fprintf(stderr,
                    "iter= %d train-error= %f test-error= %f training-time= "
                    "%lf sec inference-time= %lf sec\n",
                    i, sum_error / (eval_period * bcnn_get_batch_size(net)),
                    error_valid, bh_timer_get_msec(&t) / 1000,
                    bh_timer_get_msec(&tp) / 1000);
            fflush(stderr);
            bh_timer_start(&t);
            sum_error = 0;
            // Reschedule net for training
            bcnn_set_mode(net, BCNN_MODE_TRAIN);
        }
    }

    *error = (float)sum_error / (eval_period * bcnn_get_batch_size(net));

    return 0;
}

int run(const char *train_img, const char *train_label, const char *test_img,
        const char *test_label) {
    float error_train = 0.0f, error_test = 0.0f;
    bcnn_net *net = NULL;

    bcnn_init_net(&net, BCNN_MODE_TRAIN);

    fprintf(stderr, "Create Network...\n");
    create_network(net);

    // Setup training parameters
    bcnn_set_sgd_optimizer(net, /*learning_rate=*/0.003f, /*momentum=*/0.9f);
    bcnn_set_learning_rate_policy(net, BCNN_LR_DECAY_SIGMOID, 0.00002f, 0.f,
                                  0.f, 50000, 40000);
    bcnn_set_weight_regularizer(net, 0.0005f);

    // Setup loaders for train and validation dataset
    bcnn_set_data_loader(net, BCNN_LOAD_MNIST, train_img, train_label, test_img,
                         test_label);

    // Setup data augmentation
    bcnn_augment_data_with_shift(net, 5, 5);
    bcnn_augment_data_with_rotation(net, 30.f);

    // Finalize net config
    bcnn_compile_net(net);

    fprintf(stderr, "Start training...\n");
    if (train_mnist(net, train_img, train_label, test_img, test_label, 100000,
                    50, &error_train) != 0) {
        fprintf(stderr, "Can not perform training");
        bcnn_end_net(&net);
        return -1;
    }

    fprintf(stderr, "Start prediction...\n");
    bcnn_set_mode(net, BCNN_MODE_VALID);
    predict_mnist(net, test_img, test_label, 10000, &error_test,
                  "pred_mnist.txt");
    fprintf(stderr, "Prediction ended successfully\n");
    bcnn_end_net(&net);

    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "Usage: %s <train_img> <train_label> <test_img> <test_label>\n",
                argv[0]);
        return -1;
    }
    run(argv[1], argv[2], argv[3], argv[4]);
    return 0;
}
