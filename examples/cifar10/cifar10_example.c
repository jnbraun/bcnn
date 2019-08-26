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
#include <stdlib.h>

#include <bh/bh_timer.h>

#include "bcnn/bcnn.h"

typedef enum { SIMPLENET, RESNET18 } model_type;

static int simple_net(bcnn_net *net) {
    bcnn_set_input_shape(net, 32, 32, 3, 128);

    bcnn_add_convolutional_layer(net, 32, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "input", "conv1_1");
    bcnn_add_convolutional_layer(net, 32, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "conv1_1", "conv1_2");
    bcnn_add_convolutional_layer(net, 32, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "conv1_2", "conv1_3");
    bcnn_add_maxpool_layer(net, 2, 2, BCNN_PADDING_SAME, "conv1_3", "pool1");

    bcnn_add_convolutional_layer(net, 64, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "pool1", "conv2_1");
    bcnn_add_convolutional_layer(net, 64, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "conv2_1", "conv2_2");
    bcnn_add_convolutional_layer(net, 64, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "conv2_2", "conv2_3");
    bcnn_add_maxpool_layer(net, 2, 2, BCNN_PADDING_SAME, "conv2_3", "pool2");

    bcnn_add_fullc_layer(net, 512, BCNN_FILLER_XAVIER, BCNN_ACT_RELU, 0,
                         "pool2", "fc1");
    bcnn_add_batchnorm_layer(net, "fc1", "bn3");

    bcnn_add_fullc_layer(net, 10, BCNN_FILLER_XAVIER, BCNN_ACT_RELU, 0, "bn3",
                         "fc2");

    bcnn_add_softmax_layer(net, "fc2", "softmax");
    bcnn_add_cost_layer(net, BCNN_LOSS_EUCLIDEAN, BCNN_METRIC_ERROR_RATE, 1.0f,
                        "softmax", "label", "cost");

    return 0;
}

static int resnet18(bcnn_net *net) {
    bcnn_set_input_shape(net, 32, 32, 3, 32);

    bcnn_add_convolutional_layer(net, 64, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "input", "conv1");
    // Block 1_1
    bcnn_add_convolutional_layer(net, 64, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "conv1", "conv1_1");
    bcnn_add_convolutional_layer(net, 64, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_NONE, 0, "conv1_1", "conv1_2");
    bcnn_add_eltwise_layer(net, BCNN_ACT_RELU, "conv1", "conv1_2",
                           "conv1_add1");
    // Block 1_2
    bcnn_add_convolutional_layer(net, 64, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "conv1_add1", "conv1_3");
    bcnn_add_convolutional_layer(net, 64, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_NONE, 0, "conv1_3", "conv1_4");
    bcnn_add_eltwise_layer(net, BCNN_ACT_RELU, "conv1_add1", "conv1_4",
                           "conv1_add2");
    // Block 2_1
    bcnn_add_convolutional_layer(net, 128, 3, 2, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "conv1_add2", "conv2_1");
    bcnn_add_convolutional_layer(net, 128, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_NONE, 0, "conv2_1", "conv2_2");
    bcnn_add_convolutional_layer(net, 128, 1, 2, 0, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_NONE, 0, "conv1_add2", "conv2_res1");
    bcnn_add_eltwise_layer(net, BCNN_ACT_RELU, "conv2_res1", "conv2_2",
                           "conv2_add1");
    // Block 2_2
    bcnn_add_convolutional_layer(net, 128, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "conv2_add1", "conv2_3");
    bcnn_add_convolutional_layer(net, 128, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_NONE, 0, "conv2_3", "conv2_4");
    bcnn_add_eltwise_layer(net, BCNN_ACT_RELU, "conv2_add1", "conv2_4",
                           "conv2_add2");
    // Block 3_1
    bcnn_add_convolutional_layer(net, 256, 3, 2, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "conv2_add2", "conv3_1");
    bcnn_add_convolutional_layer(net, 256, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_NONE, 0, "conv3_1", "conv3_2");
    bcnn_add_convolutional_layer(net, 256, 1, 2, 0, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_NONE, 0, "conv2_add2", "conv3_res1");
    bcnn_add_eltwise_layer(net, BCNN_ACT_RELU, "conv3_res1", "conv3_2",
                           "conv3_add1");
    // Block 3_2
    bcnn_add_convolutional_layer(net, 256, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "conv3_add1", "conv3_3");
    bcnn_add_convolutional_layer(net, 256, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_NONE, 0, "conv3_3", "conv3_4");
    bcnn_add_eltwise_layer(net, BCNN_ACT_RELU, "conv3_add1", "conv3_4",
                           "conv3_add2");
    // Block 4_1
    bcnn_add_convolutional_layer(net, 512, 3, 2, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "conv3_add2", "conv4_1");
    bcnn_add_convolutional_layer(net, 512, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_NONE, 0, "conv4_1", "conv4_2");
    bcnn_add_convolutional_layer(net, 512, 1, 2, 0, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_NONE, 0, "conv3_add2", "conv4_res1");
    bcnn_add_eltwise_layer(net, BCNN_ACT_RELU, "conv4_res1", "conv4_2",
                           "conv4_add1");
    // Block 4_2
    bcnn_add_convolutional_layer(net, 512, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_RELU, 0, "conv4_add1", "conv4_3");
    bcnn_add_convolutional_layer(net, 512, 3, 1, 1, 1, 1, BCNN_FILLER_XAVIER,
                                 BCNN_ACT_NONE, 0, "conv4_3", "conv4_4");
    bcnn_add_eltwise_layer(net, BCNN_ACT_RELU, "conv4_add1", "conv4_4",
                           "conv4_add2");
    //////////////////
    bcnn_add_avgpool_layer(net, "conv4_add2", "pool");
    // bcnn_add_maxpool_layer(net, 4, 4, BCNN_PADDING_SAME, "conv4_add2",
    // "pool");
    bcnn_add_fullc_layer(net, 10, BCNN_FILLER_XAVIER, BCNN_ACT_NONE, 0, "pool",
                         "fc");
    bcnn_add_softmax_layer(net, "fc", "softmax");
    bcnn_add_cost_layer(net, BCNN_LOSS_EUCLIDEAN, BCNN_METRIC_ERROR_RATE, 1.0f,
                        "softmax", "label", "cost");

    return 0;
}

int create_network(bcnn_net *net, model_type type) {
    // Define the network topology
    switch (type) {
        case SIMPLENET:
            simple_net(net);
            break;
        case RESNET18:
            resnet18(net);
            break;
        default:
            simple_net(net);
            break;
    }

    return 0;
}

int predict_cifar10(bcnn_net *net, char *test_img, int nb_pred, float *avg_loss,
                    char *pred_out) {
    bcnn_set_mode(net, BCNN_MODE_VALID);
    FILE *f = fopen(pred_out, "wt");
    if (f == NULL) {
        fprintf(stderr, "[ERROR] Could not open file %s\n", pred_out);
        return -1;
    }

    int batch_size = bcnn_get_batch_size(net);
    int n = nb_pred / batch_size;
    float loss = 0.0f;
    for (int i = 0; i < n; ++i) {
        bcnn_tensor *out = NULL;
        loss += bcnn_predict_on_batch(net, &out);
        // Save predictions
        for (int j = 0; j < batch_size; ++j) {
            int out_sz = out->w * out->h * out->c;
            for (int k = 0; k < out_sz; ++k)
                fprintf(f, "%f ", out->data[j * out_sz + k]);
            fprintf(f, "\n");
        }
    }
    *avg_loss = loss / nb_pred;

    if (f != NULL) {
        fclose(f);
    }
    return 0;
}

int train_cifar10(bcnn_net *net, char *train_img, char *test_img, int nb_iter,
                  int eval_period, float *error) {
    float sum_error = 0.0f, error_valid = 0.0f;
    bh_timer t = {0}, tp = {0};

    bcnn_set_mode(net, BCNN_MODE_TRAIN);

    bh_timer_start(&t);
    for (int i = 0; i < nb_iter; ++i) {
        sum_error += bcnn_train_on_batch(net);
        if (i % eval_period == 0 && i > 0) {
            bh_timer_stop(&t);
            bh_timer_start(&tp);
            predict_cifar10(net, test_img, 10000, &error_valid,
                            "predictions_cifar10.txt");
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

int run(char *train_data, char *test_data, model_type model) {
    float error_train = 0.0f, error_test = 0.0f;
    bcnn_net *net = NULL;

    bcnn_init_net(&net, BCNN_MODE_TRAIN);
    fprintf(stderr, "Create Network...\n");
    create_network(net, model);

    // Setup training parameters
    bcnn_set_adam_optimizer(net, /*learning_rate=*/0.005f, /*beta1=*/0.9f,
                            /*beta2=*/0.999f);
    bcnn_set_learning_rate_policy(net, BCNN_LR_DECAY_SIGMOID, 0.00002f, 0.f,
                                  0.f, 1000000, 100000);
    bcnn_set_weight_regularizer(net, 0.0005f);

    // Setup loaders for train and validation dataset
    if (bcnn_set_data_loader(net, BCNN_LOAD_CIFAR10, train_data, NULL,
                             test_data, NULL) != BCNN_SUCCESS) {
        bcnn_end_net(&net);
        return -1;
    }

    // Setup data augmentation
    bcnn_augment_data_with_shift(net, 5, 5);
    bcnn_augment_data_with_rotation(net, 15.f);
    bcnn_augment_data_with_flip(net, 1, 0);
    bcnn_augment_data_with_color_adjustment(net, -60, 60, 0.6f, 1.5f);

    // Finalize net setup
    bcnn_compile_net(net);

    fprintf(stderr, "Start training...\n");
    if (train_cifar10(net, train_data, test_data, 400000, 10, &error_train) !=
        0) {
        fprintf(stderr, "Can not perform training");
        bcnn_end_net(&net);
        return -1;
    }

    fprintf(stderr, "Start prediction...\n");
    bcnn_set_mode(net, BCNN_MODE_VALID);
    predict_cifar10(net, test_data, 10000, &error_test,
                    "predictions_cifar10.txt");
    fprintf(stderr, "Prediction ended successfully\n");

    bcnn_end_net(&net);

    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <train_data> <test_data> [model]\n",
                argv[0]);
        fprintf(stderr, "\t Required:\n");
        fprintf(stderr,
                "\t\t <train_data> : path to cifar10 training "
                "binary file\n");
        fprintf(stderr,
                "\t\t <test_data> : path to cifar10 testing "
                "binary file\n");
        fprintf(stderr, "\t Optional:\n");
        fprintf(stderr, "\t\t [model]: network model type: (default: 0)\n");
        fprintf(stderr, "\t\t\t0: Simple CNN\n");
        fprintf(stderr, "\t\t\t1: Resnet18\n");
        return -1;
    }
    run(argv[1], argv[2], (model_type)(argc > 3 ? atoi(argv[3]) : 0));
    return 0;
}
