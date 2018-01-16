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

#include <bh/bh.h>
#include <bh/bh_timer.h>
#include <bh/bh_error.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>

#include "bcnn/bcnn.h"

int create_network(bcnn_net *net)
{
    net->learner.optimizer = SGD;
    net->learner.learning_rate = 0.005f;
    net->learner.gamma = 0.00002f;
    net->learner.decay = 0.0005f;
    net->learner.momentum = 0.9f;
    net->learner.policy = SIGMOID;
    net->learner.step = 40000;
    net->learner.beta1 = 0.9f;
    net->learner.beta2 = 0.999f;
    net->max_batches = 50000;

    bcnn_net_set_input_shape(net, 28, 28, 3, 128);
    
    bcnn_add_convolutional_layer(net, 32, 3, 1, 1, 0, XAVIER, RELU, 0, "input", "conv1_1");
    bcnn_add_batchnorm_layer(net, "conv1_1", "bn1_1");
    bcnn_add_convolutional_layer(net, 32, 3, 1, 1, 0, XAVIER, RELU, 0, "bn1_1", "conv1_2");
    bcnn_add_batchnorm_layer(net, "conv1_2", "bn1_2");
    bcnn_add_convolutional_layer(net, 32, 3, 1, 1, 0, XAVIER, RELU, 0, "bn1_2", "conv1_3");
    bcnn_add_batchnorm_layer(net, "conv1_3", "bn1_3");
    bcnn_add_maxpool_layer(net, 2, 2, "bn1_3", "pool1");

    bcnn_add_convolutional_layer(net, 64, 3, 1, 1, 0, XAVIER, RELU, 0, "pool1", "conv2_1");
    bcnn_add_batchnorm_layer(net, "conv2_1", "bn2_1");
    bcnn_add_convolutional_layer(net, 64, 3, 1, 1, 0, XAVIER, RELU, 0, "bn2_1", "conv2_2");
    bcnn_add_batchnorm_layer(net, "conv2_2", "bn2_2");
    bcnn_add_convolutional_layer(net, 64, 3, 1, 1, 0, XAVIER, RELU, 0, "bn2_2", "conv2_3");
    bcnn_add_batchnorm_layer(net, "conv2_3", "bn2_3");
    bcnn_add_maxpool_layer(net, 2, 2, "bn2_3", "pool2");

    bcnn_add_fullc_layer(net, 512, XAVIER, RELU, 0, "pool2", "fc1");
    bcnn_add_batchnorm_layer(net, "fc1", "bn3");

    bcnn_add_fullc_layer(net, 10, XAVIER, RELU, 0, "bn3", "fc2");

    bcnn_add_softmax_layer(net, "fc2", "softmax");
    bcnn_add_cost_layer(net, COST_ERROR, 1.0f, "softmax", "label", "cost");

    // Data augmentation
    net->data_aug.range_shift_x = 10;
    net->data_aug.range_shift_y = 10;
    net->data_aug.rotation_range = 20.0f;
    net->data_aug.max_brightness = 60;
    net->data_aug.min_brightness = -60;
    net->data_aug.max_contrast = 1.5f;
    net->data_aug.min_contrast = 0.6f;
    net->data_aug.random_fliph = 1;

    // Target
    net->prediction_type = CLASSIFICATION;

    return 0;
}


int predict_cifar10(bcnn_net *net, char *test_img, float *error,
    int nb_pred, char *pred_out)
{
    int i = 0, j = 0, n = 0, k = 0;
    float *out = NULL;
    float err = 0.0f, error_batch = 0.0f;
    FILE *f = NULL;
    bcnn_iterator data_iter = { 0 };
    int nb = net->nb_connections;
    int output_size = bcnn_tensor_get_size3d(&net->nodes[net->connections[nb - 2].dst[0]].tensor);

    bcnn_init_iterator(net, &data_iter, test_img, NULL, "cifar10");

    f = fopen(pred_out, "wt");
    if (f == NULL) {
        fprintf(stderr, "[ERROR] Could not open file %s", pred_out);
        return -1;
    }

    bcnn_compile_net(net, "predict");

    n = nb_pred / net->batch_size;
    for (i = 0; i < n; ++i) {
        bcnn_predict_on_batch(net, &data_iter, &out, &error_batch);
        err += error_batch;
        // Save predictions
        for (j = 0; j < net->batch_size; ++j) {
            for (k = 0; k < output_size; ++k)
                fprintf(f, "%f ", out[j * output_size + k]);
            fprintf(f, "\n");
        }

    }
    // Last predictions (Have to do this because batch_size is set to 16 yet the
    // number of samples of mnist test data is not a multiple of 16)
    n = nb_pred % net->batch_size;
    if (n > 0) {
        for (i = 0; i < n; ++i) {
            bcnn_predict_on_batch(net, &data_iter, &out, &error_batch);
            err += error_batch;
            // Save predictions
            for (k = 0; k < output_size; ++k)
                fprintf(f, "%f ", out[k]);
            fprintf(f, "\n");

        }
    }
    *error = err / nb_pred;

    if (f != NULL)
        fclose(f);
    bcnn_free_iterator(&data_iter);
    return 0;
}


int train_cifar10(bcnn_net *net, char *train_img,
    char *test_img, int nb_iter, int eval_period, float *error)
{
    float error_batch = 0.0f, sum_error = 0.0f, error_valid = 0.0f;
    int i = 0;
    bh_timer t = { 0 }, tp = { 0 };
    bcnn_iterator data_iter = { 0 };

    if (bcnn_init_iterator(net, &data_iter, train_img, NULL, "cifar10") != 0)
        return -1;

    bcnn_compile_net(net, "train");

    bh_timer_start(&t);
    for (i = 0; i < nb_iter; ++i) {
        bcnn_train_on_batch(net, &data_iter, &error_batch);
        sum_error += error_batch;

        if (i % eval_period == 0 && i > 0) {
            bh_timer_stop(&t);
            bh_timer_start(&tp);
            predict_cifar10(net, test_img, &error_valid, 10000, "predictions_cifar10.txt");
            bh_timer_stop(&tp);
            fprintf(stderr, "iter= %d train-error= %f test-error= %f training-time= %lf sec inference-time= %lf sec\n", i,
                sum_error / (eval_period * net->batch_size), error_valid,
                bh_timer_get_msec(&t) / 1000, bh_timer_get_msec(&tp) / 1000);
            fflush(stderr);
            bh_timer_start(&t);
            sum_error = 0;
            // Reschedule net for training
            bcnn_compile_net(net, "train");
        }
        
    }

    bcnn_free_iterator(&data_iter);
    *error = (float)sum_error / (eval_period * net->batch_size);

    return 0;
}


int run(char *train_data, char *test_data)
{
    float error_train = 0.0f, error_test = 0.0f;
    bcnn_net *net = NULL;
    
    bcnn_init_net(&net);
    bh_info("Create Network...");
    create_network(net);

    bh_info("Start training...");
    if (train_cifar10(net, train_data, test_data, 4000000, 100, &error_train) != 0)
        bh_error("Can not perform training", -1);
    
    bh_info("Start prediction...");
    predict_cifar10(net, test_data, &error_test, 10000, "predictions_cifar10.txt");
    bh_info("Prediction ended successfully");

    bcnn_end_net(&net);

    return 0;
}


int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <train_data> <test_data>\n", argv[0]);
        return -1;
    }
    run(argv[1], argv[2]);
    return 0;
}
