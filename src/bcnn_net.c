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

/* include bh helpers */
#include <bh/bh.h>
#include <bh/bh_error.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>
#include <bh/bh_timer.h>

/* include bip image processing lib */
#include <bip/bip.h>

#include "bcnn/bcnn.h"


int bcnn_init_net(bcnn_net **net)
{
    if (*net == NULL) {
        *net = (bcnn_net *)calloc(1, sizeof(bcnn_net));
    }

    return BCNN_SUCCESS;
}

int bcnn_end_net(bcnn_net **net)
{
    bcnn_free_net(*net);
    bh_free(*net);		

    return BCNN_SUCCESS;
}

int bcnn_free_tensor(bcnn_tensor *tensor)
{
    bh_free(tensor->data);
    bh_free(tensor->grad_data);
#ifdef BCNN_USE_CUDA
    bcnn_cuda_free(tensor->data_gpu);
    bcnn_cuda_free(tensor->grad_data_gpu);
#endif
    return BCNN_SUCCESS;
}

int bcnn_free_connection(bcnn_connection *conn)
{
    if (conn->layer->type != ACTIVATION && 
        conn->layer->type != DROPOUT) {
        bcnn_free_tensor(&conn->dst_tensor);
    }
    bh_free(conn->id);
    bcnn_free_layer(&conn->layer);
    return BCNN_SUCCESS;
}

int bcnn_free_net(bcnn_net *net)
{
    int i;
    bcnn_free_workload(net);
    for (i = 0; i < net->nb_connections; ++i) {
        bcnn_free_connection(&net->connections[i]);
    }
    bh_free(net->connections);
    for (i = 0; i < net->nb_finetune; ++i) {
        bh_free(net->finetune_id[i]);
    }
    bh_free(net->finetune_id);
    return BCNN_SUCCESS;
}

int bcnn_set_param(bcnn_net *net, char *name, char *val)
{
    if (strcmp(name, "input_width") == 0) {
        net->input_node.w = atoi(val);
    }
    else if (strcmp(name, "input_height") == 0) {
        net->input_node.h = atoi(val);
    }
    else if (strcmp(name, "input_channels") == 0) {
        net->input_node.c = atoi(val);
    }
    else if (strcmp(name, "batch_size") == 0) {
        net->input_node.b = atoi(val);
    }
    else if (strcmp(name, "max_batches") == 0) {
        net->max_batches = atoi(val);
    } 
    else if (strcmp(name, "loss") == 0) {
        if (strcmp(val, "error") == 0) {
            net->loss_metric = COST_ERROR;
        }
        else if (strcmp(val, "logloss") == 0) {
            net->loss_metric = COST_LOGLOSS;
        }
        else if (strcmp(val, "sse") == 0) {
            net->loss_metric = COST_SSE;
        }
        else if (strcmp(val, "mse") == 0) {
            net->loss_metric = COST_MSE;
        }
        else if (strcmp(val, "crps") == 0) {
            net->loss_metric = COST_CRPS;
        }
        else if (strcmp(val, "dice") == 0) {
            net->loss_metric = COST_DICE;
        }
        else {
            fprintf(stderr, "[WARNING] Unknown cost metric %s, going with sse\n", val);
            net->loss_metric = COST_SSE;
        }
    }
    else if (strcmp(name, "learning_policy") == 0) {
        if (strcmp(val, "sigmoid") == 0) {
            net->learner.policy = SIGMOID;
        }
        else if (strcmp(val, "constant") == 0) {
            net->learner.policy = CONSTANT;
        }
        else if (strcmp(val, "exp") == 0) {
            net->learner.policy = EXP;
        }
        else if (strcmp(val, "inv") == 0) {
            net->learner.policy = INV;
        }
        else if (strcmp(val, "step") == 0) {
            net->learner.policy = STEP;
        }
        else if (strcmp(val, "poly") == 0) {
            net->learner.policy = POLY;
        }
        else {
            net->learner.policy = CONSTANT;
        }
    }
    else if (strcmp(name, "optimizer") == 0) {
        if (strcmp(val, "sgd") == 0) {
            net->learner.optimizer = SGD;
        }
        else if (strcmp(val, "adam") == 0) {
            net->learner.optimizer = ADAM;
        }
    }
    else if (strcmp(name, "step") == 0) {
        net->learner.step = atoi(val);
    }
    else if (strcmp(name, "learning_rate") == 0) {
        net->learner.learning_rate = (float)atof(val);
    }
    else if (strcmp(name, "beta1") == 0) {
        net->learner.beta1 = (float)atof(val);
    }
    else if (strcmp(name, "beta2") == 0) {
        net->learner.beta2 = (float)atof(val);
    }
    else if (strcmp(name, "decay") == 0) {
        net->learner.decay = (float)atof(val);
    }
    else if (strcmp(name, "momentum") == 0) {
        net->learner.momentum = (float)atof(val);
    }
    else if (strcmp(name, "gamma") == 0) {
        net->learner.gamma = (float)atof(val);
    }
    else if (strcmp(name, "range_shift_x") == 0) {
        net->data_aug.range_shift_x = atoi(val);
    }
    else if (strcmp(name, "range_shift_y") == 0) {
        net->data_aug.range_shift_y = atoi(val);
    }
    else if (strcmp(name, "min_scale") == 0) {
        net->data_aug.min_scale = (float)atof(val);
    }
    else if (strcmp(name, "max_scale") == 0) {
        net->data_aug.max_scale = (float)atof(val);
    }
    else if (strcmp(name, "rotation_range") == 0) {
        net->data_aug.rotation_range = (float)atof(val);
    }
    else if (strcmp(name, "min_contrast") == 0) {
        net->data_aug.min_contrast = (float)atof(val);
    }
    else if (strcmp(name, "max_contrast") == 0) {
        net->data_aug.max_contrast = (float)atof(val);
    }
    else if (strcmp(name, "min_brightness") == 0) {
        net->data_aug.min_brightness = atoi(val);
    }
    else if (strcmp(name, "max_brightness") == 0) {
        net->data_aug.max_brightness = atoi(val);
    }
    else if (strcmp(name, "max_distortion") == 0) {
        net->data_aug.max_distortion = (float)atof(val);
    }
    else if (strcmp(name, "flip_h") == 0) {
        net->data_aug.random_fliph = 1;
    }
    else if (strcmp(name, "mean_r") == 0) {
        net->data_aug.mean_r = (float)atof(val) / 255.0f;
    }
    else if (strcmp(name, "mean_g") == 0) {
        net->data_aug.mean_g = (float)atof(val) / 255.0f;
    }
    else if (strcmp(name, "mean_b") == 0) {
        net->data_aug.mean_b = (float)atof(val) / 255.0f;
    }
    else if (strcmp(name, "swap_to_bgr") == 0) {
        net->data_aug.swap_to_bgr = atoi(val);
    }
    else if (strcmp(name, "no_input_norm") == 0) {
        net->data_aug.no_input_norm = atoi(val);
    }
    else if (strcmp(name, "prediction_type") == 0) {
        if (strcmp(val, "classif") == 0 || strcmp(val, "classification") == 0) {
            net->prediction_type = CLASSIFICATION;
        }
        else if (strcmp(val, "reg") == 0 || strcmp(val, "regression") == 0) {
            net->prediction_type = REGRESSION;
        }
        else if (strcmp(val, "heatmap") == 0 || strcmp(val, "heatmap_regression") == 0) {
            net->prediction_type = HEATMAP_REGRESSION;
        }
        else if (strcmp(val, "segmentation") == 0){
            net->prediction_type = SEGMENTATION;
        }
    }
    else if (strcmp(name, "finetune_id") == 0) {
        net->nb_finetune++;
        if (net->nb_finetune == 1) {
            net->finetune_id = (char **)calloc(net->nb_finetune, sizeof(char *));
        }
        else {
            net->finetune_id = (char **)realloc(net->finetune_id, net->nb_finetune);
        }
        bh_fill_option(&net->finetune_id[net->nb_finetune - 1], val);
    }
    return BCNN_SUCCESS;
}

int bcnn_net_add_connection(bcnn_net *net, bcnn_connection conn)
{
    net->connections = (bcnn_connection *)realloc(net->connections,
        net->nb_connections * sizeof(bcnn_connection));
    net->connections[net->nb_connections - 1] = conn;
    return BCNN_SUCCESS;
}

int bcnn_init_workload(bcnn_net *net)
{
    int i;
    int sz = bcnn_get_tensor_size(&net->input_node);
    int n = net->nb_connections;
    int k = (net->connections[n - 1].layer->type == COST ? (n - 2) : (n - 1));
    int output_size = bcnn_get_tensor_size(&net->connections[k].dst_tensor);

    net->input_buffer = (unsigned char *)calloc(net->input_node.w * net->input_node.h * net->input_node.c, 1);
    net->input_node.data = (float *)calloc(sz, sizeof(float));
    for (i = 0; i < n; ++i) {
        if (net->connections[i].layer->type == COST) {
            net->connections[i].label = (float *)calloc(output_size, sizeof(float));
        }
    }
    net->input_node.grad_data = NULL;
    net->connections[0].src_tensor.data = net->input_node.data;
    net->connections[0].src_tensor.grad_data = net->input_node.grad_data;
#ifdef BCNN_USE_CUDA
    net->input_node.data_gpu = bcnn_cuda_malloc_f32(sz);
    net->workspace_gpu = bcnn_cuda_malloc_f32(net->workspace_size);
    for (i = 0; i < n; ++i) {
        if (net->connections[i].layer->type == CONVOLUTIONAL) {
            net->connections[i].layer->conv_workspace_gpu = net->workspace_gpu;
        }
        if (net->connections[i].layer->type == COST) {
            net->connections[i].label_gpu = bcnn_cuda_malloc_f32(output_size);
        }
    }
    net->input_node.grad_data_gpu = NULL;
    net->connections[0].src_tensor.data_gpu = net->input_node.data_gpu;
    net->connections[0].src_tensor.grad_data_gpu = net->input_node.grad_data_gpu;
#endif

    return BCNN_SUCCESS;
}


int bcnn_free_workload(bcnn_net *net)
{
    int i;
    int n = net->nb_connections;

    bh_free(net->input_buffer);
    bh_free(net->input_node.data);
    for (i = 0; i < n; ++i) {
        if (net->connections[i].layer->type == COST) bh_free(net->connections[i].label);
    }
#ifdef BCNN_USE_CUDA
    bcnn_cuda_free(net->input_node.data_gpu);
    bcnn_cuda_free(net->workspace_gpu);
    for (i = 0; i < n; ++i) {
        if (net->connections[i].layer->type == COST) bcnn_cuda_free(net->connections[i].label);
    }
#endif
    return 0;
}

int bcnn_compile_net(bcnn_net *net, char *phase)
{
    int i;

    if (strcmp(phase, "train") == 0) {
        net->state = 1;
    }
    else if (strcmp(phase, "predict") == 0) {
        net->state = 0;
    }
    else {
        fprintf(stderr, "[ERROR] bcnn_compile_net: Available option are 'train' and 'predict'");
        return BCNN_INVALID_PARAMETER;
    }
    // State propagation through connections
    for (i = 0; i < net->nb_connections; ++i)
        net->connections[i].state = net->state;

    bcnn_free_workload(net);
    bcnn_init_workload(net);

    return BCNN_SUCCESS;
}


int bcnn_forward(bcnn_net *net)
{
     int i;
     int output_size = 0;
     bcnn_connection conn = { 0 };


     for (i = 0; i < net->nb_connections; ++i) {
        conn = net->connections[i];
        output_size = bcnn_get_tensor_size(&conn.dst_tensor);
#ifdef BCNN_USE_CUDA
        if (conn.dst_tensor.grad_data_gpu != NULL)
            bcnn_cuda_fill_f32(output_size, 0.0f, conn.dst_tensor.grad_data_gpu, 1);
#else
        if (conn.dst_tensor.grad_data != NULL)
            memset(conn.dst_tensor.grad_data, 0, output_size * sizeof(float));
#endif
        switch (conn.layer->type) {
        case CONVOLUTIONAL:
            bcnn_forward_conv_layer(&conn);
            break;
        case DECONVOLUTIONAL:
            bcnn_forward_deconv_layer(&conn);
            break;
        case DEPTHWISE_CONV:
            bcnn_forward_depthwise_sep_conv_layer(&conn);
            break;
        case ACTIVATION:
            bcnn_forward_activation_layer(&conn);
            break;
        case BATCHNORM:
            bcnn_forward_batchnorm_layer(&conn);
            break;
        case FULL_CONNECTED:
            bcnn_forward_fullc_layer(&conn);
            break;
        case MAXPOOL:
            bcnn_forward_maxpool_layer(&conn);
            break;
        case SOFTMAX:
            bcnn_forward_softmax_layer(&conn);
            break;
        case DROPOUT:
            bcnn_forward_dropout_layer(&conn);
            break;
        case CONCAT:
            bcnn_forward_concat_layer(net, &conn);
            break;
        case COST:
            bcnn_forward_cost_layer(&conn);
            break;
        default:
            break;
        }
    }

    return BCNN_SUCCESS;
}


int bcnn_backward(bcnn_net *net)
{
    int i;
    bcnn_connection conn = { 0 };

    for (i = net->nb_connections - 1; i >= 0; --i) {
        conn = net->connections[i];
        switch (conn.layer->type) {
        case CONVOLUTIONAL:
            bcnn_backward_conv_layer(&conn);
            break;
        case DECONVOLUTIONAL:
            bcnn_backward_deconv_layer(&conn);
            break;
        case DEPTHWISE_CONV:
            bcnn_backward_depthwise_sep_conv_layer(&conn);
            break;
        case ACTIVATION:
            bcnn_backward_activation_layer(&conn);
            break;
        case BATCHNORM:
            bcnn_backward_batchnorm_layer(&conn);
            break;
        case FULL_CONNECTED:
            bcnn_backward_fullc_layer(&conn);
            break;
        case MAXPOOL:
            bcnn_backward_maxpool_layer(&conn);
            break;
        case SOFTMAX:
            bcnn_backward_softmax_layer(&conn);
            break;
        case DROPOUT:
            bcnn_backward_dropout_layer(&conn);
            break;
        case CONCAT:
            bcnn_backward_concat_layer(net, &conn);
            break;
        case COST:
            bcnn_backward_cost_layer(&conn);
            break;
        default:
            break;
        }
    }
    return BCNN_SUCCESS;
}


int bcnn_iter_batch(bcnn_net *net, bcnn_iterator *iter)
{
    int i, j, sz = net->input_node.w * net->input_node.h * net->input_node.c, n, offset;
    int sz_img = iter->input_width * iter->input_height * iter->input_depth;
    int nb = net->nb_connections;
    int w, h, c;
    unsigned char *img_tmp = NULL;
    float *x = net->input_node.data;
    float *y = net->connections[nb - 1].label;
    float x_scale, y_scale;
    int x_pos, y_pos;
    int use_buffer_img = (net->task == TRAIN && net->state != 0 &&
        (net->data_aug.range_shift_x != 0 || net->data_aug.range_shift_y != 0 ||
        net->data_aug.rotation_range != 0 || net->data_aug.random_fliph != 0));
    bcnn_data_augment *param = &(net->data_aug);
    int input_size = bcnn_get_tensor_size(&net->input_node);
    int output_size = net->connections[nb - 2].dst_tensor.w * 
        net->connections[nb - 2].dst_tensor.h *
        net->connections[nb - 2].dst_tensor.c;

    memset(x, 0, sz * net->input_node.b * sizeof(float));
    if (net->task != PREDICT) {
        memset(y, 0, output_size * net->input_node.b * sizeof(float));
    }

    if (use_buffer_img) {
        img_tmp = (unsigned char *)calloc(/*sz*/sz_img, sizeof(unsigned char));
    }
    
    if (iter->type == ITER_MNIST || iter->type == ITER_CIFAR10) {
        for (i = 0; i < net->input_node.b; ++i) {
            //bcnn_mnist_next_iter(net, iter);
            bcnn_advance_iterator(net, iter);
            // Data augmentation
            if (net->task == TRAIN && net->state)
                bcnn_data_augmentation(iter->input_uchar, iter->input_width, iter->input_height, iter->input_depth, param, img_tmp);
            //bip_write_image("test.png", iter->input_uchar, iter->input_width, iter->input_height, iter->input_depth, iter->input_width * iter->input_depth);
            if (net->input_node.w < iter->input_width || net->input_node.h < iter->input_height) {
                bip_crop_image(iter->input_uchar, iter->input_width, iter->input_height, iter->input_width * iter->input_depth,
                    (iter->input_width - net->input_node.w) / 2, (iter->input_height - net->input_node.h) / 2,
                    net->input_buffer, net->input_node.w, net->input_node.h, net->input_node.w * net->input_node.c, net->input_node.c);
                bcnn_convert_img_to_float(net->input_buffer, net->input_node.w, net->input_node.h, net->input_node.c, param->no_input_norm,
                    param->swap_to_bgr, param->mean_r, param->mean_g, param->mean_b, x);
            }
            else
                bcnn_convert_img_to_float(iter->input_uchar, net->input_node.w, net->input_node.h, net->input_node.c, param->no_input_norm,
                    param->swap_to_bgr, param->mean_r, param->mean_g, param->mean_b, x);	
            //bip_write_image("test1.png", tmp_buf, net->input_node.w, net->input_node.h, net->input_node.c, net->input_node.w * net->input_node.c);
            x += sz;
            if (net->task != PREDICT) {
                // Load truth
                y[iter->label_int[0]] = 1;
                y += output_size;
            }
        }
    }
    else if (iter->type == ITER_BIN) {
        for (i = 0; i < net->input_node.b; ++i) {
            //bcnn_bin_iter(net, iter);
            bcnn_advance_iterator(net, iter);
            // Data augmentation
            if (net->task == TRAIN && net->state)
                bcnn_data_augmentation(iter->input_uchar, net->input_node.w, net->input_node.h, net->input_node.c, param, img_tmp);
            bcnn_convert_img_to_float(iter->input_uchar, net->input_node.w, net->input_node.h, net->input_node.c, param->no_input_norm,
                param->swap_to_bgr, param->mean_r, param->mean_g, param->mean_b, x);
            x += sz;
            if (net->task != PREDICT) {
                // Load truth
                switch (net->prediction_type) {
                case CLASSIFICATION:
                    y[(int)iter->label_float[0]] = 1;
                    y += output_size;
                    break;
                case REGRESSION:
                    for (j = 0; j < iter->label_width; ++j) {
                        y[j] = iter->label_float[j];
                    }
                    y += output_size;
                    break;
                case HEATMAP_REGRESSION:
                    // Load truth
                    w = net->connections[net->nb_connections - 2].dst_tensor.w;
                    h = net->connections[net->nb_connections - 2].dst_tensor.h;
                    c = net->connections[net->nb_connections - 2].dst_tensor.c;
                    x_scale = (float)w / (float)net->input_node.w;
                    y_scale = (float)h / (float)net->input_node.h;
                    for (j = 0; j < iter->label_width; j += 2) {
                        if (iter->label_float[j] >= 0 && iter->label_float[j + 1] >= 0) {
                            x_pos = (int)((iter->label_float[j] - net->data_aug.shift_x) * x_scale + 0.5f);
                            y_pos = (int)((iter->label_float[j + 1] - net->data_aug.shift_y) * y_scale + 0.5f);
                            // Set gaussian kernel around (x_pos, y_pos)
                            n = (j / 2) % c;
                            offset = n * w * h + (y_pos * w + x_pos);
                            if (x_pos >= 0 && x_pos < w && y_pos >= 0 && y_pos < h) {
                                y[offset] = 1.0f;
                                if (x_pos > 0) y[offset - 1] = 0.5f;
                                if (x_pos < w - 1) y[offset + 1] = 0.5f;
                                if (y_pos > 0) y[offset - w] = 0.5f;
                                if (y_pos < h - 1) y[offset + w] = 0.5f;
                                if (x_pos > 0 && y_pos > 0) y[offset - w - 1] = 0.25f;
                                if (x_pos < w - 1 && y_pos > 0) y[offset - w + 1] = 0.25f;
                                if (x_pos > 0 && y_pos < h - 1) y[offset + w - 1] = 0.25f;
                                if (x_pos < w - 1 && y_pos < h - 1) y[offset + w + 1] = 0.25f;
                            }
                        }
                    }
                    y += output_size;
                    break;
                default:
                    bh_error("Target type not implemented for this data format. Please use list format instead.", BCNN_INVALID_PARAMETER);
                }
            }
        }
    }
    else if (iter->type == ITER_LIST || iter->type == ITER_CSV) {
        for (i = 0; i < net->input_node.b; ++i) {
            //bcnn_list_iter(net, iter);
            bcnn_advance_iterator(net, iter);
            // Online data augmentation
            if (net->task == TRAIN && net->state)
                bcnn_data_augmentation(iter->input_uchar, net->input_node.w, net->input_node.h, net->input_node.c, param, img_tmp);
            bcnn_convert_img_to_float(iter->input_uchar, net->input_node.w, net->input_node.h, net->input_node.c, param->no_input_norm, 
                param->swap_to_bgr, param->mean_r, param->mean_g, param->mean_b, x);
            x += sz;
            if (net->task != PREDICT) {
                // Load truth
                switch (net->prediction_type) {
                case CLASSIFICATION:
                    y[(int)iter->label_float[0]] = 1;
                    y += output_size;
                    break;
                case REGRESSION:
                    for (j = 0; j < iter->label_width; ++j) {
                        y[j] = iter->label_float[j];
                    }
                    y += output_size;
                    break;
                case HEATMAP_REGRESSION:
                    // Load truth
                    w = net->connections[net->nb_connections - 2].dst_tensor.w;
                    h = net->connections[net->nb_connections - 2].dst_tensor.h;
                    c = net->connections[net->nb_connections - 2].dst_tensor.c;
                    x_scale = (float)w / (float)net->input_node.w;
                    y_scale = (float)h / (float)net->input_node.h;
                    for (j = 0; j < iter->label_width; j += 2) {
                        if (iter->label_float[j] >= 0 && iter->label_float[j + 1] >= 0) {
                            x_pos = (int)((iter->label_float[j] - net->data_aug.shift_x) * x_scale + 0.5f);
                            y_pos = (int)((iter->label_float[j + 1] - net->data_aug.shift_y) * y_scale + 0.5f);
                            // Set gaussian kernel around (x_pos, y_pos)
                            n = (j / 2) % c;
                            offset = n * w * h + (y_pos * w + x_pos);
                            if (x_pos >= 0 && x_pos < w && y_pos >= 0 && y_pos < h) {
                                y[offset] = 1.0f;
                                if (x_pos > 0) y[offset - 1] = 0.5f;
                                if (x_pos < w - 1) y[offset + 1] = 0.5f;
                                if (y_pos > 0) y[offset - w] = 0.5f;
                                if (y_pos < h - 1) y[offset + w] = 0.5f;
                                if (x_pos > 0 && y_pos > 0) y[offset - w - 1] = 0.25f;
                                if (x_pos < w - 1 && y_pos > 0) y[offset - w + 1] = 0.25f;
                                if (x_pos > 0 && y_pos < h - 1) y[offset + w - 1] = 0.25f;
                                if (x_pos < w - 1 && y_pos < h - 1) y[offset + w + 1] = 0.25f;
                            }
                        }
                    }
                    y += output_size;
                    break;
                case SEGMENTATION:
                    memcpy(y, iter->label_float, output_size * sizeof(float));
                    y += output_size;
                    break;
                default:
                    bh_error("Target type not implemented for this data format.", BCNN_INVALID_PARAMETER);
                }
            }

        }
    }
    if (use_buffer_img)
        bh_free(img_tmp);

#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_host2dev(net->input_node.data_gpu, net->input_node.data, input_size);
    if (net->task != PREDICT)
        bcnn_cuda_memcpy_host2dev(net->connections[nb - 1].label_gpu, net->connections[nb - 1].label, output_size
        * net->connections[nb - 1].src_tensor.b);
#endif
    return BCNN_SUCCESS;
}

int bcnn_train_on_batch(bcnn_net *net, bcnn_iterator *iter, float *loss)
{
    bcnn_iter_batch(net, iter);

    net->seen += net->input_node.b;
    // Forward
    bcnn_forward(net);
    // Back prop
    bcnn_backward(net);
    // Update network weight
    bcnn_update(net);
    *loss = net->connections[net->nb_connections - 1].dst_tensor.data[0];

    return BCNN_SUCCESS;
}

int bcnn_predict_on_batch(bcnn_net *net, bcnn_iterator *iter, float **pred, float *error)
{
    int nb = net->nb_connections;
    int en = (net->connections[nb - 1].layer->type == COST ? (nb - 2) : (nb - 1));
    int output_size = bcnn_get_tensor_size(&net->connections[en].dst_tensor);

    bcnn_iter_batch(net, iter);

    bcnn_forward(net);

#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_dev2host(net->connections[en].dst_tensor.data_gpu, net->connections[en].dst_tensor.data,
        output_size);
#endif
    (*pred) = net->connections[en].dst_tensor.data;
    *error = *(net->connections[nb - 1].dst_tensor.data);

    return BCNN_SUCCESS;
}


int bcnn_write_model(bcnn_net *net, char *filename)
{
    bcnn_layer *layer = NULL;
    int i;

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "ERROR: can't open file %s\n", filename);
        return -1;
    }

    fwrite(&net->learner.learning_rate, sizeof(float), 1, fp);
    fwrite(&net->learner.momentum, sizeof(float), 1, fp);
    fwrite(&net->learner.decay, sizeof(float), 1, fp);
    fwrite(&net->seen, sizeof(int), 1, fp);

    for (i = 0; i < net->nb_connections; ++i){
        layer = net->connections[i].layer;
        if (layer->type == CONVOLUTIONAL ||
            layer->type == DECONVOLUTIONAL ||
            layer->type == DEPTHWISE_CONV ||
            layer->type == FULL_CONNECTED) {
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(layer->weight_gpu, layer->weight, layer->weights_size);
            bcnn_cuda_memcpy_dev2host(layer->bias_gpu, layer->bias, layer->bias_size);
#endif
            fwrite(layer->bias, sizeof(float), layer->bias_size, fp);
            fwrite(layer->weight, sizeof(float), layer->weights_size, fp);
        }
        if (layer->type == BATCHNORM) {
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(layer->global_mean_gpu, layer->global_mean,
                net->connections[i].dst_tensor.c);
            bcnn_cuda_memcpy_dev2host(layer->global_variance_gpu, layer->global_variance,
                net->connections[i].dst_tensor.c);
#endif
            fwrite(layer->global_mean, sizeof(float), net->connections[i].dst_tensor.c, fp);
            fwrite(layer->global_variance, sizeof(float), net->connections[i].dst_tensor.c, fp);
        }
    }
    fclose(fp);
    return BCNN_SUCCESS;
}

int bcnn_load_model(bcnn_net *net, char *filename)
{
    FILE *fp = fopen(filename, "rb");
    bcnn_layer *layer = NULL;
    int i, j, is_ft = 0;
    size_t nb_read = 0;
    float tmp = 0.0f;
    
    if (!fp) {
        fprintf(stderr, "[ERROR] can't open file %s\n", filename);
        return -1;
    }

    nb_read = fread(&tmp, sizeof(float), 1, fp);
    nb_read = fread(&tmp, sizeof(float), 1, fp);
    nb_read = fread(&tmp, sizeof(float), 1, fp);
    nb_read = fread(&net->seen, sizeof(int), 1, fp);
    fprintf(stderr, "lr= %f ", net->learner.learning_rate);
    fprintf(stderr, "m= %f ", net->learner.momentum);
    fprintf(stderr, "decay= %f ", net->learner.decay);
    fprintf(stderr, "seen= %d\n", net->seen);

    for (i = 0; i < net->nb_connections; ++i) {
        layer = net->connections[i].layer;
        is_ft = 0;
        if (net->connections[i].id != NULL) {
            for (j = 0; j < net->nb_finetune; ++j) {
                if (strcmp(net->connections[i].id, net->finetune_id[j]) == 0)
                    is_ft = 1;
            }
        }
        if ((layer->type == CONVOLUTIONAL ||
            layer->type == DECONVOLUTIONAL ||
            layer->type == DEPTHWISE_CONV ||
            layer->type == FULL_CONNECTED) && is_ft == 0) {
            nb_read = fread(layer->bias, sizeof(float), layer->bias_size, fp);
            fprintf(stderr, "layer= %d nbread_bias= %lu bias_size_expected= %d\n", i, (unsigned long)nb_read, layer->bias_size);
            nb_read = fread(layer->weight, sizeof(float), layer->weights_size, fp);
            fprintf(stderr, "layer= %d nbread_weight= %lu weight_size_expected= %d\n", i, (unsigned long)nb_read, layer->weights_size);
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_host2dev(layer->weight_gpu, layer->weight, layer->weights_size);
            bcnn_cuda_memcpy_host2dev(layer->bias_gpu, layer->bias, layer->bias_size);
#endif	
        }
        if (layer->type == BATCHNORM) {
            nb_read = fread(layer->global_mean, sizeof(float), net->connections[i].dst_tensor.c, fp);
            fprintf(stderr, "batchnorm layer= %d nbread_mean= %lu mean_size_expected= %d\n",
                i, (unsigned long)nb_read, net->connections[i].dst_tensor.c);
            nb_read = fread(layer->global_variance, sizeof(float), net->connections[i].dst_tensor.c, fp);
            fprintf(stderr, "batchnorm layer= %d nbread_variance= %lu variance_size_expected= %d\n",
                i, (unsigned long)nb_read, net->connections[i].dst_tensor.c);
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_host2dev(layer->global_mean_gpu, layer->global_mean, net->connections[i].dst_tensor.c);
            bcnn_cuda_memcpy_host2dev(layer->global_variance_gpu, layer->global_variance, net->connections[i].dst_tensor.c);
#endif
        }
    }
    if (fp != NULL)
        fclose(fp);

    fprintf(stderr, "[INFO] Model %s loaded succesfully\n", filename);
    fflush(stdout);

    return BCNN_SUCCESS;
}



int bcnn_visualize_network(bcnn_net *net)
{
    int i, j, k, sz, w, h, c;
    bcnn_layer *layer = NULL;
    char name[256];
    FILE *ftmp = NULL;
    int nb = net->nb_connections;
        
    for (j = 0; j < net->nb_connections; ++j) {
        if (net->connections[j].layer->type == CONVOLUTIONAL) {
            w = net->connections[j].dst_tensor.w;
            h = net->connections[j].dst_tensor.h;
            c = net->connections[j].dst_tensor.c;
            sz = w * h * c;
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(net->connections[j].dst_tensor.data_gpu,
                    net->connections[j].dst_tensor.data, sz * net->input_node.b);
#endif
            for (i = 0; i < net->input_node.b / 8; ++i) {	
                layer = net->connections[j].layer;
                for (k = 0; k < net->connections[j].dst_tensor.c / 16; ++k) {
                    sprintf(name, "sample%d_layer%d_fmap%d.png", i, j, k);
                    bip_write_float_image_norm(name, net->connections[j].dst_tensor.data +
                        i * sz + k * w * h, w, h, 1, w * sizeof(float));
                }
            }
        }
        else if (net->connections[j].layer->type == FULL_CONNECTED || 
            net->connections[j].layer->type == SOFTMAX) {
            w = net->connections[j].dst_tensor.w;
            h = net->connections[j].dst_tensor.h;
            c = net->connections[j].dst_tensor.c;
            sz = w * h * c;
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(net->connections[j].dst_tensor.data_gpu,
                    net->connections[j].dst_tensor.data, sz * net->input_node.b);
#endif
            sprintf(name, "ip_%d.txt", j);
            ftmp = fopen(name, "wt");
            for (i = 0; i < net->input_node.b; ++i) {
                layer = net->connections[j].layer;
                for (k = 0; k < sz; ++k) {
                    fprintf(ftmp, "%f ", net->connections[j].dst_tensor.data[i * sz + k]);
                }
                fprintf(ftmp, "\n");
            }
            fclose(ftmp);
            if (sz == 2 && net->connections[j].layer->type == FULL_CONNECTED) {
                sz = sz * net->connections[j].src_tensor.w * net->connections[j].src_tensor.h
                    *net->connections[j].src_tensor.c;
#ifdef BCNN_USE_CUDA
                bcnn_cuda_memcpy_dev2host(net->connections[j].layer->weight_gpu,
                    net->connections[j].layer->weight, sz);
#endif
                sprintf(name, "wgt_%d.txt", j);
                ftmp = fopen(name, "wt");
                layer = net->connections[j].layer;
                for (k = 0; k < sz; ++k) {
                    fprintf(ftmp, "%f ", layer->weight[k]);
                }
                fprintf(ftmp, "\n");
                fclose(ftmp);
                sz = 2;
#ifdef BCNN_USE_CUDA
                bcnn_cuda_memcpy_dev2host(net->connections[j].layer->bias_gpu,
                    net->connections[j].layer->bias, sz);
#endif
                sprintf(name, "b_%d.txt", j);
                ftmp = fopen(name, "wt");
                layer = net->connections[j].layer;
                for (k = 0; k < sz; ++k) {
                    fprintf(ftmp, "%f ", layer->bias[k]);
                }
                fprintf(ftmp, "\n");
                fclose(ftmp);
            }
        }
    }

    return BCNN_SUCCESS;
}


int bcnn_free_layer(bcnn_layer **layer)
{
    bcnn_layer *p_layer = (*layer);
    bh_free(p_layer->indexes);
    bh_free(p_layer->weight);
    bh_free(p_layer->weight_diff);
    bh_free(p_layer->bias);
    bh_free(p_layer->bias_diff);
    bh_free(p_layer->conv_workspace);
    bh_free(p_layer->mean);
    bh_free(p_layer->diff_mean);
    bh_free(p_layer->global_mean);
    bh_free(p_layer->variance);
    bh_free(p_layer->diff_variance);
    bh_free(p_layer->global_variance);
    bh_free(p_layer->x_norm);
    bh_free(p_layer->bn_scale);
    bh_free(p_layer->bn_scale_diff);
    bh_free(p_layer->bn_workspace);
    bh_free(p_layer->rand);
    bh_free(p_layer->adam_m);
    bh_free(p_layer->adam_v);
    bh_free(p_layer->binary_weight);
    bh_free(p_layer->binary_workspace);
#ifdef BCNN_USE_CUDA
    if (p_layer->indexes_gpu)           bcnn_cuda_free(p_layer->indexes_gpu);
    if (p_layer->weight_gpu)            bcnn_cuda_free(p_layer->weight_gpu);
    if (p_layer->weight_diff_gpu)       bcnn_cuda_free(p_layer->weight_diff_gpu);
    if (p_layer->bias_gpu)              bcnn_cuda_free(p_layer->bias_gpu);
    if (p_layer->bias_diff_gpu)         bcnn_cuda_free(p_layer->bias_diff_gpu);
    if (p_layer->mean_gpu)              bcnn_cuda_free(p_layer->mean_gpu);
    if (p_layer->diff_mean_gpu)         bcnn_cuda_free(p_layer->diff_mean_gpu);
    if (p_layer->global_mean_gpu)       bcnn_cuda_free(p_layer->global_mean_gpu);
    if (p_layer->variance_gpu)          bcnn_cuda_free(p_layer->variance_gpu);
    if (p_layer->diff_variance_gpu)     bcnn_cuda_free(p_layer->diff_variance_gpu);
    if (p_layer->global_variance_gpu)   bcnn_cuda_free(p_layer->global_variance_gpu);
    if (p_layer->x_norm_gpu)            bcnn_cuda_free(p_layer->x_norm_gpu);
    if (p_layer->bn_scale_gpu)          bcnn_cuda_free(p_layer->bn_scale_gpu);
    if (p_layer->bn_scale_diff_gpu)     bcnn_cuda_free(p_layer->bn_scale_diff_gpu);
    if (p_layer->bn_workspace_gpu)      bcnn_cuda_free(p_layer->bn_workspace_gpu);
    if (p_layer->rand_gpu)              bcnn_cuda_free(p_layer->rand_gpu);
    if (p_layer->adam_m_gpu)            bcnn_cuda_free(p_layer->adam_m_gpu);
    if (p_layer->adam_v_gpu)            bcnn_cuda_free(p_layer->adam_v_gpu);
#ifdef BCNN_USE_CUDNN
    cudnnDestroyTensorDescriptor(p_layer->src_tensor_desc);
    cudnnDestroyTensorDescriptor(p_layer->dst_tensor_desc);
    cudnnDestroyTensorDescriptor(p_layer->bias_desc);
    cudnnDestroyFilterDescriptor(p_layer->filter_desc);
    cudnnDestroyConvolutionDescriptor(p_layer->conv_desc);
#endif
#endif
    bh_free(*layer);
    return BCNN_SUCCESS;
}