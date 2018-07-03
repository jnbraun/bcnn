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
#include <bh/bh_log.h>
#include <bh/bh_macros.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>
#include <bh/bh_timer.h>

/* include bip image processing lib */
#include <bip/bip.h>

#include <bh/bh_log.h>
#include "bcnn/bcnn.h"
#include "bcnn_activation_layer.h"
#include "bcnn_avgpool_layer.h"
#include "bcnn_batchnorm_layer.h"
#include "bcnn_concat_layer.h"
#include "bcnn_conv_layer.h"
#include "bcnn_cost_layer.h"
#include "bcnn_deconv_layer.h"
#include "bcnn_depthwise_conv_layer.h"
#include "bcnn_dropout_layer.h"
#include "bcnn_fc_layer.h"
#include "bcnn_mat.h"
#include "bcnn_maxpool_layer.h"
#include "bcnn_softmax_layer.h"
#include "bcnn_utils.h"
#include "bcnn_yolo.h"

bcnn_status bcnn_init_net(bcnn_net **net) {
    bcnn_net *p_net = NULL;
    if (*net == NULL) {
        *net = (bcnn_net *)calloc(1, sizeof(bcnn_net));
    }
    // Create input node
    bcnn_tensor input = {0};
    bh_strfill(&input.name, "input");
    // Input node is set to be the first node
    bcnn_net_add_tensor(*net, input);
    // Create label node
    bcnn_tensor label = {0};
    bh_strfill(&label.name, "label");
    // Input node is set to be the second node
    bcnn_net_add_tensor(*net, label);

    return BCNN_SUCCESS;
}

bcnn_status bcnn_end_net(bcnn_net **net) {
    bcnn_free_net(*net);
    bh_free(*net);

    return BCNN_SUCCESS;
}

bcnn_status bcnn_free_tensor(bcnn_tensor *tensor) {
    bh_free(tensor->data);
    bh_free(tensor->grad_data);
#ifdef BCNN_USE_CUDA
    bcnn_cuda_free(tensor->data_gpu);
    bcnn_cuda_free(tensor->grad_data_gpu);
#endif
    return BCNN_SUCCESS;
}

bcnn_status bcnn_free_net(bcnn_net *net) {
    int i;
    bcnn_free_workload(net);
    for (i = 0; i < net->num_nodes; ++i) {
        bcnn_free_node(&net->nodes[i]);
    }
    bh_free(net->nodes);
    for (i = 0; i < net->nb_finetune; ++i) {
        bh_free(net->finetune_id[i]);
    }
    bh_free(net->finetune_id);
    bcnn_net_free_tensors(net);
    return BCNN_SUCCESS;
}

void bcnn_net_set_log_context(bcnn_net *net, bh_log_callback fct,
                              bh_log_level level) {
    net->log_ctx.fct = fct;
    net->log_ctx.lvl = level;
}

int bcnn_set_param(bcnn_net *net, char *name, char *val) {
    if (strcmp(name, "input_width") == 0) {
        net->input_width = atoi(val);
    } else if (strcmp(name, "input_height") == 0) {
        net->input_height = atoi(val);
    } else if (strcmp(name, "input_channels") == 0) {
        net->input_channels = atoi(val);
    } else if (strcmp(name, "batch_size") == 0) {
        net->batch_size = atoi(val);
    } else if (strcmp(name, "max_batches") == 0) {
        net->max_batches = atoi(val);
    } else if (strcmp(name, "loss") == 0) {
        if (strcmp(val, "error") == 0) {
            net->loss_metric = COST_ERROR;
        } else if (strcmp(val, "logloss") == 0) {
            net->loss_metric = COST_LOGLOSS;
        } else if (strcmp(val, "sse") == 0) {
            net->loss_metric = COST_SSE;
        } else if (strcmp(val, "mse") == 0) {
            net->loss_metric = COST_MSE;
        } else if (strcmp(val, "crps") == 0) {
            net->loss_metric = COST_CRPS;
        } else if (strcmp(val, "dice") == 0) {
            net->loss_metric = COST_DICE;
        } else {
            BCNN_WARNING(net->log_ctx, "Unknown cost metric %s, going with sse",
                         val);
            net->loss_metric = COST_SSE;
        }
    } else if (strcmp(name, "learning_policy") == 0) {
        if (strcmp(val, "sigmoid") == 0) {
            net->learner.policy = SIGMOID;
        } else if (strcmp(val, "constant") == 0) {
            net->learner.policy = CONSTANT;
        } else if (strcmp(val, "exp") == 0) {
            net->learner.policy = EXP;
        } else if (strcmp(val, "inv") == 0) {
            net->learner.policy = INV;
        } else if (strcmp(val, "step") == 0) {
            net->learner.policy = STEP;
        } else if (strcmp(val, "poly") == 0) {
            net->learner.policy = POLY;
        } else {
            net->learner.policy = CONSTANT;
        }
    } else if (strcmp(name, "optimizer") == 0) {
        if (strcmp(val, "sgd") == 0) {
            net->learner.optimizer = SGD;
        } else if (strcmp(val, "adam") == 0) {
            net->learner.optimizer = ADAM;
        }
    } else if (strcmp(name, "step") == 0) {
        net->learner.step = atoi(val);
    } else if (strcmp(name, "learning_rate") == 0) {
        net->learner.learning_rate = (float)atof(val);
    } else if (strcmp(name, "beta1") == 0) {
        net->learner.beta1 = (float)atof(val);
    } else if (strcmp(name, "beta2") == 0) {
        net->learner.beta2 = (float)atof(val);
    } else if (strcmp(name, "decay") == 0) {
        net->learner.decay = (float)atof(val);
    } else if (strcmp(name, "momentum") == 0) {
        net->learner.momentum = (float)atof(val);
    } else if (strcmp(name, "gamma") == 0) {
        net->learner.gamma = (float)atof(val);
    } else if (strcmp(name, "range_shift_x") == 0) {
        net->data_aug.range_shift_x = atoi(val);
    } else if (strcmp(name, "range_shift_y") == 0) {
        net->data_aug.range_shift_y = atoi(val);
    } else if (strcmp(name, "min_scale") == 0) {
        net->data_aug.min_scale = (float)atof(val);
    } else if (strcmp(name, "max_scale") == 0) {
        net->data_aug.max_scale = (float)atof(val);
    } else if (strcmp(name, "rotation_range") == 0) {
        net->data_aug.rotation_range = (float)atof(val);
    } else if (strcmp(name, "min_contrast") == 0) {
        net->data_aug.min_contrast = (float)atof(val);
    } else if (strcmp(name, "max_contrast") == 0) {
        net->data_aug.max_contrast = (float)atof(val);
    } else if (strcmp(name, "min_brightness") == 0) {
        net->data_aug.min_brightness = atoi(val);
    } else if (strcmp(name, "max_brightness") == 0) {
        net->data_aug.max_brightness = atoi(val);
    } else if (strcmp(name, "max_distortion") == 0) {
        net->data_aug.max_distortion = (float)atof(val);
    } else if (strcmp(name, "flip_h") == 0) {
        net->data_aug.random_fliph = 1;
    } else if (strcmp(name, "mean_r") == 0) {
        net->data_aug.mean_r = (float)atof(val) / 255.0f;
    } else if (strcmp(name, "mean_g") == 0) {
        net->data_aug.mean_g = (float)atof(val) / 255.0f;
    } else if (strcmp(name, "mean_b") == 0) {
        net->data_aug.mean_b = (float)atof(val) / 255.0f;
    } else if (strcmp(name, "swap_to_bgr") == 0) {
        net->data_aug.swap_to_bgr = atoi(val);
    } else if (strcmp(name, "no_input_norm") == 0) {
        net->data_aug.no_input_norm = atoi(val);
    } else if (strcmp(name, "prediction_type") == 0) {
        if (strcmp(val, "classif") == 0 || strcmp(val, "classification") == 0) {
            net->prediction_type = CLASSIFICATION;
        } else if (strcmp(val, "reg") == 0 || strcmp(val, "regression") == 0) {
            net->prediction_type = REGRESSION;
        } else if (strcmp(val, "heatmap") == 0 ||
                   strcmp(val, "heatmap_regression") == 0) {
            net->prediction_type = HEATMAP_REGRESSION;
        } else if (strcmp(val, "segmentation") == 0) {
            net->prediction_type = SEGMENTATION;
        }
    } else if (strcmp(name, "finetune_id") == 0) {
        net->nb_finetune++;
        if (net->nb_finetune == 1) {
            net->finetune_id =
                (char **)calloc(net->nb_finetune, sizeof(char *));
        } else {
            net->finetune_id =
                (char **)realloc(net->finetune_id, net->nb_finetune);
        }
        bh_strfill(&net->finetune_id[net->nb_finetune - 1], val);
    }
    return BCNN_SUCCESS;
}

bcnn_status bcnn_net_add_node(bcnn_net *net, bcnn_node node) {
    bcnn_node *p_conn = NULL;
    net->num_nodes++;
    p_conn =
        (bcnn_node *)realloc(net->nodes, net->num_nodes * sizeof(bcnn_node));
    BCNN_CHECK_AND_LOG(net->log_ctx, (p_conn != NULL), BCNN_FAILED_ALLOC,
                       "Internal allocation error");
    net->nodes = p_conn;
    net->nodes[net->num_nodes - 1] = node;
    return BCNN_SUCCESS;
}

bcnn_status bcnn_free_node(bcnn_node *node) {
    bcnn_free_layer(&node->layer);
    bh_free(node->src);
    bh_free(node->dst);
    return BCNN_SUCCESS;
}

bcnn_status bcnn_net_add_tensor(bcnn_net *net, bcnn_tensor tensor) {
    bcnn_tensor *p_tensors = NULL;
    net->num_tensors++;
    p_tensors = (bcnn_tensor *)realloc(net->tensors,
                                       net->num_tensors * sizeof(bcnn_tensor));
    BCNN_CHECK_AND_LOG(net->log_ctx, (p_tensors != NULL), BCNN_FAILED_ALLOC,
                       "Internal allocation error");
    net->tensors = p_tensors;
    net->tensors[net->num_tensors - 1] = tensor;
    return BCNN_SUCCESS;
}

void bcnn_net_free_tensors(bcnn_net *net) {
    int i;
    for (i = 0; i < net->num_tensors; ++i) {
        bcnn_tensor_free(&net->tensors[i]);
    }
    bh_free(net->tensors);
}

bcnn_status bcnn_node_add_output(bcnn_net *net, bcnn_node *node, int index) {
    int *p_dst = NULL;
    node->num_dst++;
    p_dst = (int *)realloc(node->dst, node->num_dst * sizeof(int));
    BCNN_CHECK_AND_LOG(net->log_ctx, (p_dst != NULL), BCNN_FAILED_ALLOC,
                       "Internal allocation error");
    node->dst = p_dst;
    node->dst[node->num_dst - 1] = index;
    return BCNN_SUCCESS;
}

bcnn_status bcnn_node_add_input(bcnn_net *net, bcnn_node *node, int index) {
    int *p_src = NULL;
    node->num_src++;
    p_src = (int *)realloc(node->src, node->num_src * sizeof(int));
    BCNN_CHECK_AND_LOG(net->log_ctx, (p_src != NULL), BCNN_FAILED_ALLOC,
                       "Internal allocation error");
    node->src = p_src;
    node->src[node->num_src - 1] = index;
    return BCNN_SUCCESS;
}

void bcnn_net_set_input_shape(bcnn_net *net, int input_width, int input_height,
                              int input_channels, int batch_size) {
    net->input_width = input_width;
    net->input_height = input_height;
    net->input_channels = input_channels;
    net->batch_size = batch_size;
    bcnn_tensor_set_shape(&net->tensors[0], batch_size, input_channels,
                          input_height, input_width, 0);
}

bcnn_status bcnn_init_workload(bcnn_net *net) {
    int i;
    int n = net->num_nodes;
    int k = (net->nodes[n - 1].layer->type == COST ? (n - 2) : (n - 1));

    net->input_buffer = (unsigned char *)calloc(
        net->input_width * net->input_height * net->input_channels, 1);

    // Allocate tensor for input node
    bcnn_tensor_allocate(&net->tensors[0]);

#ifdef BCNN_USE_CUDA
    net->workspace_gpu = bcnn_cuda_malloc_f32(net->workspace_size);
    for (i = 0; i < n; ++i) {
        if (net->nodes[i].layer->type == CONVOLUTIONAL) {
            net->nodes[i].layer->conv_workspace_gpu = net->workspace_gpu;
        }
    }
#endif

    return BCNN_SUCCESS;
}

bcnn_status bcnn_free_workload(bcnn_net *net) {
    int i;
    int n = net->num_nodes;

    bcnn_tensor_free(&net->tensors[0]);
    bh_free(net->input_buffer);
#ifdef BCNN_USE_CUDA
    bcnn_cuda_free(net->workspace_gpu);
#endif
    return BCNN_SUCCESS;
}

bcnn_status bcnn_compile_net(bcnn_net *net, char *phase) {
    int i;

    if (strcmp(phase, "train") == 0) {
        net->state = 1;
    } else if (strcmp(phase, "predict") == 0) {
        net->state = 0;
    } else {
        BCNN_ERROR(
            net->log_ctx, BCNN_INVALID_PARAMETER,
            "bcnn_compile_net: Available option are 'train' and 'predict'");
        return BCNN_INVALID_PARAMETER;
    }
    // State propagation through nodes
    for (i = 0; i < net->num_nodes; ++i) {
        net->nodes[i].layer->net_state = net->state;
    }

    bcnn_free_workload(net);
    bcnn_init_workload(net);

    return BCNN_SUCCESS;
}

int bcnn_forward(bcnn_net *net) {
    int i, j;
    int output_size = 0;
    bcnn_node node = {0};
    for (i = 0; i < net->num_nodes; ++i) {
        node = net->nodes[i];
        for (j = 0; j < node.num_dst; ++j) {
            output_size = bcnn_tensor_size(&net->tensors[node.dst[j]]);
#ifdef BCNN_USE_CUDA
            if (net->tensors[node.dst[j]].grad_data_gpu != NULL) {
                bcnn_cuda_fill_f32(output_size, 0.0f,
                                   net->tensors[node.dst[j]].grad_data_gpu, 1);
            }
#else
            if (net->tensors[node.dst[j]].grad_data != NULL) {
                memset(net->tensors[node.dst[j]].grad_data, 0,
                       output_size * sizeof(float));
            }
#endif
        }
        switch (node.layer->type) {
            case CONVOLUTIONAL:
                bcnn_forward_conv_layer(net, &node);
                break;
            case DECONVOLUTIONAL:
                bcnn_forward_deconv_layer(net, &node);
                break;
            case DEPTHWISE_CONV:
                bcnn_forward_depthwise_sep_conv_layer(net, &node);
                break;
            case ACTIVATION:
                bcnn_forward_activation_layer(net, &node);
                break;
            case BATCHNORM:
                bcnn_forward_batchnorm_layer(net, &node);
                break;
            case FULL_CONNECTED:
                bcnn_forward_fullc_layer(net, &node);
                break;
            case MAXPOOL:
                bcnn_forward_maxpool_layer(net, &node);
                break;
            case AVGPOOL:
                bcnn_forward_avgpool_layer(net, &node);
                break;
            case SOFTMAX:
                bcnn_forward_softmax_layer(net, &node);
                break;
            case DROPOUT:
                bcnn_forward_dropout_layer(net, &node);
                break;
            case CONCAT:
                bcnn_forward_concat_layer(net, &node);
                break;
            case YOLO:
                bcnn_forward_yolo_layer(net, &node);
                break;
            case COST:
                bcnn_forward_cost_layer(net, &node);
                break;
            default:
                break;
        }
    }
    return BCNN_SUCCESS;
}

int bcnn_backward(bcnn_net *net) {
    int i;
    bcnn_node node = {0};
    for (i = net->num_nodes - 1; i >= 0; --i) {
        node = net->nodes[i];
        switch (node.layer->type) {
            case CONVOLUTIONAL:
                bcnn_backward_conv_layer(net, &node);
                break;
            case DECONVOLUTIONAL:
                bcnn_backward_deconv_layer(net, &node);
                break;
            case DEPTHWISE_CONV:
                bcnn_backward_depthwise_sep_conv_layer(net, &node);
                break;
            case ACTIVATION:
                bcnn_backward_activation_layer(net, &node);
                break;
            case BATCHNORM:
                bcnn_backward_batchnorm_layer(net, &node);
                break;
            case FULL_CONNECTED:
                bcnn_backward_fullc_layer(net, &node);
                break;
            case MAXPOOL:
                bcnn_backward_maxpool_layer(net, &node);
                break;
            case AVGPOOL:
                bcnn_backward_avgpool_layer(net, &node);
                break;
            case SOFTMAX:
                bcnn_backward_softmax_layer(net, &node);
                break;
            case DROPOUT:
                bcnn_backward_dropout_layer(net, &node);
                break;
            case CONCAT:
                bcnn_backward_concat_layer(net, &node);
                break;
            case YOLO:
                bcnn_backward_yolo_layer(net, &node);
                break;
            case COST:
                bcnn_backward_cost_layer(net, &node);
                break;
            default:
                break;
        }
    }
    return BCNN_SUCCESS;
}

int bcnn_iter_batch(bcnn_net *net, bcnn_iterator *iter) {
    int i, j, n, offset;
    int sz = net->input_width * net->input_height * net->input_channels;
    int sz_img = iter->input_width * iter->input_height * iter->input_depth;
    int nb = net->num_nodes;
    int w, h, c;
    int w_in = net->input_width;
    int h_in = net->input_height;
    int c_in = net->input_channels;
    int batch_size = net->batch_size;
    unsigned char *img_tmp = NULL;
    float *x = net->tensors[0].data;
    float *y = net->tensors[1].data;
    float x_scale, y_scale;
    int x_pos, y_pos;
    int use_buffer_img = (net->task == TRAIN && net->state != 0 &&
                          (net->data_aug.range_shift_x != 0 ||
                           net->data_aug.range_shift_y != 0 ||
                           net->data_aug.rotation_range != 0 ||
                           net->data_aug.random_fliph != 0));
    bcnn_data_augment *param = &(net->data_aug);
    int input_size = bcnn_tensor_size(&net->tensors[0]);
    int en = (net->nodes[nb - 1].layer->type == COST ? (nb - 2) : (nb - 1));
    int output_size =
        bcnn_tensor_get_size3d(&net->tensors[net->nodes[en].dst[0]]);

    memset(x, 0, sz * net->batch_size * sizeof(float));
    if (net->task != PREDICT) {
        memset(y, 0, output_size * net->batch_size * sizeof(float));
    }
    if (use_buffer_img) {
        img_tmp = (unsigned char *)calloc(sz_img, sizeof(unsigned char));
    }
    if (iter->type == ITER_MNIST || iter->type == ITER_CIFAR10) {
        for (i = 0; i < net->batch_size; ++i) {
            bcnn_iterator_next(net, iter);
            // Data augmentation
            if (net->task == TRAIN && net->state) {
                bcnn_data_augmentation(iter->input_uchar, iter->input_width,
                                       iter->input_height, iter->input_depth,
                                       param, img_tmp);
            }
            // bip_write_image("test.png", iter->input_uchar, iter->input_width,
            // iter->input_height, iter->input_depth, iter->input_width *
            // iter->input_depth);
            if (w_in < iter->input_width || h_in < iter->input_height) {
                bip_crop_image(
                    iter->input_uchar, iter->input_width, iter->input_height,
                    iter->input_width * iter->input_depth,
                    (iter->input_width - w_in) / 2,
                    (iter->input_height - h_in) / 2, net->input_buffer, w_in,
                    h_in, w_in * c_in, c_in);
                bcnn_convert_img_to_float(net->input_buffer, w_in, h_in, c_in,
                                          param->no_input_norm,
                                          param->swap_to_bgr, param->mean_r,
                                          param->mean_g, param->mean_b, x);
            } else
                bcnn_convert_img_to_float(iter->input_uchar, w_in, h_in, c_in,
                                          param->no_input_norm,
                                          param->swap_to_bgr, param->mean_r,
                                          param->mean_g, param->mean_b, x);
            // bip_write_image("test1.png", tmp_buf, w_in, h_in, c_in, w_in *
            // c_in);
            x += sz;
            if (net->task != PREDICT) {
                // Load truth
                y[iter->label_int[0]] = 1;
                y += output_size;
            }
        }
    } else if (iter->type == ITER_BIN) {
        for (i = 0; i < batch_size; ++i) {
            // bcnn_bin_iter(net, iter);
            bcnn_iterator_next(net, iter);
            // Data augmentation
            if (net->task == TRAIN && net->state)
                bcnn_data_augmentation(iter->input_uchar, w_in, h_in, c_in,
                                       param, img_tmp);
            bcnn_convert_img_to_float(iter->input_uchar, w_in, h_in, c_in,
                                      param->no_input_norm, param->swap_to_bgr,
                                      param->mean_r, param->mean_g,
                                      param->mean_b, x);
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
                        w = net->tensors[net->nodes[en].dst[0]].w;
                        h = net->tensors[net->nodes[en].dst[0]].h;
                        c = net->tensors[net->nodes[en].dst[0]].c;
                        x_scale = (float)w / (float)w_in;
                        y_scale = (float)h / (float)h_in;
                        for (j = 0; j < iter->label_width; j += 2) {
                            if (iter->label_float[j] >= 0 &&
                                iter->label_float[j + 1] >= 0) {
                                x_pos = (int)((iter->label_float[j] -
                                               net->data_aug.shift_x) *
                                                  x_scale +
                                              0.5f);
                                y_pos = (int)((iter->label_float[j + 1] -
                                               net->data_aug.shift_y) *
                                                  y_scale +
                                              0.5f);
                                // Set gaussian kernel around (x_pos, y_pos)
                                n = (j / 2) % c;
                                offset = n * w * h + (y_pos * w + x_pos);
                                if (x_pos >= 0 && x_pos < w && y_pos >= 0 &&
                                    y_pos < h) {
                                    y[offset] = 1.0f;
                                    if (x_pos > 0) y[offset - 1] = 0.5f;
                                    if (x_pos < w - 1) y[offset + 1] = 0.5f;
                                    if (y_pos > 0) y[offset - w] = 0.5f;
                                    if (y_pos < h - 1) y[offset + w] = 0.5f;
                                    if (x_pos > 0 && y_pos > 0)
                                        y[offset - w - 1] = 0.25f;
                                    if (x_pos < w - 1 && y_pos > 0)
                                        y[offset - w + 1] = 0.25f;
                                    if (x_pos > 0 && y_pos < h - 1)
                                        y[offset + w - 1] = 0.25f;
                                    if (x_pos < w - 1 && y_pos < h - 1)
                                        y[offset + w + 1] = 0.25f;
                                }
                            }
                        }
                        y += output_size;
                        break;
                    default:
                        BCNN_ERROR(
                            net->log_ctx, BCNN_INVALID_PARAMETER,
                            "Target type not implemented for this data format. "
                            "Please use list format instead.");
                }
            }
        }
    } else if (iter->type == ITER_LIST || iter->type == ITER_CSV) {
        for (i = 0; i < batch_size; ++i) {
            // bcnn_list_iter(net, iter);
            bcnn_iterator_next(net, iter);
            // Online data augmentation
            if (net->task == TRAIN && net->state)
                bcnn_data_augmentation(iter->input_uchar, w_in, h_in, c_in,
                                       param, img_tmp);
            bcnn_convert_img_to_float(iter->input_uchar, w_in, h_in, c_in,
                                      param->no_input_norm, param->swap_to_bgr,
                                      param->mean_r, param->mean_g,
                                      param->mean_b, x);
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
                        w = net->tensors[net->nodes[en].dst[0]].w;
                        h = net->tensors[net->nodes[en].dst[0]].h;
                        c = net->tensors[net->nodes[en].dst[0]].c;
                        x_scale = (float)w / (float)w_in;
                        y_scale = (float)h / (float)h_in;
                        for (j = 0; j < iter->label_width; j += 2) {
                            if (iter->label_float[j] >= 0 &&
                                iter->label_float[j + 1] >= 0) {
                                x_pos = (int)((iter->label_float[j] -
                                               net->data_aug.shift_x) *
                                                  x_scale +
                                              0.5f);
                                y_pos = (int)((iter->label_float[j + 1] -
                                               net->data_aug.shift_y) *
                                                  y_scale +
                                              0.5f);
                                // Set gaussian kernel around (x_pos, y_pos)
                                n = (j / 2) % c;
                                offset = n * w * h + (y_pos * w + x_pos);
                                if (x_pos >= 0 && x_pos < w && y_pos >= 0 &&
                                    y_pos < h) {
                                    y[offset] = 1.0f;
                                    if (x_pos > 0) y[offset - 1] = 0.5f;
                                    if (x_pos < w - 1) y[offset + 1] = 0.5f;
                                    if (y_pos > 0) y[offset - w] = 0.5f;
                                    if (y_pos < h - 1) y[offset + w] = 0.5f;
                                    if (x_pos > 0 && y_pos > 0)
                                        y[offset - w - 1] = 0.25f;
                                    if (x_pos < w - 1 && y_pos > 0)
                                        y[offset - w + 1] = 0.25f;
                                    if (x_pos > 0 && y_pos < h - 1)
                                        y[offset + w - 1] = 0.25f;
                                    if (x_pos < w - 1 && y_pos < h - 1)
                                        y[offset + w + 1] = 0.25f;
                                }
                            }
                        }
                        y += output_size;
                        break;
                    case SEGMENTATION:
                        memcpy(y, iter->label_float,
                               output_size * sizeof(float));
                        y += output_size;
                        break;
                    default:
                        BCNN_ERROR(net->log_ctx, BCNN_INVALID_PARAMETER,
                                   "Target type not implemented for this data "
                                   "format.");
                }
            }
        }
    }
    if (use_buffer_img) bh_free(img_tmp);

#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_host2dev(net->tensors[0].data_gpu, net->tensors[0].data,
                              input_size);
    if (net->task != PREDICT) {
        bcnn_cuda_memcpy_host2dev(
            net->tensors[1].data_gpu, net->tensors[1].data,
            bcnn_tensor_size(&net->tensors[net->nodes[en].dst[0]]));
    }
#endif
    return BCNN_SUCCESS;
}

int bcnn_train_on_batch(bcnn_net *net, bcnn_iterator *iter, float *loss) {
    bcnn_iter_batch(net, iter);

    net->seen += net->batch_size;
    // Forward
    bcnn_forward(net);
    // Back prop
    bcnn_backward(net);
    // Update network weight
    bcnn_update(net);
    *loss = net->tensors[net->nodes[net->num_nodes - 1].dst[0]].data[0];
    return BCNN_SUCCESS;
}

int bcnn_predict_on_batch(bcnn_net *net, bcnn_iterator *iter, float **pred,
                          float *error) {
    int nb = net->num_nodes;
    int en = (net->nodes[nb - 1].layer->type == COST ? (nb - 2) : (nb - 1));
    int output_size = bcnn_tensor_size(&net->tensors[net->nodes[en].dst[0]]);

    bcnn_iter_batch(net, iter);

    bcnn_forward(net);

#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_dev2host(net->tensors[net->nodes[en].dst[0]].data_gpu,
                              net->tensors[net->nodes[en].dst[0]].data,
                              output_size);
#endif
    (*pred) = net->tensors[net->nodes[en].dst[0]].data;
    *error = *(net->tensors[net->nodes[nb - 1].dst[0]].data);

    return BCNN_SUCCESS;
}

bcnn_status bcnn_write_model(bcnn_net *net, char *filename) {
    bcnn_layer *layer = NULL;
    int i;

    FILE *fp = NULL;
    fp = fopen(filename, "wb");
    BCNN_CHECK_AND_LOG(net->log_ctx, fp, BCNN_INVALID_PARAMETER,
                       "Could not open model file %s\n", filename);

    fwrite(&net->learner.learning_rate, sizeof(float), 1, fp);
    fwrite(&net->learner.momentum, sizeof(float), 1, fp);
    fwrite(&net->learner.decay, sizeof(float), 1, fp);
    fwrite(&net->seen, sizeof(int), 1, fp);

    for (i = 0; i < net->num_nodes; ++i) {
        layer = net->nodes[i].layer;
        if (layer->type == CONVOLUTIONAL || layer->type == DECONVOLUTIONAL ||
            layer->type == DEPTHWISE_CONV || layer->type == FULL_CONNECTED) {
            bcnn_tensor *weights = &net->tensors[net->nodes[i].src[1]];
            bcnn_tensor *biases = &net->tensors[net->nodes[i].src[2]];
            int weights_size = bcnn_tensor_size(weights);
            int biases_size = bcnn_tensor_size(biases);
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(weights->data_gpu, weights->data,
                                      weights_size);
            bcnn_cuda_memcpy_dev2host(biases->data_gpu, biases->data,
                                      biases_size);
#endif
            fwrite(biases->data, sizeof(float), biases_size, fp);
            fwrite(weights->data, sizeof(float), weights_size, fp);
            if (layer->batch_norm == 1) {
                bcnn_tensor *bn_mean = &net->tensors[net->nodes[i].src[3]];
                bcnn_tensor *bn_var = &net->tensors[net->nodes[i].src[4]];
                bcnn_tensor *bn_scales = &net->tensors[net->nodes[i].src[5]];
                int bn_mean_size = bcnn_tensor_size(bn_mean);
                int bn_var_size = bcnn_tensor_size(bn_var);
                int bn_scales_size = bcnn_tensor_size(bn_scales);
#ifdef BCNN_USE_CUDA
                bcnn_cuda_memcpy_dev2host(bn_mean->data_gpu, bn_mean->data,
                                          bn_mean_size);
                bcnn_cuda_memcpy_dev2host(bn_var->data_gpu, bn_var->data,
                                          bn_var_size);
                bcnn_cuda_memcpy_dev2host(bn_scales->data_gpu, bn_scales->data,
                                          bn_scales_size);
#endif
                fwrite(bn_mean->data, sizeof(float), bn_mean_size, fp);
                fwrite(bn_var->data, sizeof(float), bn_var_size, fp);
                fwrite(bn_scales->data, sizeof(float), bn_scales_size, fp);
            }
        }
        if (layer->type == ACTIVATION && layer->activation == PRELU) {
            bcnn_tensor *weights = &net->tensors[net->nodes[i].src[1]];
            int weights_size = bcnn_tensor_size(weights);
            fwrite(weights->data, sizeof(float), weights_size, fp);
        }
        if (layer->type == BATCHNORM) {
            bcnn_tensor *bn_mean = &net->tensors[net->nodes[i].src[1]];
            bcnn_tensor *bn_var = &net->tensors[net->nodes[i].src[2]];
            bcnn_tensor *bn_scales = &net->tensors[net->nodes[i].src[3]];
            bcnn_tensor *bn_biases = &net->tensors[net->nodes[i].src[4]];
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(bn_mean->data_gpu, bn_mean->data,
                                      net->tensors[net->nodes[i].dst[0]].c);
            bcnn_cuda_memcpy_dev2host(bn_var->data_gpu, bn_var->data,
                                      net->tensors[net->nodes[i].dst[0]].c);
            bcnn_cuda_memcpy_dev2host(bn_scales->data_gpu, bn_scales->data,
                                      net->tensors[net->nodes[i].dst[0]].c);
            bcnn_cuda_memcpy_dev2host(bn_biases->data_gpu, bn_biases->data,
                                      net->tensors[net->nodes[i].dst[0]].c);
#endif
            fwrite(bn_mean->data, sizeof(float),
                   net->tensors[net->nodes[i].dst[0]].c, fp);
            fwrite(bn_var->data, sizeof(float),
                   net->tensors[net->nodes[i].dst[0]].c, fp);
            fwrite(bn_scales->data, sizeof(float),
                   net->tensors[net->nodes[i].dst[0]].c, fp);
            fwrite(bn_biases->data, sizeof(float),
                   net->tensors[net->nodes[i].dst[0]].c, fp);
        }
    }
    fclose(fp);
    return BCNN_SUCCESS;
}

bcnn_status bcnn_load_model(bcnn_net *net, char *filename) {
    FILE *fp = NULL;
    bcnn_layer *layer = NULL;
    int i, j, is_ft = 0;
    size_t nb_read = 0;
    float tmp = 0.0f;

    fp = fopen(filename, "rb");
    BCNN_CHECK_AND_LOG(net->log_ctx, fp, BCNN_INVALID_PARAMETER,
                       "Can not open file %s\n", filename);

    nb_read = fread(&tmp, sizeof(float), 1, fp);
    nb_read = fread(&tmp, sizeof(float), 1, fp);
    nb_read = fread(&tmp, sizeof(float), 1, fp);
    nb_read = fread(&net->seen, sizeof(int), 1, fp);
    BCNN_INFO(net->log_ctx, "lr= %f ", net->learner.learning_rate);
    BCNN_INFO(net->log_ctx, "m= %f ", net->learner.momentum);
    BCNN_INFO(net->log_ctx, "decay= %f ", net->learner.decay);
    BCNN_INFO(net->log_ctx, "seen= %d\n", net->seen);

    for (i = 0; i < net->num_nodes; ++i) {
        layer = net->nodes[i].layer;
        is_ft = 0;
        if ((layer->type == CONVOLUTIONAL || layer->type == DECONVOLUTIONAL ||
             layer->type == DEPTHWISE_CONV || layer->type == FULL_CONNECTED) &&
            is_ft == 0) {
            bcnn_tensor *weights = &net->tensors[net->nodes[i].src[1]];
            bcnn_tensor *biases = &net->tensors[net->nodes[i].src[2]];
            int weights_size = bcnn_tensor_size(weights);
            int biases_size = bcnn_tensor_size(biases);
            nb_read = fread(biases->data, sizeof(float), biases_size, fp);
            BCNN_INFO(net->log_ctx,
                      "layer= %d nbread_bias= %lu bias_size_expected= %d", i,
                      (unsigned long)nb_read, biases_size);
            nb_read = fread(weights->data, sizeof(float), weights_size, fp);
            BCNN_INFO(net->log_ctx,
                      "layer= %d nbread_weight= %lu weight_size_expected= %d",
                      i, (unsigned long)nb_read, weights_size);
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_host2dev(weights->data_gpu, weights->data,
                                      weights_size);
            bcnn_cuda_memcpy_host2dev(biases->data_gpu, biases->data,
                                      biases_size);
#endif
            if (layer->batch_norm == 1) {
                bcnn_tensor *bn_mean = &net->tensors[net->nodes[i].src[3]];
                bcnn_tensor *bn_var = &net->tensors[net->nodes[i].src[4]];
                bcnn_tensor *bn_scales = &net->tensors[net->nodes[i].src[5]];
                int bn_mean_size = bcnn_tensor_size(bn_mean);
                int bn_var_size = bcnn_tensor_size(bn_var);
                int bn_scales_size = bcnn_tensor_size(bn_scales);
                nb_read = fread(bn_mean->data, sizeof(float), bn_mean_size, fp);
                nb_read = fread(bn_var->data, sizeof(float), bn_var_size, fp);
                nb_read =
                    fread(bn_scales->data, sizeof(float), bn_scales_size, fp);
#ifdef BCNN_USE_CUDA
                bcnn_cuda_memcpy_host2dev(bn_mean->data_gpu, bn_mean->data,
                                          bn_mean_size);
                bcnn_cuda_memcpy_host2dev(bn_var->data_gpu, bn_var->data,
                                          bn_var_size);
                bcnn_cuda_memcpy_host2dev(bn_scales->data_gpu, bn_scales->data,
                                          bn_scales_size);
#endif
            }
        }
        if (layer->type == ACTIVATION && layer->activation == PRELU) {
            bcnn_tensor *weights = &net->tensors[net->nodes[i].src[1]];
            int weights_size = bcnn_tensor_size(weights);
            nb_read = fread(weights->data, sizeof(float), weights_size, fp);
            BCNN_INFO(net->log_ctx, "PReLU layer= %d nbread= %lu expected= %d",
                      i, (unsigned long)nb_read, weights_size);
        }
        if (layer->type == BATCHNORM) {
            bcnn_tensor *bn_mean = &net->tensors[net->nodes[i].src[1]];
            bcnn_tensor *bn_var = &net->tensors[net->nodes[i].src[2]];
            bcnn_tensor *bn_scales = &net->tensors[net->nodes[i].src[3]];
            bcnn_tensor *bn_biases = &net->tensors[net->nodes[i].src[4]];
            int sz = net->tensors[net->nodes[i].dst[0]].c;
            nb_read = fread(bn_mean->data, sizeof(float), sz, fp);
            BCNN_INFO(
                net->log_ctx,
                "batchnorm layer= %d nbread_mean= %lu mean_size_expected= %d",
                i, (unsigned long)nb_read, sz);
            nb_read = fread(bn_var->data, sizeof(float), sz, fp);
            BCNN_INFO(net->log_ctx,
                      "batchnorm layer= %d nbread_variance= %lu "
                      "variance_size_expected= %d",
                      i, (unsigned long)nb_read, sz);
            nb_read = fread(bn_scales->data, sizeof(float), sz, fp);
            nb_read = fread(bn_biases->data, sizeof(float), sz, fp);
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_host2dev(bn_mean->data_gpu, bn_mean->data, sz);
            bcnn_cuda_memcpy_host2dev(bn_var->data_gpu, bn_var->data, sz);
            bcnn_cuda_memcpy_host2dev(bn_scales->data_gpu, bn_scales->data, sz);
            bcnn_cuda_memcpy_host2dev(bn_biases->data_gpu, bn_biases->data, sz);
#endif
        }
    }
    if (fp != NULL) fclose(fp);

    BCNN_INFO(net->log_ctx, "Model %s loaded succesfully", filename);
    fflush(stdout);

    return BCNN_SUCCESS;
}

int bcnn_visualize_network(bcnn_net *net) {
    int i, j, k, sz, w, h, c;
    bcnn_layer *layer = NULL;
    char name[256];
    FILE *ftmp = NULL;
    int nb = net->num_nodes;

    for (j = 0; j < net->num_nodes; ++j) {
        if (net->nodes[j].layer->type == CONVOLUTIONAL) {
            w = net->tensors[net->nodes[j].dst[0]].w;
            h = net->tensors[net->nodes[j].dst[0]].h;
            c = net->tensors[net->nodes[j].dst[0]].c;
            sz = w * h * c;
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(
                net->tensors[net->nodes[j].dst[0]].data_gpu,
                net->tensors[net->nodes[j].dst[0]].data,
                bcnn_tensor_size(&net->tensors[net->nodes[j].dst[0]]));
#endif
            for (i = 0; i < net->tensors[net->nodes[j].dst[0]].n / 8; ++i) {
                layer = net->nodes[j].layer;
                for (k = 0; k < net->tensors[net->nodes[j].dst[0]].c / 16;
                     ++k) {
                    sprintf(name, "sample%d_layer%d_fmap%d.png", i, j, k);
                    bip_write_float_image_norm(
                        name, net->tensors[net->nodes[j].dst[0]].data + i * sz +
                                  k * w * h,
                        w, h, 1, w * sizeof(float));
                }
            }
        } else if (net->nodes[j].layer->type == FULL_CONNECTED ||
                   net->nodes[j].layer->type == SOFTMAX) {
            w = net->tensors[net->nodes[j].dst[0]].w;
            h = net->tensors[net->nodes[j].dst[0]].h;
            c = net->tensors[net->nodes[j].dst[0]].c;
            sz = w * h * c;
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(
                net->tensors[net->nodes[j].dst[0]].data_gpu,
                net->tensors[net->nodes[j].dst[0]].data,
                bcnn_tensor_size(&net->tensors[net->nodes[j].dst[0]]));
#endif
            sprintf(name, "ip_%d.txt", j);
            ftmp = fopen(name, "wt");
            for (i = 0; i < net->tensors[net->nodes[j].dst[0]].n; ++i) {
                layer = net->nodes[j].layer;
                for (k = 0; k < sz; ++k) {
                    fprintf(
                        ftmp, "%f ",
                        net->tensors[net->nodes[j].dst[0]].data[i * sz + k]);
                }
                fprintf(ftmp, "\n");
            }
            fclose(ftmp);
            if (sz == 2 && net->nodes[j].layer->type == FULL_CONNECTED) {
                sz = bcnn_tensor_size(&net->tensors[net->nodes[j].src[0]]);
#ifdef BCNN_USE_CUDA
                bcnn_cuda_memcpy_dev2host(net->nodes[j].layer->weights.data_gpu,
                                          net->nodes[j].layer->weights.data,
                                          sz);
#endif
                sprintf(name, "wgt_%d.txt", j);
                ftmp = fopen(name, "wt");
                layer = net->nodes[j].layer;
                for (k = 0; k < sz; ++k) {
                    fprintf(ftmp, "%f ", layer->weights.data[k]);
                }
                fprintf(ftmp, "\n");
                fclose(ftmp);
                sz = 2;
#ifdef BCNN_USE_CUDA
                bcnn_cuda_memcpy_dev2host(net->nodes[j].layer->biases.data_gpu,
                                          net->nodes[j].layer->biases.data, sz);
#endif
                sprintf(name, "b_%d.txt", j);
                ftmp = fopen(name, "wt");
                layer = net->nodes[j].layer;
                for (k = 0; k < sz; ++k) {
                    fprintf(ftmp, "%f ", layer->biases.data[k]);
                }
                fprintf(ftmp, "\n");
                fclose(ftmp);
            }
        }
    }

    return BCNN_SUCCESS;
}

int bcnn_free_layer(bcnn_layer **layer) {
    bcnn_layer *p_layer = (*layer);
    bh_free(p_layer->indexes);
    bcnn_tensor_destroy(&p_layer->weights);
    bcnn_tensor_destroy(&p_layer->biases);
    bcnn_tensor_destroy(&p_layer->scales);
    bcnn_tensor_destroy(&p_layer->saved_mean);
    bcnn_tensor_destroy(&p_layer->saved_variance);
    bcnn_tensor_destroy(&p_layer->running_mean);
    bcnn_tensor_destroy(&p_layer->running_variance);
    bh_free(p_layer->conv_workspace);
    bh_free(p_layer->x_norm);
    bh_free(p_layer->bn_workspace);
    bh_free(p_layer->rand);
    bh_free(p_layer->adam_m);
    bh_free(p_layer->adam_v);
    bh_free(p_layer->binary_weight);
    bh_free(p_layer->binary_workspace);
    bh_free(p_layer->cost);
#ifdef BCNN_USE_CUDA
    if (p_layer->indexes_gpu) bcnn_cuda_free(p_layer->indexes_gpu);
    if (p_layer->x_norm_gpu) bcnn_cuda_free(p_layer->x_norm_gpu);
    if (p_layer->bn_workspace_gpu) bcnn_cuda_free(p_layer->bn_workspace_gpu);
    if (p_layer->rand_gpu) bcnn_cuda_free(p_layer->rand_gpu);
    if (p_layer->adam_m_gpu) bcnn_cuda_free(p_layer->adam_m_gpu);
    if (p_layer->adam_v_gpu) bcnn_cuda_free(p_layer->adam_v_gpu);
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
