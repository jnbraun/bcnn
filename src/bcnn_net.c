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

/* include bh helpers */
#include <bh/bh_mem.h>
#include <bh/bh_string.h>

/* include bip image processing lib */
#include <bip/bip.h>

#include "bcnn_activation_layer.h"
#include "bcnn_avgpool_layer.h"
#include "bcnn_batchnorm_layer.h"
#include "bcnn_concat_layer.h"
#include "bcnn_conv_layer.h"
#include "bcnn_cost_layer.h"
#include "bcnn_data.h"
#include "bcnn_deconv_layer.h"
#include "bcnn_depthwise_conv_layer.h"
#include "bcnn_dropout_layer.h"
#include "bcnn_eltwise_layer.h"
#include "bcnn_fc_layer.h"
#include "bcnn_lrn_layer.h"
#include "bcnn_mat.h"
#include "bcnn_maxpool_layer.h"
#include "bcnn_net.h"
#include "bcnn_softmax_layer.h"
#include "bcnn_tensor.h"
#include "bcnn_upsample_layer.h"
#include "bcnn_utils.h"
#include "bcnn_yolo.h"

bcnn_status bcnn_net_create_gemm_context(bcnn_net *net);

bcnn_status bcnn_init_net(bcnn_net **net) {
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
    // Label node is set to be the second node
    bcnn_net_add_tensor(*net, label);
    // Internal context for gemm
    BCNN_CHECK_STATUS(bcnn_net_create_gemm_context(*net));

    return BCNN_SUCCESS;
}

static inline void bcnn_net_destroy_tensors(bcnn_net *net) {
    for (int i = 0; i < net->num_tensors; ++i) {
        bcnn_tensor_destroy(&net->tensors[i]);
    }
    bh_free(net->tensors);
}

static inline void bcnn_free_node(bcnn_node *node) {
    bh_free(node->src);
    bh_free(node->dst);
    bh_free(node->param);
}

static void bcnn_free_workload(bcnn_net *net) {
    bcnn_tensor_free(&net->tensors[0]);
    bh_free(net->input_buffer);
#ifdef BCNN_USE_CUDA
    bcnn_cuda_free(net->workspace_gpu);
#endif
}

static void bcnn_free_net(bcnn_net *net) {
    bcnn_free_workload(net);
    for (int i = 0; i < net->num_nodes; ++i) {
        if (net->nodes[i].release_param) {
            net->nodes[i].release_param(&net->nodes[i]);
        }
        bcnn_free_node(&net->nodes[i]);
    }
    bh_free(net->nodes);
    bcnn_net_destroy_tensors(net);
    // Gemm context
    bh_align_free(net->gemm_ctx);
}

void bcnn_end_net(bcnn_net **net) {
    bcnn_free_net(*net);
    bh_free(*net);
}

void bcnn_set_log_context(bcnn_net *net, bcnn_log_callback fct,
                          bcnn_log_level level) {
    net->log_ctx.fct = fct;
    net->log_ctx.lvl = level;
}

bcnn_status bcnn_net_create_gemm_context(bcnn_net *net) {
    net->gemm_ctx = bh_align_calloc(sizeof(bcnn_gemm_context), 32);
    if (net->gemm_ctx) {
        return BCNN_SUCCESS;
    } else {
        return BCNN_FAILED_ALLOC;
    }
}

int bcnn_set_param(bcnn_net *net, const char *name, const char *val) {
    if (strcmp(name, "input_width") == 0) {
        net->input_width = atoi(val);
    } else if (strcmp(name, "input_height") == 0) {
        net->input_height = atoi(val);
    } else if (strcmp(name, "input_channels") == 0) {
        net->input_channels = atoi(val);
    } else if (strcmp(name, "batch_size") == 0) {
        net->batch_size = atoi(val);
    } else if (strcmp(name, "max_batches") == 0) {
        net->learner.max_batches = atoi(val);
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
    } else if (strcmp(name, "max_spots") == 0) {
        net->data_aug.max_random_spots = (float)atof(val);
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

bcnn_status bcnn_add_input(bcnn_net *net, int w, int h, int c, char *name) {
    // Create input node
    bcnn_tensor input = {0};
    bcnn_tensor_set_shape(&input, net->batch_size, c, h, w, 0);  // no gradient
    bcnn_tensor_allocate(&input, net->mode);
    bh_strfill(&input.name, name);
    // Add tensor to net
    return bcnn_net_add_tensor(net, input);
}

void bcnn_set_input_shape(bcnn_net *net, int input_width, int input_height,
                          int input_channels, int batch_size) {
    net->input_width = input_width;
    net->input_height = input_height;
    net->input_channels = input_channels;
    net->batch_size = batch_size;
    bcnn_tensor_set_shape(&net->tensors[0], batch_size, input_channels,
                          input_height, input_width, 0);
}

static bcnn_status bcnn_init_workload(bcnn_net *net) {
    net->input_buffer = (unsigned char *)calloc(
        net->input_width * net->input_height * net->input_channels, 1);

    // Allocate tensor for input node
    bcnn_tensor_allocate(&net->tensors[0], net->mode);

    // Special case for detection: allocate label tensor
    /*if (net->prediction_type == DETECTION && net->mode != PREDICT) {
        bcnn_tensor_set_shape(&net->tensors[1], net->batch_size, 1, 1,
                              BCNN_DETECTION_MAX_BOXES * 5, 0);
        bcnn_tensor_allocate(&net->tensors[1], net->mode);
    }*/

#ifdef BCNN_USE_CUDA
    net->workspace_gpu = bcnn_cuda_malloc_f32(net->workspace_size);
    for (int i = 0; i < net->num_nodes; ++i) {
        if (net->nodes[i].type == CONVOLUTIONAL) {
            bcnn_conv_param *param = (bcnn_conv_param *)(net->nodes[i].param);
            param->conv_workspace_gpu = net->workspace_gpu;
        }
    }
#endif

    return BCNN_SUCCESS;
}

bcnn_status bcnn_compile_net(bcnn_net *net) {
    bcnn_free_workload(net);
    bcnn_init_workload(net);
    return BCNN_SUCCESS;
}

int bcnn_forward(bcnn_net *net) {
    for (int i = 0; i < net->num_nodes; ++i) {
        bcnn_node *node = &net->nodes[i];
        for (int j = 0; j < node->num_dst; ++j) {
            int output_size = bcnn_tensor_size(&net->tensors[node->dst[j]]);
#ifdef BCNN_USE_CUDA
            if (net->tensors[node->dst[j]].grad_data_gpu != NULL) {
                bcnn_cuda_fill_f32(output_size, 0.0f,
                                   net->tensors[node->dst[j]].grad_data_gpu, 1);
            }
#else
            if (net->tensors[node->dst[j]].grad_data != NULL) {
                memset(net->tensors[node->dst[j]].grad_data, 0,
                       output_size * sizeof(float));
            }
#endif
        }
        node->forward(net, node);
    }
    return BCNN_SUCCESS;
}

int bcnn_backward(bcnn_net *net) {
    for (int i = net->num_nodes - 1; i >= 0; --i) {
        bcnn_node *node = &net->nodes[i];
        node->backward(net, node);
    }
    return BCNN_SUCCESS;
}

int bcnn_train_on_batch(bcnn_net *net, bcnn_loader *iter, float *loss) {
    bcnn_loader_next(net, iter);
    net->learner.seen += net->batch_size;
    // Forward
    bcnn_forward(net);
    // Back prop
    bcnn_backward(net);
    // Update network weight
    bcnn_update(net);
    if (iter->type == BCNN_LOAD_DETECTION_LIST) {
        *loss = 0;
        int n = 0;
        for (int i = 0; i < net->num_nodes; ++i) {
            if (net->nodes[i].type == YOLO) {
                bcnn_yolo_param *param =
                    (bcnn_yolo_param *)(net->nodes[i].param);
                if (param->cost) {
                    *loss += param->cost[0];
                    ++n;
                }
            }
        }
        if (n > 0) {
            *loss /= n;
        }
    } else {
        *loss = net->tensors[net->nodes[net->num_nodes - 1].dst[0]].data[0];
    }
    return BCNN_SUCCESS;
}

int bcnn_predict_on_batch(bcnn_net *net, bcnn_loader *iter, float **pred,
                          float *error) {
    int nb = net->num_nodes;
    int output_size =
        bcnn_tensor_size(&net->tensors[net->nodes[nb - 1].src[0]]);

    bcnn_loader_next(net, iter);

    bcnn_forward(net);

#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_dev2host(net->tensors[net->nodes[nb - 1].src[0]].data_gpu,
                              net->tensors[net->nodes[nb - 1].src[0]].data,
                              output_size);
#endif
    (*pred) = net->tensors[net->nodes[nb - 1].src[0]].data;
    if (net->nodes[net->num_nodes - 1].type == YOLO) {
        *error = 0;
        int n = 0;
        for (int i = 0; i < net->num_nodes; ++i) {
            if (net->nodes[i].type == YOLO) {
                bcnn_yolo_param *param =
                    (bcnn_yolo_param *)(net->nodes[i].param);
                if (param->cost) {
                    *error += param->cost[0];
                    ++n;
                }
            }
        }
        if (n > 0) {
            *error /= n;
        }
    } else {
        *error = *(net->tensors[net->nodes[nb - 1].dst[0]].data);
    }

    return BCNN_SUCCESS;
}

bcnn_status bcnn_write_model(bcnn_net *net, char *filename) {
    FILE *fp = NULL;
    fp = fopen(filename, "wb");
    BCNN_CHECK_AND_LOG(net->log_ctx, fp, BCNN_INVALID_PARAMETER,
                       "Could not open model file %s\n", filename);

    fwrite(&net->learner.learning_rate, sizeof(float), 1, fp);
    fwrite(&net->learner.momentum, sizeof(float), 1, fp);
    fwrite(&net->learner.decay, sizeof(float), 1, fp);
    fwrite(&net->learner.seen, sizeof(int), 1, fp);

    for (int i = 0; i < net->num_nodes; ++i) {
        bcnn_node *node = &net->nodes[i];
        if (node->type == CONVOLUTIONAL || node->type == DECONVOLUTIONAL ||
            node->type == DEPTHWISE_CONV || node->type == FULL_CONNECTED) {
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
            if (node->type == CONVOLUTIONAL) {
                bcnn_conv_param *param = (bcnn_conv_param *)node->param;
                if (param->batch_norm == 1) {
                    bcnn_tensor *bn_mean = &net->tensors[net->nodes[i].src[3]];
                    bcnn_tensor *bn_var = &net->tensors[net->nodes[i].src[4]];
                    bcnn_tensor *bn_scales =
                        &net->tensors[net->nodes[i].src[5]];
                    int bn_mean_size = bcnn_tensor_size(bn_mean);
                    int bn_var_size = bcnn_tensor_size(bn_var);
                    int bn_scales_size = bcnn_tensor_size(bn_scales);
#ifdef BCNN_USE_CUDA
                    bcnn_cuda_memcpy_dev2host(bn_mean->data_gpu, bn_mean->data,
                                              bn_mean_size);
                    bcnn_cuda_memcpy_dev2host(bn_var->data_gpu, bn_var->data,
                                              bn_var_size);
                    bcnn_cuda_memcpy_dev2host(bn_scales->data_gpu,
                                              bn_scales->data, bn_scales_size);
#endif
                    fwrite(bn_mean->data, sizeof(float), bn_mean_size, fp);
                    fwrite(bn_var->data, sizeof(float), bn_var_size, fp);
                    fwrite(bn_scales->data, sizeof(float), bn_scales_size, fp);
                }
            }
        }
        if (node->type == ACTIVATION) {
            bcnn_activation_param *param = (bcnn_activation_param *)node->param;
            if (param->activation == PRELU) {
                bcnn_tensor *weights = &net->tensors[net->nodes[i].src[1]];
                int weights_size = bcnn_tensor_size(weights);
                fwrite(weights->data, sizeof(float), weights_size, fp);
            }
        }
        if (node->type == BATCHNORM) {
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
    int i, j, is_ft = 0;
    size_t nb_read = 0;
    float tmp = 0.0f;

    fp = fopen(filename, "rb");
    BCNN_CHECK_AND_LOG(net->log_ctx, fp, BCNN_INVALID_PARAMETER,
                       "Can not open file %s\n", filename);

    nb_read = fread(&tmp, sizeof(float), 1, fp);
    nb_read = fread(&tmp, sizeof(float), 1, fp);
    nb_read = fread(&tmp, sizeof(float), 1, fp);
    nb_read = fread(&net->learner.seen, sizeof(int), 1, fp);
    BCNN_INFO(net->log_ctx, "lr= %f ", net->learner.learning_rate);
    BCNN_INFO(net->log_ctx, "m= %f ", net->learner.momentum);
    BCNN_INFO(net->log_ctx, "decay= %f ", net->learner.decay);
    BCNN_INFO(net->log_ctx, "seen= %d\n", net->learner.seen);

    for (i = 0; i < net->num_nodes; ++i) {
        bcnn_node *node = &net->nodes[i];
        is_ft = 0;
        if ((node->type == CONVOLUTIONAL || node->type == DECONVOLUTIONAL ||
             node->type == DEPTHWISE_CONV || node->type == FULL_CONNECTED) &&
            is_ft == 0) {
            bcnn_tensor *weights = &net->tensors[net->nodes[i].src[1]];
            bcnn_tensor *biases = &net->tensors[net->nodes[i].src[2]];
            int weights_size = bcnn_tensor_size(weights);
            int biases_size = bcnn_tensor_size(biases);
            nb_read = fread(biases->data, sizeof(float), biases_size, fp);
            BCNN_INFO(net->log_ctx,
                      "node_idx= %d nbread_bias= %lu bias_size_expected= %d", i,
                      (unsigned long)nb_read, biases_size);
            nb_read = fread(weights->data, sizeof(float), weights_size, fp);
            BCNN_INFO(
                net->log_ctx,
                "node_idx= %d nbread_weight= %lu weight_size_expected= %d", i,
                (unsigned long)nb_read, weights_size);
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_host2dev(weights->data_gpu, weights->data,
                                      weights_size);
            bcnn_cuda_memcpy_host2dev(biases->data_gpu, biases->data,
                                      biases_size);
#endif
            if (node->type == CONVOLUTIONAL) {
                bcnn_conv_param *param = (bcnn_conv_param *)node->param;
                if (param->batch_norm == 1) {
                    bcnn_tensor *bn_mean = &net->tensors[net->nodes[i].src[3]];
                    bcnn_tensor *bn_var = &net->tensors[net->nodes[i].src[4]];
                    bcnn_tensor *bn_scales =
                        &net->tensors[net->nodes[i].src[5]];
                    int bn_mean_size = bcnn_tensor_size(bn_mean);
                    int bn_var_size = bcnn_tensor_size(bn_var);
                    int bn_scales_size = bcnn_tensor_size(bn_scales);
                    nb_read =
                        fread(bn_mean->data, sizeof(float), bn_mean_size, fp);
                    nb_read =
                        fread(bn_var->data, sizeof(float), bn_var_size, fp);
                    nb_read = fread(bn_scales->data, sizeof(float),
                                    bn_scales_size, fp);
#ifdef BCNN_USE_CUDA
                    bcnn_cuda_memcpy_host2dev(bn_mean->data_gpu, bn_mean->data,
                                              bn_mean_size);
                    bcnn_cuda_memcpy_host2dev(bn_var->data_gpu, bn_var->data,
                                              bn_var_size);
                    bcnn_cuda_memcpy_host2dev(bn_scales->data_gpu,
                                              bn_scales->data, bn_scales_size);
#endif
                }
            }
        }
        if (node->type == ACTIVATION) {
            bcnn_activation_param *param = (bcnn_activation_param *)node->param;
            if (param->activation == PRELU) {
                bcnn_tensor *weights = &net->tensors[net->nodes[i].src[1]];
                int weights_size = bcnn_tensor_size(weights);
                nb_read = fread(weights->data, sizeof(float), weights_size, fp);
                BCNN_INFO(net->log_ctx, "PReLU= %d nbread= %lu expected= %d", i,
                          (unsigned long)nb_read, weights_size);
            }
        }
        if (node->type == BATCHNORM) {
            bcnn_tensor *bn_mean = &net->tensors[net->nodes[i].src[1]];
            bcnn_tensor *bn_var = &net->tensors[net->nodes[i].src[2]];
            bcnn_tensor *bn_scales = &net->tensors[net->nodes[i].src[3]];
            bcnn_tensor *bn_biases = &net->tensors[net->nodes[i].src[4]];
            int sz = net->tensors[net->nodes[i].dst[0]].c;
            nb_read = fread(bn_mean->data, sizeof(float), sz, fp);
            BCNN_INFO(net->log_ctx,
                      "batchnorm= %d nbread_mean= %lu mean_size_expected= %d",
                      i, (unsigned long)nb_read, sz);
            nb_read = fread(bn_var->data, sizeof(float), sz, fp);
            BCNN_INFO(net->log_ctx,
                      "batchnorm= %d nbread_variance= %lu "
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

bcnn_status bcnn_load_model_legacy(bcnn_net *net, char *filename) {
    FILE *fp = NULL;
    int i, j, is_ft = 0;
    size_t nb_read = 0;
    float tmp = 0.0f;

    fp = fopen(filename, "rb");
    BCNN_CHECK_AND_LOG(net->log_ctx, fp, BCNN_INVALID_PARAMETER,
                       "Can not open file %s\n", filename);

    nb_read = fread(&tmp, sizeof(float), 1, fp);
    nb_read = fread(&tmp, sizeof(float), 1, fp);
    nb_read = fread(&tmp, sizeof(float), 1, fp);
    nb_read = fread(&net->learner.seen, sizeof(int), 1, fp);
    BCNN_INFO(net->log_ctx, "lr= %f ", net->learner.learning_rate);
    BCNN_INFO(net->log_ctx, "m= %f ", net->learner.momentum);
    BCNN_INFO(net->log_ctx, "decay= %f ", net->learner.decay);
    BCNN_INFO(net->log_ctx, "seen= %d\n", net->learner.seen);

    for (i = 0; i < net->num_nodes; ++i) {
        bcnn_node *node = &net->nodes[i];
        is_ft = 0;
        if ((node->type == CONVOLUTIONAL || node->type == DECONVOLUTIONAL ||
             node->type == DEPTHWISE_CONV || node->type == FULL_CONNECTED) &&
            is_ft == 0) {
            bcnn_tensor *weights = &net->tensors[net->nodes[i].src[1]];
            bcnn_tensor *biases = &net->tensors[net->nodes[i].src[2]];
            int weights_size = bcnn_tensor_size(weights);
            int biases_size = bcnn_tensor_size(biases);
            nb_read = fread(biases->data, sizeof(float), biases_size, fp);
            BCNN_INFO(net->log_ctx,
                      "node_idx= %d nbread_bias= %lu bias_size_expected= %d", i,
                      (unsigned long)nb_read, biases_size);
            nb_read = fread(weights->data, sizeof(float), weights_size, fp);
            BCNN_INFO(
                net->log_ctx,
                "node_idx= %d nbread_weight= %lu weight_size_expected= %d", i,
                (unsigned long)nb_read, weights_size);
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_host2dev(weights->data_gpu, weights->data,
                                      weights_size);
            bcnn_cuda_memcpy_host2dev(biases->data_gpu, biases->data,
                                      biases_size);
#endif
            if (node->type == CONVOLUTIONAL) {
                bcnn_conv_param *param = (bcnn_conv_param *)node->param;
                if (param->batch_norm == 1) {
                    bcnn_tensor *bn_mean = &net->tensors[net->nodes[i].src[3]];
                    bcnn_tensor *bn_var = &net->tensors[net->nodes[i].src[4]];
                    bcnn_tensor *bn_scales =
                        &net->tensors[net->nodes[i].src[5]];
                    int bn_mean_size = bcnn_tensor_size(bn_mean);
                    int bn_var_size = bcnn_tensor_size(bn_var);
                    nb_read =
                        fread(bn_mean->data, sizeof(float), bn_mean_size, fp);
                    nb_read =
                        fread(bn_var->data, sizeof(float), bn_var_size, fp);
#ifdef BCNN_USE_CUDA
                    bcnn_cuda_memcpy_host2dev(bn_mean->data_gpu, bn_mean->data,
                                              bn_mean_size);
                    bcnn_cuda_memcpy_host2dev(bn_var->data_gpu, bn_var->data,
                                              bn_var_size);
#endif
                }
            }
        }
        if (node->type == ACTIVATION) {
            bcnn_activation_param *param = (bcnn_activation_param *)node->param;
            if (param->activation == PRELU) {
                bcnn_tensor *weights = &net->tensors[net->nodes[i].src[1]];
                int weights_size = bcnn_tensor_size(weights);
                nb_read = fread(weights->data, sizeof(float), weights_size, fp);
                BCNN_INFO(net->log_ctx, "PReLU= %d nbread= %lu expected= %d", i,
                          (unsigned long)nb_read, weights_size);
            }
        }
        if (node->type == BATCHNORM) {
            bcnn_tensor *bn_mean = &net->tensors[net->nodes[i].src[1]];
            bcnn_tensor *bn_var = &net->tensors[net->nodes[i].src[2]];
            bcnn_tensor *bn_scales = &net->tensors[net->nodes[i].src[3]];
            bcnn_tensor *bn_biases = &net->tensors[net->nodes[i].src[4]];
            int sz = net->tensors[net->nodes[i].dst[0]].c;
            nb_read = fread(bn_mean->data, sizeof(float), sz, fp);
            BCNN_INFO(net->log_ctx,
                      "batchnorm= %d nbread_mean= %lu mean_size_expected= %d",
                      i, (unsigned long)nb_read, sz);
            nb_read = fread(bn_var->data, sizeof(float), sz, fp);
            BCNN_INFO(net->log_ctx,
                      "batchnorm= %d nbread_variance= %lu "
                      "variance_size_expected= %d",
                      i, (unsigned long)nb_read, sz);
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_host2dev(bn_mean->data_gpu, bn_mean->data, sz);
            bcnn_cuda_memcpy_host2dev(bn_var->data_gpu, bn_var->data, sz);
#endif
        }
    }
    if (fp != NULL) fclose(fp);

    BCNN_INFO(net->log_ctx, "Model %s loaded succesfully", filename);
    fflush(stdout);

    return BCNN_SUCCESS;
}
#if 0
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
    bh_free(p_layer->workspace);
    bh_free(p_layer->rand);
    bh_free(p_layer->adam_m);
    bh_free(p_layer->adam_v);
    bh_free(p_layer->binary_weight);
    bh_free(p_layer->binary_workspace);
    bh_free(p_layer->cost);
    bh_free(p_layer->mask);
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
#endif
