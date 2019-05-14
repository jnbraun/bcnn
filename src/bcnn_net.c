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
#include <bh/bh_ini.h>
#include <bh/bh_macros.h>
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

bcnn_status bcnn_init_net(bcnn_net **net, bcnn_mode mode) {
    bcnn_net *p_net = (bcnn_net *)calloc(1, sizeof(bcnn_net));
    if (p_net == NULL) {
        return BCNN_FAILED_ALLOC;
    }
    p_net->mode = mode;
    // Create input node
    bcnn_tensor input = {0};
    bh_strfill(&input.name, "input");
    // Input node is set to be the first node
    bcnn_net_add_tensor(p_net, input);
    // Create label node
    bcnn_tensor label = {0};
    bh_strfill(&label.name, "label");
    // Label node is set to be the second node
    bcnn_net_add_tensor(p_net, label);
    // If required, allocate learner and data augmentation struct
    if (mode != BCNN_MODE_PREDICT) {
        p_net->learner = (bcnn_learner *)calloc(1, sizeof(bcnn_learner));
        p_net->data_aug =
            (bcnn_data_augmenter *)calloc(1, sizeof(bcnn_data_augmenter));
    }

#ifdef BCNN_USE_CUDA
    BCNN_CHECK_STATUS(bcnn_net_create_cuda_context(p_net));
#endif
#ifndef BCNN_USE_BLAS
    // Internal context for gemm
    BCNN_CHECK_STATUS(bcnn_net_create_gemm_context(p_net));
#endif
    p_net->num_threads = 1;
#ifdef BCNN_USE_OPENMP
    p_net->num_threads = bcnn_omp_get_num_threads();
    fprintf(stderr, "init num_threads %d\n", p_net->num_threads);
#endif
    *net = p_net;
    return BCNN_SUCCESS;
}

static inline void bcnn_destroy_tensors(bcnn_net *net) {
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
#ifdef BCNN_USE_CUDA
    bcnn_cuda_context *cuda_ctx = (bcnn_cuda_context *)net->cuda_ctx;
    bcnn_cuda_free(cuda_ctx->workspace_gpu);
#endif
}

static void bcnn_free_net(bcnn_net *net) {
    // Free workload
    bcnn_free_workload(net);
    // Destroy nodes
    for (int i = 0; i < net->num_nodes; ++i) {
        if (net->nodes[i].release_param) {
            net->nodes[i].release_param(&net->nodes[i]);
        }
        bcnn_free_node(&net->nodes[i]);
    }
    bh_free(net->nodes);
    // Free tensors
    bcnn_destroy_tensors(net);
    // Free data loader
    bcnn_destroy_data_loader(net);
    // Free data augmenter
    bh_free(net->data_aug);
    // Free learner
    bh_free(net->learner);
#ifdef BCNN_USE_CUDA
    // Free cuda context
    bh_free(net->cuda_ctx);
#endif
#ifndef BCNN_USE_BLAS
    // Free gemm context
    bh_align_free(net->gemm_ctx);
#endif
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

#ifdef BCNN_USE_CUDA
bcnn_status bcnn_net_create_cuda_context(bcnn_net *net) {
    net->cuda_ctx = calloc(1, sizeof(bcnn_cuda_context));
    if (net->cuda_ctx) {
        return BCNN_SUCCESS;
    } else {
        return BCNN_FAILED_ALLOC;
    }
}
#endif

void bcnn_set_num_threads(bcnn_net *net, int num_threads) {
    net->num_threads = 1;
#ifdef BCNN_USE_OPENMP
    net->num_threads = bh_clamp(num_threads, 1, 8);
    omp_set_num_threads(net->num_threads);
#endif
}

bcnn_status bcnn_net_add_node(bcnn_net *net, bcnn_node node) {
    bcnn_node *p_node = NULL;
    net->num_nodes++;
    p_node =
        (bcnn_node *)realloc(net->nodes, net->num_nodes * sizeof(bcnn_node));
    BCNN_CHECK_AND_LOG(net->log_ctx, (p_node != NULL), BCNN_FAILED_ALLOC,
                       "Internal allocation error\n");
    net->nodes = p_node;
    net->nodes[net->num_nodes - 1] = node;
    return BCNN_SUCCESS;
}

bcnn_status bcnn_net_add_tensor(bcnn_net *net, bcnn_tensor tensor) {
    bcnn_tensor *p_tensors = NULL;
    net->num_tensors++;
    p_tensors = (bcnn_tensor *)realloc(net->tensors,
                                       net->num_tensors * sizeof(bcnn_tensor));
    BCNN_CHECK_AND_LOG(net->log_ctx, (p_tensors != NULL), BCNN_FAILED_ALLOC,
                       "Internal allocation error\n");
    net->tensors = p_tensors;
    net->tensors[net->num_tensors - 1] = tensor;
    return BCNN_SUCCESS;
}

bcnn_status bcnn_add_input(bcnn_net *net, int w, int h, int c,
                           const char *name) {
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
    net->batch_size = batch_size;
    bcnn_tensor_set_shape(&net->tensors[0], batch_size, input_channels,
                          input_height, input_width, 0);
}

static bcnn_status bcnn_init_workload(bcnn_net *net) {
    // Allocate tensor for input node
    BCNN_CHECK_STATUS(bcnn_tensor_allocate(&net->tensors[0], net->mode));

#ifdef BCNN_USE_CUDA
    bcnn_cuda_context *cuda_ctx = (bcnn_cuda_context *)net->cuda_ctx;
    cuda_ctx->workspace_gpu = bcnn_cuda_malloc_f32(cuda_ctx->workspace_size);
    for (int i = 0; i < net->num_nodes; ++i) {
        if (net->nodes[i].type == BCNN_LAYER_CONV2D) {
            bcnn_conv_param *param = (bcnn_conv_param *)(net->nodes[i].param);
            param->conv_workspace_gpu = cuda_ctx->workspace_gpu;
        }
    }
#endif

    return BCNN_SUCCESS;
}

int bcnn_get_batch_size(bcnn_net *net) { return net->batch_size; }

bcnn_status bcnn_compile_net(bcnn_net *net) {
    bcnn_free_workload(net);
    return bcnn_init_workload(net);
}

static void bcnn_reset_gradients(bcnn_net *net, bcnn_node *node) {
    for (int i = 0; i < node->num_dst; ++i) {
        int sz = bcnn_tensor_size(&net->tensors[node->dst[i]]);
#ifdef BCNN_USE_CUDA
        if (net->tensors[node->dst[i]].grad_data_gpu != NULL) {
            bcnn_cuda_fill_f32(sz, 0.0f,
                               net->tensors[node->dst[i]].grad_data_gpu, 1);
        }
#else
        if (net->tensors[node->dst[i]].grad_data != NULL) {
            memset(net->tensors[node->dst[i]].grad_data, 0, sz * sizeof(float));
        }
#endif
    }
}

/* Given a tensor name, return its index in the net tensors array or -1 if not
 * found */
int bcnn_get_tensor_index_by_name(bcnn_net *net, const char *name) {
    for (int i = net->num_tensors - 1; i >= 0; --i) {
        if (strcmp(net->tensors[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

bcnn_tensor *bcnn_get_tensor_by_index(bcnn_net *net, int index) {
    if (index < 0 || (index > net->num_tensors - 1)) {
        return NULL;
    }
    bcnn_tensor *tensor = &net->tensors[index];
    return tensor;
}

bcnn_tensor *bcnn_get_tensor_by_name(bcnn_net *net, const char *name) {
    int index = bcnn_get_tensor_index_by_name(net, name);
    return bcnn_get_tensor_by_index(net, index);
}

void bcnn_forward(bcnn_net *net) {
    for (int i = 0; i < net->num_nodes; ++i) {
        bcnn_node *node = &net->nodes[i];
        if (net->mode == BCNN_MODE_TRAIN) {
            bcnn_reset_gradients(net, node);
        }
        node->forward(net, node);
    }
}

void bcnn_backward(bcnn_net *net) {
    for (int i = net->num_nodes - 1; i >= 0; --i) {
        bcnn_node *node = &net->nodes[i];
        node->backward(net, node);
    }
}

static float bcnn_get_loss(bcnn_net *net) {
    float loss = 0;
    int n = 0;
    for (int i = 0; i < net->num_nodes; ++i) {
        if (net->nodes[i].type == BCNN_LAYER_COST) {
            loss += net->tensors[net->nodes[i].dst[0]].data[0];
            ++n;
        } else if (net->nodes[i].type == BCNN_LAYER_YOLOV3) {
            bcnn_yolo_param *param = (bcnn_yolo_param *)(net->nodes[i].param);
            if (param->cost) {
                loss += param->cost[0];
                ++n;
            }
        }
    }
    if (n > 0) {
        loss /= n;
    }
    return loss;
}

float bcnn_train_on_batch(bcnn_net *net) {
    // Get next batch of data
    bcnn_loader_next(net);
    // Forward
    bcnn_forward(net);
    // Back prop
    bcnn_backward(net);
    // Update network weight
    bcnn_update(net);
    // Return the loss value
    return bcnn_get_loss(net);
}

float bcnn_predict_on_batch(bcnn_net *net, bcnn_tensor **out) {
    // Get next batch of data
    bcnn_loader_next(net);
    // Forward
    bcnn_forward(net);
    // Extract output tensor
    int out_id = net->nodes[net->num_nodes - 1].dst[0];
    if (net->nodes[net->num_nodes - 1].type == BCNN_LAYER_COST) {
        out_id = net->nodes[net->num_nodes - 1].src[0];
    }
#ifdef BCNN_USE_CUDA
    int sz = bcnn_tensor_size(&net->tensors[out_id]);
    bcnn_cuda_memcpy_dev2host(net->tensors[out_id].data_gpu,
                              net->tensors[out_id].data, sz);
#endif
    *out = &net->tensors[out_id];
    // Return the loss value
    return bcnn_get_loss(net);
}

void bcnn_set_learner(bcnn_net *net, bcnn_optimizer type, bcnn_learner params) {
    net->learner->optimizer = type;
    return;
}

bcnn_status bcnn_set_mode(bcnn_net *net, bcnn_mode mode) {
    if (net->mode == mode) {
        // Nothing changes
        return BCNN_SUCCESS;
    } else {
        // TODO: Still needs to ensure that the network allocation have been
        // done while in 'train' mode
        net->mode = mode;
        if (net->data_loader) {
            // Switch the dataset handles train / valid if required
            bcnn_switch_data_handles(net, net->data_loader);
        }
    }
    return BCNN_SUCCESS;
}

void bcnn_net_set_param(bcnn_net *net, const char *name, const char *val) {
    if (strcmp(name, "input_width") == 0 || strcmp(name, "width") == 0) {
        net->tensors[0].w = atoi(val);
    } else if (strcmp(name, "input_height") == 0 ||
               strcmp(name, "height") == 0) {
        net->tensors[0].h = atoi(val);
    } else if (strcmp(name, "input_channels") == 0 ||
               strcmp(name, "channels") == 0) {
        net->tensors[0].c = atoi(val);
    } else if (strcmp(name, "batch_size") == 0 || strcmp(name, "batch") == 0) {
        net->batch_size = atoi(val);
        net->tensors[0].n = atoi(val);
    } else if (net->learner && strcmp(name, "max_batches") == 0) {
        net->learner->max_batches = atoi(val);
    } else if (net->learner && ((strcmp(name, "learning_policy") == 0 ||
                                 strcmp(name, "decay_type") == 0))) {
        if (strcmp(val, "sigmoid") == 0) {
            net->learner->decay_type = BCNN_LR_DECAY_SIGMOID;
        } else if (strcmp(val, "constant") == 0) {
            net->learner->decay_type = BCNN_LR_DECAY_CONSTANT;
        } else if (strcmp(val, "exp") == 0) {
            net->learner->decay_type = BCNN_LR_DECAY_EXP;
        } else if (strcmp(val, "inv") == 0) {
            net->learner->decay_type = BCNN_LR_DECAY_INV;
        } else if (strcmp(val, "step") == 0) {
            net->learner->decay_type = BCNN_LR_DECAY_STEP;
        } else if (strcmp(val, "poly") == 0) {
            net->learner->decay_type = BCNN_LR_DECAY_POLY;
        } else {
            net->learner->decay_type = BCNN_LR_DECAY_CONSTANT;
        }
    } else if (net->learner && strcmp(name, "optimizer") == 0) {
        if (strcmp(val, "sgd") == 0) {
            net->learner->optimizer = BCNN_OPTIM_SGD;
        } else if (strcmp(val, "adam") == 0) {
            net->learner->optimizer = BCNN_OPTIM_ADAM;
        }
    } else if (net->learner && strcmp(name, "step") == 0) {
        net->learner->step = atoi(val);
    } else if (net->learner && strcmp(name, "learning_rate") == 0) {
        net->learner->base_learning_rate = (float)atof(val);
        net->learner->learning_rate = net->learner->base_learning_rate;
    } else if (net->learner && strcmp(name, "beta1") == 0) {
        net->learner->beta1 = (float)atof(val);
    } else if (net->learner && strcmp(name, "beta2") == 0) {
        net->learner->beta2 = (float)atof(val);
    } else if (net->learner && strcmp(name, "decay") == 0) {
        net->learner->decay = (float)atof(val);
    } else if (net->learner && strcmp(name, "momentum") == 0) {
        net->learner->momentum = (float)atof(val);
    } else if (net->learner && strcmp(name, "gamma") == 0) {
        net->learner->gamma = (float)atof(val);
    } else if (net->data_aug && strcmp(name, "range_shift_x") == 0) {
        net->data_aug->range_shift_x = atoi(val);
    } else if (net->data_aug && strcmp(name, "range_shift_y") == 0) {
        net->data_aug->range_shift_y = atoi(val);
    } else if (net->data_aug && strcmp(name, "min_scale") == 0) {
        net->data_aug->min_scale = (float)atof(val);
    } else if (net->data_aug && strcmp(name, "max_scale") == 0) {
        net->data_aug->max_scale = (float)atof(val);
    } else if (net->data_aug && strcmp(name, "rotation_range") == 0) {
        net->data_aug->rotation_range = (float)atof(val);
    } else if (net->data_aug && strcmp(name, "min_contrast") == 0) {
        net->data_aug->min_contrast = (float)atof(val);
    } else if (net->data_aug && strcmp(name, "max_contrast") == 0) {
        net->data_aug->max_contrast = (float)atof(val);
    } else if (net->data_aug && strcmp(name, "min_brightness") == 0) {
        net->data_aug->min_brightness = atoi(val);
    } else if (net->data_aug && strcmp(name, "max_brightness") == 0) {
        net->data_aug->max_brightness = atoi(val);
    } else if (net->data_aug && strcmp(name, "max_distortion") == 0) {
        net->data_aug->max_distortion = (float)atof(val);
    } else if (net->data_aug && strcmp(name, "max_spots") == 0) {
        net->data_aug->max_random_spots = (float)atof(val);
    } else if (net->data_aug && strcmp(name, "flip_h") == 0) {
        net->data_aug->random_fliph = 1;
    } else if (net->data_aug && strcmp(name, "mean_r") == 0) {
        net->data_aug->mean_r = (float)atof(val) / 255.0f;
    } else if (net->data_aug && strcmp(name, "mean_g") == 0) {
        net->data_aug->mean_g = (float)atof(val) / 255.0f;
    } else if (net->data_aug && strcmp(name, "mean_b") == 0) {
        net->data_aug->mean_b = (float)atof(val) / 255.0f;
    } else if (net->data_aug && strcmp(name, "swap_to_bgr") == 0) {
        net->data_aug->swap_to_bgr = atoi(val);
    } else if (net->data_aug && strcmp(name, "no_input_norm") == 0) {
        net->data_aug->no_input_norm = atoi(val);
    }
}

#define BCNN_MAGIC "\x42\x43\x4E\x4E"

bcnn_status bcnn_save_weights(bcnn_net *net, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    BCNN_CHECK_AND_LOG(net->log_ctx, fp, BCNN_INVALID_PARAMETER,
                       "Could not open model file %s\n", filename);

    const uint32_t major = BCNN_VERSION_MAJOR;
    const uint32_t minor = BCNN_VERSION_MINOR;
    const uint32_t patch = BCNN_VERSION_PATCH;
    fwrite(BCNN_MAGIC, 1, 4, fp);
    fwrite(&major, sizeof(uint32_t), 1, fp);
    fwrite(&minor, sizeof(uint32_t), 1, fp);
    fwrite(&patch, sizeof(uint32_t), 1, fp);

    for (int i = 0; i < net->num_nodes; ++i) {
        bcnn_node *node = &net->nodes[i];
        if (node->type == BCNN_LAYER_CONV2D ||
            node->type == BCNN_LAYER_TRANSPOSE_CONV2D ||
            node->type == BCNN_LAYER_DEPTHWISE_CONV2D ||
            node->type == BCNN_LAYER_FULL_CONNECTED) {
            bcnn_tensor *w = &net->tensors[net->nodes[i].src[1]];
            bcnn_tensor *b = &net->tensors[net->nodes[i].src[2]];
            int w_sz = bcnn_tensor_size(w);
            int b_sz = bcnn_tensor_size(b);
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(w->data_gpu, w->data, w_sz);
            bcnn_cuda_memcpy_dev2host(b->data_gpu, b->data, b_sz);
#endif
            fwrite(b->data, sizeof(float), b_sz, fp);
            fwrite(w->data, sizeof(float), w_sz, fp);
            if (node->type == BCNN_LAYER_CONV2D) {
                bcnn_conv_param *param = (bcnn_conv_param *)node->param;
                if (param->batch_norm == 1) {
                    bcnn_tensor *m = &net->tensors[net->nodes[i].src[3]];
                    bcnn_tensor *v = &net->tensors[net->nodes[i].src[4]];
                    bcnn_tensor *s = &net->tensors[net->nodes[i].src[5]];
                    int m_sz = bcnn_tensor_size(m);
                    int v_sz = bcnn_tensor_size(v);
                    int s_sz = bcnn_tensor_size(s);
#ifdef BCNN_USE_CUDA
                    bcnn_cuda_memcpy_dev2host(m->data_gpu, m->data, m_sz);
                    bcnn_cuda_memcpy_dev2host(v->data_gpu, v->data, v_sz);
                    bcnn_cuda_memcpy_dev2host(s->data_gpu, s->data, s_sz);
#endif
                    fwrite(m->data, sizeof(float), m_sz, fp);
                    fwrite(v->data, sizeof(float), v_sz, fp);
                    fwrite(s->data, sizeof(float), s_sz, fp);
                }
            }
        }
        if (node->type == BCNN_LAYER_ACTIVATION) {
            bcnn_activation_param *param = (bcnn_activation_param *)node->param;
            if (param->activation == BCNN_ACT_PRELU) {
                bcnn_tensor *w = &net->tensors[net->nodes[i].src[1]];
                int w_sz = bcnn_tensor_size(w);
                fwrite(w->data, sizeof(float), w_sz, fp);
            }
        }
        if (node->type == BCNN_LAYER_BATCHNORM) {
            bcnn_tensor *m = &net->tensors[net->nodes[i].src[1]];
            bcnn_tensor *v = &net->tensors[net->nodes[i].src[2]];
            bcnn_tensor *s = &net->tensors[net->nodes[i].src[3]];
            bcnn_tensor *b = &net->tensors[net->nodes[i].src[4]];
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(m->data_gpu, m->data,
                                      net->tensors[net->nodes[i].dst[0]].c);
            bcnn_cuda_memcpy_dev2host(v->data_gpu, v->data,
                                      net->tensors[net->nodes[i].dst[0]].c);
            bcnn_cuda_memcpy_dev2host(s->data_gpu, s->data,
                                      net->tensors[net->nodes[i].dst[0]].c);
            bcnn_cuda_memcpy_dev2host(b->data_gpu, b->data,
                                      net->tensors[net->nodes[i].dst[0]].c);
#endif
            fwrite(m->data, sizeof(float), net->tensors[net->nodes[i].dst[0]].c,
                   fp);
            fwrite(v->data, sizeof(float), net->tensors[net->nodes[i].dst[0]].c,
                   fp);
            fwrite(s->data, sizeof(float), net->tensors[net->nodes[i].dst[0]].c,
                   fp);
            fwrite(b->data, sizeof(float), net->tensors[net->nodes[i].dst[0]].c,
                   fp);
        }
    }
    fclose(fp);
    return BCNN_SUCCESS;
}

typedef struct {
    int stride;
    int pad;
    int n_filts;
    int size;
    int outputs;
    int num_groups;
    int batchnorm;
    int in_w;
    int in_h;
    int in_c;
    int num_anchors;
    int boxes_per_cell;
    int num_classes;
    int num_coords;
    int keep_routed;
    int num_srcs;
    float alpha;
    float beta;
    float k;
    float rate;
    bcnn_padding padding_type;
    bcnn_activation a;
    bcnn_filler_type init;
    bcnn_loss_metric cost;
    bcnn_loss loss;
    bcnn_mode mode;
    char **src_id;
    char *dst_id;
    int *anchors_mask;
    float *anchors;
} bcnn_layer_param;

static void bcnn_layer_param_reset(bcnn_layer_param *lp) {
    lp->stride = 1;
    lp->pad = 0;
    lp->n_filts = 1;
    lp->size = 3;
    lp->outputs = 0;
    lp->num_groups = 1;
    lp->batchnorm = 0;
    lp->in_w = 0;
    lp->in_h = 0;
    lp->in_c = 0;
    lp->num_anchors = 0;
    lp->boxes_per_cell = 0;
    lp->num_classes = 0;
    lp->num_coords = 4;
    lp->alpha = 0.f;
    lp->beta = 0.f;
    lp->k = 0.f;
    lp->rate = 1.0f;
    lp->padding_type = BCNN_PADDING_SAME;
    lp->a = BCNN_ACT_NONE;
    lp->init = BCNN_FILLER_XAVIER;
    lp->cost = BCNN_METRIC_SSE;
    lp->loss = BCNN_LOSS_EUCLIDEAN;
    lp->mode = BCNN_MODE_PREDICT;
    if (lp->keep_routed == 0 && lp->src_id) {
        bh_free(lp->src_id[0]);
    }
    lp->keep_routed = 0;
    for (int i = 1; i < lp->num_srcs; ++i) {
        bh_free(lp->src_id[i]);
    }
    if (lp->keep_routed == 0) {
        bh_free(lp->src_id);
    }
    bh_free(lp->dst_id);
    bh_free(lp->anchors_mask);
    bh_free(lp->anchors);
    lp->num_srcs = 0;
}

static bcnn_status bcnn_layer_param_set(bcnn_net *net, int section_idx,
                                        bcnn_layer_param *lp, const char *name,
                                        const char *val, int format) {
    if (strcmp(name, "dropout_rate") == 0 || strcmp(name, "rate") == 0)
        lp->rate = (float)atof(val);
    else if (strcmp(name, "filters") == 0)
        lp->n_filts = atoi(val);
    else if (strcmp(name, "size") == 0)
        lp->size = atoi(val);
    else if (strcmp(name, "stride") == 0)
        lp->stride = atoi(val);
    else if (strcmp(name, "padding") == 0) {
        if (format == 1) {
            lp->pad = atoi(val);
            if (lp->pad) {
                lp->padding_type = BCNN_PADDING_SAME;
            } else {
                lp->padding_type = BCNN_PADDING_VALID;
            }
        }
    } else if (strcmp(name, "pad") == 0) {
        if (format == 0) {  // BCNN format
            lp->pad = atoi(val);
        } else {  // Darknet format
            int pad = atoi(val);
            if (pad) {
                // TODO: Needs to ensure that kernel size has been defined
                // before padding in that case
                lp->pad = lp->size / 2;
            } else {
                lp->pad = 0;
            }
        }
    } else if (strcmp(name, "num_groups") == 0 || strcmp(name, "groups") == 0)
        lp->num_groups = atoi(val);
    else if (strcmp(name, "boxes_per_cell") == 0) {
        lp->boxes_per_cell = atoi(val);
    } else if (strcmp(name, "num_anchors") == 0 || strcmp(name, "num") == 0) {
        lp->num_anchors = atoi(val);
    } else if (strcmp(name, "num_classes") == 0 ||
               strcmp(name, "classes") == 0) {
        lp->num_classes = atoi(val);
    } else if (strcmp(name, "num_coords") == 0) {
        lp->num_coords = atoi(val);
    } else if (strcmp(name, "anchors") == 0) {
        char **str_anchors = NULL;
        int sz = bh_strsplit((char *)val, ',', &str_anchors);
        lp->anchors = (float *)calloc(sz, sizeof(float));
        for (int i = 0; i < sz; ++i) {
            lp->anchors[i] = atof(str_anchors[i]);
        }
        for (int i = 0; i < sz; ++i) {
            bh_free(str_anchors[i]);
        }
        bh_free(str_anchors);
    } else if (strcmp(name, "anchors_mask") == 0 || strcmp(name, "mask") == 0) {
        char **str_anchors_mask = NULL;
        lp->boxes_per_cell = bh_strsplit((char *)val, ',', &str_anchors_mask);
        lp->anchors_mask = (int *)calloc(lp->boxes_per_cell, sizeof(int));
        for (int i = 0; i < lp->boxes_per_cell; ++i) {
            lp->anchors_mask[i] = atoi(str_anchors_mask[i]);
        }
        for (int i = 0; i < lp->boxes_per_cell; ++i) {
            bh_free(str_anchors_mask[i]);
        }
        bh_free(str_anchors_mask);
    } else if (strcmp(name, "alpha") == 0)
        lp->alpha = atoi(val);
    else if (strcmp(name, "beta") == 0)
        lp->beta = atoi(val);
    else if (strcmp(name, "k") == 0)
        lp->k = atoi(val);
    else if (strcmp(name, "w") == 0) {
        lp->in_w = atoi(val);
    } else if (strcmp(name, "h") == 0) {
        lp->in_h = atoi(val);
    } else if (strcmp(name, "c") == 0) {
        lp->in_c = atoi(val);
    } else if (strcmp(name, "bn") == 0 || strcmp(name, "batchnorm") == 0 ||
               strcmp(name, "batch_normalize") == 0) {
        lp->batchnorm = atoi(val);
    } else if (strcmp(name, "src") == 0) {
        char **srcids = NULL;
        int num_srcids = bh_strsplit((char *)val, ',', &srcids);
        lp->num_srcs = num_srcids;
        lp->src_id = (char **)calloc(num_srcids, sizeof(char *));
        for (int i = 0; i < lp->num_srcs; ++i) {
            bh_strfill(&lp->src_id[i], srcids[i]);
        }
        for (int i = 0; i < num_srcids; ++i) {
            bh_free(srcids[i]);
        }
        bh_free(srcids);
    } else if (strcmp(name, "dst") == 0)
        bh_strfill(&lp->dst_id, val);
    else if (strcmp(name, "output") == 0)
        lp->outputs = atoi(val);
    else if (strcmp(name, "padding_type") == 0) {
        if (strcmp(val, "same") == 0)
            lp->padding_type = BCNN_PADDING_SAME;
        else if (strcmp(val, "valid") == 0)
            lp->padding_type = BCNN_PADDING_VALID;
        else if (strcmp(val, "caffe") == 0)
            lp->padding_type = BCNN_PADDING_CAFFE;
    } else if (strcmp(name, "function") == 0 ||
               strcmp(name, "activation") == 0) {
        if (strcmp(val, "relu") == 0)
            lp->a = BCNN_ACT_RELU;
        else if (strcmp(val, "tanh") == 0)
            lp->a = BCNN_ACT_TANH;
        else if (strcmp(val, "ramp") == 0)
            lp->a = BCNN_ACT_RAMP;
        else if (strcmp(val, "clamp") == 0)
            lp->a = BCNN_ACT_CLAMP;
        else if (strcmp(val, "softplus") == 0)
            lp->a = BCNN_ACT_SOFTPLUS;
        else if (strcmp(val, "leaky_relu") == 0 || strcmp(val, "lrelu") == 0 ||
                 strcmp(val, "leaky") == 0)
            lp->a = BCNN_ACT_LRELU;
        else if (strcmp(val, "prelu") == 0)
            lp->a = BCNN_ACT_PRELU;
        else if (strcmp(val, "abs") == 0)
            lp->a = BCNN_ACT_ABS;
        else if (strcmp(val, "none") == 0 || strcmp(val, "linear") == 0)
            lp->a = BCNN_ACT_NONE;
        else {
            BCNN_WARNING(net->log_ctx,
                         "Unknown activation type %s, going with ReLU\n", val);
            lp->a = BCNN_ACT_RELU;
        }
    } else if (strcmp(name, "init") == 0) {
        if (strcmp(val, "xavier") == 0)
            lp->init = BCNN_FILLER_XAVIER;
        else if (strcmp(val, "msra") == 0)
            lp->init = BCNN_FILLER_MSRA;
        else {
            BCNN_WARNING(net->log_ctx,
                         "Unknown init type %s, going with xavier init\n", val);
            lp->init = BCNN_FILLER_XAVIER;
        }
    } else if (strcmp(name, "metric") == 0) {
        if (strcmp(val, "error") == 0)
            lp->cost = BCNN_METRIC_ERROR_RATE;
        else if (strcmp(val, "logloss") == 0)
            lp->cost = BCNN_METRIC_LOGLOSS;
        else if (strcmp(val, "sse") == 0)
            lp->cost = BCNN_METRIC_SSE;
        else if (strcmp(val, "mse") == 0)
            lp->cost = BCNN_METRIC_MSE;
        else if (strcmp(val, "crps") == 0)
            lp->cost = BCNN_METRIC_CRPS;
        else if (strcmp(val, "dice") == 0)
            lp->cost = BCNN_METRIC_DICE;
        else {
            BCNN_WARNING(net->log_ctx,
                         "Unknown cost metric %s, going with sse\n", val);
            lp->cost = BCNN_METRIC_SSE;
        }
    } else if (strcmp(name, "loss") == 0) {
        if (strcmp(val, "l2") == 0 || strcmp(val, "euclidean") == 0) {
            lp->loss = BCNN_LOSS_EUCLIDEAN;
        } else if (strcmp(val, "lifted_struct_similarity") == 0) {
            lp->loss = BCNN_LOSS_LIFTED_STRUCT;
        } else {
            BCNN_WARNING(net->log_ctx,
                         "Unknown loss %s, going with euclidean loss\n", val);
            lp->loss = BCNN_LOSS_EUCLIDEAN;
        }
    } else if (strcmp(name, "layers") == 0) {  // Darknet format
        char **str_layers = NULL;
        int sz = bh_strsplit((char *)val, ',', &str_layers);
        if (sz > 0) {
            lp->num_srcs = sz;
            lp->src_id = (char **)calloc(lp->num_srcs, sizeof(char *));
            lp->keep_routed = 1;
            int l1 = atoi(str_layers[0]);
            char lid[32];
            for (int i = 0; i < lp->num_srcs; ++i) {
                int l = atoi(str_layers[i]);
                char lid[32];
                if (l >= 0) {
                    snprintf(lid, sizeof(lid), "lid%d", l + 1);
                    bh_strfill(&lp->src_id[i], lid);
                } else {
                    snprintf(lid, sizeof(lid), "lid%d", section_idx + l);
                    bh_strfill(&lp->src_id[i], lid);
                }
            }
            if (sz > 1) {
                lp->keep_routed = 0;
            }
            for (int i = 0; i < sz; ++i) {
                bh_free(str_layers[i]);
            }
            bh_free(str_layers);
        }
    } else if (strcmp(name, "from") == 0) {  // Darknet format
        lp->num_srcs = 2;
        lp->src_id = (char **)calloc(lp->num_srcs, sizeof(char *));
        char lid[32];
        snprintf(lid, sizeof(lid), "lid%d", section_idx - 1);
        bh_strfill(&lp->src_id[0], lid);
        int l = atoi(val);
        if (l >= 0) {
            snprintf(lid, sizeof(lid), "lid%d", l + 1);
            bh_strfill(&lp->src_id[1], lid);
        } else {
            snprintf(lid, sizeof(lid), "lid%d", section_idx + l);
            bh_strfill(&lp->src_id[1], lid);
        }
    }
    return BCNN_SUCCESS;
}

static bcnn_status bcnn_add_layer(bcnn_net *net, const char *name,
                                  const bcnn_layer_param *lp) {
    // If first layer to be added, check that input dimensions are valid
    if (net->num_nodes == 2) {
        BCNN_CHECK_AND_LOG(net->log_ctx,
                           net->tensors[0].w > 0 && net->tensors[0].h > 0 &&
                               net->tensors[0].c > 0,
                           BCNN_INVALID_PARAMETER,
                           "Input's width, height and "
                           "channels must be > 0\n");
        BCNN_CHECK_AND_LOG(net->log_ctx, net->tensors[0].n > 0,
                           BCNN_INVALID_PARAMETER, "Batch size must be > 0\n");
    }
    BCNN_CHECK_AND_LOG(net->log_ctx, lp->src_id && lp->src_id[0],
                       BCNN_INVALID_PARAMETER,
                       "Invalid input node name. "
                       "Hint: Are you sure that 'src' field is correctly "
                       "setup?\n");
    if (strcmp(name, "[input]") == 0) {
        bcnn_add_input(net, lp->in_w, lp->in_h, lp->in_c, lp->src_id[0]);
    } else if (strcmp(name, "[conv]") == 0 ||
               strcmp(name, "[convolutional]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_convolutional_layer(
            net, lp->n_filts, lp->size, lp->stride, lp->pad, lp->num_groups,
            lp->batchnorm, lp->init, lp->a, 0, lp->src_id[0], lp->dst_id);
    } else if (strcmp(name, "[deconv]") == 0 ||
               strcmp(name, "[deconvolutional]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_deconvolutional_layer(net, lp->n_filts, lp->size, lp->stride,
                                       lp->pad, lp->init, lp->a, lp->src_id[0],
                                       lp->dst_id);
    } else if (strcmp(name, "[depthwise-conv]") == 0 ||
               strcmp(name, "[dw-conv]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_depthwise_conv_layer(net, lp->size, lp->stride, lp->pad, 0,
                                      lp->init, lp->a, lp->src_id[0],
                                      lp->dst_id);
    } else if (strcmp(name, "[activation]") == 0 || strcmp(name, "[nl]") == 0) {
        bcnn_add_activation_layer(net, lp->a, lp->src_id[0]);
    } else if (strcmp(name, "[batchnorm]") == 0 || strcmp(name, "[bn]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_batchnorm_layer(net, lp->src_id[0], lp->dst_id);
    } else if (strcmp(name, "[lrn]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_lrn_layer(net, lp->size, lp->alpha, lp->beta, lp->k,
                           lp->src_id[0], lp->dst_id);
    } else if (strcmp(name, "[connected]") == 0 ||
               strcmp(name, "[fullconnected]") == 0 ||
               strcmp(name, "[fc]") == 0 || strcmp(name, "[ip]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_fullc_layer(net, lp->outputs, lp->init, lp->a, 0,
                             lp->src_id[0], lp->dst_id);
    } else if (strcmp(name, "[softmax]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_softmax_layer(net, lp->src_id[0], lp->dst_id);
    } else if (strcmp(name, "[max]") == 0 || strcmp(name, "[maxpool]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_maxpool_layer(net, lp->size, lp->stride, lp->padding_type,
                               lp->src_id[0], lp->dst_id);
    } else if (strcmp(name, "[avgpool]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_avgpool_layer(net, lp->src_id[0], lp->dst_id);
    } else if (strcmp(name, "[upsample]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_upsample_layer(net, lp->stride, lp->src_id[0], lp->dst_id);
    } else if (strcmp(name, "[dropout]") == 0) {
        bcnn_add_dropout_layer(net, lp->rate, lp->src_id[0]);
    } else if (strcmp(name, "[concat]") == 0 || strcmp(name, "[route]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        if (lp->src_id != NULL) {
            bcnn_add_concat_layer(net, lp->num_srcs, lp->src_id, lp->dst_id);
        }
    } else if (strcmp(name, "[eltwise]") == 0 ||
               strcmp(name, "[shortcut]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        if (lp->src_id != NULL) {
            bcnn_add_eltwise_layer(net, lp->a, lp->src_id[0], lp->src_id[1],
                                   lp->dst_id);
        }
    } else if (strcmp(name, "[yolo]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Cost layer: invalid input node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly "
                           "setup?\n");
        bcnn_add_yolo_layer(net, lp->boxes_per_cell, lp->num_classes,
                            lp->num_coords, lp->num_anchors, lp->anchors_mask,
                            lp->anchors, lp->src_id[0], lp->dst_id);
    } else if (strcmp(name, "[cost]") == 0) {
        BCNN_CHECK_AND_LOG(
            net->log_ctx, lp->src_id[0], BCNN_INVALID_PARAMETER,
            "Cost layer: invalid input node name. "
            "Hint: Are you sure that 'src' field is correctly setup?\n");
        BCNN_CHECK_AND_LOG(
            net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
            "Cost layer: invalid input node name. "
            "Hint: Are you sure that 'dst' field is correctly setup?\n");
        bcnn_add_cost_layer(net, lp->loss, lp->cost, 1.0f, lp->src_id[0],
                            "label", lp->dst_id);
    } else {
        BCNN_ERROR(net->log_ctx, BCNN_INVALID_PARAMETER, "Unknown Layer %s\n",
                   name);
    }
    return BCNN_SUCCESS;
}

bcnn_status bcnn_load_net(bcnn_net *net, const char *config_path,
                          const char *model_path) {
    int format = 0;  // default is BCNN
    if (model_path != NULL) {
        // Try to infer the input model format according to
        // the model file extension
        char **tok = NULL;
        int num_toks = bh_strsplit((char *)model_path, '.', &tok);
        BCNN_CHECK_AND_LOG(net->log_ctx, num_toks >= 2, BCNN_INVALID_DATA,
                           "File %s needs to have an extension (.bcnnmodel OR "
                           ".onnx OR .weights)\n",
                           model_path);
        if (strcmp(tok[num_toks - 1], "weights") == 0) {
            format = 1;  // Darknet model
        } else if (strcmp(tok[num_toks - 1], "onnx") == 0) {
            format = 2;  // ONNX model
        }
        for (int i = 0; i < num_toks; ++i) {
            bh_free(tok[i]);
        }
        bh_free(tok);
    }
    if (format == 0 || format == 1) {
        bh_ini_parser *config = bh_ini_parser_create(config_path);
        if (config == NULL) {
            return BCNN_INVALID_PARAMETER;
        }
        if (config->num_sections == 0 || config->sections == NULL) {
            bcnn_log(net->log_ctx, BCNN_LOG_ERROR, "Empty config file %s\n",
                     config_path);
            bh_ini_parser_destroy(config);
            return BCNN_INVALID_PARAMETER;
        }
        if (strcmp(config->sections[0].name, "[net]") != 0 &&
            strcmp(config->sections[0].name, "[network]") != 0) {
            bcnn_log(net->log_ctx, BCNN_LOG_ERROR,
                     "Invalid config file %s: First section must be [net] or "
                     "[network]\n",
                     config_path);
            bh_ini_parser_destroy(config);
            return BCNN_INVALID_PARAMETER;
        }
        if (config->sections[0].keys == NULL ||
            config->sections[0].num_keys == 0) {
            bcnn_log(net->log_ctx, BCNN_LOG_ERROR,
                     "Invalid config file %s: empty section [net]\n",
                     config_path);
            bh_ini_parser_destroy(config);
            return BCNN_INVALID_PARAMETER;
        }
        // Parse network parameters
        for (int i = 0; i < config->sections[0].num_keys; ++i) {
            /*fprintf(stderr, "%s %s\n", config->sections[0].keys[i].name,
                    config->sections[0].keys[i].val);*/
            bcnn_net_set_param(net, config->sections[0].keys[i].name,
                               config->sections[0].keys[i].val);
        }
        // Parse layers
        bcnn_layer_param lp = {0};
        bcnn_layer_param_reset(&lp);
        for (int i = 1; i < config->num_sections; ++i) {
            // Parse layers parameters
            for (int j = 0; j < config->sections[i].num_keys; ++j) {
                /*fprintf(stderr, "%s %s\n",
                   config->sections[i].keys[j].name,
                        config->sections[i].keys[j].val);*/
                if (bcnn_layer_param_set(net, i, &lp,
                                         config->sections[i].keys[j].name,
                                         config->sections[i].keys[j].val,
                                         format) != BCNN_SUCCESS) {
                    bh_ini_parser_destroy(config);
                    return BCNN_INVALID_PARAMETER;
                }
            }
            if (format == 1) {
                if (lp.src_id == NULL) {
                    lp.num_srcs = 1;
                    lp.src_id = (char **)calloc(lp.num_srcs, sizeof(char *));
                    char lid[32];
                    snprintf(lid, sizeof(lid), "lid%d", i - 1);
                    bh_strfill(&lp.src_id[0], lid);
                }
                if (lp.dst_id == NULL) {
                    char lid[32];
                    snprintf(lid, sizeof(lid), "lid%d", i);
                    bh_strfill(&lp.dst_id, lid);
                }
            }
            // Add layer
            if (bcnn_add_layer(net, config->sections[i].name, &lp) !=
                BCNN_SUCCESS) {
                bh_ini_parser_destroy(config);
                return BCNN_INVALID_PARAMETER;
            }
            bcnn_layer_param_reset(&lp);
        }
        bh_ini_parser_destroy(config);
    }
    // Load weights
    if (model_path != NULL) {
        BCNN_INFO(net->log_ctx, "Loading pre-trained model %s\n", model_path);
        BCNN_CHECK_STATUS(bcnn_load_weights(net, model_path));
    }
    return BCNN_SUCCESS;
}

static bcnn_status bcnn_load_conv_weights(bcnn_net *net, bcnn_node *node,
                                          FILE *fp, int format) {
    bcnn_tensor *w = &net->tensors[node->src[1]];  // weights
    bcnn_tensor *b = &net->tensors[node->src[2]];  // biases
    int w_sz = bcnn_tensor_size(w);
    int b_sz = bcnn_tensor_size(b);
    int nr = 0;
    BCNN_CHECK_AND_LOG(
        net->log_ctx, (nr = fread(b->data, sizeof(float), b_sz, fp)) == b_sz,
        BCNN_INVALID_MODEL,
        "Inconsistent biases size %s: expected %d but found %lu\n", b->name,
        b_sz, (unsigned long)nr);
    if (format == 0) {
        BCNN_CHECK_AND_LOG(
            net->log_ctx,
            (nr = fread(w->data, sizeof(float), w_sz, fp)) == w_sz,
            BCNN_INVALID_MODEL,
            "Inconsistent weights size %s: expected %d but found %lu\n",
            w->name, w_sz, (unsigned long)nr);
    }
    if (node->type == BCNN_LAYER_CONV2D) {
        bcnn_conv_param *param = (bcnn_conv_param *)node->param;
        if (param->batch_norm == 1) {
            bcnn_tensor *m = &net->tensors[node->src[3]];  // means
            bcnn_tensor *v = &net->tensors[node->src[4]];  // variances
            bcnn_tensor *s = &net->tensors[node->src[5]];  // scales
            int m_sz = bcnn_tensor_size(m);
            int v_sz = bcnn_tensor_size(v);
            int s_sz = bcnn_tensor_size(s);
            if (format == 1) {
                BCNN_CHECK_AND_LOG(
                    net->log_ctx,
                    (nr = fread(s->data, sizeof(float), s_sz, fp)) == s_sz,
                    BCNN_INVALID_MODEL,
                    "Inconsistent batchnorm scales size: "
                    "expected %d but found %lu\n",
                    s_sz, (unsigned long)nr);
            }
            BCNN_CHECK_AND_LOG(
                net->log_ctx,
                (nr = fread(m->data, sizeof(float), m_sz, fp)) == m_sz,
                BCNN_INVALID_MODEL,
                "Inconsistent batchnorm means size: expected %d but found "
                "%lu\n",
                m_sz, (unsigned long)nr);
            BCNN_CHECK_AND_LOG(net->log_ctx, (nr = fread(v->data, sizeof(float),
                                                         v_sz, fp)) == v_sz,
                               BCNN_INVALID_MODEL,
                               "Inconsistent batchnorm variances size: "
                               "expected %d but found %lu\n",
                               v_sz, (unsigned long)nr);
            if (format == 0) {
                BCNN_CHECK_AND_LOG(
                    net->log_ctx,
                    (nr = fread(s->data, sizeof(float), s_sz, fp)) == s_sz,
                    BCNN_INVALID_MODEL,
                    "Inconsistent batchnorm scales size: "
                    "expected %d but found %lu\n",
                    s_sz, (unsigned long)nr);
            }
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_host2dev(m->data_gpu, m->data, m_sz);
            bcnn_cuda_memcpy_host2dev(v->data_gpu, v->data, v_sz);
            bcnn_cuda_memcpy_host2dev(s->data_gpu, s->data, s_sz);
#endif
        }
    }
    if (format == 1) {
        BCNN_CHECK_AND_LOG(
            net->log_ctx,
            (nr = fread(w->data, sizeof(float), w_sz, fp)) == w_sz,
            BCNN_INVALID_MODEL,
            "Inconsistent weights size %s: expected %d but found %lu\n",
            w->name, w_sz, (unsigned long)nr);
    }
#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_host2dev(w->data_gpu, w->data, w_sz);
    bcnn_cuda_memcpy_host2dev(b->data_gpu, b->data, b_sz);
#endif
    return BCNN_SUCCESS;
}

static bcnn_status bcnn_load_batchnorm_weights(bcnn_net *net, bcnn_node *node,
                                               FILE *fp, int format) {
    bcnn_tensor *m = &net->tensors[node->src[1]];  // means
    bcnn_tensor *v = &net->tensors[node->src[2]];  // variances
    bcnn_tensor *s = &net->tensors[node->src[3]];  // scales
    bcnn_tensor *b = &net->tensors[node->src[4]];  // biases
    int sz = net->tensors[node->dst[0]].c;
    int nr = 0;
    if (format == 1) {
        BCNN_CHECK_AND_LOG(
            net->log_ctx, (nr = fread(s->data, sizeof(float), sz, fp)) == sz,
            BCNN_INVALID_MODEL,
            "Inconsistent scales size: expected %d but found %lu\n", sz,
            (unsigned long)nr);
    }
    BCNN_CHECK_AND_LOG(net->log_ctx,
                       (nr = fread(m->data, sizeof(float), sz, fp)) == sz,
                       BCNN_INVALID_MODEL,
                       "Inconsistent means size: expected %d but found %lu\n",
                       sz, (unsigned long)nr);
    BCNN_CHECK_AND_LOG(
        net->log_ctx, (nr = fread(v->data, sizeof(float), sz, fp)) == sz,
        BCNN_INVALID_MODEL,
        "Inconsistent variances size: expected %d but found %lu\n", sz,
        (unsigned long)nr);
    if (format == 0) {
        BCNN_CHECK_AND_LOG(
            net->log_ctx, (nr = fread(s->data, sizeof(float), sz, fp)) == sz,
            BCNN_INVALID_MODEL,
            "Inconsistent scales size: expected %d but found %lu\n", sz,
            (unsigned long)nr);
        BCNN_CHECK_AND_LOG(
            net->log_ctx, (nr = fread(b->data, sizeof(float), sz, fp)) == sz,
            BCNN_INVALID_MODEL,
            "Inconsistent biases size: expected %d but found %lu\n", sz,
            (unsigned long)nr);
    }
#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_host2dev(m->data_gpu, m->data, sz);
    bcnn_cuda_memcpy_host2dev(v->data_gpu, v->data, sz);
    bcnn_cuda_memcpy_host2dev(s->data_gpu, s->data, sz);
    bcnn_cuda_memcpy_host2dev(b->data_gpu, b->data, sz);
#endif
    return BCNN_SUCCESS;
}

static bcnn_status bcnn_load_prelu_weights(bcnn_net *net, bcnn_node *node,
                                           FILE *fp, int format) {
    bcnn_tensor *w = &net->tensors[node->src[1]];
    int w_sz = bcnn_tensor_size(w);
    int nr = 0;
    BCNN_CHECK_AND_LOG(
        net->log_ctx, (nr = fread(w->data, sizeof(float), w_sz, fp)) == w_sz,
        BCNN_INVALID_MODEL,
        "Inconsistent prelu weights size: expected %d but found %lu\n", w_sz,
        (unsigned long)nr);
    return BCNN_SUCCESS;
}

static void bcnn_transpose(float *a, int rows, int cols) {
    float *transpose = (float *)calloc(rows * cols, sizeof(float));
    int x, y;
    for (x = 0; x < rows; ++x) {
        for (y = 0; y < cols; ++y) {
            transpose[y * rows + x] = a[x * cols + y];
        }
    }
    memcpy(a, transpose, rows * cols * sizeof(float));
    free(transpose);
}

static bcnn_status bcnn_load_fullc_weights(bcnn_net *net, bcnn_node *node,
                                           FILE *fp, int format,
                                           int need_transpose) {
    bcnn_tensor *w = &net->tensors[node->src[1]];  // weights
    bcnn_tensor *b = &net->tensors[node->src[2]];  // biases
    int w_sz = bcnn_tensor_size(w);
    int b_sz = bcnn_tensor_size(b);
    int nr = 0;
    BCNN_CHECK_AND_LOG(
        net->log_ctx, (nr = fread(b->data, sizeof(float), b_sz, fp)) == b_sz,
        BCNN_INVALID_MODEL,
        "Inconsistent biases size %s: expected %d but found %lu\n", b->name,
        b_sz, (unsigned long)nr);
    BCNN_CHECK_AND_LOG(
        net->log_ctx, (nr = fread(w->data, sizeof(float), w_sz, fp)) == w_sz,
        BCNN_INVALID_MODEL,
        "Inconsistent weights size %s: expected %d but found %lu\n", w->name,
        w_sz, (unsigned long)nr);
    if (need_transpose) {
        bcnn_transpose(w->data, bcnn_tensor_size3d(&net->tensors[node->src[0]]),
                       bcnn_tensor_size3d(&net->tensors[node->dst[0]]));
    }
#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_host2dev(w->data_gpu, w->data, w_sz);
    bcnn_cuda_memcpy_host2dev(b->data_gpu, b->data, b_sz);
#endif
    return BCNN_SUCCESS;
}

static int bcnn_model_find_format(const char *filename) {
    int format = 0;  // default is BCNN
    // Parse filename extension
    char **toks = NULL;
    int ntoks = bh_strsplit((char *)filename, '.', &toks);
    if (strcmp(toks[ntoks - 1], "weights") == 0) {
        format = 1;  // Darknet
    } else if (strcmp(toks[ntoks - 1], "onnx") == 0) {
        format = 2;  // onnx
    }
    for (int i = 0; i < ntoks; ++i) {
        bh_free(toks[i]);
    }
    bh_free(toks);
    return format;
}

bcnn_status bcnn_load_weights(bcnn_net *net, const char *filename) {
    int format = bcnn_model_find_format(filename);
    FILE *fp = fopen(filename, "rb");
    BCNN_CHECK_AND_LOG(net->log_ctx, fp, BCNN_INVALID_PARAMETER,
                       "Can not open file %s\n", filename);
    int need_transpose = 0;
    if (format == 0) {  // bcnn
        char magic[4];
        uint32_t major, minor, patch;
        size_t nr = fread(magic, 1, 4, fp);
        nr = fread(&major, sizeof(uint32_t), 1, fp);
        nr = fread(&minor, sizeof(uint32_t), 1, fp);
        nr = fread(&patch, sizeof(uint32_t), 1, fp);
        if (strncmp(magic, BCNN_MAGIC, 4) != 0) {
            bcnn_log(net->log_ctx, BCNN_LOG_ERROR,
                     "Invalid format for model file %s\n", filename);
            fclose(fp);
            return BCNN_INVALID_MODEL;
        }
        BCNN_INFO(net->log_ctx, "BCNN version %d.%d.%d used for model %s\n",
                  major, minor, patch, filename);
    } else if (format == 1) {  // Darknet
        int major;
        int minor;
        int revision;
        size_t nr = fread(&major, sizeof(int), 1, fp);
        nr = fread(&minor, sizeof(int), 1, fp);
        nr = fread(&revision, sizeof(int), 1, fp);
        uint64_t num_samples_seen = 0;
        if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000) {
            size_t lseen = 0;
            nr = fread(&lseen, sizeof(uint64_t), 1, fp);
            num_samples_seen = (uint64_t)lseen;
        } else {
            int iseen = 0;
            nr = fread(&iseen, sizeof(int), 1, fp);
            num_samples_seen = (uint64_t)iseen;
        }
        BCNN_INFO(net->log_ctx, "Darknet version %d.%d seen %ld\n", major,
                  minor, num_samples_seen);

        need_transpose = (major > 1000) || (minor > 1000);
    } else {
        bcnn_log(net->log_ctx, BCNN_LOG_ERROR,
                 "Model file %s format is not yet supported\n", filename);
        fclose(fp);
        return BCNN_INVALID_MODEL;
    }

    for (int i = 0; i < net->num_nodes; ++i) {
        bcnn_node *node = &net->nodes[i];
        if (node->type == BCNN_LAYER_CONV2D ||
            node->type == BCNN_LAYER_TRANSPOSE_CONV2D ||
            node->type == BCNN_LAYER_DEPTHWISE_CONV2D) {
            bcnn_load_conv_weights(net, node, fp, format);
        } else if (node->type == BCNN_LAYER_ACTIVATION) {
            bcnn_activation_param *param = (bcnn_activation_param *)node->param;
            if (param->activation == BCNN_ACT_PRELU && format == 0) {
                bcnn_load_prelu_weights(net, node, fp, 0);
            }
        } else if (node->type == BCNN_LAYER_BATCHNORM) {
            bcnn_load_batchnorm_weights(net, node, fp, format);
        } else if (node->type == BCNN_LAYER_FULL_CONNECTED) {
            bcnn_load_fullc_weights(net, node, fp, format, need_transpose);
        }
    }
    if (fp != NULL) {
        fclose(fp);
    }

    BCNN_INFO(net->log_ctx, "Model %s loaded succesfully\n", filename);

    return BCNN_SUCCESS;
}