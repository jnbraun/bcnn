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

#include <bh/bh_log.h>
#include <bh/bh_macros.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>
#include <bh/bh_timer.h>

#include <bip/bip.h>

#include "bcnn/bcnn.h"
#include "bcnn_cl.h"
#include "bcnn_tensor.h"
#include "bcnn_utils.h"
#include "bcnn_yolo.h"

typedef struct {
    char *name;
    char *val;
} bcnn_parser_key;

typedef struct {
    int num_keys;
    char *name;
    bcnn_parser_key *keys;
} bcnn_parser_section;

typedef struct {
    int num_sections;
    bcnn_parser_section *sections;
} bcnn_parser_ini;

int bcnn_parser_add_key(bcnn_parser_section *section, const char *line) {
    if (section == NULL) {
        fprintf(stderr, "[ERROR] No valid section for key %s\n", line);
    }
    char **tok = NULL;
    int num_toks = bh_strsplit((char *)line, '=', &tok);
    if (num_toks != 2) {
        fprintf(stderr, "[ERROR] Invalid key section %s\n", line);
        return -1;
    }
    bcnn_parser_key key = {0};
    bh_strfill(&key.name, tok[0]);
    bh_strfill(&key.val, tok[1]);
    for (int i = 0; i < num_toks; ++i) {
        bh_free(tok[i]);
    }
    bh_free(tok);
    section->num_keys++;
    bcnn_parser_key *p_keys = (bcnn_parser_key *)realloc(
        section->keys, section->num_keys * sizeof(bcnn_parser_key));
    if (p_keys == NULL) {
        fprintf(stderr, "[ERROR] Failed allocation\n");
        return -1;
    }
    section->keys = p_keys;
    section->keys[section->num_keys - 1] = key;
    return 0;
}

int bcnn_parser_add_section(bcnn_parser_ini *config, const char *line) {
    bcnn_parser_section section = {0};
    config->num_sections++;
    bcnn_parser_section *p_sections = (bcnn_parser_section *)realloc(
        config->sections, config->num_sections * sizeof(bcnn_parser_section));
    if (p_sections == NULL) {
        fprintf(stderr, "[ERROR] Failed allocation\n");
        return -1;
    }
    bh_strfill(&section.name, line);
    config->sections = p_sections;
    config->sections[config->num_sections - 1] = section;
    return 0;
}

void bcnn_parser_free(bcnn_parser_ini *config) {
    for (int i = 0; i < config->num_sections; ++i) {
        for (int j = 0; j < config->sections[i].num_keys; ++j) {
            bh_free(config->sections[i].keys[j].name);
            bh_free(config->sections[i].keys[j].val);
        }
        bh_free(config->sections[i].keys);
        bh_free(config->sections[i].name);
    }
    bh_free(config->sections);
    bh_free(config);
}

bcnn_parser_ini *bcnn_parser_read_ini(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "[ERROR] Could not open file: %s\n", filename);
        return NULL;
    }
    char *line = NULL;
    bcnn_parser_ini *config =
        (bcnn_parser_ini *)calloc(1, sizeof(bcnn_parser_ini));
    bcnn_parser_section *section = NULL;
    while ((line = bh_fgetline(file)) != NULL) {
        bh_strstrip(line);
        switch (line[0]) {
            case '[': {
                if (bcnn_parser_add_section(config, line) != 0) {
                    fprintf(stderr, "[ERROR] Failed to parse config file %s\n",
                            filename);
                    bh_free(line);
                    bcnn_parser_free(config);
                    return NULL;
                }
                section = &config->sections[config->num_sections - 1];
                break;
            }
            case '!':
            case '\0':
            case '#':
            case ';':
                break;
            default: {
                if (bcnn_parser_add_key(section, line) != 0) {
                    fprintf(stderr, "[ERROR] Failed to parse config file %s\n",
                            filename);
                    bh_free(line);
                    bcnn_parser_free(config);
                    return NULL;
                }
                break;
            }
        }
        bh_free(line);
    }
    fclose(file);
    return config;
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
    char *src_id;
    char *dst_id;
    char *src_id2;
    int *anchors_mask;
    float *anchors;
} layer_param;

static void reset_layer_param(layer_param *lp) {
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
    bh_free(lp->src_id);
    bh_free(lp->src_id2);
    bh_free(lp->dst_id);
    bh_free(lp->anchors_mask);
    bh_free(lp->anchors);
}

static void set_layer_param(bcnn_net *net, layer_param *lp, const char *name,
                            const char *val) {
    if (strcmp(name, "dropout_rate") == 0 || strcmp(name, "rate") == 0)
        lp->rate = (float)atof(val);
    else if (strcmp(name, "filters") == 0)
        lp->n_filts = atoi(val);
    else if (strcmp(name, "size") == 0)
        lp->size = atoi(val);
    else if (strcmp(name, "stride") == 0)
        lp->stride = atoi(val);
    else if (strcmp(name, "pad") == 0)
        lp->pad = atoi(val);
    else if (strcmp(name, "num_groups") == 0)
        lp->num_groups = atoi(val);
    else if (strcmp(name, "boxes_per_cell") == 0) {
        lp->boxes_per_cell = atoi(val);
    } else if (strcmp(name, "num_anchors") == 0) {
        lp->num_anchors = atoi(val);
    } else if (strcmp(name, "num_classes") == 0) {
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
    } else if (strcmp(name, "anchors_mask") == 0) {
        char **str_anchors_mask = NULL;
        int sz = bh_strsplit((char *)val, ',', &str_anchors_mask);
        lp->anchors_mask = (int *)calloc(sz, sizeof(int));
        for (int i = 0; i < sz; ++i) {
            lp->anchors_mask[i] = atoi(str_anchors_mask[i]);
        }
        for (int i = 0; i < sz; ++i) {
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
    } else if (strcmp(name, "bn") == 0 || strcmp(name, "batchnorm") == 0) {
        lp->batchnorm = atoi(val);
    } else if (strcmp(name, "src") == 0) {
        char **srcids = NULL;
        int num_srcids = bh_strsplit((char *)val, ',', &srcids);
        bh_strfill(&lp->src_id, srcids[0]);
        if (num_srcids > 1) {
            bh_strfill(&lp->src_id2, srcids[1]);
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
    } else if (strcmp(name, "function") == 0) {
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
        else if (strcmp(val, "leaky_relu") == 0 || strcmp(val, "lrelu") == 0)
            lp->a = BCNN_ACT_LRELU;
        else if (strcmp(val, "prelu") == 0)
            lp->a = BCNN_ACT_PRELU;
        else if (strcmp(val, "abs") == 0)
            lp->a = BCNN_ACT_ABS;
        else if (strcmp(val, "none") == 0)
            lp->a = BCNN_ACT_NONE;
        else {
            BCNN_WARNING(net->log_ctx,
                         "Unknown activation type %s, going with ReLU", val);
            lp->a = BCNN_ACT_RELU;
        }
    } else if (strcmp(name, "init") == 0) {
        if (strcmp(val, "xavier") == 0)
            lp->init = BCNN_FILLER_XAVIER;
        else if (strcmp(val, "msra") == 0)
            lp->init = BCNN_FILLER_MSRA;
        else {
            BCNN_WARNING(net->log_ctx,
                         "Unknown init type %s, going with xavier init", val);
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
            BCNN_WARNING(net->log_ctx, "Unknown cost metric %s, going with sse",
                         val);
            lp->cost = BCNN_METRIC_SSE;
        }
    } else if (strcmp(name, "loss") == 0) {
        if (strcmp(val, "l2") == 0 || strcmp(val, "euclidean") == 0) {
            lp->loss = BCNN_LOSS_EUCLIDEAN;
        } else if (strcmp(val, "lifted_struct_similarity") == 0) {
            lp->loss = BCNN_LOSS_LIFTED_STRUCT;
        } else {
            BCNN_WARNING(net->log_ctx,
                         "Unknown loss %s, going with euclidean loss", val);
            lp->loss = BCNN_LOSS_EUCLIDEAN;
        }
    }
}

static bcnn_status add_layer(bcnn_net *net, const char *name, int num,
                             const layer_param *lp) {
    if (num == 1) {
        BCNN_CHECK_AND_LOG(net->log_ctx,
                           net->tensors[0].w > 0 && net->tensors[0].h > 0 &&
                               net->tensors[0].c > 0,
                           BCNN_INVALID_PARAMETER,
                           "Input's width, height and "
                           "channels must be > 0\n");
        BCNN_CHECK_AND_LOG(net->log_ctx, net->tensors[0].n > 0,
                           BCNN_INVALID_PARAMETER, "Batch size must be > 0\n");
    }
    BCNN_CHECK_AND_LOG(net->log_ctx, lp->src_id, BCNN_INVALID_PARAMETER,
                       "Invalid input node name."
                       "Hint: Are you sure that 'src' field is correctly "
                       "setup?\n");
    if (strcmp(name, "[input]") == 0) {
        bcnn_add_input(net, lp->in_w, lp->in_h, lp->in_c, lp->src_id);
    } else if (strcmp(name, "[conv]") == 0 ||
               strcmp(name, "[convolutional]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_convolutional_layer(
            net, lp->n_filts, lp->size, lp->stride, lp->pad, lp->num_groups,
            lp->batchnorm, lp->init, lp->a, 0, lp->src_id, lp->dst_id);
    } else if (strcmp(name, "[deconv]") == 0 ||
               strcmp(name, "[deconvolutional]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_deconvolutional_layer(net, lp->n_filts, lp->size, lp->stride,
                                       lp->pad, lp->init, lp->a, lp->src_id,
                                       lp->dst_id);
    } else if (strcmp(name, "[depthwise-conv]") == 0 ||
               strcmp(name, "[dw-conv]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_depthwise_conv_layer(net, lp->size, lp->stride, lp->pad, 0,
                                      lp->init, lp->a, lp->src_id, lp->dst_id);
    } else if (strcmp(name, "[activation]") == 0 || strcmp(name, "[nl]") == 0) {
        bcnn_add_activation_layer(net, lp->a, lp->src_id);
    } else if (strcmp(name, "[batchnorm]") == 0 || strcmp(name, "[bn]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_batchnorm_layer(net, lp->src_id, lp->dst_id);
    } else if (strcmp(name, "[lrn]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_lrn_layer(net, lp->size, lp->alpha, lp->beta, lp->k,
                           lp->src_id, lp->dst_id);
    } else if (strcmp(name, "[connected]") == 0 ||
               strcmp(name, "[fullconnected]") == 0 ||
               strcmp(name, "[fc]") == 0 || strcmp(name, "[ip]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_fullc_layer(net, lp->outputs, lp->init, lp->a, 0, lp->src_id,
                             lp->dst_id);
    } else if (strcmp(name, "[softmax]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_softmax_layer(net, lp->src_id, lp->dst_id);
    } else if (strcmp(name, "[max]") == 0 || strcmp(name, "[maxpool]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_maxpool_layer(net, lp->size, lp->stride, lp->padding_type,
                               lp->src_id, lp->dst_id);
    } else if (strcmp(name, "[avgpool]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_avgpool_layer(net, lp->src_id, lp->dst_id);
    } else if (strcmp(name, "[dropout]") == 0) {
        bcnn_add_dropout_layer(net, lp->rate, lp->src_id);
    } else if (strcmp(name, "[concat]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_concat_layer(net, lp->src_id, lp->src_id2, lp->dst_id);
    } else if (strcmp(name, "[eltwise]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?\n");
        bcnn_add_eltwise_layer(net, lp->a, lp->src_id, lp->src_id2, lp->dst_id);
    } else if (strcmp(name, "[yolo]") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
                           "Cost layer: invalid input node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly "
                           "setup?\n");
        bcnn_add_yolo_layer(net, lp->boxes_per_cell, lp->num_classes,
                            lp->num_coords, lp->num_anchors, lp->anchors_mask,
                            lp->anchors, lp->src_id, lp->dst_id);
    } else if (strcmp(name, "[cost]") == 0) {
        BCNN_CHECK_AND_LOG(
            net->log_ctx, lp->src_id, BCNN_INVALID_PARAMETER,
            "Cost layer: invalid input node name. "
            "Hint: Are you sure that 'src' field is correctly setup?\n");
        BCNN_CHECK_AND_LOG(
            net->log_ctx, lp->dst_id, BCNN_INVALID_PARAMETER,
            "Cost layer: invalid input node name. "
            "Hint: Are you sure that 'dst' field is correctly setup?\n");
        bcnn_add_cost_layer(net, lp->loss, lp->cost, 1.0f, lp->src_id, "label",
                            lp->dst_id);
    } else {
        BCNN_ERROR(net->log_ctx, BCNN_INVALID_PARAMETER, "Unknown Layer %s\n",
                   name);
    }
    return BCNN_SUCCESS;
}

bcnn_status bcnn_load_net(bcnn_net *net, const char *config_path,
                          const char *model_path) {
    bcnn_parser_ini *config = bcnn_parser_read_ini(config_path);
    if (config == NULL) {
        return BCNN_INVALID_PARAMETER;
    }
    if (config->num_sections == 0 || config->sections == NULL) {
        bcnn_log(net->log_ctx, BCNN_LOG_ERROR, "Empty config file %s\n",
                 config_path);
        bcnn_parser_free(config);
        return BCNN_INVALID_PARAMETER;
    }
    if (strcmp(config->sections[0].name, "[net]") != 0 &&
        strcmp(config->sections[0].name, "[network]") != 0) {
        bcnn_log(net->log_ctx, BCNN_LOG_ERROR,
                 "Invalid config file %s: First section must be [net] or "
                 "[network]\n",
                 config_path);
        bcnn_parser_free(config);
        return BCNN_INVALID_PARAMETER;
    }
    if (config->sections[0].keys == NULL || config->sections[0].num_keys == 0) {
        bcnn_log(net->log_ctx, BCNN_LOG_ERROR,
                 "Invalid config file %s: empty section [net]\n", config_path);
        bcnn_parser_free(config);
        return BCNN_INVALID_PARAMETER;
    }
    // Parse network parameters
    for (int i = 0; i < config->sections[0].num_keys; ++i) {
        /*fprintf(stderr, "%s %s\n", config->sections[0].keys[i].name,
                config->sections[0].keys[i].val);*/
        bcnn_set_param(net, config->sections[0].keys[i].name,
                       config->sections[0].keys[i].val);
    }
    // Parse layers
    layer_param lp = {0};
    reset_layer_param(&lp);
    for (int i = 1; i < config->num_sections; ++i) {
        // Parse layers parameters
        for (int j = 0; j < config->sections[i].num_keys; ++j) {
            /*fprintf(stderr, "%s %s\n", config->sections[i].keys[j].name,
                    config->sections[i].keys[j].val);*/
            set_layer_param(net, &lp, config->sections[i].keys[j].name,
                            config->sections[i].keys[j].val);
        }
        // Add layer
        add_layer(net, config->sections[i].name, i, &lp);
        reset_layer_param(&lp);
    }
    bcnn_parser_free(config);
    return BCNN_SUCCESS;
}

bcnn_status bcnn_cl_load_param(const char *config_path, bcnn_cl_param *param) {
    bcnn_parser_ini *config = bcnn_parser_read_ini(config_path);
    if (config == NULL) {
        return BCNN_INVALID_PARAMETER;
    }
    if (config->num_sections == 0 || config->sections == NULL) {
        fprintf(stderr, "Empty config file %s\n", config_path);
        bcnn_parser_free(config);
        return BCNN_INVALID_PARAMETER;
    }
    for (int i = 0; i < config->sections[0].num_keys; ++i) {
        const char *name = config->sections[0].keys[i].name;
        const char *val = config->sections[0].keys[i].val;
        if (strcmp(name, "data_format") == 0) {
            if (strcmp(val, "mnist") == 0) {
                param->data_format = BCNN_LOAD_MNIST;
            } else if (strcmp(val, "cifar10") == 0) {
                param->data_format = BCNN_LOAD_CIFAR10;
            } else if (strcmp(val, "classif") == 0 ||
                       strcmp(val, "classification") == 0) {
                param->data_format = BCNN_LOAD_CLASSIFICATION_LIST;
            } else if (strcmp(val, "reg") == 0 ||
                       strcmp(val, "regression") == 0) {
                param->data_format = BCNN_LOAD_REGRESSION_LIST;
            } else if (strcmp(val, "detection") == 0) {
                param->data_format = BCNN_LOAD_DETECTION_LIST;
            } else {
                fprintf(stderr,
                        "Invalid parameter %s for 'data_format', "
                        "available parameters: "
                        "mnist, cifar10, classif, reg, detection",
                        name);
                return BCNN_INVALID_PARAMETER;
            }
        } else if (strcmp(name, "input_model") == 0) {
            bh_strfill(&param->input_model, val);
        } else if (strcmp(name, "output_model") == 0) {
            bh_strfill(&param->output_model, val);
        } else if (strcmp(name, "out_pred") == 0) {
            bh_strfill(&param->pred_out, val);
        } else if (strcmp(name, "eval_test") == 0) {
            param->eval_test = atoi(val);
        } else if (strcmp(name, "eval_period") == 0) {
            param->eval_period = atoi(val);
        } else if (strcmp(name, "save_model") == 0) {
            param->save_model = atoi(val);
        } else if (strcmp(name, "num_pred") == 0) {
            param->num_pred = atoi(val);
        } else if (strcmp(name, "source_train") == 0) {
            bh_strfill(&param->train_input, val);
        } else if (strcmp(name, "label_train") == 0) {
            bh_strfill(&param->path_train_label, val);
        } else if (strcmp(name, "source_test") == 0) {
            bh_strfill(&param->test_input, val);
        } else if (strcmp(name, "label_test") == 0) {
            bh_strfill(&param->path_test_label, val);
        }
    }
    param->num_pred = (param->num_pred > 0 ? param->num_pred : 1);
    param->eval_test = (param->eval_test > 0 ? param->eval_test : 0);
    param->eval_period = (param->eval_period > 0 ? param->eval_period : 100);
    param->save_model = (param->save_model > 0 ? param->save_model : 1000);
    bcnn_parser_free(config);
    return BCNN_SUCCESS;
}

bcnn_status bcnn_cl_train(bcnn_net *net, bcnn_cl_param *param, float *error) {
    float sum_error = 0.0f, error_valid = 0.0f;
    int nb_iter = net->learner->max_batches;
    int batch_size = net->batch_size;
    bh_timer t = {0};
    char chk_pt_path[1024];

    bh_timer_start(&t);
    for (int i = 0; i < nb_iter; ++i) {
        sum_error += bcnn_train_on_batch(net);

        if (i % param->eval_period == 0 && i > 0) {
            bh_timer_stop(&t);
            if (param->eval_test) {
                BCNN_CHECK_STATUS(bcnn_set_mode(net, BCNN_MODE_VALID));
                bcnn_cl_predict(net, param, &error_valid);
                BCNN_CHECK_STATUS(bcnn_set_mode(net, BCNN_MODE_TRAIN));
                BCNN_INFO(net->log_ctx,
                          "iter-batches= %d train-error= %f test-error= %f "
                          "training-time= %lf sec\n",
                          i, sum_error / (param->eval_period * batch_size),
                          error_valid, bh_timer_get_msec(&t) / 1000);
            } else {
                BCNN_INFO(
                    net->log_ctx,
                    "iter-batches= %d train-error= %f training-time= %lf sec\n",
                    i, sum_error / (param->eval_period * batch_size),
                    bh_timer_get_msec(&t) / 1000);
            }
            fflush(stderr);
            bh_timer_start(&t);
            sum_error = 0;
        }
        if (i % param->save_model == 0 && i > 0) {
            sprintf(chk_pt_path, "%s_iter%d.bcnnmodel", param->output_model, i);
            bcnn_write_model(net, chk_pt_path);
        }
    }

    *error = (float)sum_error / (param->eval_period * batch_size);

    return BCNN_SUCCESS;
}

bcnn_status bcnn_cl_predict(bcnn_net *net, bcnn_cl_param *param, float *error) {
    float err = 0.0f;
    FILE *f = NULL;
    int batch_size = net->batch_size;
    char out_pred_name[128] = {0};
    int output_size = bcnn_tensor_size3d(
        &net->tensors[net->nodes[net->num_nodes - 2].dst[0]]);
    unsigned char *dump_img = (unsigned char *)calloc(
        net->tensors[0].w * net->tensors[0].h * 3, sizeof(unsigned char));

    if (param->pred_out != NULL) {
        f = fopen(param->pred_out, "wt");
        if (f == NULL) {
            BCNN_ERROR(net->log_ctx, BCNN_INVALID_PARAMETER,
                       "Could not open file %s\n", param->pred_out);
        }
    }

    int n = param->num_pred / batch_size;
    for (int i = 0; i < n; ++i) {
        bcnn_tensor *out = NULL;
        err += bcnn_predict_on_batch(net, &out);
        // Dump predictions
        if (param->pred_out) {
            if (net->data_loader->type == BCNN_LOAD_DETECTION_LIST) {
                for (int b = 0; b < net->batch_size; ++b) {
                    int num_dets = 0;
                    bcnn_output_detection *dets = bcnn_yolo_get_detections(
                        net, b, net->tensors[0].w, net->tensors[0].h,
                        net->tensors[0].w, net->tensors[0].h, 0.5, 1,
                        &num_dets);
                    int sz = bcnn_tensor_size3d(&net->tensors[0]);
                    if (net->tensors[0].c == 3) {
                        memcpy(dump_img, net->tensors[0].data + b * sz, sz);
                    } else if (net->tensors[0].c == 1) {
                        for (int p = 0;
                             p < bcnn_tensor_size2d(&net->tensors[0]); ++p) {
                            dump_img[3 * p] =
                                127 * (net->tensors[0].data[b * sz + p] + 1);
                            dump_img[3 * p + 1] =
                                127 * (net->tensors[0].data[b * sz + p] + 1);
                            dump_img[3 * p + 2] =
                                127 * (net->tensors[0].data[b * sz + p] + 1);
                        }
                    }
                    int sz_label = bcnn_tensor_size3d(&net->tensors[1]);
                    if (net->nodes[net->num_nodes - 1].type ==
                        BCNN_LAYER_YOLOV3) {
                        bcnn_node *node = &net->nodes[net->num_nodes - 1];
                        bcnn_yolo_param *yolo_param =
                            (bcnn_yolo_param *)node->param;
                        for (int j = 0; j < yolo_param->classes; ++j) {
                            // truth
                            unsigned char green[3] = {0, 255, 0};
                            for (int t = 0; t < BCNN_DETECTION_MAX_BOXES; ++t) {
                                bcnn_draw_color_box(
                                    dump_img, net->tensors[0].w,
                                    net->tensors[0].h,
                                    net->tensors[1].data[b * sz_label + t * 5],
                                    net->tensors[1]
                                        .data[b * sz_label + t * 5 + 1],
                                    net->tensors[1]
                                        .data[b * sz_label + t * 5 + 2],
                                    net->tensors[1]
                                        .data[b * sz_label + t * 5 + 3],
                                    green);
                            }
                        }
                    }
                    for (int d = 0; d < num_dets; ++d) {
                        if (dets[d].prob[0] > 0) {
                            unsigned char blue[3] = {0, 0, 255};
                            bcnn_draw_color_box(dump_img, net->tensors[0].w,
                                                net->tensors[0].h, dets[d].x,
                                                dets[d].y, dets[d].w, dets[d].h,
                                                blue);
                        }
                    }
                    sprintf(out_pred_name, "det_%d.png", b);
                    bip_write_image(out_pred_name, dump_img, net->tensors[0].w,
                                    net->tensors[0].h, 3,
                                    net->tensors[0].w * 3);
                    bh_free(dets);
                }
            } else {
                int out_sz = out->w * out->h * out->c;
                for (int j = 0; j < net->batch_size; ++j) {
                    for (int k = 0; k < out_sz; ++k) {
                        fprintf(f, "%f ", out->data[j * out_sz + k]);
                    }
                    fprintf(f, "\n");
                }
            }
        }
    }
    *error = err / param->num_pred;
    bh_free(dump_img);

    if (f != NULL) {
        fclose(f);
    }
    return BCNN_SUCCESS;
}

void bcnn_cl_free_param(bcnn_cl_param *param) {
    bh_free(param->input_model);
    bh_free(param->output_model);
    bh_free(param->pred_out);
    bh_free(param->train_input);
    bh_free(param->test_input);
    bh_free(param->path_train_label);
    bh_free(param->path_test_label);
}

void show_usage(char *argv) {
    fprintf(stderr, "Usage: %s <mode> <config> [gpu_device]\n", argv);
    fprintf(stderr, "\t Required:\n");
    fprintf(stderr,
            "\t\t <mode>: network mode. Possible values are: 'train', "
            "'valid', 'predict'"
            "\t\t <config>: configuration file. See example here: ");
    fprintf(stderr,
            "https://github.com/jnbraun/bcnn/blob/master/examples/mnist_cl/"
            "mnist.cfg\n");
    fprintf(stderr, "\t Optional:\n");
    fprintf(stderr, "\t\t [gpu_device]: Gpu device id. Default: 0.\n");
}

int main(int argc, char **argv) {
    if (argc < 3) {
        show_usage(argv[0]);
        return -1;
    }
#ifdef BCNN_USE_CUDA
    if (argc == 4) {
        bcnn_cuda_set_device(atoi(argv[3]));
    }
#endif

    bcnn_mode mode;
    if (strcmp(argv[1], "train") == 0) {
        mode = BCNN_MODE_TRAIN;
    } else if (strcmp(argv[1], "valid") == 0) {
        mode = BCNN_MODE_VALID;
    } else if (strcmp(argv[1], "predict") == 0) {
        mode = BCNN_MODE_PREDICT;
    } else {
        fprintf(stderr, "Invalid mode %s\n", argv[1]);
        show_usage(argv[0]);
        return -1;
    }
    // Initialize network from config file
    bcnn_net *net = NULL;
    BCNN_CHECK_STATUS(bcnn_init_net(&net, mode));
    BCNN_CHECK_STATUS(bcnn_load_net(net, argv[2], NULL));
    bcnn_cl_param param = {0};
    BCNN_CHECK_STATUS(bcnn_cl_load_param(argv[2], &param));
    BCNN_CHECK_STATUS(bcnn_set_data_loader(
        net, param.data_format, param.train_input, param.path_train_label,
        param.test_input, param.path_test_label));
    BCNN_CHECK_STATUS(bcnn_compile_net(net));
    if (net->mode == BCNN_MODE_TRAIN) {
        if (param.input_model != NULL) {
            fprintf(stderr, "[INFO] Loading pre-trained model %s\n",
                    param.input_model);
            BCNN_CHECK_STATUS(bcnn_load_model(net, param.input_model));
        }
        BCNN_INFO(net->log_ctx, "Start training...\n");
        float error_train = 0.0f, error_valid = 0.0f;
        BCNN_CHECK_STATUS(bcnn_cl_train(net, &param, &error_train));
        if (param.pred_out != NULL) {
            BCNN_CHECK_STATUS(bcnn_set_mode(net, BCNN_MODE_VALID));
            BCNN_CHECK_STATUS(bcnn_cl_predict(net, &param, &error_valid));
            BCNN_CHECK_STATUS(bcnn_set_mode(net, BCNN_MODE_TRAIN));
        }
        if (param.output_model != NULL) {
            BCNN_CHECK_STATUS(bcnn_write_model(net, param.output_model));
        }
        BCNN_INFO(net->log_ctx, "Training ended successfully\n");
    } else if (net->mode == BCNN_MODE_VALID || net->mode == BCNN_MODE_PREDICT) {
        if (param.input_model != NULL) {
            BCNN_CHECK_STATUS(bcnn_load_model(net, param.input_model));
        } else {
            BCNN_ERROR(net->log_ctx, BCNN_INVALID_PARAMETER,
                       "No model in input. Precise which model to use in "
                       "config file "
                       "with field 'input_model'\n");
        }
        BCNN_INFO(net->log_ctx, "Start prediction...\n");
        float error_test = 0.0f;
        BCNN_CHECK_STATUS(bcnn_cl_predict(net, &param, &error_test));
        BCNN_INFO(net->log_ctx, "Prediction ended successfully\n");
    }
    bcnn_end_net(&net);
    bcnn_cl_free_param(&param);
    return 0;
}