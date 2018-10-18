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

#include <bh/bh_log.h>
#include <bh/bh_macros.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>
#include <bh/bh_timer.h>

#include <bip/bip.h>

#include "bcnn/bcnn.h"
#include "bcnn/bcnn_cl.h"

int bcnncl_init_from_config(bcnn_net *net, char *config_file,
                            bcnncl_param *param) {
    FILE *file = NULL;
    char *line = NULL, *curr_layer = NULL;
    char **tok = NULL;
    int nb_lines = 0, nb_layers = 0;
    bcnn_padding padding_type = PADDING_SAME;
    int stride = 1, pad = 0, n_filts = 1, size = 3, outputs = 0, num_groups = 1,
        batchnorm = 0;
    int in_w = 0, in_h = 0, in_c = 0;
    float alpha, beta, k;
    bcnn_activation a = NONE;
    bcnn_filler_type init = XAVIER;
    bcnn_loss_metric cost = COST_SSE;
    bcnn_loss loss = EUCLIDEAN_LOSS;
    float rate = 1.0f;
    int n_tok;
    char *src_id = NULL, *dst_id = NULL, *src_id2 = NULL;

    file = fopen(config_file, "rt");
    if (file == 0) {
        fprintf(stderr, "Couldn't open file: %s\n", config_file);
        exit(-1);
    }

    BCNN_INFO(net->log_ctx, "Network architecture");
    while ((line = bh_fgetline(file)) != 0) {
        nb_lines++;
        bh_strstrip(line);
        switch (line[0]) {
            case '{':
                if (nb_layers > 0) {
                    if (nb_layers == 1) {
                        BCNN_CHECK_AND_LOG(
                            net->log_ctx,
                            net->input_width > 0 && net->input_height > 0 &&
                                net->input_channels > 0,
                            BCNN_INVALID_PARAMETER,
                            "Input's width, height and channels must be > 0");
                        BCNN_CHECK_AND_LOG(net->log_ctx, net->batch_size > 0,
                                           BCNN_INVALID_PARAMETER,
                                           "Batch size must be > 0");
                        bcnn_net_set_input_shape(
                            net, net->input_width, net->input_height,
                            net->input_channels, net->batch_size);
                    }
                    BCNN_CHECK_AND_LOG(
                        net->log_ctx, src_id, BCNN_INVALID_PARAMETER,
                        "Invalid input node name. "
                        "Hint: Are you sure that 'src' field is correctly "
                        "setup?");
                    if (strcmp(curr_layer, "{input}") == 0) {
                        bcnn_net_add_input(net, in_w, in_h, in_c, src_id);
                    } else if (strcmp(curr_layer, "{conv}") == 0 ||
                               strcmp(curr_layer, "{convolutional}") == 0) {
                        BCNN_CHECK_AND_LOG(
                            net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                            "Invalid output node name. "
                            "Hint: Are you sure that 'dst' field is "
                            "correctly setup?");
                        bcnn_add_convolutional_layer(
                            net, n_filts, size, stride, pad, num_groups,
                            batchnorm, init, a, 0, src_id, dst_id);
                    } else if (strcmp(curr_layer, "{deconv}") == 0 ||
                               strcmp(curr_layer, "{deconvolutional}") == 0) {
                        BCNN_CHECK_AND_LOG(
                            net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                            "Invalid output node name. "
                            "Hint: Are you sure that 'dst' field is "
                            "correctly setup?");
                        bcnn_add_deconvolutional_layer(net, n_filts, size,
                                                       stride, pad, init, a,
                                                       src_id, dst_id);
                    } else if (strcmp(curr_layer, "{depthwise-conv}") == 0 ||
                               strcmp(curr_layer, "{dw-conv}") == 0) {
                        BCNN_CHECK_AND_LOG(
                            net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                            "Invalid output node name. "
                            "Hint: Are you sure that 'dst' field is "
                            "correctly setup?");
                        bcnn_add_depthwise_sep_conv_layer(
                            net, size, stride, pad, 0, init, a, src_id, dst_id);
                    } else if (strcmp(curr_layer, "{activation}") == 0 ||
                               strcmp(curr_layer, "{nl}") == 0) {
                        bcnn_add_activation_layer(net, a, src_id);
                    } else if (strcmp(curr_layer, "{batchnorm}") == 0 ||
                               strcmp(curr_layer, "{bn}") == 0) {
                        BCNN_CHECK_AND_LOG(
                            net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                            "Invalid output node name. "
                            "Hint: Are you sure that 'dst' field is "
                            "correctly setup?");
                        bcnn_add_batchnorm_layer(net, src_id, dst_id);
                    } else if (strcmp(curr_layer, "{lrn}") == 0) {
                        BCNN_CHECK_AND_LOG(
                            net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                            "Invalid output node name. "
                            "Hint: Are you sure that 'dst' field is "
                            "correctly setup?");
                        bcnn_add_lrn_layer(net, size, alpha, beta, k, src_id,
                                           dst_id);
                    } else if (strcmp(curr_layer, "{connected}") == 0 ||
                               strcmp(curr_layer, "{fullconnected}") == 0 ||
                               strcmp(curr_layer, "{fc}") == 0 ||
                               strcmp(curr_layer, "{ip}") == 0) {
                        BCNN_CHECK_AND_LOG(
                            net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                            "Invalid output node name. "
                            "Hint: Are you sure that 'dst' field is "
                            "correctly setup?");
                        bcnn_add_fullc_layer(net, outputs, init, a, 0, src_id,
                                             dst_id);
                    } else if (strcmp(curr_layer, "{softmax}") == 0) {
                        BCNN_CHECK_AND_LOG(
                            net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                            "Invalid output node name. "
                            "Hint: Are you sure that 'dst' field is "
                            "correctly setup?");
                        bcnn_add_softmax_layer(net, src_id, dst_id);
                    } else if (strcmp(curr_layer, "{max}") == 0 ||
                               strcmp(curr_layer, "{maxpool}") == 0) {
                        BCNN_CHECK_AND_LOG(
                            net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                            "Invalid output node name. "
                            "Hint: Are you sure that 'dst' field is "
                            "correctly setup?");
                        bcnn_add_maxpool_layer(net, size, stride, padding_type,
                                               src_id, dst_id);
                    } else if (strcmp(curr_layer, "{dropout}") == 0) {
                        bcnn_add_dropout_layer(net, rate, src_id);
                    } else if (strcmp(curr_layer, "{concat}") == 0) {
                        bcnn_add_concat_layer(net, src_id, src_id2, dst_id);
                    } else {
                        BCNN_ERROR(net->log_ctx, BCNN_INVALID_PARAMETER,
                                   "Unknown Layer %s", curr_layer);
                    }
                    bh_free(curr_layer);
                    bh_free(src_id);
                    bh_free(src_id2);
                    bh_free(dst_id);
                    a = NONE;
                }
                curr_layer = line;
                nb_layers++;
                break;
            case '!':
            case '\0':
            case '#':
                bh_free(line);
                break;
            default:
                n_tok = bh_strsplit(line, '=', &tok);
                BCNN_CHECK_AND_LOG(net->log_ctx, (n_tok == 2),
                                   BCNN_INVALID_PARAMETER,
                                   "Wrong format option in config file");
                if (strcmp(tok[0], "task") == 0) {
                    if (strcmp(tok[1], "train") == 0)
                        param->task = TRAIN;
                    else if (strcmp(tok[1], "predict") == 0) {
                        param->task = PREDICT;
                        net->task = PREDICT;
                    } else
                        BCNN_ERROR(
                            net->log_ctx, BCNN_INVALID_PARAMETER,
                            "Invalid parameter for task, available parameters: "
                            "TRAIN, PREDICT");
                } else if (strcmp(tok[0], "data_format") == 0)
                    bh_strfill(&param->data_format, tok[1]);
                else if (strcmp(tok[0], "input_model") == 0)
                    bh_strfill(&param->input_model, tok[1]);
                else if (strcmp(tok[0], "output_model") == 0)
                    bh_strfill(&param->output_model, tok[1]);
                else if (strcmp(tok[0], "out_pred") == 0)
                    bh_strfill(&param->pred_out, tok[1]);
                else if (strcmp(tok[0], "eval_test") == 0)
                    param->eval_test = atoi(tok[1]);
                else if (strcmp(tok[0], "eval_period") == 0)
                    param->eval_period = atoi(tok[1]);
                else if (strcmp(tok[0], "save_model") == 0)
                    param->save_model = atoi(tok[1]);
                else if (strcmp(tok[0], "nb_pred") == 0)
                    param->nb_pred = atoi(tok[1]);
                else if (strcmp(tok[0], "source_train") == 0)
                    bh_strfill(&param->train_input, tok[1]);
                else if (strcmp(tok[0], "label_train") == 0)
                    bh_strfill(&param->path_train_label, tok[1]);
                else if (strcmp(tok[0], "source_test") == 0)
                    bh_strfill(&param->test_input, tok[1]);
                else if (strcmp(tok[0], "label_test") == 0)
                    bh_strfill(&param->path_test_label, tok[1]);
                else if (strcmp(tok[0], "dropout_rate") == 0 ||
                         strcmp(tok[0], "rate") == 0)
                    rate = (float)atof(tok[1]);
                else if (strcmp(tok[0], "filters") == 0)
                    n_filts = atoi(tok[1]);
                else if (strcmp(tok[0], "size") == 0)
                    size = atoi(tok[1]);
                else if (strcmp(tok[0], "stride") == 0)
                    stride = atoi(tok[1]);
                else if (strcmp(tok[0], "pad") == 0)
                    pad = atoi(tok[1]);
                else if (strcmp(tok[0], "num_groups") == 0)
                    num_groups = atoi(tok[1]);
                else if (strcmp(tok[0], "alpha") == 0)
                    alpha = atoi(tok[1]);
                else if (strcmp(tok[0], "beta") == 0)
                    beta = atoi(tok[1]);
                else if (strcmp(tok[0], "k") == 0)
                    k = atoi(tok[1]);
                else if (strcmp(tok[0], "w") == 0) {
                    in_w = atoi(tok[1]);
                } else if (strcmp(tok[0], "h") == 0) {
                    in_h = atoi(tok[1]);
                } else if (strcmp(tok[0], "c") == 0) {
                    in_c = atoi(tok[1]);
                } else if (strcmp(tok[0], "bn") == 0 ||
                           strcmp(tok[0], "batchnorm") == 0) {
                    batchnorm = atoi(tok[1]);
                } else if (strcmp(tok[0], "src") == 0) {
                    char **srcids = NULL;
                    int num_srcids = bh_strsplit(tok[1], ',', &srcids);
                    bh_strfill(&src_id, srcids[0]);
                    if (num_srcids > 1) {
                        bh_strfill(&src_id2, srcids[1]);
                    }
                } else if (strcmp(tok[0], "dst") == 0)
                    bh_strfill(&dst_id, tok[1]);
                else if (strcmp(tok[0], "output") == 0)
                    outputs = atoi(tok[1]);
                else if (strcmp(tok[0], "padding_type") == 0) {
                    if (strcmp(tok[1], "same") == 0)
                        padding_type = PADDING_SAME;
                    else if (strcmp(tok[1], "valid") == 0)
                        padding_type = PADDING_VALID;
                    else if (strcmp(tok[1], "caffe") == 0)
                        padding_type = PADDING_CAFFE;
                } else if (strcmp(tok[0], "function") == 0) {
                    if (strcmp(tok[1], "relu") == 0)
                        a = RELU;
                    else if (strcmp(tok[1], "tanh") == 0)
                        a = TANH;
                    else if (strcmp(tok[1], "ramp") == 0)
                        a = RAMP;
                    else if (strcmp(tok[1], "clamp") == 0)
                        a = CLAMP;
                    else if (strcmp(tok[1], "softplus") == 0)
                        a = SOFTPLUS;
                    else if (strcmp(tok[1], "leaky_relu") == 0 ||
                             strcmp(tok[1], "lrelu") == 0)
                        a = LRELU;
                    else if (strcmp(tok[1], "prelu") == 0)
                        a = PRELU;
                    else if (strcmp(tok[1], "abs") == 0)
                        a = ABS;
                    else if (strcmp(tok[1], "none") == 0)
                        a = NONE;
                    else {
                        BCNN_WARNING(
                            net->log_ctx,
                            "Unknown activation type %s, going with ReLU",
                            tok[1]);
                        a = RELU;
                    }
                } else if (strcmp(tok[0], "init") == 0) {
                    if (strcmp(tok[1], "xavier") == 0)
                        init = XAVIER;
                    else if (strcmp(tok[1], "msra") == 0)
                        init = MSRA;
                    else {
                        BCNN_WARNING(
                            net->log_ctx,
                            "Unknown init type %s, going with xavier init",
                            tok[1]);
                        init = XAVIER;
                    }
                } else if (strcmp(tok[0], "metric") == 0) {
                    if (strcmp(tok[1], "error") == 0)
                        cost = COST_ERROR;
                    else if (strcmp(tok[1], "logloss") == 0)
                        cost = COST_LOGLOSS;
                    else if (strcmp(tok[1], "sse") == 0)
                        cost = COST_SSE;
                    else if (strcmp(tok[1], "mse") == 0)
                        cost = COST_MSE;
                    else if (strcmp(tok[1], "crps") == 0)
                        cost = COST_CRPS;
                    else if (strcmp(tok[1], "dice") == 0)
                        cost = COST_DICE;
                    else {
                        BCNN_WARNING(net->log_ctx,
                                     "Unknown cost metric %s, going with sse",
                                     tok[1]);
                        cost = COST_SSE;
                    }
                } else if (strcmp(tok[0], "loss") == 0) {
                    if (strcmp(tok[1], "l2") == 0 ||
                        strcmp(tok[1], "euclidean") == 0) {
                        loss = EUCLIDEAN_LOSS;
                    } else if (strcmp(tok[1], "lifted_struct_similarity") ==
                               0) {
                        loss = LIFTED_STRUCT_SIMILARITY_SOFTMAX_LOSS;
                    } else {
                        BCNN_WARNING(
                            net->log_ctx,
                            "Unknown loss %s, going with euclidean loss",
                            tok[1]);
                        loss = EUCLIDEAN_LOSS;
                    }
                } else
                    bcnn_set_param(net, tok[0], tok[1]);

                bh_free(tok[0]);
                bh_free(tok[1]);
                bh_free(tok);
                bh_free(line);
                break;
        }
    }
    // Add cost layer
    if (strcmp(curr_layer, "{cost}") == 0) {
        BCNN_CHECK_AND_LOG(
            net->log_ctx, src_id, BCNN_INVALID_PARAMETER,
            "Cost layer: invalid input node name. "
            "Hint: Are you sure that 'src' field is correctly setup?");
        BCNN_CHECK_AND_LOG(
            net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
            "Cost layer: invalid input node name. "
            "Hint: Are you sure that 'dst' field is correctly setup?");
        bcnn_add_cost_layer(net, loss, cost, 1.0f, src_id, "label", dst_id);
    } else {
        BCNN_ERROR(net->log_ctx, BCNN_INVALID_PARAMETER,
                   "Error in config file: last layer must be a cost layer");
    }
    bh_free(src_id);
    bh_free(src_id2);
    bh_free(dst_id);
    bh_free(curr_layer);
    fclose(file);

    param->eval_period = (param->eval_period > 0 ? param->eval_period : 100);

    fflush(stderr);
    return 0;
}

int bcnncl_train(bcnn_net *net, bcnncl_param *param, float *error) {
    float error_batch = 0.0f, sum_error = 0.0f, error_valid = 0.0f;
    int i = 0, nb_iter = net->max_batches;
    int batch_size = net->batch_size;
    bh_timer t = {0};
    bcnn_iterator iter_data = {0};
    char chk_pt_path[1024];

    if (bcnn_iterator_initialize(net, &iter_data, param->train_input,
                                 param->path_train_label,
                                 param->data_format) != 0) {
        return -1;  // TODO: proper error return
    }

    bcnn_compile_net(net, "train");

    bh_timer_start(&t);
    for (i = 0; i < nb_iter; ++i) {
        bcnn_train_on_batch(net, &iter_data, &error_batch);
        sum_error += error_batch;

        if (i % param->eval_period == 0 && i > 0) {
            bh_timer_stop(&t);
            if (param->eval_test) {
                bcnncl_predict(net, param, &error_valid, 1);
                fprintf(stderr,
                        "iter= %d train-error= %f test-error= %f "
                        "training-time= %lf sec\n",
                        i, sum_error / (param->eval_period * batch_size),
                        error_valid, bh_timer_get_msec(&t) / 1000);
            } else {
                fprintf(stderr,
                        "iter= %d train-error= %f training-time= %lf sec\n", i,
                        sum_error / (param->eval_period * batch_size),
                        bh_timer_get_msec(&t) / 1000);
            }
            fflush(stderr);
            bh_timer_start(&t);
            sum_error = 0;
            if (param->eval_test) bcnn_compile_net(net, "train");
        }
        if (i % param->save_model == 0 && i > 0) {
            sprintf(chk_pt_path, "%s_iter%d.dat", param->output_model, i);
            bcnn_write_model(net, chk_pt_path);
        }
    }

    bcnn_iterator_terminate(&iter_data);
    *error = (float)sum_error / (param->eval_period * batch_size);

    return BCNN_SUCCESS;
}

int bcnncl_predict(bcnn_net *net, bcnncl_param *param, float *error,
                   int dump_pred) {
    int i = 0, j = 0, n = 0, k = 0;
    float *out = NULL;
    float err = 0.0f, error_batch = 0.0f;
    FILE *f = NULL;
    int batch_size = net->batch_size;
    char out_pred_name[128] = {0};
    bcnn_iterator iter_data = {0};
    int out_w = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].w;
    int out_h = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].h;
    int out_c = net->tensors[net->nodes[net->num_nodes - 2].dst[0]].c;
    int output_size = out_w * out_h * out_c;

    if (bcnn_iterator_initialize(net, &iter_data, param->test_input,
                                 param->path_test_label,
                                 param->data_format) != 0)
        return -1;

    if (dump_pred) {
        if (net->prediction_type != HEATMAP_REGRESSION &&
            net->prediction_type != SEGMENTATION) {
            f = fopen(param->pred_out, "wt");
            if (f == NULL) {
                fprintf(stderr, "[ERROR] bcnn_predict: Can't open file %s",
                        param->pred_out);
                return -1;
            }
        }
    }

    bcnn_compile_net(net, "predict");

    n = param->nb_pred / batch_size;
    for (i = 0; i < n; ++i) {
        bcnn_predict_on_batch(net, &iter_data, &out, &error_batch);
        err += error_batch;
        // Dump predictions
        if (dump_pred) {
            if (net->prediction_type == HEATMAP_REGRESSION ||
                net->prediction_type == SEGMENTATION) {
                for (j = 0; j < net->batch_size; ++j) {
                    for (k = 0; k < out_c; ++k) {
                        sprintf(out_pred_name, "%d_%d.png",
                                i * net->batch_size + j, k);
                        bip_write_float_image(
                            out_pred_name,
                            out + j * out_w * out_h * out_c + k * out_w * out_h,
                            out_w, out_h, 1, out_w * sizeof(float));
                    }
                }
            } else {
                for (j = 0; j < net->batch_size; ++j) {
                    for (k = 0; k < output_size; ++k)
                        fprintf(f, "%f ", out[j * output_size + k]);
                    fprintf(f, "\n");
                }
            }
        }
    }
    // Process last instances
    n = param->nb_pred % net->batch_size;
    if (n > 0) {
        for (i = 0; i < n; ++i) {
            bcnn_predict_on_batch(net, &iter_data, &out, &error_batch);
            err += error_batch;
            // Dump predictions
            if (dump_pred) {
                if (net->prediction_type == HEATMAP_REGRESSION ||
                    net->prediction_type == SEGMENTATION) {
                    for (k = 0; k < out_c; ++k) {
                        sprintf(out_pred_name, "%d_%d.png", i, k);
                        bip_write_float_image(out_pred_name,
                                              out + k * out_w * out_h, out_w,
                                              out_h, 1, out_w * sizeof(float));
                    }
                } else {
                    for (k = 0; k < output_size; ++k) fprintf(f, "%f ", out[k]);
                    fprintf(f, "\n");
                }
            }
        }
    }
    *error = err / param->nb_pred;

    if (f != NULL) fclose(f);
    bcnn_iterator_terminate(&iter_data);
    return BCNN_SUCCESS;
}

int bcnncl_free_param(bcnncl_param *param) {
    bh_free(param->input_model);
    bh_free(param->output_model);
    bh_free(param->pred_out);
    bh_free(param->train_input);
    bh_free(param->test_input);
    bh_free(param->data_format);
    bh_free(param->path_train_label);
    bh_free(param->path_test_label);
    return BCNN_SUCCESS;
}

int run(char *config_file) {
    bcnn_net *net = NULL;
    bcnncl_param param = {0};
    float error_train = 0.0f, error_valid = 0.0f, error_test = 0.0f;

    bcnn_init_net(&net);
    // Initialize network from config file
    BCNN_CHECK_STATUS(bcnncl_init_from_config(net, config_file, &param));

    if (param.task == TRAIN) {
        if (param.input_model != NULL) {
            fprintf(stderr, "[INFO] Loading pre-trained model %s\n",
                    param.input_model);
            bcnn_load_model(net, param.input_model);
        }
        BCNN_INFO(net->log_ctx, "Start training...");
        BCNN_CHECK_STATUS(bcnncl_train(net, &param, &error_train));
        if (param.pred_out != NULL) {
            BCNN_CHECK_STATUS(bcnncl_predict(net, &param, &error_valid, 1));
        }
        if (param.output_model != NULL) {
            bcnn_write_model(net, param.output_model);
        }
        BCNN_INFO(net->log_ctx, "Training ended successfully");
    } else if (param.task == PREDICT) {
        if (param.input_model != NULL)
            bcnn_load_model(net, param.input_model);
        else {
            BCNN_ERROR(
                net->log_ctx, BCNN_INVALID_PARAMETER,
                "No model in input. Inform which model to use in config file "
                "with field 'input_model'");
        }
        BCNN_INFO(net->log_ctx, "Start prediction...");
        BCNN_CHECK_STATUS(bcnncl_predict(net, &param, &error_test, 1));
        BCNN_INFO(net->log_ctx, "Prediction ended successfully");
    }
    bcnn_end_net(&net);
    bcnncl_free_param(&param);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config> [gpu_device]\n", argv[0]);
        fprintf(stderr, "\t Required:\n");
        fprintf(stderr,
                "\t\t <config>: configuration file. See example here: ");
        fprintf(stderr,
                "https://github.com/jnbraun/bcnn/blob/master/examples/mnist_cl/"
                "mnist.cfg\n");
        fprintf(stderr, "\t Optional:\n");
        fprintf(stderr, "\t\t [gpu_device]: Gpu device id. Default: 0.\n");
        return -1;
    }
#ifdef BCNN_USE_CUDA
    if (argc == 3) {
        bcnn_cuda_set_device(atoi(argv[2]));
    }
#endif
    run(argv[1]);
    return 0;
}
