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

#include <bh/bh_ini.h>
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

bcnn_status bcnn_cl_load_param(const char *config_path, bcnn_cl_param *param) {
    bh_ini_parser *config = bh_ini_parser_create(config_path);
    if (config == NULL) {
        return BCNN_INVALID_PARAMETER;
    }
    if (config->num_sections == 0 || config->sections == NULL) {
        fprintf(stderr, "Empty config file %s\n", config_path);
        bh_ini_parser_destroy(config);
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
    bh_ini_parser_destroy(config);
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
                BCNN_INFO(net->log_ctx,
                          "iter-batches= %d train-error= %f training-time= "
                          "%lf sec\n",
                          i, sum_error / (param->eval_period * batch_size),
                          bh_timer_get_msec(&t) / 1000);
            }
            fflush(stderr);
            bh_timer_start(&t);
            sum_error = 0;
        }
        if (i % param->save_model == 0 && i > 0) {
            sprintf(chk_pt_path, "%s_iter%d.bcnnmodel", param->output_model, i);
            bcnn_save_weights(net, chk_pt_path);
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
            "'valid', 'predict'\n"
            "\t\t <config>: configuration file. See example here: ");
    fprintf(stderr,
            "https://github.com/jnbraun/bcnn/blob/master/examples/mnist_cl/"
            "mnist.conf\n");
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
    bcnn_cl_param param = {0};
    BCNN_CHECK_STATUS(bcnn_cl_load_param(argv[2], &param));
    BCNN_CHECK_STATUS(bcnn_load_net(net, argv[2], param.input_model));
    BCNN_CHECK_STATUS(bcnn_set_data_loader(
        net, param.data_format, param.train_input, param.path_train_label,
        param.test_input, param.path_test_label));
    BCNN_CHECK_STATUS(bcnn_compile_net(net));
    if (net->mode == BCNN_MODE_TRAIN) {
        /*if (param.input_model != NULL) {
            fprintf(stderr, "[INFO] Loading pre-trained model %s\n",
                    param.input_model);
            BCNN_CHECK_STATUS(bcnn_load_weights(net, param.input_model));
        }*/
        BCNN_INFO(net->log_ctx, "Start training...\n");
        float error_train = 0.0f, error_valid = 0.0f;
        BCNN_CHECK_STATUS(bcnn_cl_train(net, &param, &error_train));
        if (param.pred_out != NULL) {
            BCNN_CHECK_STATUS(bcnn_set_mode(net, BCNN_MODE_VALID));
            BCNN_CHECK_STATUS(bcnn_cl_predict(net, &param, &error_valid));
            BCNN_CHECK_STATUS(bcnn_set_mode(net, BCNN_MODE_TRAIN));
        }
        if (param.output_model != NULL) {
            BCNN_CHECK_STATUS(bcnn_save_weights(net, param.output_model));
        }
        BCNN_INFO(net->log_ctx, "Training ended successfully\n");
    } else if (net->mode == BCNN_MODE_VALID || net->mode == BCNN_MODE_PREDICT) {
        if (param.input_model != NULL) {
            BCNN_CHECK_STATUS(bcnn_load_weights(net, param.input_model));
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