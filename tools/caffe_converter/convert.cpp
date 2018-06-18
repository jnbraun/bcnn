#include <limits.h>
#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include <bh/bh.h>
#include <bh/bh_error.h>
#include <bh/bh_string.h>

/* bcnn include */
#include <bcnn/bcnn.h>
#include <bcnn/bcnn_cl.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

/* Caffe include */
#ifndef CPU_ONLY
#define CPU_ONLY
#endif
#include <caffe/proto/caffe.pb.h>
#include <caffe/blob.hpp>
#include <caffe/layers/conv_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>
#include <caffe/net.hpp>
#include <caffe/util/io.hpp>

using namespace caffe;

static int _init_from_config(bcnn_net *net, char *config_file,
                             bcnncl_param *param) {
    FILE *file = NULL;
    char *line = NULL, *curr_layer = NULL;
    char **tok = NULL;
    int nb_lines = 0, nb_layers = 0;
    int stride = 1, pad = 0, n_filts = 1, size = 3, outputs = 0;
    bcnn_activation a = NONE;
    bcnn_filler_type init = XAVIER;
    bcnn_loss_metric cost = COST_SSE;
    float rate = 1.0f;
    int n_tok;
    char *src_id = NULL;
    char *dst_id = NULL;

    file = fopen(config_file, "rt");
    if (file == 0) {
        fprintf(stderr, "Couldn't open file: %s\n", config_file);
        exit(-1);
    }

    bh_info("Network architecture");
    while ((line = bh_fgetline(file)) != 0) {
        nb_lines++;
        bh_strstrip(line);
        switch (line[0]) {
            case '{':
                if (nb_layers > 0) {
                    if (nb_layers == 1) {
                        bh_assert(
                            net->input_width > 0 && net->input_height > 0 &&
                                net->input_channels > 0,
                            "Input's width, height and channels must be > 0",
                            BCNN_INVALID_PARAMETER);
                        bh_assert(net->batch_size > 0, "Batch size must be > 0",
                                  BCNN_INVALID_PARAMETER);
                    }
                    if (strcmp(curr_layer, "{conv}") == 0 ||
                        strcmp(curr_layer, "{convolutional}") == 0) {
                        bcnn_add_convolutional_layer(net, n_filts, size, stride,
                                                     pad, 0, init, a, 0, src_id,
                                                     dst_id);
                        /*fprintf(stderr, "out_c= %d %d %d %d\n", n_filts, size,
                           net->nodes[net->num_nodes - 1].src_tensor.c,
                            net->nodes[net->num_nodes -
                           1].layer->weights_size);*/
                    } else if (strcmp(curr_layer, "{deconv}") == 0 ||
                               strcmp(curr_layer, "{deconvolutional}") == 0) {
                        bcnn_add_deconvolutional_layer(net, n_filts, size,
                                                       stride, pad, init, a,
                                                       src_id, dst_id);
                    } else if (strcmp(curr_layer, "{depthwise-conv}") == 0 ||
                               strcmp(curr_layer, "{dw-conv}") == 0) {
                        // bcnn_add_deconvolutional_layer(net, n_filts, size,
                        // stride, pad, init, a, src_id);
                        bcnn_add_depthwise_sep_conv_layer(
                            net, size, stride, pad, 0, init, a, src_id, dst_id);
                    } else if (strcmp(curr_layer, "{activation}") == 0 ||
                               strcmp(curr_layer, "{nl}") == 0) {
                        bcnn_add_activation_layer(net, a, src_id);
                    } else if (strcmp(curr_layer, "{batchnorm}") == 0 ||
                               strcmp(curr_layer, "{bn}") == 0) {
                        bcnn_add_batchnorm_layer(net, src_id, dst_id);
                    } else if (strcmp(curr_layer, "{connected}") == 0 ||
                               strcmp(curr_layer, "{fullconnected}") == 0 ||
                               strcmp(curr_layer, "{fc}") == 0 ||
                               strcmp(curr_layer, "{ip}") == 0) {
                        bcnn_add_fullc_layer(net, outputs, init, a, 0, src_id,
                                             dst_id);
                    } else if (strcmp(curr_layer, "{softmax}") == 0) {
                        bcnn_add_softmax_layer(net, src_id, dst_id);
                    } else if (strcmp(curr_layer, "{max}") == 0 ||
                               strcmp(curr_layer, "{maxpool}") == 0) {
                        bcnn_add_maxpool_layer(net, size, stride, PADDING_CAFFE,
                                               src_id, dst_id);
                    } else if (strcmp(curr_layer, "{dropout}") == 0) {
                        bcnn_add_dropout_layer(net, rate, src_id);
                    } else {
                        fprintf(stderr, "[ERROR] Unknown Layer %s\n",
                                curr_layer);
                        return BCNN_INVALID_PARAMETER;
                    }
                    bh_free(curr_layer);
                    bh_free(src_id);
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
                bh_assert(n_tok == 2, "Wrong format option in config file",
                          BCNN_INVALID_PARAMETER);
                if (strcmp(tok[0], "task") == 0) {
                    if (strcmp(tok[1], "train") == 0)
                        param->task = (bcnn_task)0;
                    else if (strcmp(tok[1], "predict") == 0)
                        param->task = (bcnn_task)1;
                    else
                        bh_error(
                            "Invalid parameter for task, available parameters: "
                            "TRAIN, PREDICT",
                            BCNN_INVALID_PARAMETER);
                } else if (strcmp(tok[0], "data_format") == 0)
                    bh_fill_option(&param->data_format, tok[1]);
                else if (strcmp(tok[0], "input_model") == 0)
                    bh_fill_option(&param->input_model, tok[1]);
                else if (strcmp(tok[0], "output_model") == 0)
                    bh_fill_option(&param->output_model, tok[1]);
                else if (strcmp(tok[0], "out_pred") == 0)
                    bh_fill_option(&param->pred_out, tok[1]);
                else if (strcmp(tok[0], "eval_test") == 0)
                    param->eval_test = atoi(tok[1]);
                else if (strcmp(tok[0], "eval_period") == 0)
                    param->eval_period = atoi(tok[1]);
                else if (strcmp(tok[0], "save_model") == 0)
                    param->save_model = atoi(tok[1]);
                else if (strcmp(tok[0], "nb_pred") == 0)
                    param->nb_pred = atoi(tok[1]);
                else if (strcmp(tok[0], "source_train") == 0)
                    bh_fill_option(&param->train_input, tok[1]);
                else if (strcmp(tok[0], "label_train") == 0)
                    bh_fill_option(&param->path_train_label, tok[1]);
                else if (strcmp(tok[0], "source_test") == 0)
                    bh_fill_option(&param->test_input, tok[1]);
                else if (strcmp(tok[0], "label_test") == 0)
                    bh_fill_option(&param->path_test_label, tok[1]);
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
                else if (strcmp(tok[0], "src") == 0)
                    bh_fill_option(&src_id, tok[1]);
                else if (strcmp(tok[0], "dst") == 0)
                    bh_fill_option(&dst_id, tok[1]);
                else if (strcmp(tok[0], "output") == 0)
                    outputs = atoi(tok[1]);
                else if (strcmp(tok[0], "function") == 0) {
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
                    else if (strcmp(tok[1], "abs") == 0)
                        a = ABS;
                    else if (strcmp(tok[1], "none") == 0)
                        a = NONE;
                    else {
                        fprintf(stderr,
                                "[WARNING] Unknown activation type %s, going "
                                "with ReLU\n",
                                tok[1]);
                        a = RELU;
                    }
                } else if (strcmp(tok[0], "init") == 0) {
                    if (strcmp(tok[1], "xavier") == 0)
                        init = XAVIER;
                    else if (strcmp(tok[1], "msra") == 0)
                        init = MSRA;
                    else {
                        fprintf(stderr,
                                "[WARNING] Unknown init type %s, going with "
                                "xavier init\n",
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
                        fprintf(stderr,
                                "[WARNING] Unknown cost metric %s, going with "
                                "sse\n",
                                tok[1]);
                        cost = COST_SSE;
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
        std::string label_id = "label";
        bcnn_add_cost_layer(net, cost, 1.0f, src_id, (char *)label_id.c_str(),
                            dst_id);
    } else
        bh_error("Error in config file: last layer must be a cost layer",
                 BCNN_INVALID_PARAMETER);
    bh_free(curr_layer);
    bh_free(src_id);
    bh_free(dst_id);
    fclose(file);

    param->eval_period = (param->eval_period > 0 ? param->eval_period : 100);

    fflush(stderr);
    return 0;
}

static int _free_param(bcnncl_param *param) {
    bh_free(param->input_model);
    bh_free(param->output_model);
    bh_free(param->pred_out);
    bh_free(param->train_input);
    bh_free(param->test_input);
    bh_free(param->data_format);
    bh_free(param->path_train_label);
    bh_free(param->path_test_label);
    return 0;
}

static int _write_model(bcnn_net *net, char *filename) {
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

    for (i = 0; i < net->num_nodes; ++i) {
        layer = net->nodes[i].layer;
        if (layer->type == CONVOLUTIONAL || layer->type == DECONVOLUTIONAL ||
            layer->type == DEPTHWISE_CONV || layer->type == FULL_CONNECTED) {
            fwrite(layer->bias, sizeof(float), layer->bias_size, fp);
            fwrite(layer->weight, sizeof(float), layer->weights_size, fp);
        }
        if (layer->type == BATCHNORM) {
            fwrite(layer->global_mean, sizeof(float),
                   net->tensors[net->nodes[i].dst[0]].tensor.c, fp);
            fwrite(layer->global_variance, sizeof(float),
                   net->tensors[net->nodes[i].dst[0]].tensor.c, fp);
        }
    }
    fclose(fp);
    return 0;
}

int main(int argc, char **argv) {
    bcnn_net *net = NULL;
    bcnncl_param param = {0};

    if (argc < 5) {
        fprintf(stderr,
                "Usage: .exe <prototxt> <caffemodel> <bcnnconf> <bcnnmodel>\n");
        return -1;
    }

    shared_ptr<caffe::Net<float> > caffe_net;

    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    caffe_net.reset(new caffe::Net<float>(argv[1], caffe::TEST));
    caffe_net->CopyTrainedLayersFrom(argv[2]);

    bcnn_init_net(&net);
    _init_from_config(net, argv[3], &param);

    const vector<caffe::shared_ptr<caffe::Layer<float> > > &caffe_layers =
        caffe_net->layers();
    const vector<string> &layer_names = caffe_net->layer_names();

    /*for (int i = 0; i < net->num_nodes; ++i) {
        fprintf(stderr, "layer %d wsz = %d bsz = %d\n", i,
    net->nodes[i].layer->weights_size,
            net->nodes[i].layer->bias_size);
    }*/

    for (size_t i = 1; i < layer_names.size(); ++i) {
        // caffe first layer is actually the input data layer
        int i_bcnn = i - 1;
        if (caffe::ConvolutionLayer<float> *caffe_layer =
                dynamic_cast<caffe::ConvolutionLayer<float> *>(
                    caffe_layers[i].get())) {
            vector<caffe::shared_ptr<caffe::Blob<float> > > &blobs =
                caffe_layer->blobs();
            caffe::Blob<float> &caffe_weight = *blobs[0];
            caffe::Blob<float> &caffe_bias = *blobs[1];
            fprintf(
                stderr,
                "Converting weights of convolution layer %s: found %d ...\n",
                layer_names[i].c_str(), caffe_weight.count());
            // fprintf(stderr, "%ld %ld %ld %ld\n", caffe_weight.num(),
            // caffe_weight.channels(), caffe_weight.height(),
            // caffe_weight.width());
            int d_sz = caffe_weight.channels() * caffe_weight.height() *
                       caffe_weight.width();
            float ratio0 = 0.0f;
            if (net->nodes[i_bcnn].layer->weights_size ==
                caffe_weight.count() /** caffe_weight.num()*/) {
                for (int n = 0; n < caffe_weight.num(); n++) {
                    for (int c = 0; c < caffe_weight.channels(); c++) {
                        for (int h = 0; h < caffe_weight.height(); h++) {
                            for (int w = 0; w < caffe_weight.width(); w++) {
                                float data;
                                if (i == 1) {
                                    data = caffe_weight.data_at(n, 2 - c, h, w);
                                } else {
                                    data = caffe_weight.data_at(n, c, h, w);
                                }
                                /*net->nodes[i_bcnn].layer->weight[n *
                                   caffe_weight.count() +
                                    (c * caffe_weight.height() + h) *
                                   caffe_weight.width() + w] = data;*/
                                /*if (data < 0.001f) {
                                    ratio0 += 1.0f;
                                }*/
                                net->nodes[i_bcnn]
                                    .layer
                                    ->weight[n * d_sz +
                                             (c * caffe_weight.height() + h) *
                                                 caffe_weight.width() +
                                             w] = data;
                            }  // width
                        }      // height
                    }          // channel
                }              // num
                // fprintf(stderr, "ratio0= %f caffe_bias.num() %d = %d\n",
                // ratio0 / caffe_weight.count(),  i, caffe_bias.num());
                for (int b = 0; b < caffe_bias.num(); b++) {
                    net->nodes[i_bcnn].layer->bias[b] =
                        caffe_bias.data_at(b, 0, 0, 0);
                }
                fprintf(stderr, "Weights of layer %s succesfully converted\n",
                        layer_names[i].c_str());
            } else {
                fprintf(stderr,
                        "[WARNING] Weights size not compatible for layer %s: "
                        "found %d * %d, expected %d skipping...\n",
                        layer_names[i].c_str(), caffe_weight.count(),
                        caffe_weight.num(),
                        net->nodes[i_bcnn].layer->weights_size);
            }
        } else if (caffe::InnerProductLayer<float> *caffe_layer =
                       dynamic_cast<caffe::InnerProductLayer<float> *>(
                           caffe_layers[i].get())) {
            vector<caffe::shared_ptr<caffe::Blob<float> > > &blobs =
                caffe_layer->blobs();
            caffe::Blob<float> &caffe_weight = *blobs[0];
            caffe::Blob<float> &caffe_bias = *blobs[1];
            fprintf(stderr,
                    "Converting weights of inner product layer %s: found %d * "
                    "%d...\n",
                    layer_names[i].c_str(), caffe_weight.channels(),
                    caffe_weight.num());
            /*bh_assert(net.nodes[i].layer->weights_size ==
               caffe_weight.channels() * caffe_weight.num(),
                "Weights size not compatible", -1);*/

            if (net->nodes[i_bcnn].layer->weights_size ==
                caffe_weight.channels() * caffe_weight.num()) {
                for (int n = 0; n < caffe_weight.num(); n++) {
                    for (int c = 0; c < caffe_weight.channels(); c++) {
                        net->nodes[i_bcnn]
                            .layer->weight[n * caffe_weight.channels() + c] =
                            caffe_weight.data_at(n, c, 0, 0);
                    }
                }
                // fprintf(stderr, "caffe_bias.count() %d = %d\n", i,
                // caffe_bias.count());
                for (int b = 0; b < caffe_bias.count(); b++) {
                    net->nodes[i_bcnn].layer->bias[b] =
                        caffe_bias.data_at(b, 0, 0, 0);
                }
                fprintf(stderr, "Weights of layer %s succesfully converted\n",
                        layer_names[i].c_str());
            } else {
                fprintf(stderr,
                        "[WARNING] Weights size not compatible for layer %s: "
                        "found %d * %d, expected %d skipping...\n",
                        layer_names[i].c_str(), caffe_weight.channels(),
                        caffe_weight.num(),
                        net->nodes[i_bcnn].layer->weights_size);
            }
        }
    }

    _write_model(net, argv[4]);

    bcnn_end_net(&net);
    _free_param(&param);

    return 0;
}