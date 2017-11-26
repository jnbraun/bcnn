#include <string>
#include <cstring>
#include <vector>

#include <bh/bh.h>
#include <bh/bh_error.h>
#include <bh/bh_string.h>

/* bcnn include */
#include <bcnn/bcnn.h>

/* Caffe include */
#include <caffe/blob.hpp>
#include <caffe/net.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>
#include <caffe/layers/conv_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>

using namespace caffe;


typedef struct {
    char                        *train_input;       /**< Path to train file. */
    char                        *test_input;        /**< Path to test/validation file. */
    char                        *input_model;       /**< Path to input model. */
    char                        *output_model;      /**< Path to output model. */
    char                        *pred_out;          /**< Path to output prediction file. */
    bcnn_task                   task;               /**< Task to process. */
    bcnn_target                 prediction_type;    /**< Type of prediction to make. */
    bcnn_data_format            data_format;        /**< Data format. */
    int                         save_model;         /**< Periodicity of model saving. */
    int                         nb_pred;            /**< Number of samples to be predicted in test file. */
    int                         eval_period;        /**< Periodicity of evaluating the train/test error. */
    int                         eval_test;          /**< Set to 1 if evaluation of test database is asked. */
} bcnn_param;

int init_from_config(bcnn_net *net, char *config_file, bcnn_param *param)
{
    FILE *file = NULL;
    char *line = NULL, *curr_layer = NULL;
    char **tok = NULL;
    int nb_lines = 0, nb_layers = 0;
    int w = 0, h = 0, c = 0, stride = 1, pad = 0, n_filts = 1, batch_norm = 0, input_size = 0, size = 3, outputs = 0;
    bcnn_activation a = RELU;
    bcnn_weights_init init = XAVIER;
    bcnn_loss_metric cost = COST_SSE;
    float scale = 1.0f, rate = 1.0f;
    int input_shape[3] = { 0 };
    int n_tok;
    int concat_index = 0;
    int nb_connections;


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
                    bh_assert(net->input_node.w > 0 &&
                        net->input_node.h > 0 && net->input_node.c > 0,
                        "Input's width, height and channels must be > 0", BCNN_INVALID_PARAMETER);
                    bh_assert(net->input_node.b > 0, "Batch size must be > 0", BCNN_INVALID_PARAMETER);
                }
                if (strcmp(curr_layer, "{conv}") == 0 ||
                    strcmp(curr_layer, "{convolutional}") == 0) {
                    bcnn_add_convolutional_layer(net, n_filts, size, stride, pad, 0, init, a);
                }
                else if (strcmp(curr_layer, "{deconv}") == 0 ||
                    strcmp(curr_layer, "{deconvolutional}") == 0) {
                    bcnn_add_deconvolutional_layer(net, n_filts, size, stride, pad, init, a);
                }
                else if (strcmp(curr_layer, "{activation}") == 0 ||
                    strcmp(curr_layer, "{nl}") == 0) {
                    bcnn_add_activation_layer(net, a);
                }
                else if (strcmp(curr_layer, "{batchnorm}") == 0 ||
                    strcmp(curr_layer, "{bn}") == 0) {
                    bcnn_add_batchnorm_layer(net);
                }
                else if (strcmp(curr_layer, "{connected}") == 0 ||
                    strcmp(curr_layer, "{fullconnected}") == 0 ||
                    strcmp(curr_layer, "{fc}") == 0 ||
                    strcmp(curr_layer, "{ip}") == 0) {
                    bcnn_add_fullc_layer(net, outputs, init, a);
                }
                else if (strcmp(curr_layer, "{softmax}") == 0) {
                    bcnn_add_softmax_layer(net);
                }
                else if (strcmp(curr_layer, "{max}") == 0 ||
                    strcmp(curr_layer, "{maxpool}") == 0) {
                    bcnn_add_maxpool_layer(net, size, stride);
                }
                else if (strcmp(curr_layer, "{dropout}") == 0) {
                    bcnn_add_dropout_layer(net, rate);
                }
                else {
                    fprintf(stderr, "[ERROR] Unknown Layer %s\n", curr_layer);
                    return BCNN_INVALID_PARAMETER;
                }
                bh_free(curr_layer);
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
            bh_assert(n_tok == 2, "Wrong format option in config file", BCNN_INVALID_PARAMETER);
            if (strcmp(tok[0], "task") == 0) {
                if (strcmp(tok[1], "train") == 0) param->task = (bcnn_task)0;
                else if (strcmp(tok[1], "predict") == 0)  param->task = PREDICT;
                else bh_error("Invalid option for task, available options: TRAIN, PREDICT", BCNN_INVALID_PARAMETER);
            }
            else if (strcmp(tok[0], "data_format") == 0) {
                if (strcmp(tok[1], "img") == 0) param->data_format = IMG;
                else if (strcmp(tok[1], "csv") == 0) param->data_format = CSV;
                else if (strcmp(tok[1], "mnist") == 0) param->data_format = MNIST;
            }
            else if (strcmp(tok[0], "input_model") == 0) bh_fill_option(&param->input_model, tok[1]);
            else if (strcmp(tok[0], "output_model") == 0) bh_fill_option(&param->output_model, tok[1]);
            else if (strcmp(tok[0], "out_pred") == 0) bh_fill_option(&param->pred_out, tok[1]);
            else if (strcmp(tok[0], "eval_test") == 0) param->eval_test = atoi(tok[1]);
            else if (strcmp(tok[0], "eval_period") == 0) param->eval_period = atoi(tok[1]);
            else if (strcmp(tok[0], "save_model") == 0) param->save_model = atoi(tok[1]);
            else if (strcmp(tok[0], "nb_pred") == 0) param->nb_pred = atoi(tok[1]);
            else if (strcmp(tok[0], "source_train") == 0) bh_fill_option(&param->train_input, tok[1]);
            else if (strcmp(tok[0], "source_test") == 0) bh_fill_option(&param->test_input, tok[1]);
            else if (strcmp(tok[0], "dropout_rate") == 0 || strcmp(tok[0], "rate") == 0) rate = (float)atof(tok[1]);
            else if (strcmp(tok[0], "with") == 0) concat_index = atoi(tok[1]);
            else if (strcmp(tok[0], "batch_norm") == 0) batch_norm = atoi(tok[1]);
            else if (strcmp(tok[0], "filters") == 0) n_filts = atoi(tok[1]);
            else if (strcmp(tok[0], "size") == 0) size = atoi(tok[1]);
            else if (strcmp(tok[0], "stride") == 0) stride = atoi(tok[1]);
            else if (strcmp(tok[0], "pad") == 0) pad = atoi(tok[1]);
            else if (strcmp(tok[0], "output") == 0) outputs = atoi(tok[1]);
            else if (strcmp(tok[0], "function") == 0) {
                if (strcmp(tok[1], "relu") == 0) a = RELU;
                else if (strcmp(tok[1], "tanh") == 0) a = TANH;
                else if (strcmp(tok[1], "ramp") == 0) a = RAMP;
                else if (strcmp(tok[1], "clamp") == 0) a = CLAMP;
                else if (strcmp(tok[1], "softplus") == 0) a = SOFTPLUS;
                else if (strcmp(tok[1], "leaky_relu") == 0 || strcmp(tok[1], "lrelu") == 0) a = LRELU;
                else if (strcmp(tok[1], "abs") == 0) a = ABS;
                else {
                    fprintf(stderr, "[WARNING] Unknown activation type %s, going with ReLU\n", tok[1]);
                    a = RELU;
                }
            }
            else if (strcmp(tok[0], "init") == 0) {
                if (strcmp(tok[1], "xavier") == 0) init = XAVIER;
                else if (strcmp(tok[1], "msra") == 0) init = MSRA;
                else {
                    fprintf(stderr, "[WARNING] Unknown activation type %s, going with xavier init\n", tok[1]);
                    init = XAVIER;
                }
            }
            else if (strcmp(tok[0], "metric") == 0) {
                if (strcmp(tok[1], "error") == 0) cost = COST_ERROR;
                else if (strcmp(tok[1], "logloss") == 0) cost = COST_LOGLOSS;
                else if (strcmp(tok[1], "sse") == 0) cost = COST_SSE;
                else if (strcmp(tok[1], "mse") == 0) cost = COST_MSE;
                else if (strcmp(tok[1], "crps") == 0) cost = COST_CRPS;
                else if (strcmp(tok[1], "dice") == 0) cost = COST_DICE;
                else {
                    fprintf(stderr, "[WARNING] Unknown cost metric %s, going with sse\n", tok[1]);
                    cost = COST_SSE;
                }
            }
            else
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
        bcnn_add_cost_layer(net, cost, 1.0f);
    }
    else
        bh_error("Error in config file: last layer must be a cost layer", BCNN_INVALID_PARAMETER);
    bh_free(curr_layer);
    fclose(file);
    nb_connections = net->nb_connections;

    param->eval_period = (param->eval_period > 0 ? param->eval_period : 100);

    fflush(stderr);
    return 0;
}

int free_param(bcnn_param *param)
{
    bh_free(param->input_model);
    bh_free(param->output_model);
    bh_free(param->pred_out);
    bh_free(param->train_input);
    bh_free(param->test_input);
    return 0;
}


int main(int argc, char **argv)
{
    bcnn_net *net = NULL;
    bcnn_param param = { 0 };
    
    if (argc < 5) {
        fprintf(stderr, "Usage: .exe <prototxt> <caffemodel> <bcnnconf> <bcnnmodel>\n");
        return -1;
    }
    
    shared_ptr<caffe::Net<float> > caffe_net;
    
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    caffe_net.reset(new caffe::Net<float>(argv[1], caffe::TEST));
    caffe_net->CopyTrainedLayersFrom(argv[2]);

    bcnn_init_net(&net);
    init_from_config(net, argv[3], &param);

    const vector<caffe::shared_ptr<caffe::Layer<float> > >& caffe_layers = caffe_net->layers();
    const vector<string> & layer_names = caffe_net->layer_names();

    for (size_t i = 0; i < layer_names.size(); ++i) {
        if (caffe::ConvolutionLayer<float> *caffe_layer =
            dynamic_cast<caffe::ConvolutionLayer<float> *>(caffe_layers[i].get())) {
            fprintf(stderr, "Converting weights of convolution layer %s\n", layer_names[i].c_str());

            vector<caffe::shared_ptr<caffe::Blob<float> > >& blobs = caffe_layer->blobs();
            caffe::Blob<float> &caffe_weight = *blobs[0];
            caffe::Blob<float> &caffe_bias = *blobs[1];

            /*bh_assert(net.connections[i].layer->weights_size == caffe_weight.count() * caffe_weight.num(),
                "Weights size not compatible", -1);*/
            if (net->connections[i].layer->weights_size == caffe_weight.count() * caffe_weight.num()) {
                for (int n = 0; n < caffe_weight.num(); n++) {
                    for (int c = 0; c < caffe_weight.channels(); c++) {
                        for (int h = 0; h < caffe_weight.height(); h++) {
                            for (int w = 0; w < caffe_weight.width(); w++) {
                                float data;
                                if (i == 0) {
                                    data = caffe_weight.data_at(n, 2 - c, h, w);
                                }
                                else {
                                    data = caffe_weight.data_at(n, c, h, w);
                                }
                                net->connections[i].layer->weight[n * caffe_weight.count() +
                                    (c * caffe_weight.height() + h) * caffe_weight.width() + w] = data;
                            } // width
                        } // height
                    } // channel
                } // num
                for (int b = 0; b < caffe_bias.count(); b++) {
                    net->connections[i].layer->bias[b] = caffe_bias.data_at(b, 0, 0, 0);
                }
                fprintf(stderr, "Weights of layer %s succesfully converted\n", layer_names[i].c_str());
            }
            else {
                fprintf(stderr, "[WARNING] Weights size not compatible for layer %s: skipping...",
                    layer_names[i].c_str());
            }
        }
        else if (caffe::InnerProductLayer<float> *caffe_layer =
            dynamic_cast<caffe::InnerProductLayer<float> *>(caffe_layers[i].get())) {
            fprintf(stderr, "Converting weights of inner product layer %s\n", layer_names[i].c_str());

            vector<caffe::shared_ptr<caffe::Blob<float> > >& blobs = caffe_layer->blobs();
            caffe::Blob<float> &caffe_weight = *blobs[0];
            caffe::Blob<float> &caffe_bias = *blobs[1];

            /*bh_assert(net.connections[i].layer->weights_size == caffe_weight.channels() * caffe_weight.num(),
                "Weights size not compatible", -1);*/

            if (net->connections[i].layer->weights_size == caffe_weight.channels() * caffe_weight.num()) {
                for (int n = 0; n < caffe_weight.num(); n++) {
                    for (int c = 0; c < caffe_weight.channels(); c++) {
                        net->connections[i].layer->weight[n * caffe_weight.channels() + c] = caffe_weight.data_at(n, c, 0, 0);
                    }
                }

                for (int b = 0; b < caffe_bias.count(); b++) {
                    net->connections[i].layer->bias[b] = caffe_bias.data_at(b, 0, 0, 0);
                }
                fprintf(stderr, "Weights of layer %s succesfully converted\n", layer_names[i].c_str());
            }
            else {
                fprintf(stderr, "[WARNING] Weights size not compatible for layer %s: skipping...",
                    layer_names[i].c_str());
            }
        }
    }

    bcnn_write_model(net, argv[4]);

    bcnn_end_net(&net);
    free_param(&param);

    return 0;
}