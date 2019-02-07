#include <map>

/* bcnn include */
#include <bcnn/bcnn.h>
//#include <bcnn/bcnn_cl.h>
#include <bcnn_activation_layer.h>
#include <bcnn_net.h>
#include <bcnn_tensor.h>
#include <bcnn_utils.h>
#include <bh/bh_macros.h>
#include <bh/bh_string.h>
#include <bip/bip.h>
#include "bcnn_avgpool_layer.h"
#include "bcnn_conv_layer.h"
#include "bcnn_deconv_layer.h"
#include "bcnn_depthwise_conv_layer.h"
#include "bcnn_fc_layer.h"
#include "bcnn_maxpool_layer.h"

/* tflite generated flatbuffers */
#include "schema_generated.h"

using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;
using flatbuffers::Vector;

//#define CHECK_REFERENCE

typedef struct {
    bcnn_tensor* p_tensor;
    bool need_to_write;
} buffer_to_write;

typedef struct {
    char* train_input;      /**< Path to train file. */
    char* test_input;       /**< Path to test/validation file. */
    char* path_train_label; /**< Path to label train file (used for mnist format
                               only). */
    char* path_test_label;  /**< Path to label test file (used for mnist format
                               only). */
    char* input_model;      /**< Path to input model. */
    char* output_model;     /**< Path to output model. */
    char* pred_out;         /**< Path to output prediction file. */
    bcnn_loader_type data_format;
    int save_model;  /**< Periodicity of model saving. */
    int nb_pred;     /**< Number of samples to be predicted in test file. */
    int eval_period; /**< Periodicity of evaluating the train/test error. */
    int eval_test;   /**< Set to 1 if evaluation of test database is asked. */
} config_param;

// Small function to simulate a reshape function as bcnn does not have/need it
// and it is needed by tflite in somes particular cases such as FC+PRelu.
// As it is not part of a bcnn graph representation, it is the caller's
// responsability to call this function when needed.
void add_reshape_node(bcnn_net* net, int dst_n, int dst_c, int dst_h, int dst_w,
                      char* src_id, char* dst_id) {
    bcnn_node node = {0};
    bcnn_tensor dst_tensor = {0};

    if (net->num_nodes > 0) {
        int is_src_node_found = 0;
        for (int i = net->num_tensors - 1; i >= 0; --i) {
            if (strcmp(net->tensors[i].name, src_id) == 0) {
                bcnn_node_add_input(net, &node, i);
                is_src_node_found = 1;
                break;
            }
        }
        if (!is_src_node_found) {
            fprintf(stderr, "ERROR: Invalid input node name %s", src_id);
            return;
        }
    } else {
        bcnn_node_add_input(net, &node, 0);
    }
    bcnn_tensor_set_shape(&dst_tensor, dst_n, dst_c, dst_h, dst_w, 1);
    bcnn_tensor_allocate(&dst_tensor, net->mode);
    bh_strfill(&dst_tensor.name, dst_id);
    // Add node to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(net, &node, net->num_tensors - 1);
    node.type = RESHAPE;
    bcnn_net_add_node(net, node);
    return;
}

void run_bcnn_reference(bcnn_net* net, char* img_path) {
    unsigned char* img = NULL;
    int w, h, c;
    bip_load_image(img_path, &img, &w, &h, &c);
    if (net->tensors[0].c != c) {
        fprintf(stderr, "ERROR: Wrong number of channels %d. Expected %d\n", c,
                net->tensors[0].c);
        free(img);
        return;
    }
    if (net->tensors[0].w != w || net->tensors[0].h != h) {
        unsigned char* img_rz = (unsigned char*)calloc(
            net->tensors[0].w * net->tensors[0].h * net->tensors[0].c, 1);
        bip_resize_bilinear(img, w, h, w * c, img_rz, net->tensors[0].w,
                            net->tensors[0].h, net->tensors[0].w * c, c);
        fprintf(stderr, "Input resized from %d x %d to %d x %d\n", w, h,
                net->tensors[0].w, net->tensors[0].h);
        free(img);
        img = img_rz;
    }
    bcnn_convert_img_to_float(img, net->tensors[0].w, net->tensors[0].h,
                              net->tensors[0].c, 1 / 127.5f, 0, 127.5f, 127.5f,
                              127.5f, net->tensors[0].data);
    for (int i = 0; i < 5; ++i) {
        fprintf(stderr, "%f ", net->tensors[0].data[i]);
    }
    fprintf(stderr, "\n");
    bcnn_forward(net);
    bh_free(img);
#if 0
    for (int i = 2; i < net->num_tensors; ++i) {
        float* y = net->tensors[i].data;
        fprintf(stderr, "tensor %d %s %d %d %d ", i, net->tensors[i].name,
                net->tensors[i].w, net->tensors[i].h, net->tensors[i].c);
        int sp_dim = net->tensors[i].w * net->tensors[i].h;
        for (int i = 0; i < 5; ++i) {
            fprintf(stderr, "%f ", y[i * sp_dim]);
        }
        fprintf(stderr, "\n");
    }
#endif
    return;
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::Tensor>>>
export_tensors(bcnn_net* net, FlatBufferBuilder* builder,
               std::vector<buffer_to_write>* buffers_to_write) {
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensor_vector;
    tensor_vector.reserve(net->num_tensors);
    fprintf(stderr, "net->num_tensors %d\n", net->num_tensors);
    for (int i = 0; i < net->num_tensors; ++i) {
        int buffer_index = buffers_to_write->size();
        buffers_to_write->push_back(
            {.p_tensor = &net->tensors[i], .need_to_write = false});
        // shape: [batch size, height, width, number of channels] (That's
        // Tensorflow's NHWC)
        std::vector<int> shape = {net->tensors[i].n, net->tensors[i].h,
                                  net->tensors[i].w, net->tensors[i].c};
        for (int n = 0; n < net->num_nodes; ++n) {
            if (net->nodes[n].type == FULL_CONNECTED) {
                if (net->nodes[n].src[1] == i) {
                    // Special case for FullyConnected as TFlite only
                    // supports 2-D tensor as FullyConnected weights shape
                    shape.clear();
                    shape = {net->tensors[i].n, net->tensors[i].c *
                                                    net->tensors[i].h *
                                                    net->tensors[i].w};
                }
            } else if (net->nodes[n].type == ACTIVATION) {
                bcnn_activation_param* param =
                    (bcnn_activation_param*)net->nodes[n].param;
                if (param->activation == PRELU) {
                    if (net->nodes[n].src[1] == i) {
                        // Special case for Prelu as TFlite only
                        // supports 3-D tensor as Prelu slope(=alpha) shape
                        shape.clear();
                        shape = {1, 1, net->tensors[i].c * net->tensors[i].h *
                                           net->tensors[i].w};
                    }
                }
            }
        }
        tensor_vector.push_back(tflite::CreateTensorDirect(
            *builder, &shape, tflite::TensorType_FLOAT32, buffer_index,
            net->tensors[i].name,
            /*quantization=*/0));
    }
    fprintf(stderr, "tensor_vector size %ld\n", tensor_vector.size());
    return builder->CreateVector(tensor_vector);
}

flatbuffers::Offset<flatbuffers::Vector<int32_t>> export_input_tensors(
    bcnn_net* net, FlatBufferBuilder* builder) {
    std::vector<int32_t> inputs;
    // Currently, bcnn is limited to one input
    inputs.push_back(0);
    return builder->CreateVector<int32_t>(inputs);
}

flatbuffers::Offset<flatbuffers::Vector<int32_t>> export_output_tensors(
    bcnn_net* net, FlatBufferBuilder* builder) {
    std::vector<int32_t> outputs;
    // Define every tensor as output. This is done because the simple_arena
    // memory management code of tflite has obviously some bugs so this
    // will prevent tflite to re-use the same memory chunk for different
    // tensors.
    for (int i = 2; i < net->num_tensors; ++i) {
        outputs.push_back(i);
    }
    return builder->CreateVector<int32_t>(outputs);
}

static const std::map<bcnn_layer_type, tflite::BuiltinOperator>
    bcnn_tflite_ops_map = {
        {CONVOLUTIONAL, tflite::BuiltinOperator_CONV_2D},
        {DECONVOLUTIONAL, tflite::BuiltinOperator_TRANSPOSE_CONV},
        {DEPTHWISE_CONV, tflite::BuiltinOperator_DEPTHWISE_CONV_2D},
        {FULL_CONNECTED, tflite::BuiltinOperator_FULLY_CONNECTED},
        {MAXPOOL, tflite::BuiltinOperator_MAX_POOL_2D},
        {AVGPOOL, tflite::BuiltinOperator_AVERAGE_POOL_2D},
        {SOFTMAX, tflite::BuiltinOperator_SOFTMAX},
        {CONCAT, tflite::BuiltinOperator_CONCATENATION},
        {RESHAPE, tflite::BuiltinOperator_RESHAPE}};

static const std::map<bcnn_activation, tflite::BuiltinOperator>
    bcnn_tflite_act_map = {{RELU, tflite::BuiltinOperator_RELU},
                           {LOGISTIC, tflite::BuiltinOperator_LOGISTIC},
                           {PRELU, tflite::BuiltinOperator_PRELU}};

static const std::map<bcnn_activation, tflite::ActivationFunctionType>
    bcnn_tflite_fused_act_map = {{NONE, tflite::ActivationFunctionType_NONE},
                                 {RELU, tflite::ActivationFunctionType_RELU}};

flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>>>
export_operator_codes(bcnn_net* net, FlatBufferBuilder* builder) {
    std::vector<flatbuffers::Offset<tflite::OperatorCode>> opcode_vector;
    opcode_vector.reserve(net->num_nodes);
    for (int i = 0; i < net->num_nodes; ++i) {
        if (net->nodes[i].type != ACTIVATION) {
            if (bcnn_tflite_ops_map.count(net->nodes[i].type) > 0) {
                opcode_vector.push_back(tflite::CreateOperatorCode(
                    *builder, bcnn_tflite_ops_map.at(net->nodes[i].type), 0,
                    /*op_version=*/1));
            } else {
                fprintf(stderr, "[ERROR] The operator %d is not supported",
                        net->nodes[i].type);
                return builder->CreateVector(opcode_vector);
            }
        } else {
            bcnn_activation_param* param = (bcnn_activation_param*)param;
            if (bcnn_tflite_act_map.count(param->activation) > 0) {
                opcode_vector.push_back(tflite::CreateOperatorCode(
                    *builder, bcnn_tflite_act_map.at(param->activation), 0,
                    /*op_version=*/1));
            } else {
                fprintf(stderr, "[ERROR] The activation %d is not supported",
                        net->nodes[i].type);
                return builder->CreateVector(opcode_vector);
            }
        }
    }
    return builder->CreateVector(opcode_vector);
}

// This function infers the corresponding padding scheme of tflite
// considering the inputs/outputs tensors dimensions of the targeted layer.
tflite::Padding resolve_padding(bcnn_net* net, int i, int size, int stride) {
    tflite::Padding pad = tflite::Padding_SAME;
    int in_w = net->tensors[net->nodes[i].src[0]].w;
    int in_h = net->tensors[net->nodes[i].src[0]].h;
    int out_w = net->tensors[net->nodes[i].dst[0]].w;
    int out_h = net->tensors[net->nodes[i].dst[0]].h;
    int out_w_same = (in_w + stride - 1) / stride;
    int out_h_same = (in_h + stride - 1) / stride;
    int out_w_valid = (in_w - size + stride) / stride;
    int out_h_valid = (in_h - size + stride) / stride;
    fprintf(stderr, "bcnn %d same %d valid %d\n", out_w, out_w_same,
            out_w_valid);
    if (out_w_valid == out_w && out_h_valid == out_h) {
        pad = tflite::Padding_VALID;
        fprintf(stderr, "Going with VALID padding\n");
    } else {
        fprintf(stderr, "Going with SAME padding\n");
    }
    return pad;
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::Operator>>>
export_operators(bcnn_net* net, std::vector<buffer_to_write>* buffers_to_write,
                 FlatBufferBuilder* builder) {
    std::vector<flatbuffers::Offset<tflite::Operator>> op_vector;
    for (int i = 0; i < net->num_nodes; ++i) {
        // We need to check manually for each type of layer, the inputs /
        // outputs
        std::vector<int32_t> inputs;
        fprintf(stderr, "node %d\n", i);
        for (int j = 0; j < net->nodes[i].num_src; ++j) {
            fprintf(stderr, "inputs %d tensor %d w %d h %d c %d\n", j,
                    net->nodes[i].src[j], net->tensors[net->nodes[i].src[j]].w,
                    net->tensors[net->nodes[i].src[j]].h,
                    net->tensors[net->nodes[i].src[j]].c);
            inputs.push_back(net->nodes[i].src[j]);
            if (net->nodes[i].type == CONVOLUTIONAL ||
                net->nodes[i].type == DECONVOLUTIONAL ||
                net->nodes[i].type == DEPTHWISE_CONV ||
                net->nodes[i].type == FULL_CONNECTED) {
                // Set the weights and bias buffer writable
                buffers_to_write->at(net->nodes[i].src[1]).need_to_write = true;
                buffers_to_write->at(net->nodes[i].src[2]).need_to_write = true;
            } else if (net->nodes[i].type == ACTIVATION) {
                bcnn_activation_param* param =
                    (bcnn_activation_param*)net->nodes[i].param;
                if (param->activation == PRELU) {
                    buffers_to_write->at(net->nodes[i].src[1]).need_to_write =
                        true;
                }
            }
        }
        std::vector<int32_t> outputs;
        for (int j = 0; j < net->nodes[i].num_dst; ++j) {
            fprintf(stderr, "outputs %d tensor %d\n", j, net->nodes[i].dst[j]);
            outputs.push_back(net->nodes[i].dst[j]);
        }
        tflite::BuiltinOptions type;
        flatbuffers::Offset<void> offset;
        if (net->nodes[i].type != ACTIVATION) {
            switch (net->nodes[i].type) {
                case CONVOLUTIONAL: {
                    bcnn_conv_param* param = (bcnn_conv_param*)param;
                    tflite::Padding pad =
                        resolve_padding(net, i, param->size, param->stride);
                    flatbuffers::Offset<tflite::Conv2DOptions> conv2d_options =
                        tflite::CreateConv2DOptions(
                            *builder, pad, param->stride, param->stride,
                            bcnn_tflite_fused_act_map.at(param->activation),
                            /*dilation=*/1,
                            /*dilation=*/1);
                    type = tflite::BuiltinOptions_Conv2DOptions;
                    offset = conv2d_options.Union();
                    break;
                }
                case DECONVOLUTIONAL: {
                    bcnn_deconv_param* param = (bcnn_deconv_param*)param;
                    flatbuffers::Offset<tflite::TransposeConvOptions>
                        trans_conv_options = tflite::CreateTransposeConvOptions(
                            *builder, tflite::Padding_SAME, param->stride,
                            param->stride);
                    type = tflite::BuiltinOptions_TransposeConvOptions;
                    offset = trans_conv_options.Union();
                    break;
                }
                case DEPTHWISE_CONV: {
                    bcnn_depthwise_conv_param* param =
                        (bcnn_depthwise_conv_param*)param;
                    flatbuffers::Offset<tflite::DepthwiseConv2DOptions>
                        dw_conv_options = tflite::CreateDepthwiseConv2DOptions(
                            *builder, tflite::Padding_SAME, param->stride,
                            param->stride,
                            /*depth_multiplier=*/1,
                            bcnn_tflite_fused_act_map.at(param->activation));
                    type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
                    offset = dw_conv_options.Union();
                    break;
                }
                case FULL_CONNECTED: {
                    bcnn_fullc_param* param = (bcnn_fullc_param*)param;
                    flatbuffers::Offset<tflite::FullyConnectedOptions>
                        fc_options = tflite::CreateFullyConnectedOptions(
                            *builder,
                            bcnn_tflite_fused_act_map.at(param->activation));
                    type = tflite::BuiltinOptions_FullyConnectedOptions;
                    offset = fc_options.Union();
                    break;
                }
                case MAXPOOL: {
                    bcnn_maxpool_param* param = (bcnn_maxpool_param*)param;
                    tflite::Padding pad =
                        resolve_padding(net, i, param->size, param->stride);
                    flatbuffers::Offset<tflite::Pool2DOptions> maxpool_options =
                        tflite::CreatePool2DOptions(
                            *builder, pad, param->stride, param->stride,
                            param->size, param->size,
                            bcnn_tflite_fused_act_map.at(NONE));
                    type = tflite::BuiltinOptions_Pool2DOptions;
                    offset = maxpool_options.Union();
                    break;
                }
                case AVGPOOL: {
                    tflite::Padding pad = resolve_padding(
                        net, i, /*size=*/net->tensors[net->nodes[i].dst[0]].w,
                        /*stride=*/net->tensors[net->nodes[i].dst[0]].w);
                    flatbuffers::Offset<tflite::Pool2DOptions> avgpool_options =
                        tflite::CreatePool2DOptions(
                            *builder, pad, net->tensors[net->nodes[i].dst[0]].w,
                            net->tensors[net->nodes[i].dst[0]].h,
                            net->tensors[net->nodes[i].dst[0]].w,
                            net->tensors[net->nodes[i].dst[0]].h,
                            bcnn_tflite_fused_act_map.at(NONE));
                    type = tflite::BuiltinOptions_Pool2DOptions;
                    offset = avgpool_options.Union();
                    break;
                }
                case SOFTMAX: {
                    flatbuffers::Offset<tflite::SoftmaxOptions> options =
                        tflite::CreateSoftmaxOptions(*builder,
                                                     /*beta=*/1.0f);
                    type = tflite::BuiltinOptions_SoftmaxOptions;
                    offset = options.Union();
                    break;
                }
                case RESHAPE: {
                    std::vector<int32_t> new_shape = {
                        net->tensors[net->nodes[i].dst[0]].n,
                        net->tensors[net->nodes[i].dst[0]].h,
                        net->tensors[net->nodes[i].dst[0]].w,
                        net->tensors[net->nodes[i].dst[0]].c};
                    flatbuffers::Offset<tflite::ReshapeOptions>
                        reshape_options = tflite::CreateReshapeOptionsDirect(
                            *builder, &new_shape);
                    type = tflite::BuiltinOptions_ReshapeOptions;
                    offset = reshape_options.Union();
                    break;
                }
            }
            op_vector.push_back(tflite::CreateOperator(
                *builder, i, builder->CreateVector(inputs),
                builder->CreateVector(outputs), type, offset, 0,
                tflite::CustomOptionsFormat_FLEXBUFFERS));
        } else {  // Activation layer
            op_vector.push_back(tflite::CreateOperator(
                *builder, i, builder->CreateVector(inputs),
                builder->CreateVector(outputs), tflite::BuiltinOptions_NONE, 0,
                0, ::tflite::CustomOptionsFormat_FLEXBUFFERS));
        }
    }
    return builder->CreateVector(op_vector);
}

// bcnn layout: NCHW
// Tflite layout: NHWC
void convert_nchw_to_nhwc(float* src, int w, int h, int c, int n, float* dst) {
    for (int b = 0; b < n; ++b) {
        int offset = b * c * h * w;
        for (int k = 0; k < c; ++k) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    dst[offset + c * (x + w * y) + k] =
                        src[offset + w * (h * k + y) + x];
                }
            }
        }
    }
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>>
export_buffers(bcnn_net* net, std::vector<buffer_to_write>& buffers_to_write,
               FlatBufferBuilder* builder) {
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffer_vector;
    size_t index = 0;
    for (buffer_to_write buf_ptr : buffers_to_write) {
        if (buf_ptr.need_to_write) {  // weights and bias
            float* nhwc_buf =
                (float*)calloc(buf_ptr.p_tensor->w * buf_ptr.p_tensor->h *
                                   buf_ptr.p_tensor->c * buf_ptr.p_tensor->n,
                               sizeof(float));
            convert_nchw_to_nhwc(buf_ptr.p_tensor->data, buf_ptr.p_tensor->w,
                                 buf_ptr.p_tensor->h, buf_ptr.p_tensor->c,
                                 buf_ptr.p_tensor->n, nhwc_buf);
            uint8_t* dst_data = (uint8_t*)nhwc_buf;
            size_t size = buf_ptr.p_tensor->w * buf_ptr.p_tensor->h *
                          buf_ptr.p_tensor->c * buf_ptr.p_tensor->n *
                          sizeof(float);
            flatbuffers::Offset<flatbuffers::Vector<uint8_t>> data_buffer =
                builder->CreateVector(dst_data, size);
            buffer_vector.push_back(
                tflite::CreateBuffer(*builder, data_buffer));
            free(nhwc_buf);
            fprintf(stderr, "Wrote tensor %s of size %ld (%ld bytes)\n",
                    buf_ptr.p_tensor->name, size / sizeof(float), size);
        } else {
            flatbuffers::Offset<flatbuffers::Vector<uint8_t>> data_buffer = 0;
            buffer_vector.push_back(
                tflite::CreateBuffer(*builder, data_buffer));
        }
        index++;
    }
    return builder->CreateVector(buffer_vector);
}

void convert_bcnn_to_flatbuffers_tflite(bcnn_net* net,
                                        const char* out_tflite_path) {
    flatbuffers::FlatBufferBuilder builder(/*initial_size=*/10240);

    // Export tensors
    std::vector<buffer_to_write> buffers_to_write;

    auto tensors = export_tensors(net, &builder, &buffers_to_write);
    auto inputs = export_input_tensors(net, &builder);
    auto outputs = export_output_tensors(net, &builder);
    auto op_codes = export_operator_codes(net, &builder);
    auto ops = export_operators(net, &buffers_to_write, &builder);
    auto subgraph =
        tflite::CreateSubGraph(builder, tensors, inputs, outputs, ops);
    std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {subgraph};

    auto buffers = export_buffers(net, buffers_to_write, &builder);
    auto description = builder.CreateString("BCNN Converted.");
    auto new_model_location = tflite::CreateModel(
        builder, /*TFLITE_SCHEMA_VERSION=*/3, op_codes,
        builder.CreateVector(subgraphs), description, buffers);
    tflite::FinishModelBuffer(builder, new_model_location);
    const uint8_t* buffer = builder.GetBufferPointer();
    int size = builder.GetSize();
    // Save to file
    FILE* f_tflite = NULL;
    f_tflite = fopen(out_tflite_path, "wb");
    fwrite(buffer, sizeof(uint8_t), size, f_tflite);
    fclose(f_tflite);
}

int add_layer(bcnn_net* net, char* curr_layer, int stride, int pad,
              int num_groups, bcnn_padding padding_type, int n_filts, int size,
              int outputs, bcnn_activation a, float rate, bcnn_loss_metric cost,
              bcnn_filler_type init, bcnn_loss loss, char* src_id,
              char* dst_id) {
    BCNN_CHECK_AND_LOG(net->log_ctx, src_id, BCNN_INVALID_PARAMETER,
                       "Invalid input node name. "
                       "Hint: Are you sure that 'src' field is correctly "
                       "setup?");
    if (strcmp(curr_layer, "{conv}") == 0 ||
        strcmp(curr_layer, "{convolutional}") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?");
        bcnn_add_convolutional_layer(net, n_filts, size, stride, pad,
                                     num_groups, 0, init, a, 0, src_id, dst_id);
    } else if (strcmp(curr_layer, "{deconv}") == 0 ||
               strcmp(curr_layer, "{deconvolutional}") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?");
        bcnn_add_deconvolutional_layer(net, n_filts, size, stride, pad, init, a,
                                       src_id, dst_id);
    } else if (strcmp(curr_layer, "{depthwise-conv}") == 0 ||
               strcmp(curr_layer, "{dw-conv}") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?");
        bcnn_add_depthwise_conv_layer(net, size, stride, pad, 0, init, a,
                                      src_id, dst_id);
    } else if (strcmp(curr_layer, "{activation}") == 0 ||
               strcmp(curr_layer, "{nl}") == 0) {
        bcnn_add_activation_layer(net, a, src_id);
    } else if (strcmp(curr_layer, "{batchnorm}") == 0 ||
               strcmp(curr_layer, "{bn}") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?");
        bcnn_add_batchnorm_layer(net, src_id, dst_id);
    } else if (strcmp(curr_layer, "{connected}") == 0 ||
               strcmp(curr_layer, "{fullconnected}") == 0 ||
               strcmp(curr_layer, "{fc}") == 0 ||
               strcmp(curr_layer, "{ip}") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?");
        bcnn_add_fullc_layer(net, outputs, init, a, 0, src_id, dst_id);
    } else if (strcmp(curr_layer, "{softmax}") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?");
        bcnn_add_softmax_layer(net, src_id, dst_id);
    } else if (strcmp(curr_layer, "{max}") == 0 ||
               strcmp(curr_layer, "{maxpool}") == 0) {
        BCNN_CHECK_AND_LOG(net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
                           "Invalid output node name. "
                           "Hint: Are you sure that 'dst' field is "
                           "correctly setup?");
        bcnn_add_maxpool_layer(net, size, stride, padding_type, src_id, dst_id);
    } else if (strcmp(curr_layer, "{dropout}") == 0) {
        bcnn_add_dropout_layer(net, rate, src_id);
    } else if (strcmp(curr_layer, "{cost}") == 0) {
        BCNN_CHECK_AND_LOG(
            net->log_ctx, dst_id, BCNN_INVALID_PARAMETER,
            "Cost layer: invalid input node name. "
            "Hint: Are you sure that 'dst' field is correctly setup?");
        bcnn_add_cost_layer(net, loss, cost, 1.0f, src_id, (char*)"label",
                            dst_id);
    } else {
        BCNN_ERROR(net->log_ctx, BCNN_INVALID_PARAMETER, "Unknown Layer %s",
                   curr_layer);
    }
    return BCNN_SUCCESS;
}

int init_from_config(bcnn_net* net, char* config_file, config_param* param) {
    FILE* file = NULL;
    char *line = NULL, *curr_layer = NULL;
    char** tok = NULL;
    int nb_lines = 0, nb_layers = 0;
    bcnn_padding padding_type;
    int num_groups = 1;
    int stride = 1, pad = 0, n_filts = 1, size = 3, outputs = 0;
    bcnn_activation a = NONE;
    bcnn_filler_type init = XAVIER;
    bcnn_loss_metric cost = COST_SSE;
    bcnn_loss loss = EUCLIDEAN_LOSS;
    float rate = 1.0f;
    int n_tok;
    char *src_id = NULL, *dst_id = NULL;

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
                        bcnn_set_input_shape(
                            net, net->input_width, net->input_height,
                            net->input_channels, net->batch_size);
                    }
                    add_layer(net, curr_layer, stride, pad, num_groups,
                              padding_type, n_filts, size, outputs, a, rate,
                              cost, init, loss, src_id, dst_id);
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
                BCNN_CHECK_AND_LOG(net->log_ctx, (n_tok == 2),
                                   BCNN_INVALID_PARAMETER,
                                   "Wrong format option in config file");
                if (strcmp(tok[0], "task") == 0) {
                    if (strcmp(tok[1], "train") == 0)
                        net->mode = TRAIN;
                    else if (strcmp(tok[1], "predict") == 0) {
                        net->mode = PREDICT;
                    } else
                        BCNN_ERROR(
                            net->log_ctx, BCNN_INVALID_PARAMETER,
                            "Invalid parameter for task, available parameters: "
                            "TRAIN, PREDICT");
                } else if (strcmp(tok[0], "data_format") == 0) {
                    if (strcmp(tok[1], "mnist") == 0) {
                        param->data_format = BCNN_LOAD_MNIST;
                    } else if (strcmp(tok[1], "cifar10") == 0) {
                        param->data_format = BCNN_LOAD_CIFAR10;
                    } else if (strcmp(tok[1], "classif") == 0 ||
                               strcmp(tok[1], "classification") == 0) {
                        param->data_format = BCNN_LOAD_CLASSIFICATION_LIST;
                    } else if (strcmp(tok[1], "reg") == 0 ||
                               strcmp(tok[1], "regression") == 0) {
                        param->data_format = BCNN_LOAD_REGRESSION_LIST;
                    } else if (strcmp(tok[1], "detection") == 0) {
                        param->data_format = BCNN_LOAD_DETECTION_LIST;
                    } else {
                        BCNN_ERROR(net->log_ctx, BCNN_INVALID_PARAMETER,
                                   "Invalid parameter for 'data_format', "
                                   "available parameters: "
                                   "mnist, cifar10, classif, reg, detection");
                    }
                } else if (strcmp(tok[0], "input_model") == 0)
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
                else if (strcmp(tok[0], "num_groups") == 0) {
                    num_groups = atoi(tok[1]);
                } else if (strcmp(tok[0], "pad") == 0)
                    pad = atoi(tok[1]);
                else if (strcmp(tok[0], "padding_type") == 0) {
                    if (strcmp(tok[1], "same") == 0)
                        padding_type = PADDING_SAME;
                    else if (strcmp(tok[1], "valid") == 0)
                        padding_type = PADDING_VALID;
                    else if (strcmp(tok[1], "caffe") == 0)
                        padding_type = PADDING_CAFFE;
                } else if (strcmp(tok[0], "src") == 0)
                    bh_strfill(&src_id, tok[1]);
                else if (strcmp(tok[0], "dst") == 0)
                    bh_strfill(&dst_id, tok[1]);
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
                        loss = LIFTED_STRUCT_LOSS;
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
    // Add last layer
    add_layer(net, curr_layer, stride, pad, num_groups, padding_type, n_filts,
              size, outputs, a, rate, cost, init, loss, src_id, dst_id);
    bh_free(src_id);
    bh_free(dst_id);
    bh_free(curr_layer);
    fclose(file);

    param->eval_period = (param->eval_period > 0 ? param->eval_period : 100);

    fflush(stderr);
    return 0;
}

#ifdef CHECK_REFERENCE
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <in.bcnnmodel> <img>\n", argv[0]);
        return -1;
    }
    bcnn_net* net = NULL;
    bcnn_init_net(&net);
    init_from_config(net, argv[1], &param);
    bcnn_load_model(net, argv[2]);
    run_bcnn_reference(net, argv[2]);
    bcnn_end_net(&net);
    return 0;
}
#else
int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <in.bcnnconf> <in.bcnnmodel> <out.tflite>\n",
                argv[0]);
        return -1;
    }
    bcnn_net* net = NULL;
    config_param param;
    bcnn_init_net(&net);
    init_from_config(net, argv[1], &param);
    bcnn_load_model(net, argv[2]);
    convert_bcnn_to_flatbuffers_tflite(net, argv[3]);
    bcnn_end_net(&net);
    return 0;
}
#endif