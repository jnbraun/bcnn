#include <map>

/* bcnn include */
#include <bcnn/bcnn.h>
#include <bcnn/bcnn_cl.h>
#include <bh/bh.h>
#include <bh/bh_error.h>
#include <bh/bh_string.h>
#include <bip/bip.h>

/* tflite generated flatbuffers */
#include "schema_generated.h"

using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;
using flatbuffers::Vector;

typedef struct {
    bcnn_tensor* p_tensor;
    bool need_to_write;
} buffer_to_write;

void load_test_model(bcnn_net* net, char* model_path) {
    bcnn_net_set_input_shape(net, 40, 40, 1, 1);
    bcnn_add_convolutional_layer(net, 32, 3, 1, 1, 0, XAVIER, RELU, 0,
                                 (char*)"input", (char*)"conv1");
    bcnn_add_convolutional_layer(net, 32, 3, 1, 1, 0, XAVIER, RELU, 0,
                                 (char*)"conv1", (char*)"conv2");
    bcnn_add_maxpool_layer(net, 2, 2, (char*)"conv2", (char*)"pool1");
    bcnn_add_convolutional_layer(net, 64, 3, 1, 1, 0, XAVIER, RELU, 0,
                                 (char*)"pool1", (char*)"conv3");
    bcnn_add_convolutional_layer(net, 64, 3, 1, 1, 0, XAVIER, RELU, 0,
                                 (char*)"conv3", (char*)"conv4");
    bcnn_add_maxpool_layer(net, 2, 2, (char*)"conv4", (char*)"pool2");
    bcnn_add_fullc_layer(net, 128, XAVIER, RELU, 0, (char*)"pool2",
                         (char*)"fc1");
    bcnn_add_fullc_layer(net, 5, XAVIER, NONE, 0, (char*)"fc1", (char*)"fc2");
    bcnn_load_model(net, model_path);
    bcnn_compile_net(net, (char*)"predict");
    return;
}

void run_bcnn_reference(bcnn_net* net, char* img_path) {
    unsigned char* img = NULL;
    int w, h, c;
    bip_load_image(img_path, &img, &w, &h, &c);
    // bh_check(c == 1, "Error: Need gray image as input");
    // bh_check(w == 40 && h == 40, "Error: input size must be 40x40");

    bcnn_convert_img_to_float2(img, w, h, c, 1 / 127.5f, 0, 127.5f, 127.5f,
                               127.5f, net->tensors[0].data);
    for (int i = 0; i < 5; ++i) {
        fprintf(stderr, "%f ", net->tensors[0].data[i]);
    }
    fprintf(stderr, "\n");
    bcnn_forward(net);

    for (int i = 0; i < 5; ++i) {
        fprintf(stderr, "%f ", net->tensors[2].data[i]);
    }
    fprintf(stderr, "\n");
    bcnn_forward(net);
    for (int i = 0; i < 5; ++i) {
        fprintf(stderr, "%f ", net->tensors[3].data[i]);
    }
    fprintf(stderr, "\n");
    bcnn_forward(net);
    for (int i = 0; i < 5; ++i) {
        fprintf(stderr, "%f ", net->tensors[4].data[i]);
    }
    fprintf(stderr, "\n");
    bcnn_forward(net);
    for (int i = 0; i < 5; ++i) {
        fprintf(stderr, "%f ", net->tensors[5].data[i]);
    }
    fprintf(stderr, "\n");
    bcnn_forward(net);

    float* out = net->tensors[net->num_tensors - 1].data;
    for (int i = 0; i < 5; ++i) {
        fprintf(stderr, "%f ", out[i]);
    }
    fprintf(stderr, "\n");
    bh_free(img);

    return;
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::Tensor>>>
export_tensors(
    bcnn_net* net, FlatBufferBuilder* builder,
    std::vector</*bcnn_tensor**/ buffer_to_write>* buffers_to_write) {
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
            if (net->nodes[n].layer->type == FULL_CONNECTED) {
                if (net->nodes[n].src[1] == i) {
                    // Special case for weights of FullyConnected as TFlite only
                    // accepts 2-D tensor
                    shape.clear();
                    shape = {net->tensors[i].n, net->tensors[i].c};
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
    // Super hacky
    outputs.push_back(/*net->num_tensors - 1*/ 4);
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
        {CONCAT, tflite::BuiltinOperator_CONCATENATION}};

static const std::map<bcnn_activation, tflite::BuiltinOperator>
    bcnn_tflite_act_map = {{RELU, tflite::BuiltinOperator_RELU},
                           {LOGISTIC, tflite::BuiltinOperator_LOGISTIC},
                           {PRELU, tflite::BuiltinOperator_PRELU}};

static const std::map<bcnn_activation, tflite::ActivationFunctionType>
    bcnn_tflite_fused_act_map = {{NONE, tflite::ActivationFunctionType_NONE},
                                 {RELU, tflite::ActivationFunctionType_RELU}};

flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>>>
export_operator_codes(
    bcnn_net* net,
    /*const std::map<bcnn_layer_type, tflite::BuiltinOperator>&
        bcnn_tflite_ops_map,
    const std::map<bcnn_activation, tflite::BuiltinOperator>&
        bcnn_tflite_act_map,*/
    FlatBufferBuilder* builder) {
    /*std::map<bcnn_layer_type, BuiltinOperator> bcnn_tflite_ops_map;
    std::map<bcnn_activation, BuiltinOperator> bcnn_tflite_act_map;
    bcnn_tflite_ops_map[CONVOLUTIONAL] = BuiltinOperator_CONV_2D;
    bcnn_tflite_ops_map[DECONVOLUTIONAL] = BuiltinOperator_TRANSPOSE_CONV;
    bcnn_tflite_ops_map[DEPTHWISE_CONV] = BuiltinOperator_DEPTHWISE_CONV_2D;
    bcnn_tflite_ops_map[FULL_CONNECTED] = BuiltinOperator_FULLY_CONNECTED;
    bcnn_tflite_ops_map[MAXPOOL] = BuiltinOperator_MAX_POOL_2D;
    bcnn_tflite_ops_map[AVGPOOL] = BuiltinOperator_AVERAGE_POOL_2D;
    bcnn_tflite_ops_map[SOFTMAX] = BuiltinOperator_SOFTMAX;
    bcnn_tflite_ops_map[CONCAT] = BuiltinOperator_CONCATENATION;
    bcnn_tflite_act_map[RELU] = BuiltinOperator_RELU;
    bcnn_tflite_act_map[LOGISTIC] = BuiltinOperator_LOGISTIC;
    bcnn_tflite_act_map[PRELU] = BuiltinOperator_PRELU;*/

    std::vector<flatbuffers::Offset<tflite::OperatorCode>> opcode_vector;
    opcode_vector.reserve(net->num_nodes);
    for (int i = 0; i < net->num_nodes; ++i) {
        if (net->nodes[i].layer->type != ACTIVATION) {
            if (bcnn_tflite_ops_map.count(net->nodes[i].layer->type) > 0) {
                opcode_vector.push_back(tflite::CreateOperatorCode(
                    *builder, bcnn_tflite_ops_map.at(net->nodes[i].layer->type),
                    0,
                    /*op_version=*/1));
            } else {
                fprintf(stderr, "[ERROR] The operator %d is not supported",
                        net->nodes[i].layer->type);
                return builder->CreateVector(opcode_vector);
            }
        } else {
            if (bcnn_tflite_act_map.count(net->nodes[i].layer->activation) >
                0) {
                opcode_vector.push_back(tflite::CreateOperatorCode(
                    *builder, bcnn_tflite_ops_map.at(net->nodes[i].layer->type),
                    0,
                    /*op_version=*/1));
            } else {
                fprintf(stderr, "[ERROR] The activation %d is not supported",
                        net->nodes[i].layer->type);
                return builder->CreateVector(opcode_vector);
            }
        }
    }
    return builder->CreateVector(opcode_vector);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::Operator>>>
export_operators(bcnn_net* net, std::vector<buffer_to_write>* buffers_to_write,
                 FlatBufferBuilder* builder) {
    std::vector<flatbuffers::Offset<tflite::Operator>> op_vector;
    for (int i = 0; i < net->num_nodes; ++i) {
        bcnn_layer* layer = net->nodes[i].layer;
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
            if (layer->type == CONVOLUTIONAL ||
                layer->type == DECONVOLUTIONAL ||
                layer->type == DEPTHWISE_CONV ||
                layer->type == FULL_CONNECTED) {
                // Set the weights and bias buffer writable
                buffers_to_write->at(net->nodes[i].src[1]).need_to_write = true;
                buffers_to_write->at(net->nodes[i].src[2]).need_to_write = true;
            }
        }
        std::vector<int32_t> outputs;
        for (int j = 0; j < net->nodes[i].num_dst; ++j) {
            fprintf(stderr, "outputs %d tensor %d\n", j, net->nodes[i].dst[j]);
            outputs.push_back(net->nodes[i].dst[j]);
        }
        tflite::BuiltinOptions type;
        flatbuffers::Offset<void> offset;
        if (layer->type != ACTIVATION) {
            switch (layer->type) {
                case CONVOLUTIONAL: {
                    flatbuffers::Offset<tflite::Conv2DOptions> conv2d_options =
                        tflite::CreateConv2DOptions(
                            *builder, tflite::Padding_SAME, layer->stride,
                            layer->stride,
                            bcnn_tflite_fused_act_map.at(layer->activation),
                            /*dilation=*/1,
                            /*dilation=*/1);
                    type = tflite::BuiltinOptions_Conv2DOptions;
                    offset = conv2d_options.Union();
                    break;
                }
                case DECONVOLUTIONAL: {
                    flatbuffers::Offset<tflite::TransposeConvOptions>
                        trans_conv_options = tflite::CreateTransposeConvOptions(
                            *builder, tflite::Padding_SAME, layer->stride,
                            layer->stride);
                    type = tflite::BuiltinOptions_TransposeConvOptions;
                    offset = trans_conv_options.Union();
                    break;
                }
                case DEPTHWISE_CONV: {
                    flatbuffers::Offset<tflite::DepthwiseConv2DOptions>
                        dw_conv_options = tflite::CreateDepthwiseConv2DOptions(
                            *builder, tflite::Padding_SAME, layer->stride,
                            layer->stride,
                            /*depth_multiplier=*/1,
                            bcnn_tflite_fused_act_map.at(layer->activation));
                    type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
                    offset = dw_conv_options.Union();
                    break;
                }
                case FULL_CONNECTED: {
                    flatbuffers::Offset<tflite::FullyConnectedOptions>
                        fc_options = tflite::CreateFullyConnectedOptions(
                            *builder,
                            bcnn_tflite_fused_act_map.at(layer->activation));
                    type = tflite::BuiltinOptions_FullyConnectedOptions;
                    offset = fc_options.Union();
                    break;
                }
                case MAXPOOL: {
                    flatbuffers::Offset<tflite::Pool2DOptions> maxpool_options =
                        tflite::CreatePool2DOptions(
                            *builder, tflite::Padding_VALID, layer->stride,
                            layer->stride, layer->size, layer->size,
                            bcnn_tflite_fused_act_map.at(layer->activation));
                    type = tflite::BuiltinOptions_Pool2DOptions;
                    offset = maxpool_options.Union();
                    break;
                }
                case AVGPOOL: {
                    flatbuffers::Offset<tflite::Pool2DOptions> avgpool_options =
                        tflite::CreatePool2DOptions(
                            *builder, tflite::Padding_VALID, layer->stride,
                            layer->stride, layer->size, layer->size,
                            bcnn_tflite_fused_act_map.at(layer->activation));
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
                    /*case PRELU:
                        // TODO: graph transformation as :PRelu(x) = Relu(x)
                       +
                        // (-alpha * Relu(-x))
                        fprintf(stderr, "PRelu not supported currently\n");
                        break;*/
            }
        } else {  // Activation layer
            fprintf(stderr, "Only fused activations are supported currently\n");
        }
        op_vector.push_back(
            tflite::CreateOperator(*builder, i, builder->CreateVector(inputs),
                                   builder->CreateVector(outputs), type, offset,
                                   0, tflite::CustomOptionsFormat_FLEXBUFFERS));
    }
    return builder->CreateVector(op_vector);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>>
export_buffers(
    bcnn_net* net, /*const std::vector<const Array*>& buffers_to_write*/
    std::vector</*bcnn_tensor**/ buffer_to_write>& buffers_to_write,
    FlatBufferBuilder* builder) {
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffer_vector;
    size_t index = 0;
    for (buffer_to_write buf_ptr : buffers_to_write) {
        if (buf_ptr.need_to_write) {  // weights and bias
            uint8_t* dst_data = (uint8_t*)buf_ptr.p_tensor->data;
            size_t size = buf_ptr.p_tensor->w * buf_ptr.p_tensor->h *
                          buf_ptr.p_tensor->c * buf_ptr.p_tensor->n *
                          sizeof(float);
            flatbuffers::Offset<flatbuffers::Vector<uint8_t>> data_buffer =
                builder->CreateVector(dst_data, size);
            buffer_vector.push_back(
                tflite::CreateBuffer(*builder, data_buffer));
        } else {
            flatbuffers::Offset<flatbuffers::Vector<uint8_t>> data_buffer = 0;
            buffer_vector.push_back(
                tflite::CreateBuffer(*builder, data_buffer));
        }
        index++;
    }
    return builder->CreateVector(buffer_vector);
    /*for (const Array* array_ptr : buffers_to_write) {
        const Array& array = *array_ptr;
        Offset<Vector<uint8_t>> data_buffer =
            DataBuffer::Serialize(array, builder);
        buffer_vector.push_back(CreateBuffer(*builder, data_buffer));
        index++;
    }*/
}

void convert_bcnn_to_flatbuffers_tflite(bcnn_net* net,
                                        const char* out_tflite_path) {
    flatbuffers::FlatBufferBuilder builder(/*initial_size=*/10240);

    // Export tensors
    std::vector</*bcnn_tensor**/ buffer_to_write> buffers_to_write;
    // bcnn_tensor empty_array;
    // buffers_to_write.push_back(&empty_array);

    auto tensors = export_tensors(net, &builder, &buffers_to_write);
    auto inputs = export_input_tensors(net, &builder);
    auto outputs = export_output_tensors(net, &builder);
    auto op_codes = export_operator_codes(net, /*bcnn_tflite_ops_map,
                                          bcnn_tflite_act_map,*/ &builder);
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
    // Finish builder
    // TODO Finish(); ...
    const uint8_t* buffer = builder.GetBufferPointer();
    int size = builder.GetSize();
    // Save to file
    FILE* f_tflite = NULL;
    f_tflite = fopen(out_tflite_path, "wb");
    fwrite(buffer, sizeof(uint8_t), size, f_tflite);
    fclose(f_tflite);
}

#define CHECK_REFERENCE
#ifdef CHECK_REFERENCE
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <in.bcnnmodel> <img>\n", argv[0]);
        return -1;
    }
    bcnn_net* net = NULL;
    bcnn_init_net(&net);
    load_test_model(net, argv[1]);
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
    bcnn_init_net(&net);
    load_test_model(net, argv[2]);
    convert_bcnn_to_flatbuffers_tflite(net, argv[3]);
    bcnn_end_net(&net);
    return 0;
}
#endif