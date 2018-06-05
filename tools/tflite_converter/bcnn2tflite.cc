/* bcnn include */
#include <bcnn/bcnn.h>
#include <bcnn/bcnn_cl.h>
#include <bh/bh.h>
#include <bh/bh_error.h>
#include <bh/bh_string.h>

/* tflite generated flatbuffers */
#include "schema_generated.h"

void load_mnist_model(bcnn_net* net, const char* model_path) {
    bcnn_net_set_input_shape(net, 28, 28, 1, 16);
    bcnn_add_convolutional_layer(net, 32, 3, 1, 1, 0, XAVIER, RELU, 0, "input",
                                 "conv1");
    bcnn_add_batchnorm_layer(net, "conv1", "bn1");
    bcnn_add_maxpool_layer(net, 2, 2, "bn1", "pool1");
    bcnn_add_convolutional_layer(net, 32, 3, 1, 1, 0, XAVIER, RELU, 0, "pool1",
                                 "conv2");
    bcnn_add_batchnorm_layer(net, "conv2", "bn2");
    bcnn_add_maxpool_layer(net, 2, 2, "bn2", "pool2");
    bcnn_add_fullc_layer(net, 256, XAVIER, RELU, 0, "pool2", "fc1");
    bcnn_add_batchnorm_layer(net, "fc1", "bn3");
    bcnn_add_fullc_layer(net, 10, XAVIER, RELU, 0, "bn3", "fc2");
    bcnn_add_softmax_layer(net, "fc2", "softmax");
    bcnn_load_model(net, model_path);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::Tensor>>>
export_tensors(bcnn_net* net, FlatBufferBuilder* builder,
               std::vector<const bcnn_tensor*>* buffers_to_write) {
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensor_vector;
    tensor_vector.reserve(net->num_tensors);
    for (int i = 0; i < net->num_tensors; ++i) {
        int buffer_index = buffers_to_write->size();
        buffers_to_write->push_back(&net->tensors[i]);
        // shape: [batch size, height, width, number of channels] (That's
        // Tensorflow's NHWC)
        std::vector<int> shape = {net->tensors[i].n, net->tensors[i].c,
                                  net->tensors[i].h, net->tensors[i].w};
        tensor_vector[i] = tflite::CreateTensorDirect(
            *builder, &shape, tflite::TensorType_FLOAT32, buffer_index,
            net->tensors[i].name,
            /*quantization=*/0);
    }
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
    outputs.push_back(net->tensors[net->num_tensors - 1]);
    return builder->CreateVector<int32_t>(outputs);
}

static const std::map<bcnn_layer_type, tflite::BuiltinOperator>
    bcnn_tflite_ops_map = {{CONVOLUTIONAL, BuiltinOperator_CONV_2D},
                           {DECONVOLUTIONAL, BuiltinOperator_TRANSPOSE_CONV},
                           {DEPTHWISE_CONV, BuiltinOperator_DEPTHWISE_CONV_2D},
                           {FULL_CONNECTED, BuiltinOperator_FULLY_CONNECTED},
                           {MAXPOOL, BuiltinOperator_MAX_POOL_2D},
                           {AVGPOOL, BuiltinOperator_AVERAGE_POOL_2D},
                           {SOFTMAX, BuiltinOperator_SOFTMAX},
                           {CONCAT, BuiltinOperator_CONCATENATION}};

static const std::map<bcnn_activation, BuiltinOperator> bcnn_tflite_act_map = {
    {RELU, BuiltinOperator_RELU},
    {LOGISTIC, BuiltinOperator_LOGISTIC},
    {PRELU, BuiltinOperator_PRELU}};

flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>>>
export_operator_codes(
    bcnn_net* net,
    const std::map<bcnn_layer_type, BuiltinOperator>& bcnn_tflite_ops_map,
    const std::map<bcnn_activation, BuiltinOperator>& bcnn_tflite_act_map,
    const FlatBufferBuilder* builder) {
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
                opcode_vector[i] = tflite::CreateOperatorCode(
                    *builder, bcnn_tflite_ops_map[net->nodes[i].layer->type], 0,
                    /*op_version=*/1);
            } else {
                fprintf(stderr, "[ERROR] The operator %d is not supported",
                        net->nodes[i].layer->type);
                return nullptr;
            }
        } else {
            if (bcnn_tflite_act_map.count(net->nodes[i].layer->type) > 0) {
                opcode_vector[i] = tflite::CreateOperatorCode(
                    *builder, bcnn_tflite_ops_map[net->nodes[i].layer->type], 0,
                    /*op_version=*/1);
            } else {
                fprintf(stderr, "[ERROR] The activation %d is not supported",
                        net->nodes[i].layer->type);
                return nullptr;
            }
        }
    }
    return builder->CreateVector(opcode_vector);
}

flatbuffers::Offset<Vector<flatbuffers::Offset<tflite::Operator>>>
export_operators(
    bcnn_net* net,
    const std::map<bcnn_layer_type, BuiltinOperator>& bcnn_tflite_ops_map,
    const std::map<bcnn_activation, BuiltinOperator>& bcnn_tflite_act_map,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type,
    const details::OperatorsMap& operators_map,
    const details::TensorsMap& tensors_map, FlatBufferBuilder* builder) {
    std::vector<flatbuffers::Offset<tflite::Operator>> op_vector;
    for (int i = 0; i < net->num_nodes; ++i) {
        // We need to check manually for each type of layer, the inputs /
        // outputs
        std::vector<int32_t> inputs;
        for (int j = 0; j < net->nodes[i].num_src; ++j) {
            inputs.push_back(net->tensors[net->nodes[i].src[j]]);
        }
        std::vector<int32_t> outputs;
        for (int j = 0; j < net->nodes[i].num_dst; ++j) {
            outputs.push_back(net->tensors[net->nodes[i].dst[j]]);
        }
    }
    // TODO WIP....
    for (const auto& op : model.operators) {
        std::vector<int32_t> inputs;
        for (const string& input : op->inputs) {
            // -1 is the ID for optional tensor in TFLite output
            int id = model.IsOptionalArray(input) ? -1 : tensors_map.at(input);
            inputs.push_back(id);
        }
        std::vector<int32_t> outputs;
        for (const string& output : op->outputs) {
            outputs.push_back(tensors_map.at(output));
        }

        int op_index = operators_map.at(GetOperatorKey(*op, ops_by_type));

        // This is a custom op unless we can find it in ops_by_type, and even
        // then
        // it could be a custom op (such as kTensorFlowUnsupported).

        auto options = Options::Custom(0);
        if (ops_by_type.count(op->type) != 0) {
            options = ops_by_type.at(op->type)->Serialize(*op, builder);
        }
        // The only supported CustomOptionFormat is FLEXBUFFERS now.
        op_vector.push_back(CreateOperator(
            *builder, op_index, builder->CreateVector(inputs),
            builder->CreateVector(outputs), options.type, options.builtin,
            options.custom, ::tflite::CustomOptionsFormat_FLEXBUFFERS));
    }

    return builder->CreateVector(op_vector);
}

void convert_bcnn_to_flatbuffers_tflite(bcnn_net* net,
                                        const char* out_tflite_path) {
    flatbuffers::FlatBufferBuilder builder(/*initial_size=*/10240);

    // Export tensors
    std::vector<const bcnn_tensor*> buffers_to_write;
    bcnn_tensor empty_array;
    buffers_to_write.push_back(&bcnn_tensor);

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

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <in.bcnnconf> <in.bcnnmodel> <out.tflite>\n",
                argv[0]);
        return -1;
    }

    bcnn_net* net = NULL;
    bcnn_init_net(&net);

    load_mnist_model(net, argv[2]);

    bcnn_end_net(&net);

    return 0;
}