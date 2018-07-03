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

#include <bh/bh_log.h>
#include <bh/bh_macros.h>
#include <bh/bh_string.h>

/* bcnn include */
#include <bcnn/bcnn.h>
#include <bcnn/bcnn_cl.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

/* Caffe proto include */
#include <caffe/proto/caffe.pb.h>

using namespace caffe;

static bool read_proto_from_text(const char* filepath,
                                 google::protobuf::Message* message) {
    std::ifstream fs(filepath, std::ifstream::in);
    if (!fs.is_open()) {
        fprintf(stderr, "[ERROR] Failed to open %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    bool success = google::protobuf::TextFormat::Parse(&input, message);

    fs.close();

    return success;
}

static bool read_proto_from_binary(const char* filepath,
                                   google::protobuf::Message* message) {
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open()) {
        fprintf(stderr, "[ERROR] Failed to open %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
}

int main(int argc, char** argv) {
    FILE *f_conf = NULL, *f_dat = NULL;

    if (argc < 5) {
        fprintf(stderr,
                "Usage: %s <prototxt> <caffemodel> <bcnnconf> <bcnnmodel>\n",
                argv[0]);
        return -1;
    }

    caffe::NetParameter proto;
    caffe::NetParameter net;

    // load
    bool s0 = read_proto_from_text(argv[1], &proto);
    if (!s0) {
        fprintf(stderr, "read_proto_from_text failed\n");
        return -1;
    }

    bool s1 = read_proto_from_binary(argv[2], &net);
    if (!s1) {
        fprintf(stderr, "read_proto_from_binary failed\n");
        return -1;
    }

    f_conf = fopen(argv[3], "wt");
    if (f_conf == NULL) {
        fprintf(stderr, "[ERROR] Failed to open %s\n", argv[3]);
        return -2;
    }
    f_dat = fopen(argv[4], "wb");
    if (f_dat == NULL) {
        fprintf(stderr, "[ERROR] Failed to open %s\n", argv[3]);
        return -3;
    }

    float unknown = -1.0f;
    int seen = 0;
    fwrite(&unknown, sizeof(float), 1, f_dat);
    fwrite(&unknown, sizeof(float), 1, f_dat);
    fwrite(&unknown, sizeof(float), 1, f_dat);
    fwrite(&seen, sizeof(int), 1, f_dat);

    for (int i = 0; i < proto.layer_size(); ++i) {
        const caffe::LayerParameter& layer = proto.layer(i);
        int layer_id;
        for (layer_id = 0; layer_id < net.layer_size(); layer_id++) {
            if (net.layer(layer_id).name() == layer.name()) {
                break;
            }
        }

        if (layer.type() == "BatchNorm") {
            const caffe::LayerParameter& binlayer = net.layer(layer_id);

            const caffe::BlobProto& mean_blob = binlayer.blobs(0);
            const caffe::BlobProto& var_blob = binlayer.blobs(1);
            fprintf(f_conf, "\n{bn}\n");
            const caffe::BatchNormParameter& batch_norm_param =
                layer.batch_norm_param();
            float eps = batch_norm_param.eps();

            std::vector<float> ones(mean_blob.data_size(), 1.f);
            fwrite(ones.data(), sizeof(float), ones.size(), f_dat);  // slope

            if (binlayer.blobs_size() < 3) {
                fwrite(mean_blob.data().data(), sizeof(float),
                       mean_blob.data_size(), f_dat);
                float tmp;
                for (int j = 0; j < var_blob.data_size(); j++) {
                    tmp = var_blob.data().data()[j] + eps;
                    fwrite(&tmp, sizeof(float), 1, f_dat);
                }
            } else {
                float scale_factor = 1 / binlayer.blobs(2).data().data()[0];
                // premultiply scale_factor to mean and variance
                float tmp;
                for (int j = 0; j < mean_blob.data_size(); j++) {
                    tmp = mean_blob.data().data()[j] * scale_factor;
                    fwrite(&tmp, sizeof(float), 1, f_dat);
                }
                for (int j = 0; j < var_blob.data_size(); j++) {
                    tmp = var_blob.data().data()[j] * scale_factor + eps;
                    fwrite(&tmp, sizeof(float), 1, f_dat);
                }
            }

            std::vector<float> zeros(mean_blob.data_size(), 0.f);
            fwrite(zeros.data(), sizeof(float), zeros.size(), f_dat);  // bias
        } else if (layer.type() == "Concat") {
            const caffe::ConcatParameter& concat_param = layer.concat_param();
            if (concat_param.axis()) {
                fprintf(stderr,
                        "[WARNING] Only concatenation along channels is "
                        "supported in current bcnn");
            }
            fprintf(f_conf, "\n{concat}\n");
        } else if (layer.type() == "Convolution") {
            const caffe::LayerParameter& binlayer = net.layer(layer_id);
            const caffe::BlobProto& weight_blob = binlayer.blobs(0);
            const caffe::ConvolutionParameter& convolution_param =
                layer.convolution_param();
            fprintf(f_conf, "\n{conv}\n");
            fprintf(f_conf, "filters=%d\n", convolution_param.num_output());
            fprintf(f_conf, "size=%d\n", convolution_param.kernel_size(0));
            fprintf(f_conf, "stride=%d\n", convolution_param.stride_size() != 0
                                               ? convolution_param.stride(0)
                                               : 1);
            fprintf(f_conf, "pad=%d\n", convolution_param.pad_size() != 0
                                            ? convolution_param.pad(0)
                                            : 0);
            // Write bias
            if (convolution_param.bias_term()) {
                const caffe::BlobProto& bias_blob = binlayer.blobs(1);
                fwrite(bias_blob.data().data(), sizeof(float),
                       bias_blob.data_size(), f_dat);
            } else {
                std::vector<float> zeros(convolution_param.num_output(), 0.0f);
                fwrite(zeros.data(), sizeof(float), zeros.size(), f_dat);
            }
            // Write weights
            const caffe::BlobProto& weights_blob = binlayer.blobs(0);
            fwrite(weights_blob.data().data(), sizeof(float),
                   weights_blob.data_size(), f_dat);
        } else if (layer.type() == "Dropout") {
            const caffe::DropoutParameter& dropout_param =
                layer.dropout_param();
            fprintf(f_conf, "\n{dropout}\n");
            float scale = 1.0f - dropout_param.dropout_ratio();
            fprintf(f_conf, " rate=%f\n", scale);

        } else if (layer.type() == "InnerProduct") {
            const caffe::LayerParameter& binlayer = net.layer(layer_id);
            const caffe::InnerProductParameter& inner_product_param =
                layer.inner_product_param();
            fprintf(f_conf, "\n{connected}\n");
            fprintf(f_conf, "output=%d\n", inner_product_param.num_output());
            // Write bias
            if (inner_product_param.bias_term()) {
                const caffe::BlobProto& bias_blob = binlayer.blobs(1);
                fwrite(bias_blob.data().data(), sizeof(float),
                       bias_blob.data_size(), f_dat);
            } else {
                std::vector<float> zeros(inner_product_param.num_output(),
                                         0.0f);
                fwrite(zeros.data(), sizeof(float), zeros.size(), f_dat);
            }
            // Write weights
            const caffe::BlobProto& weights_blob = binlayer.blobs(0);
            fwrite(weights_blob.data().data(), sizeof(float),
                   weights_blob.data_size(), f_dat);
        } else if (layer.type() == "Input") {
            const caffe::InputParameter& input_param = layer.input_param();
            const caffe::BlobShape& bs = input_param.shape(0);
            if (bs.dim_size() != 4) {
                fprintf(stderr, "[WARNING] Unexpected input shape %d\n",
                        bs.dim_size());
            }
            fprintf(f_conf, "input_width=%ld\n", bs.dim(3));
            fprintf(f_conf, "input_height=%ld\n", bs.dim(2));
            fprintf(f_conf, "input_channels=%ld\n", bs.dim(1));
        } else if (layer.type() == "Pooling") {
            const caffe::PoolingParameter& pooling_param =
                layer.pooling_param();
            fprintf(f_conf, "\n{maxpool}\n");
            fprintf(f_conf, "size=%d\n", pooling_param.kernel_size());
            fprintf(f_conf, "stride=%d\n", pooling_param.stride());
        } else if (layer.type() == "PReLU") {
            const caffe::LayerParameter& binlayer = net.layer(layer_id);
            const caffe::BlobProto& slope_blob = binlayer.blobs(0);
            fprintf(f_conf, "\n{activation}\n");
            fprintf(f_conf, "function=prelu\n");
            fwrite(slope_blob.data().data(), sizeof(float),
                   slope_blob.data_size(), f_dat);
        } else if (layer.type() == "ReLU") {
            const caffe::ReLUParameter& relu_param = layer.relu_param();
            fprintf(f_conf, "\n{activation}\n");
            fprintf(f_conf, "function=relu\n");
        } else if (layer.type() == "Softmax") {
            const caffe::SoftmaxParameter& softmax_param =
                layer.softmax_param();
            fprintf(f_conf, "\n{softmax}\n");
        }

        // Src and Dst Nodes names
        int num_src = layer.bottom_size();
        int num_dst = layer.top_size();
        if (num_src > 0) {
            fprintf(f_conf, "src=%s", layer.bottom(0).c_str());
            for (int j = 1; j < num_src; ++j) {
                fprintf(f_conf, ",%s", layer.bottom(j).c_str());
            }
            fprintf(f_conf, "\n");
        }
        if (num_dst > 0) {
            fprintf(f_conf, "dst=%s", layer.top(0).c_str());
            for (int j = 1; j < num_dst; ++j) {
                fprintf(f_conf, ",%s", layer.top(j).c_str());
            }
            fprintf(f_conf, "\n");
        }
    }

    if (f_conf) fclose(f_conf);
    if (f_dat) fclose(f_dat);

    return 0;
}
