#include <string>
#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <bh/bh_log.h>
#include <bh/bh_macros.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>
#include <bh/bh_timer.h>
#include <bip/bip.h>

#include "bcnn/bcnn.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"

static void transpose_matrix(float *a, int rows, int cols) {
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

void load_yolo_weights(bcnn_net *net, char *model) {
    FILE *fp = NULL;
    fp = fopen(model, "rb");
    if (fp == NULL) {
        fprintf(stderr, "[ERROR] Could not open file %s\n", model);
    }
    int major;
    int minor;
    int revision;
    size_t nr = fread(&major, sizeof(int), 1, fp);
    nr = fread(&minor, sizeof(int), 1, fp);
    nr = fread(&revision, sizeof(int), 1, fp);
    if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000) {
        size_t lseen = 0;
        nr = fread(&lseen, sizeof(uint64_t), 1, fp);
        net->seen = lseen;
    } else {
        int iseen = 0;
        nr = fread(&iseen, sizeof(int), 1, fp);
        net->seen = iseen;
    }
    fprintf(stderr, "version %d.%d seen %d\n", major, minor, net->seen);
    int transpose = (major > 1000) || (minor > 1000);
    for (int i = 0; i < net->num_nodes; ++i) {
        bcnn_layer *layer = net->nodes[i].layer;
        if (layer->type == CONVOLUTIONAL) {
            bcnn_tensor *weights = &net->tensors[net->nodes[i].src[1]];
            bcnn_tensor *biases = &net->tensors[net->nodes[i].src[2]];
            int weights_size = bcnn_tensor_size(weights);
            int biases_size = bcnn_tensor_size(biases);
            int nb_read = fread(biases->data, sizeof(float), biases_size, fp);
            BCNN_INFO(net->log_ctx,
                      "layer= %d nbread_bias= %lu bias_size_expected= %d", i,
                      (unsigned long)nb_read, biases_size);
            if (layer->batch_norm == 1) {
                bcnn_tensor *bn_mean = &net->tensors[net->nodes[i].src[3]];
                bcnn_tensor *bn_var = &net->tensors[net->nodes[i].src[4]];
                bcnn_tensor *bn_scales = &net->tensors[net->nodes[i].src[5]];
                int bn_mean_size = bcnn_tensor_size(bn_mean);
                int bn_var_size = bcnn_tensor_size(bn_var);
                int sz = net->tensors[net->nodes[i].dst[0]].c;
                int bn_scales_size = bcnn_tensor_size(bn_scales);
                nb_read =
                    fread(bn_scales->data, sizeof(float), bn_scales_size, fp);
                BCNN_INFO(
                    net->log_ctx,
                    "layer= %d nbread_scales= %lu scales_size_expected= %d", i,
                    (unsigned long)nb_read, bn_scales_size);
                nb_read = fread(bn_mean->data, sizeof(float), sz, fp);
                BCNN_INFO(net->log_ctx,
                          "layer= %d nbread_mean= %lu mean_size_expected= "
                          "%d",
                          i, (unsigned long)nb_read, sz);
                nb_read = fread(bn_var->data, sizeof(float), sz, fp);
                BCNN_INFO(net->log_ctx,
                          "layer= %d nbread_variance= %lu "
                          "variance_size_expected= %d",
                          i, (unsigned long)nb_read, sz);
#ifdef BCNN_USE_CUDA
                bcnn_cuda_memcpy_host2dev(bn_mean->data_gpu, bn_mean->data,
                                          bn_mean_size);
                bcnn_cuda_memcpy_host2dev(bn_var->data_gpu, bn_var->data,
                                          bn_var_size);
                bcnn_cuda_memcpy_host2dev(bn_scales->data_gpu, bn_scales->data,
                                          bn_scales_size);
#endif
            }
            nb_read = fread(weights->data, sizeof(float), weights_size, fp);
            BCNN_INFO(net->log_ctx,
                      "layer= %d nbread_weight= %lu weight_size_expected= %d",
                      i, (unsigned long)nb_read, weights_size);
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_host2dev(weights->data_gpu, weights->data,
                                      weights_size);
            bcnn_cuda_memcpy_host2dev(biases->data_gpu, biases->data,
                                      biases_size);
#endif
        } else if (layer->type == FULL_CONNECTED) {
            bcnn_tensor *weights = &net->tensors[net->nodes[i].src[1]];
            bcnn_tensor *biases = &net->tensors[net->nodes[i].src[2]];
            int weights_size = bcnn_tensor_size(weights);
            int biases_size = bcnn_tensor_size(biases);
            int nb_read = fread(biases->data, sizeof(float), biases_size, fp);
            BCNN_INFO(net->log_ctx,
                      "layer= %d nbread_bias= %lu bias_size_expected= %d", i,
                      (unsigned long)nb_read, biases_size);
            nb_read = fread(weights->data, sizeof(float), weights_size, fp);
            BCNN_INFO(net->log_ctx,
                      "layer= %d nbread_weight= %lu weight_size_expected= %d",
                      i, (unsigned long)nb_read, weights_size);
            if (transpose) {
                transpose_matrix(
                    weights->data,
                    bcnn_tensor_size3d(&net->tensors[net->nodes[i].src[0]]),
                    bcnn_tensor_size3d(&net->tensors[net->nodes[i].dst[0]]));
            }
            if (layer->batch_norm == 1) {
                bcnn_tensor *bn_mean = &net->tensors[net->nodes[i].src[3]];
                bcnn_tensor *bn_var = &net->tensors[net->nodes[i].src[4]];
                bcnn_tensor *bn_scales = &net->tensors[net->nodes[i].src[5]];
                int bn_mean_size = bcnn_tensor_size(bn_mean);
                int bn_var_size = bcnn_tensor_size(bn_var);
                int bn_scales_size = bcnn_tensor_size(bn_scales);
                nb_read =
                    fread(bn_scales->data, sizeof(float), bn_scales_size, fp);
                nb_read = fread(bn_mean->data, sizeof(float), bn_mean_size, fp);
                nb_read = fread(bn_var->data, sizeof(float), bn_var_size, fp);
#ifdef BCNN_USE_CUDA
                bcnn_cuda_memcpy_host2dev(bn_mean->data_gpu, bn_mean->data,
                                          bn_mean_size);
                bcnn_cuda_memcpy_host2dev(bn_var->data_gpu, bn_var->data,
                                          bn_var_size);
                bcnn_cuda_memcpy_host2dev(bn_scales->data_gpu, bn_scales->data,
                                          bn_scales_size);
#endif
            }
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_host2dev(weights->data_gpu, weights->data,
                                      weights_size);
            bcnn_cuda_memcpy_host2dev(biases->data_gpu, biases->data,
                                      biases_size);
#endif
        } else if (layer->type == BATCHNORM) {
            bcnn_tensor *bn_mean = &net->tensors[net->nodes[i].src[1]];
            bcnn_tensor *bn_var = &net->tensors[net->nodes[i].src[2]];
            bcnn_tensor *bn_scales = &net->tensors[net->nodes[i].src[3]];
            bcnn_tensor *bn_biases = &net->tensors[net->nodes[i].src[4]];
            int sz = net->tensors[net->nodes[i].dst[0]].c;
            int nb_read = fread(bn_scales->data, sizeof(float), sz, fp);
            nb_read = fread(bn_mean->data, sizeof(float), sz, fp);
            BCNN_INFO(
                net->log_ctx,
                "batchnorm layer= %d nbread_mean= %lu mean_size_expected= %d",
                i, (unsigned long)nb_read, sz);
            nb_read = fread(bn_var->data, sizeof(float), sz, fp);
            BCNN_INFO(net->log_ctx,
                      "batchnorm layer= %d nbread_variance= %lu "
                      "variance_size_expected= %d",
                      i, (unsigned long)nb_read, sz);
// nb_read = fread(bn_biases->data, sizeof(float), sz, fp);
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_host2dev(bn_mean->data_gpu, bn_mean->data, sz);
            bcnn_cuda_memcpy_host2dev(bn_var->data_gpu, bn_var->data, sz);
            bcnn_cuda_memcpy_host2dev(bn_scales->data_gpu, bn_scales->data, sz);
// bcnn_cuda_memcpy_host2dev(bn_biases->data_gpu, bn_biases->data, sz);
#endif
        }
    }

    fclose(fp);
    return;
}

void setup_yolo_tiny_net(bcnn_net *net, int input_width, int input_height,
                         char *model) {
    int num_classes = 80;
    int boxes_per_cell = 3;
    bcnn_net_set_input_shape(net, input_width, input_height, 3, 1);
    bcnn_add_convolutional_layer(net, 16, 3, 1, 1, 1, 1, XAVIER, LRELU, 0,
                                 (char *)"input", (char *)"conv1");
    bcnn_add_maxpool_layer(net, 2, 2, PADDING_SAME, (char *)"conv1",
                           (char *)"pool1");

    bcnn_add_convolutional_layer(net, 32, 3, 1, 1, 1, 1, XAVIER, LRELU, 0,
                                 (char *)"pool1", (char *)"conv2");
    bcnn_add_maxpool_layer(net, 2, 2, PADDING_SAME, (char *)"conv2",
                           (char *)"pool2");

    bcnn_add_convolutional_layer(net, 64, 3, 1, 1, 1, 1, XAVIER, LRELU, 0,
                                 (char *)"pool2", (char *)"conv3");
    bcnn_add_maxpool_layer(net, 2, 2, PADDING_SAME, (char *)"conv3",
                           (char *)"pool3");

    bcnn_add_convolutional_layer(net, 128, 3, 1, 1, 1, 1, XAVIER, LRELU, 0,
                                 (char *)"pool3", (char *)"conv4");
    bcnn_add_maxpool_layer(net, 2, 2, PADDING_SAME, (char *)"conv4",
                           (char *)"pool4");

    bcnn_add_convolutional_layer(net, 256, 3, 1, 1, 1, 1, XAVIER, LRELU, 0,
                                 (char *)"pool4", (char *)"conv5");
    bcnn_add_maxpool_layer(net, 2, 2, PADDING_SAME, (char *)"conv5",
                           (char *)"pool5");

    bcnn_add_convolutional_layer(net, 512, 3, 1, 1, 1, 1, XAVIER, LRELU, 0,
                                 (char *)"pool5", (char *)"conv6");
    bcnn_add_maxpool_layer(net, 2, 1, PADDING_SAME, (char *)"conv6",
                           (char *)"pool6");

    bcnn_add_convolutional_layer(net, 1024, 3, 1, 1, 1, 1, XAVIER, LRELU, 0,
                                 (char *)"pool6", (char *)"conv7");
    bcnn_add_convolutional_layer(net, 256, 1, 1, 0, 1, 1, XAVIER, LRELU, 0,
                                 (char *)"conv7", (char *)"conv8");
    bcnn_add_convolutional_layer(net, 512, 3, 1, 1, 1, 1, XAVIER, LRELU, 0,
                                 (char *)"conv8", (char *)"conv9");
    bcnn_add_convolutional_layer(net, (boxes_per_cell * (5 + num_classes)), 1,
                                 1, 0, 1, 0, XAVIER, NONE, 0, (char *)"conv9",
                                 (char *)"conv10");
    int mask[3] = {3, 4, 5};
    float anchors[12] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
    bcnn_add_yolo_layer(net, boxes_per_cell, num_classes, 4, 6, mask, anchors,
                        (char *)"conv10", (char *)"yolo1");
    bcnn_add_convolutional_layer(net, 128, 1, 1, 0, 1, 1, XAVIER, LRELU, 0,
                                 (char *)"conv8", (char *)"conv11");
    bcnn_add_upsample_layer(net, 2, (char *)"conv11", (char *)"conv11_up");
    bcnn_add_concat_layer(net, (char *)"conv11_up", (char *)"conv5",
                          (char *)"conv12");
    bcnn_add_convolutional_layer(net, 256, 3, 1, 1, 1, 1, XAVIER, LRELU, 0,
                                 (char *)"conv12", (char *)"conv13");
    bcnn_add_convolutional_layer(net, (boxes_per_cell * (5 + num_classes)), 1,
                                 1, 0, 1, 0, XAVIER, NONE, 0, (char *)"conv13",
                                 (char *)"conv14");
    int mask2[3] = {0, 1, 2};
    float anchors2[12] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
    bcnn_add_yolo_layer(net, boxes_per_cell, num_classes, 4, 6, mask2, anchors2,
                        (char *)"conv14", (char *)"yolo2");
    bcnn_compile_net(net, (char *)"predict");
    // Load yolo parameters
    load_yolo_weights(net, model);
}

#ifdef USE_OPENCV
void prepare_frame(cv::Mat frame, float *img, int w, int h) {
#else
void prepare_frame(unsigned char *frame, int w_frame, int h_frame, float *img,
                   int w, int h) {
#endif
    if (!img) {
        return;
    }
#ifdef USE_OPENCV
    int new_w = frame.cols;
    int new_h = frame.rows;
    if (((float)w / frame.cols) < ((float)h / frame.rows)) {
        new_w = w;
        new_h = (frame.rows * w) / frame.cols;
    } else {
        new_h = h;
        new_w = (frame.cols * h) / frame.rows;
    }
#else
    int new_w = w_frame;
    int new_h = h_frame;
    if (((float)w / w_frame) < ((float)h / h_frame)) {
        new_w = w;
        new_h = (h_frame * w) / w_frame;
    } else {
        new_h = h;
        new_w = (w_frame * h) / h_frame;
    }
#endif
    unsigned char *img_rz =
        (unsigned char *)calloc(new_w * new_h * 3, sizeof(unsigned char));
#ifdef USE_OPENCV
    bip_resize_bilinear(frame.data, frame.cols, frame.rows, frame.step, img_rz,
                        new_w, new_h, new_w * 3, 3);
#else
    bip_resize_bilinear(frame, w_frame, h_frame, w_frame * 3, img_rz, new_w,
                        new_h, new_w * 3, 3);
#endif
    unsigned char *canvas =
        (unsigned char *)calloc(w * h * 3, sizeof(unsigned char));
    for (int i = 0; i < w * h * 3; ++i) {
        canvas[i] = 128;
    }
    int x_offset = (w - new_w) / 2;
    int y_offset = (h - new_h) / 2;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < new_h; ++y) {
            for (int x = 0; x < new_w; ++x) {
                canvas[((y + y_offset) * w + (x + x_offset)) * 3 + c] =
                    img_rz[(y * new_w + x) * 3 + c];
            }
        }
    }
    bcnn_convert_img_to_float2(canvas, w, h, 3, 1.0f / 255.0f, 1, 0.0f, 0.0f,
                               0.0f, img);
    bh_free(img_rz);
    bh_free(canvas);
    return;
}

void free_detection_results(yolo_detection *dets, int num_dets) {
    for (int i = 0; i < num_dets; ++i) {
        free(dets[i].prob);
        free(dets[i].mask);
    }
}

static int nms_comparator(const void *pa, const void *pb) {
    yolo_detection a = *(yolo_detection *)pa;
    yolo_detection b = *(yolo_detection *)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0)
        return 1;
    else if (diff > 0)
        return -1;
    return 0;
}

static void do_nms_obj(yolo_detection *dets, int total, int classes,
                       float thresh) {
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            yolo_detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (i = 0; i < total; ++i) {
        dets[i].sort_class = -1;
    }

    qsort(dets, total, sizeof(yolo_detection), nms_comparator);
    for (i = 0; i < total; ++i) {
        if (dets[i].objectness == 0) {
            continue;
        }
        yolo_box a = dets[i].bbox;
        for (j = i + 1; j < total; ++j) {
            if (dets[j].objectness == 0) {
                continue;
            }
            yolo_box b = dets[j].bbox;
            if (box_iou(a, b) > thresh) {
                dets[j].objectness = 0;
                for (k = 0; k < classes; ++k) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

yolo_detection *run_inference(int w_frame, int h_frame, bcnn_net *net,
                              int *num_dets) {
    float nms_tresh = 0.45f;
#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_host2dev(net->tensors[0].data_gpu, net->tensors[0].data,
                              bcnn_tensor_size(&net->tensors[0]));
#endif
    bh_timer t = {0};
    bh_timer_start(&t);
    bcnn_forward(net);
    bh_timer_stop(&t);
    fprintf(stderr, "time= %lf msec\n", bh_timer_get_msec(&t));
    // Get yolo_detection boxes
    int ndets = 0;
    yolo_detection *dets =
        bcnn_yolo_get_detections(net, 0, w_frame, h_frame, net->input_width,
                                 net->input_height, 0.5, 1, &ndets);
    // Non max suppression
    int num_classes = net->nodes[net->num_nodes - 1].layer->classes;
    do_nms_obj(dets, ndets, num_classes, nms_tresh);
    *num_dets = ndets;
    return dets;
}

#ifdef USE_OPENCV
bool open_video(std::string video_path, cv::VideoCapture &capture) {
    if (video_path == "0") {
        capture.open(0);
    } else if (video_path == "1") {
        capture.open(1);
    } else {
        capture.open(video_path);
    }
    if (!capture.isOpened()) {
        fprintf(stderr, "Failed to open %s\n", video_path.c_str());
        return false;
    } else {
        return true;
    }
}
#endif

static std::string str_objs[80] = {
    "person",      "bicycle",    "car",          "motorbike",   "aeroplane",
    "bus",         "train",      "truck",        "boat",        "traffic",
    "fire",        "stop",       "parking",      "bench",       "bird",
    "cat",         "dog",        "horse",        "sheep",       "cow",
    "elephant",    "bear",       "zebra",        "giraffe",     "backpack",
    "umbrella",    "handbag",    "tie",          "suitcase",    "frisbee",
    "skis",        "snowboard",  "sports",       "kite",        "baseball",
    "baseball",    "skateboard", "surfboard",    "tennis",      "bottle",
    "wine",        "cup",        "fork",         "knife",       "spoon",
    "bowl",        "banana",     "apple",        "sandwich",    "orange",
    "broccoli",    "carrot",     "hot",          "pizza",       "donut",
    "cake",        "chair",      "sofa",         "pottedplant", "bed",
    "diningtable", "toilet",     "tvmonitor",    "laptop",      "mouse",
    "remote",      "keyboard",   "cell",         "microwave",   "oven",
    "toaster",     "sink",       "refrigerator", "book",        "clock",
    "vase",        "scissors",   "teddy",        "hair",        "toothbrush"};

#ifdef USE_OPENCV
void display_detections(cv::Mat &frame, yolo_detection *dets, int num_dets,
                        float thresh, int num_classes) {
    for (int i = 0; i < num_dets; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            if (dets[i].prob[j] > thresh) {
                int x_tl = (dets[i].bbox.x - dets[i].bbox.w / 2) * frame.cols;
                int y_tl = (dets[i].bbox.y - dets[i].bbox.h / 2) * frame.rows;
                /*fprintf(stderr,
                        "det %d class %d bbox x %f y %f w %f h %f x_tl %d y_tl "
                        "%d wimg %f himg %f\n",
                        i, j, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w,
                        dets[i].bbox.h, x_tl, y_tl, dets[i].bbox.w * frame.cols,
                        dets[i].bbox.h * frame.rows);*/
                cv::Rect box = cv::Rect(x_tl, y_tl, dets[i].bbox.w * frame.cols,
                                        dets[i].bbox.h * frame.rows);
                int r, g, b;
                b = (j % 6) * 51;
                g = ((80 - j) % 11) * 25;
                r = (j % 4) * 70 + 45;
                cv::rectangle(frame, box, cv::Scalar(r, g, b), 2, 8, 0);
                cv::putText(frame, str_objs[j], cv::Point(x_tl, y_tl - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(r, g, b),
                            2.0);
            }
        }
    }
}
#endif

void print_detections(int w, int h, yolo_detection *dets, int num_dets,
                      float thresh, int num_classes) {
    for (int i = 0; i < num_dets; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            if (dets[i].prob[j] > thresh) {
                int x_tl = (dets[i].bbox.x - dets[i].bbox.w / 2) * w;
                int y_tl = (dets[i].bbox.y - dets[i].bbox.h / 2) * h;
                fprintf(stderr,
                        "det %d class %s bbox x %f y %f w %f h %f x_tl %d y_tl "
                        "%d\n",
                        i, str_objs[j].c_str(), dets[i].bbox.x, dets[i].bbox.y,
                        dets[i].bbox.w, dets[i].bbox.h, x_tl, y_tl);
            }
        }
    }
}

void show_usage(int argc, char **argv) {
    fprintf(stderr, "Usage: ./%s <mode> <video path/source> <model>\n",
            argv[0]);
    fprintf(stderr,
            "\t<mode>: can either be 'img' or 'video' for video on disk or "
            "webcam stream.\n");
}

int run(int argc, char **argv) {
    // Init net
    bcnn_net *net = NULL;
    bcnn_init_net(&net);
    // Setup net and weights
    int w = 416, h = 416;
    setup_yolo_tiny_net(net, w, h, argv[3]);

    int out_sz = bcnn_tensor_size(&net->tensors[net->num_tensors - 1]);
    if (strcmp(argv[1], "video") == 0) {
#ifdef USE_OPENCV
        cv::VideoCapture cap;
        if (!open_video(argv[2], cap)) {
            return -1;
        }
        cv::Mat frame;
        cap >> frame;
        while (!frame.empty()) {
            cap >> frame;
            prepare_frame(frame, net->tensors[0].data, w, h);
            int num_dets = 0;
            yolo_detection *dets =
                run_inference(frame.cols, frame.rows, net, &num_dets);
            display_detections(frame, dets, num_dets, 0.45, 80);
            cv::imshow("yolov3-tiny example", frame);
            free_detection_results(dets, num_dets);
            free(dets);
            int q = cv::waitKey(10);
            if (q == 27) {
                break;
            }
        }
#else
        fprintf(stderr,
                "[ERROR] OpenCV is required for the webcam live example.");
        return -1;
#endif
    } else if (strcmp(argv[1], "img") == 0) {
#ifdef USE_OPENCV
        cv::Mat img = cv::imread(argv[2]);
        if (img.empty()) {
            fprintf(stderr, "[ERROR] Failed to open image %s\n", argv[2]);
            return -1;
        }
#else
        unsigned char *img = NULL;
        int w_frame, h_frame, c_frame;
        int ret = bip_load_image(argv[2], &img, &w_frame, &h_frame, &c_frame);
        if (c_frame == 1) {
            fprintf(stderr, "[ERROR] Gray images are not supported\n");
            return -1;
        }
        if (ret != BIP_SUCCESS) {
            fprintf(stderr, "[ERROR] Failed to open image %s\n", argv[2]);
            return -1;
        }
#endif
        int num_dets = 0;
#ifdef USE_OPENCV
        prepare_frame(img, net->tensors[0].data, w, h);
        yolo_detection *dets =
            run_inference(img.cols, img.rows, net, &num_dets);
#else
        prepare_frame(img, w_frame, h_frame, net->tensors[0].data, w, h);
        yolo_detection *dets = run_inference(w_frame, h_frame, net, &num_dets);
#endif
#ifdef USE_OPENCV
        display_detections(img, dets, num_dets, 0.45, 80);
        std::string in_path = argv[2];
        std::string out_path = in_path + "_dets.png";
        cv::imwrite(out_path, img);
#else
        print_detections(w_frame, h_frame, dets, num_dets, 0.45, 80);
#endif
        free_detection_results(dets, num_dets);
        free(dets);
    } else {
        fprintf(stderr, "[ERROR] Incorrect mode %s. Should be 'img' or 'video'",
                argv[1]);
        show_usage(argc, argv);
        return -1;
    }

    bcnn_end_net(&net);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        show_usage(argc, argv);
        return 1;
    }
    run(argc, argv);

    return 0;
}