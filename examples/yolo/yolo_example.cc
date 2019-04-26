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
#include "bcnn_conv_layer.h"
#include "bcnn_mat.h"
#include "bcnn_tensor.h"
#include "bcnn_utils.h"
#include "bcnn_yolo.h"

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
    cv::imwrite("toto.png", frame);
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
    /*bcnn_convert_img_to_float(canvas, w, h, 3, 1.0f / 127.5f, 1, 127.5f,
       127.5f,
                              127.5f, img);*/
    bcnn_convert_img_to_float(canvas, w, h, 3, 1.0f / 255.0f, 1, 0, 0, 0, img);
    bh_free(img_rz);
    bh_free(canvas);
    return;
}

void free_detection_results(bcnn_output_detection *dets, int num_dets) {
    for (int i = 0; i < num_dets; ++i) {
        free(dets[i].prob);
        free(dets[i].mask);
    }
}

bcnn_output_detection *run_inference(int w_frame, int h_frame, bcnn_net *net,
                                     int *num_dets) {
    float nms_tresh = 0.45f;
#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_host2dev((float *)net->tensors[0].data_gpu,
                              net->tensors[0].data,
                              bcnn_tensor_size(&net->tensors[0]));
#endif
    bh_timer t = {0};
    bh_timer_start(&t);
    bcnn_forward(net);
    bh_timer_stop(&t);
    fprintf(stderr, "time= %lf msec\n", bh_timer_get_msec(&t));
    // Get bcnn_output_detection boxes
    int nd = 0;
    bcnn_output_detection *dets =
        bcnn_yolo_get_detections(net, 0, w_frame, h_frame, net->tensors[0].w,
                                 net->tensors[0].h, 0.5, 1, &nd);
    *num_dets = nd;
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

static std::string str_objs[9] = {"paper", "scissors", "rock",
                                  "OK",    "thumb",    "index",
                                  "shape", "number6",  "number3"};

/*static std::string str_objs[80] = {
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
    "vase",        "scissors",   "teddy",        "hair",        "toothbrush"};*/

#ifdef USE_OPENCV
void display_detections(cv::Mat &frame, bcnn_output_detection *dets,
                        int num_dets, float thresh, int num_classes) {
    fprintf(stderr, "nd %d nc %d\n", num_dets, num_classes);
    for (int i = 0; i < num_dets; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            if (dets[i].prob[j] > thresh) {
                int x_tl = (dets[i].x - dets[i].w / 2) * frame.cols;
                int y_tl = (dets[i].y - dets[i].h / 2) * frame.rows;
                fprintf(stderr, "x_tl %f y_tl %f w %f h %f\n", dets[i].x,
                        dets[i].y, dets[i].w, dets[i].h);
                cv::Rect box = cv::Rect(x_tl, y_tl, dets[i].w * frame.cols,
                                        dets[i].h * frame.rows);
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
#else
void display_detections(unsigned char *img, int w, int h, int c,
                        bcnn_output_detection *dets, int num_dets, float thresh,
                        int num_classes, std::string outname) {
    uint8_t *dump_img = (uint8_t *)calloc(w * h * c, sizeof(uint8_t));
    memcpy(dump_img, img, w * h * c);
    for (int d = 0; d < num_dets; ++d) {
        for (int j = 0; j < num_classes; ++j) {
            if (dets[d].prob[j] > thresh) {
                unsigned char color[3] = {(j % 6) * 51, ((80 - j) % 11) * 25,
                                          (j % 4) * 70 + 45};
                bcnn_draw_color_box(dump_img, w, h, dets[d].x, dets[d].y,
                                    dets[d].w, dets[d].h, color);
            }
        }
    }
    bip_write_image((char *)(outname.c_str()), dump_img, w, h, c, c * w);
    bh_free(dump_img);
}
#endif

void show_usage(int argc, char **argv) {
    fprintf(stderr, "Usage: ./%s <mode> <input> <config> <model>\n", argv[0]);
    fprintf(
        stderr,
        "\t<mode>: can either be 'img' or 'video' for video on disk or "
        "webcam stream.\n"
        "\t<input>: path to video or img path or source if webcam is used.\n"
        "\t<config>: can either be 'yolov3-tiny.cfg' or "
        "'yolov3.cfg'.\n"
        "\t<model>: can either be 'yolov3-tiny.weights' or "
        "'yolov3.weights'.\n");
}

int main(int argc, char **argv) {
    if (argc < 4) {
        show_usage(argc, argv);
        return 1;
    }
    // Init net
    bcnn_net *net = NULL;
    bcnn_init_net(&net, BCNN_MODE_PREDICT);
    // Load net config and weights
    if (bcnn_load_net(net, argv[3], argv[4]) != BCNN_SUCCESS) {
        bcnn_end_net(&net);
        return -1;
    }
    if (bcnn_compile_net(net) != BCNN_SUCCESS) {
        bcnn_end_net(&net);
        return -1;
    }
    int out_sz = bcnn_tensor_size(&net->tensors[net->num_tensors - 1]);
    fprintf(stderr, "out_sz %d\n", out_sz);
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
            prepare_frame(frame, net->tensors[0].data, net->tensors[0].w,
                          net->tensors[0].h);
            int num_dets = 0;
            bcnn_output_detection *dets =
                run_inference(frame.cols, frame.rows, net, &num_dets);
            display_detections(frame, dets, num_dets, 0.45, /*80*/ 9);
            cv::imshow("yolov3 example", frame);
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
        prepare_frame(img, net->tensors[0].data, net->tensors[0].w,
                      net->tensors[0].h);
        bcnn_output_detection *dets =
            run_inference(img.cols, img.rows, net, &num_dets);
#else
        prepare_frame(img, w_frame, h_frame, net->tensors[0].data,
                      net->tensors[0].w, net->tensors[0].h);
        bcnn_output_detection *dets =
            run_inference(w_frame, h_frame, net, &num_dets);
#endif
        for (int i = 2; i < net->num_tensors; ++i) {
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host((float *)net->tensors[i].data_gpu,
                                      net->tensors[i].data,
                                      bcnn_tensor_size(&net->tensors[i]));
#endif
            fprintf(stderr, "%d : %s %f %f %f\n", i, net->tensors[i].name,
                    net->tensors[i].data[0], net->tensors[i].data[10],
                    net->tensors[i].data[100]);
        }
        std::string in_path = argv[2];
        std::string out_path = in_path + "_dets.png";
#ifdef USE_OPENCV
        display_detections(img, dets, num_dets, 0.45, 80);
        cv::imwrite(out_path, img);
#else
        display_detections(img, w_frame, h_frame, c_frame, dets, num_dets, 0.45,
                           /*80*/ 9, out_path);
#endif
        free_detection_results(dets, num_dets);
        free(dets);
    } else {
        fprintf(stderr, "[ERROR] Incorrect mode %s. Should be 'img' or 'video'",
                argv[1]);
        show_usage(argc, argv);
        bcnn_end_net(&net);
        return -1;
    }

    bcnn_end_net(&net);
    return 0;
}