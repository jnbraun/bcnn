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

#ifndef BCNN_H
#define BCNN_H

/* Cuda include */
#if 0
#ifdef BCNN_USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#ifdef BCNN_USE_CUDNN
#include <cudnn.h>
#endif  // BCNN_USE_CUDNN
#endif  // BCNN_USE_CUDA
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef BCNN_USE_AVX
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#if defined(__GNUC__) || (defined(_MSC_VER) && (_MSC_VER >= 1600))
#include <stdint.h>
#else
#if (_MSC_VER < 1300)
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#else
typedef signed __int8 int8_t;
typedef signed __int16 int16_t;
typedef signed __int32 int32_t;
typedef unsigned __int8 uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
#endif
typedef signed __int64 int64_t;
typedef unsigned __int64 uint64_t;
#endif

/** Convenient macros */
#define BCNN_CHECK(exp, err) \
    do {                     \
        if (!(exp)) {        \
            return (err);    \
        }                    \
    } while (0)

#define BCNN_CHECK_AND_LOG(ctx, exp, err, fmt, ...)                \
    do {                                                           \
        if (!(exp)) {                                              \
            bcnn_log((ctx), BCNN_LOG_ERROR, (fmt), ##__VA_ARGS__); \
            return (err);                                          \
        }                                                          \
    } while (0)

#define BCNN_CHECK_STATUS(s)         \
    do {                             \
        bcnn_status ret = (s);       \
        if ((ret) != BCNN_SUCCESS) { \
            return (ret);            \
        }                            \
    } while (0)

#define BCNN_ERROR(ctx, err, fmt, ...)                         \
    do {                                                       \
        bcnn_log((ctx), BCNN_LOG_ERROR, (fmt), ##__VA_ARGS__); \
        return (err);                                          \
    } while (0)

#define BCNN_INFO(ctx, fmt, ...)                              \
    do {                                                      \
        bcnn_log((ctx), BCNN_LOG_INFO, (fmt), ##__VA_ARGS__); \
    } while (0)

#define BCNN_WARNING(ctx, fmt, ...)                              \
    do {                                                         \
        bcnn_log((ctx), BCNN_LOG_WARNING, (fmt), ##__VA_ARGS__); \
    } while (0)

//#define USE_GRID
//#define USE_MASKHP
//#define USE_HPONLY
#define USE_2EYES

/**
 * \brief Enum of error codes.
 */
typedef enum {
    BCNN_SUCCESS,
    BCNN_INVALID_PARAMETER,
    BCNN_INVALID_DATA,
    BCNN_FAILED_ALLOC,
    BCNN_INTERNAL_ERROR,
    BCNN_CUDA_FAILED_ALLOC,
    BCNN_UNKNOWN_ERROR
} bcnn_status;

/**
 * \brief Enum of available tasks.
 */
typedef enum { UNKNOWN, PREDICT, TRAIN, VALID } bcnn_state;

typedef enum {
    CLASSIFICATION,
    REGRESSION,
    HEATMAP_REGRESSION,
    SEGMENTATION,
    DETECTION
} bcnn_target;

typedef enum {
    ITER_BIN,
    ITER_LIST,
    ITER_CSV,
    ITER_MNIST,
    ITER_CIFAR10,
    ITER_MULTI
} bcnn_iterator_type;

typedef struct {
    int n_samples;
    bcnn_iterator_type type;
    FILE *f_input;
    FILE *f_label;
    FILE *f_list;
    int n_iter;
    int input_width;
    int input_height;
    int input_depth;
    unsigned char *input_uchar;
    unsigned char *input_uchar2;
    unsigned char *input_uchar3;
    unsigned char *input_uchar4;
    float input_float[7];
    int label_width;
    int *label_int;
    float *label_float;
    unsigned char *label_uchar;
} bcnn_iterator;

/**
 * \brief Structure for online data augmentation parameters.
 */
typedef enum { LABEL_INT, LABEL_FLOAT, LABEL_IMG } bcnn_label_type;

/**
 * \brief Structure for online data augmentation parameters.
 */
typedef struct {
    int range_shift_x;    /**< X-shift allowed range (chosen between
                             [-range_shift_x / 2; range_shift_x / 2]). */
    int range_shift_y;    /**< Y-shift allowed range (chosen between
                             [-range_shift_y / 2; range_shift_y / 2]). */
    int random_fliph;     /**< If !=0, randomly (with probability of 0.5) apply
                             horizontal flip to image. */
    float min_scale;      /**< Minimum scale factor allowed. */
    float max_scale;      /**< Maximum scale factor allowed. */
    float rotation_range; /**< Rotation angle allowed range (chosen between
                             [-rotation_range / 2; rotation_range / 2]).
                             Expressed in degree. */
    int min_brightness; /**< Minimum brightness factor allowed (additive factor,
                           range [-255;255]). */
    int max_brightness; /**< Maximum brightness factor allowed (additive factor,
                           range [-255;255]). */
    float min_contrast; /**< Minimum contrast allowed (mult factor). */
    float max_contrast; /**< Maximum contrast allowed (mult factor). */
    int use_precomputed;  /**< Flag set to 1 if the parameters to be applied are
                             those already set. */
    float scale;          /**< Current scale factor. */
    int shift_x;          /**< Current x-shift. */
    int shift_y;          /**< Current y-shift. */
    float rotation;       /**< Current rotation angle. */
    int brightness;       /**< Current brightness factor. */
    float contrast;       /**< Current contrast factor. */
    float max_distortion; /**< Maximum distortion factor allowed. */
    float distortion;     /**< Current distortion factor. */
    float distortion_kx;  /**< Current distortion x kernel. */
    float distortion_ky;  /**< Current distortion y kernel. */
    int apply_fliph;      /**< Current flip flag. */
    float mean_r;
    float mean_g;
    float mean_b;
    int swap_to_bgr;
    int no_input_norm;    /**< If set to 1, Input data range is not normalized
                             between [-1;1] */
    int max_random_spots; /**< Add a random number between [0;max_random_spots]
                             of saturated blobs. */
} bcnn_data_augment;

/**
 * \brief Enum of learning policies.
 */
typedef enum { CONSTANT, STEP, INV, EXP, POLY, SIGMOID } bcnn_lr_policy;

/**
 * \brief Enum of optimization methods.
 */
typedef enum { SGD, ADAM } bcnn_optimizer;

/**
 * \brief Structure to handle learner method and parameters.
 */
typedef struct {
    float momentum;      /**< Momentum parameter */
    float decay;         /**< Decay parameter */
    float learning_rate; /**< Base learning rate */
    float gamma;
    float scale;
    float power;
    float beta1; /**< Parameter for Adam optimizer */
    float beta2; /**< Parameter for Adam optimizer */
    int step;
    bcnn_optimizer optimizer; /**< Optimization method */
    bcnn_lr_policy policy;    /**< Learning rate policy */
} bcnn_learner;

/**
 * \brief Enum of available layers types.
 */
typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    DEPTHWISE_CONV, /**< Depthwise convolution */
    ACTIVATION,
    FULL_CONNECTED,
    MAXPOOL,
    AVGPOOL,
    SOFTMAX,
    DROPOUT,
    BATCHNORM,
    LRN,
    CONCAT,
    ELTWISE,
    UPSAMPLE,
    YOLO,
    RESHAPE,
    COST
} bcnn_layer_type;

/**
 * \brief Enum of available activations functions (non-linearities).
 */
typedef enum {
    NONE,
    TANH,
    RELU,
    RAMP,
    SOFTPLUS,
    LRELU, /**< Leaky relu (alpha (negative slope) set to 0.01) */
    ABS,
    CLAMP,
    PRELU, /**< Parametric ReLU */
    LOGISTIC
} bcnn_activation;

/**
 * \brief Enum of available loss functions.
 */
typedef enum {
    EUCLIDEAN_LOSS,
    LIFTED_STRUCT_SIMILARITY_SOFTMAX_LOSS
} bcnn_loss;

/**
 * \brief Enum of available error metrics.
 */
typedef enum {
    COST_ERROR,   /**< Error rate (classification only) */
    COST_LOGLOSS, /**< Multi-class Logloss (classification only) */
    COST_SSE,     /**< Sum-squared error */
    COST_MSE,     /**< Mean-squared error */
    COST_CRPS,    /**< Continuous Ranked Probability Score */
    COST_DICE     /**< Sørensen–Dice index: metric for image segmentation */
} bcnn_loss_metric;

/**
 * \brief Available padding types for pooling
 */
typedef enum {
    PADDING_SAME,
    PADDING_VALID,
    PADDING_CAFFE /**< Caffe-like padding for compatibility purposes */
} bcnn_padding;

typedef void (*bcnn_log_callback)(const char *fmt, ...);

/* Logging */

/**
 * Available log levels
 */
typedef enum {
    BCNN_LOG_INFO = 0,
    BCNN_LOG_WARNING = 1,
    BCNN_LOG_ERROR = 2,
    BCNN_LOG_SILENT = 3
} bcnn_log_level;

typedef struct bcnn_log_context {
    bcnn_log_callback fct;
    bcnn_log_level lvl;
} bcnn_log_context;

static const int align_offset_ = 32;
/**
 * Tensor struct
 */
typedef struct {
    int n;        // Batch size
    int c;        // Number of channels = depth
    int h;        // Height
    int w;        // Width
    float *data;  // Pointer to data
#ifndef BCNN_DEPLOY_ONLY
    float *grad_data;  // Pointer to gradient data
#endif
#ifdef BCNN_USE_CUDA
    float *data_gpu;  // Pointer to data on gpu
#ifndef BCNN_DEPLOY_ONLY
    float *grad_data_gpu;  // Pointer to gradient data on gpu
#endif
#endif
    int has_grad;  // if has gradient data or not
    char *name;
} bcnn_tensor;

// The different type of tensor initialization.
// This is ususally used to randomly initialize the weights/bias of one layer
typedef enum bcnn_filler_type {
    FIXED,   // Fill with constant set by value
    XAVIER,  // Xavier init
    MSRA     // MSRA init
} bcnn_filler_type;

typedef struct tensor_filler {
    int range;
    float value;
    bcnn_filler_type type;
} bcnn_tensor_filler;

void bcnn_tensor_create(bcnn_tensor *t, int n, int c, int h, int w,
                        int has_grad, char *name, int net_state);

void bcnn_tensor_fill(bcnn_tensor *t, bcnn_tensor_filler filler);

void bcnn_tensor_destroy(bcnn_tensor *t);

void bcnn_tensor_set_shape(bcnn_tensor *t, int n, int c, int h, int w,
                           int has_grad);

void bcnn_tensor_allocate(bcnn_tensor *t, int net_state);

void bcnn_tensor_free(bcnn_tensor *t);

int bcnn_tensor_size(bcnn_tensor *tensor);

int bcnn_tensor_size3d(bcnn_tensor *t);

int bcnn_tensor_size2d(bcnn_tensor *t);

struct bcnn_net;
typedef struct bcnn_net bcnn_net;

struct bcnn_node;
typedef struct bcnn_node bcnn_node;

struct bcnn_node {
    int num_src;
    int *src;  // 'num_src' tensors indexes (net->tensors array)
    int num_dst;
    int *dst;  // 'num_dst' tensors indexes (net->tensors array)
    size_t param_size;
    bcnn_layer_type type;
    void *param;
    void (*forward)(struct bcnn_net *net, struct bcnn_node *node);
    void (*backward)(struct bcnn_net *net, struct bcnn_node *node);
    void (*update)(struct bcnn_net *net, struct bcnn_node *node);
    void (*release_param)(struct bcnn_node *node);
};

struct bcnn_net {
    int input_width;
    int input_height;
    int input_channels;
    int batch_size;
    int max_batches;              /**< Maximum number of batches during training
                                   (=iterations) */
    bcnn_loss_metric loss_metric; /**< Loss metric for evaluation */
    bcnn_learner learner;         /**< Learner/optimizer parameters */
    int seen; /**< Number of instances seen by the network */
    int num_nodes;
    bcnn_node *nodes;
    int num_tensors;      /**< Number of tensors hold in the network */
    bcnn_tensor *tensors; /**< Array of tensors hold in the network */
    bcnn_target prediction_type;
    bcnn_data_augment data_aug; /**< Parameters for online data augmentation */
    bcnn_state state;
    int nb_finetune;
    char **finetune_id;
    unsigned char *input_buffer;
    int workspace_size;
    float *workspace;
#ifdef BCNN_USE_CUDA
    float *workspace_gpu;
#endif
    bcnn_log_context log_ctx;
    void *gemm_ctx;
};

/* Logging */
void bcnn_log(bcnn_log_context ctx, bcnn_log_level level, const char *fmt, ...);
void bcnn_net_set_log_context(bcnn_net *net, bcnn_log_callback fct,
                              bcnn_log_level level);

/**
 * Set the shape of the primary input tensor
 */
void bcnn_net_set_input_shape(bcnn_net *net, int input_width, int input_height,
                              int input_channels, int batch_size);

/**
 * Add extra input tensors to the network
 */
bcnn_status bcnn_net_add_input(bcnn_net *net, int w, int h, int c, char *name);

bcnn_status bcnn_net_add_node(bcnn_net *net, bcnn_node node);
bcnn_status bcnn_free_node(bcnn_node *node);
void bcnn_net_free_nodes(bcnn_net *net);
void bcnn_net_destroy_tensors(bcnn_net *net);

bcnn_status bcnn_node_add_input(bcnn_net *net, bcnn_node *node, int index);
bcnn_status bcnn_node_add_output(bcnn_net *net, bcnn_node *node, int index);

bcnn_status bcnn_net_add_tensor(bcnn_net *net, bcnn_tensor tensor);

bcnn_status bcnn_init_net(bcnn_net **net);
bcnn_status bcnn_end_net(bcnn_net **net);

int bcnn_set_param(bcnn_net *net, char *name, char *val);

bcnn_status bcnn_compile_net(bcnn_net *net);

int bcnn_iterator_initialize(bcnn_net *net, bcnn_iterator *iter,
                             char *path_input, char *path_label, char *type);
int bcnn_iterator_next(bcnn_net *net, bcnn_iterator *iter);
int bcnn_iterator_terminate(bcnn_iterator *iter);

/* Load / Write model */
bcnn_status bcnn_load_model(bcnn_net *net, char *filename);
bcnn_status bcnn_write_model(bcnn_net *net, char *filename);
/* For compatibility with older versions */
bcnn_status bcnn_load_model_legacy(bcnn_net *net, char *filename);

bcnn_status bcnn_init_workload(bcnn_net *net);
bcnn_status bcnn_free_workload(bcnn_net *net);

/* Conv layer */
bcnn_status bcnn_add_convolutional_layer(bcnn_net *net, int n, int size,
                                         int stride, int pad, int num_groups,
                                         int batch_norm, bcnn_filler_type init,
                                         bcnn_activation activation,
                                         int quantize, char *src_id,
                                         char *dst_id);

/* Deconv layer */
bcnn_status bcnn_add_deconvolutional_layer(bcnn_net *net, int n, int size,
                                           int stride, int pad,
                                           bcnn_filler_type init,
                                           bcnn_activation activation,
                                           char *src_id, char *dst_id);

/* Depthwise separable conv layer */
bcnn_status bcnn_add_depthwise_conv_layer(bcnn_net *net, int size, int stride,
                                          int pad, int batch_norm,
                                          bcnn_filler_type init,
                                          bcnn_activation activation,
                                          char *src_id, char *dst_id);

/* Batchnorm layer */
bcnn_status bcnn_add_batchnorm_layer(bcnn_net *net, char *src_id, char *dst_id);

/* Local Response normalization layer */
bcnn_status bcnn_add_lrn_layer(bcnn_net *net, int local_size, float alpha,
                               float beta, float k, char *src_id, char *dst_id);

/* Full-connected layer */
bcnn_status bcnn_add_fullc_layer(bcnn_net *net, int output_size,
                                 bcnn_filler_type init,
                                 bcnn_activation activation, int quantize,
                                 char *src_id, char *dst_id);

/* Activation layer */
bcnn_status bcnn_add_activation_layer(bcnn_net *net, bcnn_activation type,
                                      char *id);

/* Softmax layer */
bcnn_status bcnn_add_softmax_layer(bcnn_net *net, char *src_id, char *dst_id);

/* Max-Pooling layer */
bcnn_status bcnn_add_maxpool_layer(bcnn_net *net, int size, int stride,
                                   bcnn_padding padding, char *src_id,
                                   char *dst_id);

/* Average pooling layer */
bcnn_status bcnn_add_avgpool_layer(bcnn_net *net, char *src_id, char *dst_id);

/* Concat layer */
bcnn_status bcnn_add_concat_layer(bcnn_net *net, char *src_id1, char *src_id2,
                                  char *dst_id);

/* Elementwise addition layer */
bcnn_status bcnn_add_eltwise_layer(bcnn_net *net, bcnn_activation activation,
                                   char *src_id1, char *src_id2, char *dst_id);

/* Dropout layer */
bcnn_status bcnn_add_dropout_layer(bcnn_net *net, float rate, char *id);

/* Upsample layer */
bcnn_status bcnn_add_upsample_layer(bcnn_net *net, int size, char *src_id,
                                    char *dst_id);

/* Cost layer */
bcnn_status bcnn_add_cost_layer(bcnn_net *net, bcnn_loss loss,
                                bcnn_loss_metric loss_metric, float scale,
                                char *src_id, char *label_id, char *dst_id);

/* YOLO */
#define BCNN_DETECTION_MAX_BOXES 50

/* TODO: move to private header */
bcnn_status bcnn_data_iter_detection(bcnn_net *net, bcnn_iterator *iter);

typedef struct { float x, y, w, h; } yolo_box;

typedef struct yolo_detection {
    yolo_box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} yolo_detection;

bcnn_status bcnn_add_yolo_layer(bcnn_net *net, int num_boxes_per_cell,
                                int classes, int coords, int total, int *mask,
                                float *anchors, char *src_id, char *dst_id);
yolo_detection *bcnn_yolo_get_detections(bcnn_net *net, int batch, int w, int h,
                                         int netw, int neth, float thresh,
                                         int relative, int *num_dets);

// Temporary put these functions here
float overlap(float x1, float w1, float x2, float w2);
float box_intersection(yolo_box a, yolo_box b);
float box_union(yolo_box a, yolo_box b);
float box_iou(yolo_box a, yolo_box b);

/* Core network routines */
int bcnn_update(bcnn_net *net);
int bcnn_sgd_optimizer(bcnn_net *net, bcnn_node *node, int batch_size,
                       float learning_rate, float momentum, float decay);
int bcnn_visualize_network(bcnn_net *net);
int bcnn_forward(bcnn_net *net);
int bcnn_backward(bcnn_net *net);

/* General routines for training / predict */
int bcnn_train_on_batch(bcnn_net *net, bcnn_iterator *iter, float *loss);
int bcnn_predict_on_batch(bcnn_net *net, bcnn_iterator *iter, float **pred,
                          float *error);

/* Free routines */
bcnn_status bcnn_free_net(bcnn_net *cnn);

/* Helpers */
int bcnn_load_image_from_csv(bcnn_net *net, char *str, int w, int h, int c,
                             unsigned char **img);
int bcnn_load_image_from_path(bcnn_net *net, char *path, int w, int h, int c,
                              unsigned char *img, int *x_shift, int *y_shift);
int bcnn_load_image_from_memory(bcnn_net *net, unsigned char *buffer,
                                int buffer_size, int w, int h, int c,
                                unsigned char **img, int *x_shift,
                                int *y_shift);
int bcnn_data_augmentation(unsigned char *img, int width, int height, int depth,
                           bcnn_data_augment *param, unsigned char *buffer);

int bcnn_iter_batch(bcnn_net *net, bcnn_iterator *iter);

int bcnn_convert_img_to_float(unsigned char *src, int w, int h, int c,
                              int no_input_norm, int swap_to_bgr, float mean_r,
                              float mean_g, float mean_b, float *dst);
// TODO replace bcnn_convert_img_to_float by version 2
void bcnn_convert_img_to_float2(unsigned char *src, int w, int h, int c,
                                float norm_coeff, int swap_to_bgr, float mean_r,
                                float mean_g, float mean_b, float *dst);

#ifdef __cplusplus
}
#endif

#endif
