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

#ifdef __cplusplus
extern "C" {
#endif

/*************************************************************************
 * Preprocessor / compiler stuff
 ************************************************************************/

#include <stddef.h>
#include <stdint.h>

/**
 * BCNN_DLL must be defined by applications that are linking against the DLL
 * version of the library. BCNN_BUILD_SHARED is defined when building the DLL /
 * shared / dynamic library.
 */
#if defined(BCNN_DLL) && defined(BCNN_BUILD_SHARED)
#error "BCNN_DLL and BCNN_BUILD_SHARED can not be both defined"
#endif

#if defined(_WIN32) && defined(BCNN_BUILD_SHARED)
/* Build as a dll */
#define BCNN_API __declspec(dllexport)
#elif defined(_WIN32) && defined(BCNN_DLL)
/* Call as dll */
#define BCNN_API __declspec(dllimport)
#elif defined(__GNUC__) && defined(BCNN_BUILD_SHARED)
/* Build as a shared / dynamic library */
#define BCNN_API __attribute__((visibility("default")))
#else
/* Build or call as a static library */
#define BCNN_API
#endif

/* Major version number */
#define BCNN_VERSION_MAJOR 0
/* Minor version number */
#define BCNN_VERSION_MINOR 2
/* Patch version number */
#define BCNN_VERSION_PATCH 0
/* Version number */
#define BCNN_VERSION \
    (BCNN_VERSION_MAJOR * 10000 + BCNN_VERSION_MINOR * 100 + BCNN_VERSION_PATCH)

/*************************************************************************
 * Forward declarations
 ************************************************************************/

/** Net struct (main BCNN object) */
typedef struct bcnn_net bcnn_net;

/** Tensor struct */
typedef struct bcnn_tensor bcnn_tensor;

/** Object detection result struct */
typedef struct bcnn_output_detection bcnn_output_detection;

/*************************************************************************
 * BCNN types
 ************************************************************************/

/**
 * Error codes.
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
 * The available modes that allow to switch between a inference-only framework
 * to a full training capable framework.
 */
typedef enum {
    BCNN_MODE_PREDICT, /* Deployment mode: Inference only (no back-propagation,
                          no groundtruth, no cost evaluation) */
    BCNN_MODE_TRAIN,   /* Training mode: Back-propagation, parameters update,
                          evaluation against ground-truth */
    BCNN_MODE_VALID    /* Evaluation mode: Forward pass and evaluation against
                          groundtruth (required a cost layer) */
} bcnn_mode;

/**
 * Data loader format
 */
typedef enum {
    BCNN_LOAD_MNIST,
    BCNN_LOAD_CIFAR10,
    BCNN_LOAD_CLASSIFICATION_LIST,
    BCNN_LOAD_REGRESSION_LIST,
    BCNN_LOAD_DETECTION_LIST,
    BCNN_NUM_LOADERS
} bcnn_loader_type;

/**
 * Enum of learning rate decay policies.
 */
typedef enum {
    BCNN_LR_DECAY_CONSTANT,
    BCNN_LR_DECAY_STEP,
    BCNN_LR_DECAY_INV,
    BCNN_LR_DECAY_EXP,
    BCNN_LR_DECAY_POLY,
    BCNN_LR_DECAY_SIGMOID
} bcnn_lr_decay;

/**
 * Enum of available layers types.
 */
typedef enum {
    BCNN_LAYER_CONV2D,
    BCNN_LAYER_TRANSPOSE_CONV2D,
    BCNN_LAYER_DEPTHWISE_CONV2D, /* Depthwise convolution */
    BCNN_LAYER_ACTIVATION,
    BCNN_LAYER_FULL_CONNECTED,
    BCNN_LAYER_MAXPOOL,
    BCNN_LAYER_AVGPOOL,
    BCNN_LAYER_SOFTMAX,
    BCNN_LAYER_DROPOUT,
    BCNN_LAYER_BATCHNORM,
    BCNN_LAYER_LRN,
    BCNN_LAYER_CONCAT,
    BCNN_LAYER_ELTWISE,
    BCNN_LAYER_UPSAMPLE,
    BCNN_LAYER_YOLOV3,
    BCNN_LAYER_RESHAPE,
    BCNN_LAYER_COST
} bcnn_layer_type;

/**
 * Enum of available activations functions (non-linearities).
 */
typedef enum {
    BCNN_ACT_NONE,
    BCNN_ACT_TANH,
    BCNN_ACT_RELU,
    BCNN_ACT_RAMP,
    BCNN_ACT_SOFTPLUS,
    BCNN_ACT_LRELU, /* Leaky relu (alpha (negative slope) set to 0.01) */
    BCNN_ACT_ABS,
    BCNN_ACT_CLAMP,
    BCNN_ACT_PRELU, /* Parametric ReLU */
    BCNN_ACT_LOGISTIC
} bcnn_activation;

/**
 *  Enum of available loss functions.
 */
typedef enum { BCNN_LOSS_EUCLIDEAN, BCNN_LOSS_LIFTED_STRUCT } bcnn_loss;

/**
 *  Enum of available error metrics.
 */
typedef enum {
    BCNN_METRIC_ERROR_RATE, /* Error rate (classification only) */
    BCNN_METRIC_LOGLOSS,    /* Multi-class Logloss (classification only) */
    BCNN_METRIC_SSE,        /* Sum-squared error */
    BCNN_METRIC_MSE,        /* Mean-squared error */
    BCNN_METRIC_CRPS,       /* Continuous Ranked Probability Score */
    BCNN_METRIC_DICE /* Sørensen–Dice index: metric for image segmentation
                      */
} bcnn_loss_metric;

/**
 * Available padding types
 *
 * Note: Currently used for pooling operation only. Convolutional-like
 * operations take explicit padding as input parameters.
 */
typedef enum {
    BCNN_PADDING_SAME,
    BCNN_PADDING_VALID,
    BCNN_PADDING_CAFFE /* Caffe-like padding for compatibility purposes */
} bcnn_padding;

/**
 *  Enum of optimization methods.
 */
typedef enum { BCNN_OPTIM_SGD, BCNN_OPTIM_ADAM } bcnn_optimizer;

/**
 * Available log levels.
 */
typedef enum {
    BCNN_LOG_INFO = 0,
    BCNN_LOG_WARNING = 1,
    BCNN_LOG_ERROR = 2,
    BCNN_LOG_SILENT = 3
} bcnn_log_level;

/**
 * The different type of tensor initialization.
 * This is ususally used to randomly initialize the weights/bias of one layer
 */
typedef enum bcnn_filler_type {
    BCNN_FILLER_FIXED,  /* Fill with constant value. For internal use only */
    BCNN_FILLER_XAVIER, /* Xavier init */
    BCNN_FILLER_MSRA    /* MSRA init */
} bcnn_filler_type;

/* Max number of bounding boxes for detection */
#define BCNN_DETECTION_MAX_BOXES 50

/* Logging callback */
typedef void (*bcnn_log_callback)(const char *fmt, ...);

/**
 * Tensor structure.
 * Data layout is NCHW.
 */
struct bcnn_tensor {
    int n;            /* Batch size */
    int c;            /* Number of channels = depth */
    int h;            /* Spatial height */
    int w;            /* Spatial width */
    float *data;      /* Pointer to data */
    float *grad_data; /* Pointer to gradient data */
#ifdef BCNN_USE_CUDA
    float *data_gpu;      /* Pointer to data on gpu */
    float *grad_data_gpu; /* Pointer to gradient data on gpu */
#endif
    int has_grad; /* If has gradient data or not */
    char *name;   /* Tensor name */
};

/**
 * Detection output struct.
 */
struct bcnn_output_detection {
    int num_classes;
    float x, y, w, h;
    float *prob;
    float *mask;
    float objectness;
};

/*************************************************************************
 * BCNN API functions
 ************************************************************************/

/**
 *  This function creates an bcnn_net instance and needs to be called
 * before any other BCNN functions applied to this bcnn_net instance. In order
 * to free the bcnn_net instance, the function bcnn_end_net needs to be called
 * before exiting the application.
 *
 * @return 'BCNN_SUCCESS' if successful initialization or 'BCNN_FAILED_ALLOC'
 * otherwise
 */
BCNN_API bcnn_status bcnn_init_net(bcnn_net **net, bcnn_mode mode);

/**
 * This function frees any allocated ressources in the bcnn_net instance
 * and destroys the instance itself (net pointer is set to NULL after being
 * freed).
 */
BCNN_API void bcnn_end_net(bcnn_net **net);

/**
 * Set the logging context.
 *
 * @param[in]   net     Pointer to net handle.
 * @param[in]   fct     Callback to user defined log function. If NULL, default
 *                      logging to stderr will be used.
 * @param[in]   level   Log level.
 */
BCNN_API void bcnn_set_log_context(bcnn_net *net, bcnn_log_callback fct,
                                   bcnn_log_level level);

/**
 * Set the shape of the primary input tensor. The primary input tensor holds the
 * default name 'input'.
 *
 * @param[in]   net         Pointer to net handle.
 * @param[in]   width       Input tensor width.
 * @param[in]   height      Input tensor height.
 * @param[in]   channels    Input tensor depth (= number of channels).
 * @param[in]   batch_size  Set the batch size (will be the same for each
 *                          tensor).
 */
BCNN_API void bcnn_set_input_shape(bcnn_net *net, int width, int height,
                                   int channels, int batch_size);

/**
 * Defines an input tensor to the network.
 *
 * @param[in]   width       Tensor width.
 * @param[in]   height      Tensor height.
 * @param[in]   channels    Tensor depth (= number of channels).
 * @param[in]   name        Tensor name.
 *
 * @return 'BCNN_SUCCESS' if successful initialization or 'BCNN_FAILED_ALLOC'
 * otherwise
 */
BCNN_API bcnn_status bcnn_add_input(bcnn_net *net, int width, int height,
                                    int channels, const char *name);

/**
 * Returns the batch size used for training / validation
 */
BCNN_API int bcnn_get_batch_size(bcnn_net *net);

/**
 * Finalizes the net configuration.
 * This function needs to be called after everything has been setup (the model
 * architecture, the dataset loader, the data augmentation, the training
 * configuration and the model weights) have been setup and before effectively
 * starting the model training or inference.
 */
BCNN_API bcnn_status bcnn_compile_net(bcnn_net *net);

/* Load the model parameters from disk */
BCNN_API bcnn_status bcnn_load_model(bcnn_net *net, const char *filename);
/* For compatibility with older versions */
BCNN_API bcnn_status bcnn_load_model_legacy(bcnn_net *net,
                                            const char *filename);

/* Write the model on disk */
BCNN_API bcnn_status bcnn_write_model(bcnn_net *net, const char *filename);

/* Setup Adam optimizer */
BCNN_API void bcnn_set_adam_optimizer(bcnn_net *net, float learning_rate,
                                      float beta1, float beta2);
/* Setup SGD optimizer with momentum */
BCNN_API void bcnn_set_sgd_optimizer(bcnn_net *net, float learning_rate,
                                     float momentum);

/* Set the learning rate decay policy */
BCNN_API void bcnn_set_learning_rate_policy(bcnn_net *net,
                                            bcnn_lr_decay decay_type,
                                            float gamma, float scale,
                                            float power, int max_batches,
                                            int step);

/* Weight regularization */
BCNN_API void bcnn_set_weight_regularizer(bcnn_net *net, float weight_decay);

/* Conv layer */
BCNN_API bcnn_status bcnn_add_convolutional_layer(
    bcnn_net *net, int n, int size, int stride, int pad, int num_groups,
    int batch_norm, bcnn_filler_type init, bcnn_activation activation,
    int quantize, const char *src_id, const char *dst_id);

/* Transposed convolution 2d layer */
BCNN_API bcnn_status bcnn_add_deconvolutional_layer(
    bcnn_net *net, int n, int size, int stride, int pad, bcnn_filler_type init,
    bcnn_activation activation, const char *src_id, const char *dst_id);

/* Depthwise convolution layer */
BCNN_API bcnn_status bcnn_add_depthwise_conv_layer(
    bcnn_net *net, int size, int stride, int pad, int batch_norm,
    bcnn_filler_type init, bcnn_activation activation, const char *src_id,
    const char *dst_id);

/* Batchnorm layer */
BCNN_API bcnn_status bcnn_add_batchnorm_layer(bcnn_net *net, const char *src_id,
                                              const char *dst_id);

/* Local Response normalization layer */
BCNN_API bcnn_status bcnn_add_lrn_layer(bcnn_net *net, int local_size,
                                        float alpha, float beta, float k,
                                        const char *src_id, const char *dst_id);

/* Fully-connected layer */
BCNN_API bcnn_status bcnn_add_fullc_layer(bcnn_net *net, int output_size,
                                          bcnn_filler_type init,
                                          bcnn_activation activation,
                                          int quantize, const char *src_id,
                                          const char *dst_id);

/* Activation layer */
BCNN_API bcnn_status bcnn_add_activation_layer(bcnn_net *net,
                                               bcnn_activation type,
                                               const char *id);

/* Softmax layer */
BCNN_API bcnn_status bcnn_add_softmax_layer(bcnn_net *net, const char *src_id,
                                            const char *dst_id);

/* Max-Pooling layer */
BCNN_API bcnn_status bcnn_add_maxpool_layer(bcnn_net *net, int size, int stride,
                                            bcnn_padding padding,
                                            const char *src_id,
                                            const char *dst_id);

/* Average pooling layer */
BCNN_API bcnn_status bcnn_add_avgpool_layer(bcnn_net *net, const char *src_id,
                                            const char *dst_id);

/* Concat layer */
BCNN_API bcnn_status bcnn_add_concat_layer(bcnn_net *net, const char *src_id1,
                                           const char *src_id2,
                                           const char *dst_id);

/* Elementwise addition layer */
BCNN_API bcnn_status bcnn_add_eltwise_layer(bcnn_net *net,
                                            bcnn_activation activation,
                                            const char *src_id1,
                                            const char *src_id2,
                                            const char *dst_id);

/* Dropout layer */
BCNN_API bcnn_status bcnn_add_dropout_layer(bcnn_net *net, float rate,
                                            const char *id);

/* Upsample layer */
BCNN_API bcnn_status bcnn_add_upsample_layer(bcnn_net *net, int size,
                                             const char *src_id,
                                             const char *dst_id);

/* Cost layer */
BCNN_API bcnn_status bcnn_add_cost_layer(bcnn_net *net, bcnn_loss loss,
                                         bcnn_loss_metric loss_metric,
                                         float scale, const char *src_id,
                                         const char *label_id,
                                         const char *dst_id);
/* Yolo output layer */
BCNN_API bcnn_status bcnn_add_yolo_layer(bcnn_net *net, int num_boxes_per_cell,
                                         int classes, int coords, int total,
                                         int *mask, float *anchors,
                                         const char *src_id,
                                         const char *dst_id);

/* Return the detection results of a Yolo-like model */
BCNN_API bcnn_output_detection *bcnn_yolo_get_detections(
    bcnn_net *net, int batch, int w, int h, int netw, int neth, float thresh,
    int relative, int *num_dets);

/**
 * Convert an image (represented as an array of unsigned char) to floating point
 * values. Also perform a mean substraction and rescale the values
 * according to the following formula:
 * output_val = (input_pixel - mean) * norm_coeff
 *
 * Note: If the image has less than 3 channels, only the first mean values are
 * considered (up to the number of channels)
 *
 * @param[in]   src             Pointer to input image.
 * @param[in]   w               Image width.
 * @param[in]   h               Image height.
 * @param[in]   c               Number of channels of input image.
 * @param[in]   norm_coeff      Multiplicative factor to rescale input values
 * @param[in]   swap_to_bgr     Swap Red and Blue channels (Default layout is
 *                              RGB).
 * @param[in]   mean_r          Value to be substracted to first channel pixels
 *                              (red).
 * @param[in]   mean_g          Value to be substracted to second channel pixels
 *                              (green).
 * @param[in]   mean_b          Value to be substracted to third channel pixels
 * `                            (blue).
 * @param[out]  dst             Pointer to output float values array.
 */
BCNN_API void bcnn_convert_img_to_float(unsigned char *src, int w, int h, int c,
                                        float norm_coeff, int swap_to_bgr,
                                        float mean_r, float mean_g,
                                        float mean_b, float *dst);

/**
 * Perform the model prediction on the provided input data and computes the
 * loss if cost layers are defined.
 */
BCNN_API void bcnn_forward(bcnn_net *net);

/**
 * Back-propagate the gradients of the loss w.r.t. the parameters of the model.
 */
BCNN_API void bcnn_backward(bcnn_net *net);

/**
 * Update the model parameters according to the learning configuration and the
 * calculated gradients.
 */
BCNN_API void bcnn_update(bcnn_net *net);

/**
 * Convenient wrapper to compute the different steps required to train one batch
 * of data.
 * This functions performs the following:
 * - Load the next data batch (and performs data augmentation if required)
 * - Compute the forward pass given the loaded data batch
 * - Compute the back-propagation of the gradients
 * - Update the model parameters
 * - Return the loss according to the error metric
 *
 * The common use-case for this function is to be called inside a training loop
 * See: examples/mnist/mnist_example.c for a real-case example.
 */
BCNN_API float bcnn_train_on_batch(bcnn_net *net);

/**
 * Wrapper function to compute the inference pass only on a data batch.
 * This functions performs the following:
 * - Load the next data batch (and performs data augmentation if required)
 * - Compute the forward pass given the loaded data batch
 *
 * Return the loss value and the output tensor.
 */
BCNN_API float bcnn_predict_on_batch(bcnn_net *net, bcnn_tensor **out);

/**
 * Setup the dataset loader given the paths to training and testing dataset and
 * according to the data type.
 */
BCNN_API bcnn_status bcnn_set_data_loader(bcnn_net *net, bcnn_loader_type type,
                                          const char *train_path_data,
                                          const char *train_path_extra,
                                          const char *test_path_data,
                                          const char *test_path_extra);

/**
 * Set the online data augmentation parameters that will be applied on the
 * training data.
 */
BCNN_API bcnn_status bcnn_set_data_augmentation(
    bcnn_net *net, int width_shift_range, int height_shift_range,
    float rotation_range, float min_scale, float max_scale, int horizontal_flip,
    int min_brightness, int max_brightness, float min_constrast,
    float max_contrast, float distortion, int add_blobs);

/**
 * Set the network mode.
 * This function can be called on the same bcnn_net object with different modes
 * and will check internally if the requested mode is compatible with the
 * network current state.
 */
BCNN_API bcnn_status bcnn_set_mode(bcnn_net *net, bcnn_mode mode);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_H
