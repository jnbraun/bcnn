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

/****************************************************************************
 * Preprocessor / compiler stuff
 ***************************************************************************/

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

/****************************************************************************
 * Forward declarations
 ***************************************************************************/

/** Net struct (main BCNN object) */
typedef struct bcnn_net bcnn_net;

/** Tensor struct */
typedef struct bcnn_tensor bcnn_tensor;

/** Object detection result struct */
typedef struct bcnn_output_detection bcnn_output_detection;

/****************************************************************************
 * BCNN types
 ***************************************************************************/

/**
 * Error codes.
 */
typedef enum {
    BCNN_SUCCESS,
    BCNN_INVALID_PARAMETER,
    BCNN_INVALID_DATA,
    BCNN_INVALID_MODEL,
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
                          groundtruth (requires a cost layer) */
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

/* Function signature for logging callback */
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
    int has_grad;     /* If has gradient data or not */
    char *name;       /* Tensor name */
    float *data;      /* Pointer to data */
    float *grad_data; /* Pointer to gradient data */
#ifdef BCNN_USE_CUDA
    float *data_gpu;      /* Pointer to data on gpu */
    float *grad_data_gpu; /* Pointer to gradient data on gpu */
#endif
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

/****************************************************************************
 * BCNN functions API
 ***************************************************************************/

/**
 * \brief Creates a net object.
 *
 * This function creates an bcnn_net instance and needs to be called
 * before any other BCNN functions applied to this bcnn_net instance. In order
 * to free the bcnn_net instance, the function bcnn_end_net needs to be called
 * before exiting the application.
 *
 * \param[in]   net     Pointer to net handle to be created.
 *
 * \return 'BCNN_SUCCESS' if successful initialization or 'BCNN_FAILED_ALLOC'
 * otherwise
 */
BCNN_API bcnn_status bcnn_init_net(bcnn_net **net, bcnn_mode mode);

/**
 * \brief Destroys the net object.
 *
 * \param[in]   net     Pointer to net handle to be destroyed.
 *
 * This function frees any allocated ressources in the bcnn_net instance
 * and destroys the instance itself (net pointer is set to NULL after being
 * freed).
 */
BCNN_API void bcnn_end_net(bcnn_net **net);

/**
 * \brief Sets the logging context.
 *
 * \param[in]   net     Pointer to net instance.
 * \param[in]   fct     Callback to user defined log function. If NULL, default
 *                      logging to stderr will be used.
 * \param[in]   level   Log level.
 */
BCNN_API void bcnn_set_log_context(bcnn_net *net, bcnn_log_callback fct,
                                   bcnn_log_level level);

/**
 * \brief Sets the number of threads for BCNN to use (maximal: 8).
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   num_threads     Number of threads to use. BCNN_USE_OPENMP must
 *                              be defined.
 */
BCNN_API void bcnn_set_num_threads(bcnn_net *net, int num_threads);

/**
 * \brief Gets the number of threads currently used by the net instance. Use
 * 'bcnn_set_num_threads' to change the current number of threads.
 *
 * \param[in]   net             Pointer to net instance.
 *
 * \return Number of threads used by the net instance.
 */
BCNN_API int bcnn_get_num_threads(bcnn_net *net);

/**
 * \brief Sets the shape of the primary input tensor.
 *
 * The primary input tensor holds the default name 'input'.
 *
 * \param[in]   net         Pointer to net instance.
 * \param[in]   width       Input tensor width.
 * \param[in]   height      Input tensor height.
 * \param[in]   channels    Input tensor depth (= number of channels).
 * \param[in]   batch_size  Set the batch size (will be the same for each
 *                          tensor).
 */
BCNN_API void bcnn_set_input_shape(bcnn_net *net, int width, int height,
                                   int channels, int batch_size);

/**
 * \brief Defines an input tensor to the network.
 *
 * \param[in]   net         Pointer to net instance.
 * \param[in]   width       Tensor width.
 * \param[in]   height      Tensor height.
 * \param[in]   channels    Tensor depth (= number of channels).
 * \param[in]   name        Tensor name.
 *
 * \return BCNN_SUCCESS if successful initialization or BCNN_FAILED_ALLOC
 * otherwise
 */
BCNN_API bcnn_status bcnn_add_input(bcnn_net *net, int width, int height,
                                    int channels, const char *name);

/**
 * \brief Returns the batch size used for training / validation
 *
 * \param[in]   net         Pointer to net instance.
 *
 * \return The batch size value.
 */
BCNN_API int bcnn_get_batch_size(bcnn_net *net);

/**
 * \brief Resizes the network according to given input width, height and
 * channels.
 *
 * \note This function is valid only when applied to a fully convolutionnal
 * network.
 *
 * \param[in]   net                  Pointer to net instance.
 * \param[in]   w                    New input width.
 * \param[in]   h                    New input height.
 * \param[in]   c                    New input number of channels.
 * \param[in]   need_realloc         Set to '1' if a memory allocation is
 * needed, '0' otherwise.
 *
 * \return      Possible error include BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_resize_net(bcnn_net *net, int w, int h, int c,
                                     int need_realloc);

/**
 * \brief Finalizes the net configuration.
 *
 * This function needs to be called after everything has been setup (the model
 * architecture, the dataset loader, the data augmentation, the training
 * configuration and the model weights) have been setup and before effectively
 * starting the model training or inference.
 *
 * \param[in]   net         Pointer to net instance.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_compile_net(bcnn_net *net);

/**
 * \brief Loads the model weights from disk.
 *
 * \param[in]   net           Pointer to net instance.
 * \param[in]   model_path    Path to the model weights to be loaded.
 *
 * \return Possible erros include BCNN_INVALID_MODEL and BCNN_INVALID_PARAMETER.
 */
BCNN_API bcnn_status bcnn_load_weights(bcnn_net *net, const char *model_path);

/**
 * \brief Defines the net architecture and loads the model weights if required.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   config_path     Path to configuration file that defines the net
 *                              architecture.
 * \param[in]   model_path      Path to the model weights to be loaded.
 *
 * \note The model weights should be consistent with the net architecture
 * defined in the configuration file.
 *
 * \return Possible erros include
 * BCNN_INVALID_MODEL and BCNN_INVALID_PARAMETER.
 */
BCNN_API bcnn_status bcnn_load_net(bcnn_net *net, const char *config_path,
                                   const char *model_path);

/**
 * \brief Writes the model weights on disk.
 *
 * \param[in]   net         Pointer to net instance.
 * \param[in]   filename    Path where to save the model weights.
 *
 * \return BCNN_INVALID_PARAMETER if file failed to be open.
 */
BCNN_API bcnn_status bcnn_save_weights(bcnn_net *net, const char *filename);

/**
 * \brief Setups the dataset loader.
 *
 * \param[in]   net                 Pointer to net instance.
 * \param[in]   type                The loader type.
 * \param[in]   train_path_data     Path to training dataset.
 * \param[in]   train_path_extra    Optional path to extra training dataset.
 *                                  NULL if not needed.
 * \param[in]   test_path_data      Path to validataion
 *                                  dataset
 * \param[in]   test_path_extra     Optional path to extra validation
 *                                  dataset. NULL if not needed.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER, BCNN_INVALID_DATA and
 * BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_set_data_loader(bcnn_net *net, bcnn_loader_type type,
                                          const char *train_path_data,
                                          const char *train_path_extra,
                                          const char *test_path_data,
                                          const char *test_path_extra);

/**
 * \brief Generates random shifts (i.e. translations) on inputs.
 *
 * \param[in]   net                 Pointer to net instance.
 * \param[in]   width_shift_range   Horizontal shift range in pixels.
 * \param[in]   height_shift_range  Vertical shift range in pixels.
 */
BCNN_API void bcnn_augment_data_with_shift(bcnn_net *net, int width_shift_range,
                                           int height_shift_range);

/**
 * \brief Generates random scalings on inputs.
 *
 * \param[in]   net                 Pointer to net instance.
 * \param[in]   min_scale           Minimal scale coefficient.
 * \param[in]   max_scale           Maximal scale coefficent.
 */
BCNN_API void bcnn_augment_data_with_scale(bcnn_net *net, float min_scale,
                                           float max_scale);

/**
 * \brief Generates random rotations on inputs.
 *
 * \param[in]   net                 Pointer to net instance.
 * \param[in]   rotation_range      Rotation angle range in degree. Angle will
 *                                  randomly be sampled in [-rotation_range / 2;
 *                                  rotation_range / 2].
 */
BCNN_API void bcnn_augment_data_with_rotation(bcnn_net *net,
                                              float rotation_range);

/**
 * \brief Generates random flips on inputs.
 *
 * \param[in]   net                 Pointer to net instance.
 * \param[in]   horizontal_flip     If set to 1, will randomly flip inputs
 *                                  horizontally.
 * \param[in]   vertical_flip       If set to 1, will randomly flip inputs
 *                                  vertically. Not implemented.
 */
BCNN_API void bcnn_augment_data_with_flip(bcnn_net *net, int horizontal_flip,
                                          int vertical_flip);

/**
 * \brief Generates random brightness and contrast adjustments on inputs.
 *
 * \param[in]   net                 Pointer to net instance.
 * \param[in]   min_brightness      Minimal brightness additive factor (range in
 *                                  [-255;255]).
 * \param[in]   max_brightness      Maximal brightness additive factor (range in
 *                                  [-255;255]).
 * \param[in]   min_contrast        Minimal contrast scale factor.
 * \param[in]   max_contrast        Maximal contrast scale factor.
 */
BCNN_API void bcnn_augment_data_with_color_adjustment(bcnn_net *net,
                                                      int min_brightness,
                                                      int max_brightness,
                                                      float min_constrast,
                                                      float max_contrast);

/**
 * \brief Generates random saturated blobs on inputs.
 *
 * \param[in]   net                 Pointer to net instance.
 * \param[in]   max_blobs           Maximal number of blobs generated on an
 *                                  input i.e. will randomly generates
 *                                  [0;max_blobs] blobs.
 */
BCNN_API void bcnn_augment_data_with_blobs(bcnn_net *net, int max_blobs);

/**
 * \brief Generates random 2d-perlin-noise based distortion on inputs.
 *
 * \param[in]   net                 Pointer to net instance.
 * \param[in]   distortion          Distortion intensity.
 */
BCNN_API void bcnn_augment_data_with_distortion(bcnn_net *net,
                                                float distortion);

/**
 * \brief Sets the network mode.
 *
 * This function can be called on the same bcnn_net object with different modes
 * and will check internally if the requested mode is compatible with the
 * network current state.
 *
 * \param[in]   net         Pointer to net instance.
 * \param[in]   mode        The network mode to be set.
 */
BCNN_API bcnn_status bcnn_set_mode(bcnn_net *net, bcnn_mode mode);

/**
 * \brief Setups Adam optimizer.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[int]  learning_rate   Initial learning rate.
 * \param[int]  beta1           Exponential decay rate for the first moment
 *                              estimates. Default: 0.9.
 * \param[in]   beta2           Exponential decay rate for the second moment
 *                              estimates. Default: 0.999.
 */
BCNN_API void bcnn_set_adam_optimizer(bcnn_net *net, float learning_rate,
                                      float beta1, float beta2);
/**
 * \brief Setups SGD optimizer with momentum.
 *
 * \param[in]   net             Pointer to net instance
 * \param[int]  learning_rate   Initial learning rate.
 * \param[in]   momentum        Exponentially decay rate for the weighted
 *                              gradients averages. Default: 0.9.
 */
BCNN_API void bcnn_set_sgd_optimizer(bcnn_net *net, float learning_rate,
                                     float momentum);

/**
 * \brief Sets the learning rate decay policy.
 *
 * \param[in]   net         Pointer to net instance.
 * \param[in]   decay_type  Decay policy. \see bcnn_lr_decay
 * \param[in]   gamma
 * \param[in]   scale
 * \param[in]   power
 * \param[in]   max_batches
 * \param[in]   step
 */
BCNN_API void bcnn_set_learning_rate_policy(bcnn_net *net,
                                            bcnn_lr_decay decay_type,
                                            float gamma, float scale,
                                            float power, int max_batches,
                                            int step);

/**
 * \brief Setups the weights L2 regularization.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   weight_decay    Weight decay coefficient.
 */
BCNN_API void bcnn_set_weight_regularizer(bcnn_net *net, float weight_decay);

/**
 * Converts an image (represented as an array of unsigned char) to floating
 * point values and fill a tensor. Also perform a mean substraction and rescale
 * the values according to the following formula:
 * output_val = (input_pixel - mean) * norm_coeff
 *
 * \note: If the image has less than 3 channels, only the first mean values are
 * considered (up to the number of channels)
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   src             Pointer to input image pixels data.
 * \param[in]   w               Image width.
 * \param[in]   h               Image height.
 * \param[in]   c               Number of channels of input image.
 * \param[in]   norm_coeff      Multiplicative factor to rescale input values
 * \param[in]   swap_to_bgr     Swap Red and Blue channels (Default layout is
 *                              RGB).
 * \param[in]   mean_r          Value to be substracted to first channel pixels
 *                              (red).
 * \param[in]   mean_g          Value to be substracted to second channel pixels
 *                              (green).
 * \param[in]   mean_b          Value to be substracted to third channel pixels
 *                              (blue).
 * \param[in]   tensor_index    Index of the tensor to be filled in.
 * \param[in]   batch_index     Position of the tensor in the data batch.
 */
BCNN_API bcnn_status bcnn_fill_tensor_with_image(
    bcnn_net *net, const uint8_t *src, int w, int h, int c, float norm_coeff,
    int swap_to_bgr, float mean_r, float mean_g, float mean_b, int tensor_index,
    int batch_index);

/**
 * \brief Computes the model prediction on the current batch data and computes
 * the loss if cost layers are defined.
 *
 * \param[in]   net             Pointer to net instance.
 */
BCNN_API void bcnn_forward(bcnn_net *net);

/**
 * \brief Back-propagates the gradients of the loss w.r.t. the model weights.
 *
 * \param[in]   net             Pointer to net instance.
 */
BCNN_API void bcnn_backward(bcnn_net *net);

/**
 * \brief Updates the model parameters according to the learner configuration
 * and the calculated gradients.
 *
 * \param[in]   net            Pointer to net instance.
 */
BCNN_API void bcnn_update(bcnn_net *net);

/**
 * \brief Convenient wrapper to compute the different steps required to train
 * one batch of data.
 *
 * This functions performs the following:
 * - Load the next data batch (and performs data augmentation if required)
 * - Compute the forward pass given the loaded data batch
 * - Compute the back-propagation of the gradients
 * - Update the model parameters
 * - Return the loss according to the error metric
 *
 * The common use-case for this function is to be called inside a training loop
 * See: examples/mnist/mnist_example.c for a real-case example.
 *
 * \param[in]   net            Pointer to net instance.
 *
 * \return The loss value.
 */
BCNN_API float bcnn_train_on_batch(bcnn_net *net);

/**
 * \brief Wrapper function to compute the inference pass only on a data batch.
 *
 * This functions performs the following:
 * - Load the next data batch (and performs data augmentation if required)
 * - Compute the forward pass given the loaded data batch
 *
 * \param[in]   net            Pointer to net instance.
 * \param[out]  out            Pointer to output tensor hold in the net
 *                             instance. It must *not* be allocated neither be
 *                             freed by the user.
 *
 * \return The loss value.
 */
BCNN_API float bcnn_predict_on_batch(bcnn_net *net, bcnn_tensor **out);

/**
 * \brief Gets the output results of an object detection model.
 *
 * \param[in]   net         Pointer to net instance.
 * \param[in]   batch       Batch size.
 * \param[in]   width       Input image width.
 * \param[in]   height      Input image height.
 * \param[in]   netw        Input tensor width.
 * \param[in]   neth        Input tensor height.
 * \param[in]   thresh      Threshold between [0;1] above which detections will
 *                          be kept.
 * \param[in]   relative    If set to 1, will scale the boxes dimensions to the
 *                          input image.
 * \param[out]  num_dets    Number of detected boxes.
 *
 * \return Array of 'num_dets' detected boxes.
 */
BCNN_API bcnn_output_detection *bcnn_yolo_get_detections(
    bcnn_net *net, int batch, int width, int height, int netw, int neth,
    float thresh, int relative, int *num_dets);

/**
 * \brief Gets a tensor's index, given its name.
 *
 * \param[in]   net         Pointer to net instance.
 * \param[in]   name        Tensor name.
 *
 * \return Tensor index. Returns -1 is invalid name.
 */
BCNN_API int bcnn_get_tensor_index_by_name(bcnn_net *net, const char *name);

/**
 * \brief Gets a pointer to a tensor struct, given its index.
 *
 * \param[in]   net         Pointer to net instance.
 * \param[in]   index       Tensor index (accessible via
 *                          'bcnn_get_tensor_index_by_name').
 *
 * \return A pointer to tensor struct. Returns NULL if index is invalid.
 */
BCNN_API bcnn_tensor *bcnn_get_tensor_by_index(bcnn_net *net, int index);

/**
 * \brief Gets a pointer to a tensor struct, given its name.
 *
 * \note For performance-critical code, it is better to use a combination
 * of 'bcnn_get_tensor_index_by_name' and 'bcnn_get_tensor_by_index' instead of
 * this function i.e. prefer using:
 *
 * int index = bcnn_get_tensor_index_by_name(some_net, "some_tensor");
 * for (;;) {bcnn_tensor *pt = bcnn_get_tensor_by_index(some_net, index);}
 *
 * instead of:
 * for (;;) {bcnn_tensor *pt = bcnn_get_tensor_by_name(some_net,
 * "some_tensor");}
 *
 * \param[in]   net         Pointer to net instance.
 * \param[in]   name        Tensor name.
 *
 * \return A pointer to tensor struct. Returns NULL is name is invalid.
 */
BCNN_API bcnn_tensor *bcnn_get_tensor_by_name(bcnn_net *net, const char *name);

/****************************************************************************
 * BCNN layers API
 ***************************************************************************/

/**
 * \brief 2D-Convolutional layer.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   num_filters     Number of output filters.
 * \param[in]   size            Kernel size.
 * \param[in]   stride          Stride value.
 * \param[in]   pad             Padding value.
 * \param[in]   num_groups      Number of groups of input feature maps channels.
 * \param[in]   batch_norm      If set to 1, will fuse a batch normalization
 *                              layer.
 * \param[in]   init            Weights initialization type. Used only for
 *                              training.
 * \param[in]   activation      Type of the fused activation.
 * \param[in]   quantize        Not implemented. Reserved for future versions.
 * \param[in]   src_id          Input tensor name.
 * \param[in]   dst_id          Output tensor name.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_add_convolutional_layer(
    bcnn_net *net, int num_filters, int size, int stride, int pad,
    int num_groups, int batch_norm, bcnn_filler_type init,
    bcnn_activation activation, int quantize, const char *src_id,
    const char *dst_id);

/**
 * \brief Transposed 2D-convolution layer.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   num_filters     Number of output filters.
 * \param[in]   size            Kernel size.
 * \param[in]   stride          Stride value.
 * \param[in]   pad             Padding value.
 * \param[in]   init            Weights initialization type. Used only for
 *                              training.
 * \param[in]   activation      Type of the fused activation.
 * \param[in]   src_id          Input tensor name.
 * \param[in]   dst_id          Output tensor name.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_add_deconvolutional_layer(
    bcnn_net *net, int num_filters, int size, int stride, int pad,
    bcnn_filler_type init, bcnn_activation activation, const char *src_id,
    const char *dst_id);

/**
 * \brief Depthwise 2D-convolution layer.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   size            Kernel size.
 * \param[in]   stride          Stride value.
 * \param[in]   pad             Padding value.
 * \param[in]   batch_norm      If set to 1, will fuse a batch normalization
 *                              layer.
 * \param[in]   init            Weights initialization type. Used only for
 *                              training.
 * \param[in]   activation      Type of the fused activation.
 * \param[in]   src_id          Input tensor name.
 * \param[in]   dst_id          Output tensor name.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_add_depthwise_conv_layer(
    bcnn_net *net, int size, int stride, int pad, int batch_norm,
    bcnn_filler_type init, bcnn_activation activation, const char *src_id,
    const char *dst_id);

/**
 * \brief Batch-normalization layer.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   src_id          Input tensor name.
 * \param[in]   dst_id          Output tensor name.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_add_batchnorm_layer(bcnn_net *net, const char *src_id,
                                              const char *dst_id);

/**
 * \brief Local Response Normalization layer.
 *
 * Normalizes over adjacent channels according to the formula:
 * output = input * (k + alpha * sum2) ^ (-beta)
 * with sum2 the squared sum of inputs within local_size channels.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   local_size      Width of the normalization window.
 * \param[in]   alpha           Scale factor.
 * \param[in]   beta            Exponent factor.
 * \param[in]   k               Bias value.
 * \param[in]   src_id          Input tensor name.
 * \param[in]   dst_id          Output tensor name.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_add_lrn_layer(bcnn_net *net, int local_size,
                                        float alpha, float beta, float k,
                                        const char *src_id, const char *dst_id);

/**
 * \brief Dense aka fully-connected layer.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   output_size     Tensor output size (i.e number of output
 *                              channels).
 * \param[in]   init            Weights initialization type. Used only for
 *                              training.
 * \param[in]   activation      Type of the fused activation.
 * \param[in]   quantize        Not implemented. Reserved for future versions.
 * \param[in]   src_id          Input tensor name.
 * \param[in]   dst_id          Output tensor name.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_add_fullc_layer(bcnn_net *net, int output_size,
                                          bcnn_filler_type init,
                                          bcnn_activation activation,
                                          int quantize, const char *src_id,
                                          const char *dst_id);

/**
 * \brief Activation layer.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   type            Activation function.
 * \param[in]   src_id          Input / output tensor name (inplace layer).
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_add_activation_layer(bcnn_net *net,
                                               bcnn_activation type,
                                               const char *id);

/**
 * \brief Softmax layer.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   src_id          Input tensor name.
 * \param[in]   dst_id          Output tensor name.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_add_softmax_layer(bcnn_net *net, const char *src_id,
                                            const char *dst_id);

/**
 * \brief Max-Pooling layer.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   size            Kernel size.
 * \param[in]   stride          Stride value.
 * \param[in]   pad             Padding type. \see bcnn_padding
 * \param[in]   src_id          Input tensor name.
 * \param[in]   dst_id          Output tensor name.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_add_maxpool_layer(bcnn_net *net, int size, int stride,
                                            bcnn_padding padding,
                                            const char *src_id,
                                            const char *dst_id);

/**
 * \brief Average pooling layer.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   src_id          Input tensor name.
 * \param[in]   dst_id          Output tensor name.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_add_avgpool_layer(bcnn_net *net, const char *src_id,
                                            const char *dst_id);

/**
 * \brief Concatenation layer.
 * Concatenates input tensors along the channel axis.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   num_src         Number of input tensors to be concatenated.
 * \param[in]   src_ids         Array of input tensors names.
 * \param[in]   dst_id          Output tensor name.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
bcnn_status bcnn_add_concat_layer(bcnn_net *net, int num_src,
                                  char *const *src_ids, const char *dst_id);

/**
 * \brief Elementwise addition layer.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   activation      Fused activation type.
 * \param[in]   src_id1         First input tensor name.
 * \param[in]   src_id2         Second input tensor name.
 * \param[in]   dst_id          Output tensor name.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_add_eltwise_layer(bcnn_net *net,
                                            bcnn_activation activation,
                                            const char *src_id1,
                                            const char *src_id2,
                                            const char *dst_id);

/**
 * \brief Dropout layer.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   rate            Dropout ratio.
 * \param[in]   id              Input / output tensor name (inplace layer).
 */
BCNN_API bcnn_status bcnn_add_dropout_layer(bcnn_net *net, float rate,
                                            const char *id);

/**
 * \brief Upsampling layer.
 *
 * Upsamples the input tensor by a scale factor 'size'.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   size            Upsampling factor.
 * \param[in]   src_id          Input tensor name.
 * \param[in]   dst_id          Output tensor name.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_add_upsample_layer(bcnn_net *net, int size,
                                             const char *src_id,
                                             const char *dst_id);

/**
 * \brief Cost layer.
 *
 * Computes the loss according to the objective function and the error metric.
 *
 * \param[in]   net             Pointer to net instance.
 * \param[in]   loss            Loss objective type.
 * \param[in]   loss_metric     Error metric type.
 * \param[in]   scale           Participation ratio of this specific loss layer
 *                              towards the total network loss (in case of
 *                              several loss layers).
 * \param[in]   src_id          Input tensor name.
 * \param[in]   label_id        Label tensor name. Default name: 'label'
 * \param[in]   dst_id          Output tensor name.
 *
 * \return Possible errors include BCNN_INVALID_PARAMETER and BCNN_FAILED_ALLOC.
 */
BCNN_API bcnn_status bcnn_add_cost_layer(bcnn_net *net, bcnn_loss loss,
                                         bcnn_loss_metric loss_metric,
                                         float scale, const char *src_id,
                                         const char *label_id,
                                         const char *dst_id);
/**
 * \brief Yolo v3 output layer.
 *
 * \param[in]   net                 Pointer to net instance.
 * \param[in]   num_boxes_per_cell  Max number of possibles detections per cell.
 * \param[in]   classes             Number of object classes.
 * \param[in]   coords              Number of coordinates to define a detection
 *                                  box. Default: 4.
 * \param[in]   total               Number of anchors to considerate.
 * \param[in]   mask                Anchors indexes to considerate.
 * \param[in]   anchors             Anchors coordinates (prior box coordinates).
 * \param[in]   src_id              Input tensor name.
 * \param[in]   dst_id              Output tensor name.
 */
BCNN_API bcnn_status bcnn_add_yolo_layer(bcnn_net *net, int num_boxes_per_cell,
                                         int num_classes, int coords, int total,
                                         int *mask, float *anchors,
                                         const char *src_id,
                                         const char *dst_id);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_H
