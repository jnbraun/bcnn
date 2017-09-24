/*
* Copyright (c) 2016 Jean-Noel Braun.
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
#ifdef BCNN_USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#ifdef BCNN_USE_CUDNN
#include <cudnn.h>
#endif
#endif


#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <limits.h>
#include <time.h>
#ifdef BCNN_USE_SSE2
#include <emmintrin.h> // SSE2
#endif

#include "bh/bh_error.h"

#if defined(__GNUC__) || (defined(_MSC_VER) && (_MSC_VER >= 1600))
#include <stdint.h>
#else
#  if (_MSC_VER < 1300)
typedef signed char       int8_t;
typedef signed short      int16_t;
typedef signed int        int32_t;
typedef unsigned char     uint8_t;
typedef unsigned short    uint16_t;
typedef unsigned int      uint32_t;
#  else
typedef signed __int8     int8_t;
typedef signed __int16    int16_t;
typedef signed __int32    int32_t;
typedef unsigned __int8   uint8_t;
typedef unsigned __int16  uint16_t;
typedef unsigned __int32  uint32_t;
#  endif
typedef signed __int64    int64_t;
typedef unsigned __int64  uint64_t;
#endif

/*typedef struct bcnn_context {
	bh_logctx *log_hdl;
} bcnn_context;*/

//typedef struct bcnn_context *bcnn_context_handle;

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
typedef enum {
	TRAIN,
	PREDICT,
} bcnn_task;

typedef enum {
	CLASSIFICATION,
	REGRESSION,
	HEATMAP_REGRESSION,
	SEGMENTATION
} bcnn_target;


typedef enum {
	ITER_BIN,
	ITER_LIST,
	ITER_CSV,
	ITER_MNIST,
	ITER_CIFAR10
} bcnn_iterator_type;


typedef struct {
	int	n_samples;
	bcnn_iterator_type type;
	FILE *f_input;
	FILE *f_label;
	FILE *f_list;
	int n_iter;
	int input_width;
	int input_height;
	int input_depth;
	unsigned char *input_uchar;
	int label_width;
	int *label_int;
	float *label_float;
	unsigned char *label_uchar;
} bcnn_iterator;


/**
 * \brief Structure for online data augmentation parameters.
 */
typedef enum {
	LABEL_INT,
	LABEL_FLOAT,
	LABEL_IMG
} bcnn_label_type;

typedef struct {
	int			n_samples;
	int			width;
	int			label_width;
	bcnn_label_type		label_type;
	unsigned char		*data;
} bcnn_data;

/**
 * \brief Structure for online data augmentation parameters.
 */
typedef struct {
	int			range_shift_x;			/**< X-shift allowed range (chosen between [-range_shift_x / 2; range_shift_x / 2]). */
	int			range_shift_y;			/**< Y-shift allowed range (chosen between [-range_shift_y / 2; range_shift_y / 2]). */	
	int			random_fliph;			/**< If !=0, randomly (with probability of 0.5) apply horizontal flip to image. */	
	float		min_scale;				/**< Minimum scale factor allowed. */
	float		max_scale;				/**< Maximum scale factor allowed. */
	float		rotation_range;			/**< Rotation angle allowed range (chosen between [-rotation_range / 2; rotation_range / 2]). Expressed in degree. */
	int			min_brightness;			/**< Minimum brightness factor allowed (additive factor, range [-255;255]). */
	int			max_brightness;			/**< Maximum brightness factor allowed (additive factor, range [-255;255]). */
	float		min_contrast;			/**< Minimum contrast allowed (mult factor). */
	float		max_contrast;			/**< Maximum contrast allowed (mult factor). */
	int			use_precomputed;		/**< Flag set to 1 if the parameters to be applied are those already set. */
	float		scale;					/**< Current scale factor. */
	int			shift_x;				/**< Current x-shift. */
	int			shift_y;				/**< Current y-shift. */
	float		rotation;				/**< Current rotation angle. */
	int			brightness;				/**< Current brightness factor. */
	float		contrast;				/**< Current contrast factor. */
	float		max_distortion;			/**< Maximum distortion factor allowed. */
	float		distortion;				/**< Current distortion factor. */
	float		distortion_kx;			/**< Current distortion x kernel. */
	float		distortion_ky;			/**< Current distortion y kernel. */
	float		mean_r;
	float		mean_g;
	float		mean_b;
	int			swap_to_bgr;
} bcnn_data_augment;


/**
 * \brief Enum of learning policies.
 */
typedef enum {
	CONSTANT,
	STEP,
	INV,
	EXP,
	POLY,
	SIGMOID
} bcnn_lr_policy;

/**
 * \brief Enum of optimization methods.
 */
typedef enum {
	SGD,
	ADAM
} bcnn_optimizer;

/**
 * \brief Structure to handle learner method and parameters.
 */
typedef struct {
	float			momentum;				/**< Momentum parameter */
	float			decay;					/**< Decay parameter */
	float			learning_rate;				/**< Base learning rate */
	float			gamma;
	float			scale;
	float			power;
	float			beta1;					/**< Parameter for Adam optimizer */
	float			beta2;					/**< Parameter for Adam optimizer */
	int				step;
	bcnn_optimizer	optimizer;				/**< Optimization method */
	bcnn_lr_policy	policy;					/**< Learning rate policy */
} bcnn_learner;


/**
 * \brief Enum of available layers types.
 */
typedef enum {
	CONVOLUTIONAL,
	DECONVOLUTIONAL,
	ACTIVATION,
	FULL_CONNECTED,
	MAXPOOL,
	SOFTMAX,
	DROPOUT,
	BATCHNORM,
	CONCAT,
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
	LRELU,				/**< Leaky relu (alpha (negative slope) set to 0.01) */
	ABS,
	CLAMP
} bcnn_activation;

/**
 * \brief Enum of available weight inializations modes.
 */
typedef enum {
	XAVIER,				/**< Xavier weight init */
	MSRA				/**< MSRA weight init */
} bcnn_weights_init;


typedef enum {
	L2,
	HUBER
} bcnn_loss;


/**
 * \brief Enum of available loss metrics.
 */
typedef enum {
	COST_ERROR,			/**< Error rate (classification only) */
	COST_LOGLOSS,			/**< Multi-class Logloss (classification only) */
	COST_SSE,			/**< Sum-squared error */
	COST_MSE,			/**< Mean-squared error */
	COST_CRPS,			/**< Continuous Ranked Probability Score */
	COST_DICE			/**< Sørensen–Dice index: metric for image segmentation */
} bcnn_loss_metric;

/**
 * \brief Structure for handling the current workload through the network (data batch).
 */
typedef struct {
	float	*input;
	float	*input2;
	float	*truth;
#ifdef BCNN_USE_CUDA
	float	*input_gpu;
	float	*input2_gpu;
	float	*truth_gpu;
#endif
	float	*diff;
	float	*diff2;
	int		train;
	int		batch_size;
} bcnn_workload;


/* Experimental */
typedef struct {
	int		b;				/**< Batch size */
	int		w;
	int		h;
	int		c;
	float	*data;
	float	*grad_data;
#ifdef BCNN_USE_CUDA
	float	*data_gpu;
	float	*grad_data_gpu;
#endif
} bcnn_tensor;

/**
* \brief Structure defining a generic layer.
*/
typedef struct {
	int					num;
	int					size;
	int					stride;
	int					pad;
	int					quantize;
	bcnn_layer_type		type;
	bcnn_activation		activation;
	bcnn_loss_metric	loss_metric;
	float				dropout_rate;
	float				scale;
	int					concat_index;
	int					weights_size;
	float				*weight;
	float				*weight_diff;
#ifdef BCNN_USE_CUDA
	float				*weight_gpu;
	float				*weight_diff_gpu;
#endif
	int					bias_size;
	float				*bias;
	float				*bias_diff;
#ifdef BCNN_USE_CUDA
	float				*bias_gpu;
	float				*bias_diff_gpu;
#endif
	int					*indexes;
	float				*conv_workspace;
	float				*rand;
#ifdef BCNN_USE_CUDA
	int					*indexes_gpu;
	float				*conv_workspace_gpu;
	float				*rand_gpu;
#endif
	float				*bn_workspace;
	float				*mean;
	float				*variance;
	float				*global_mean;
	float				*global_variance;
	float				*diff_mean;
	float				*diff_variance;
	float				*x_norm;
	float				*bn_scale;
	float				*bn_scale_diff;
#ifdef BCNN_USE_CUDA
	float				*mean_gpu;
	float				*variance_gpu;
	float				*global_mean_gpu;
	float				*global_variance_gpu;
	float				*diff_mean_gpu;
	float				*diff_variance_gpu;
	float				*bn_workspace_gpu;
	float				*x_norm_gpu;
	float				*bn_scale_gpu;
	float				*bn_scale_diff_gpu;
#endif
	float				*adam_m;		/**< Adam optimizer: first moment gradient */
	float				*adam_v;		/**< Adam optimizer: second moment gradient */
#ifdef BCNN_USE_CUDA
	float				*adam_m_gpu;	/**< Adam optimizer: first moment gradient */
	float				*adam_v_gpu;	/**< Adam optimizer: second moment gradient */
#endif
	unsigned int		*binary_weight;
	unsigned int		*binary_workspace;
#ifdef BCNN_USE_CUDA
#ifdef BCNN_USE_CUDNN
	cudnnTensorDescriptor_t				src_tensor_desc;
	cudnnTensorDescriptor_t				dst_tensor_desc;
	cudnnTensorDescriptor_t				src_tensor_desc_diff;
	cudnnTensorDescriptor_t				dst_tensor_desc_diff;
	cudnnFilterDescriptor_t				filter_desc;
	cudnnFilterDescriptor_t				filter_desc_diff;
	cudnnTensorDescriptor_t				bias_desc;
	cudnnTensorDescriptor_t				bias_desc_diff;
	cudnnConvolutionDescriptor_t		conv_desc;
	cudnnPoolingDescriptor_t			pooling_desc;
	cudnnConvolutionFwdAlgo_t			fwd_algo;
	cudnnConvolutionBwdDataAlgo_t		bwd_data_algo;
	cudnnConvolutionBwdFilterAlgo_t 	bwd_filter_algo;
	size_t								workspace_size;
#endif
#endif
} bcnn_layer;

/**
* \brief Structure handling the network architecture and generic parameters.
*/
typedef struct {
	int			state; // 1: train / 0: predict
	bcnn_tensor		src_tensor;
	bcnn_tensor		dst_tensor;
	bcnn_layer		*layer;
	float			*label;
	char			*id;
#ifdef BCNN_USE_CUDA
	float			*label_gpu;
#endif
} bcnn_connection;

/**
* \brief Structure handling the network architecture and generic parameters.
*/
typedef struct {
	int					max_batches;		/**< Maximum number of batches during training (=iterations) */ 
	bcnn_loss_metric	loss_metric;		/**< Loss metric for evaluation */
	bcnn_learner		learner;			/**< Learner/optimizer parameters */
	int					seen;				/**< Number of instances seen by the network */
	int					nb_connections;
	bcnn_connection		*connections;
	bcnn_target			prediction_type;
	bcnn_data_augment   data_aug;			/**< Parameters for online data augmentation */
	bcnn_task			task;
	int					state;
	bcnn_tensor			input_node;
	int					nb_finetune;
	char				**finetune_id;
	unsigned char		*input_buffer;
} bcnn_net;

/* Define for binarized layers */
#define BITS_IN_CHAR 8
#define BITS_IN_UINT32 (sizeof(uint32_t) * BITS_IN_CHAR)
#define BIT_SET(var, pos, val) var |= (val << pos)

int bcnn_net_add_connection(bcnn_net *net, bcnn_connection conn);

int bcnn_get_tensor_size(bcnn_tensor *tensor);

int bcnn_init_net(bcnn_net **net);
int bcnn_end_net(bcnn_net **net);

int bcnn_set_param(bcnn_net *net, char *name, char *val);

int bcnn_compile_net(bcnn_net *net, char *phase);

//int bcnn_init_mnist_iterator(bcnn_iterator *iter, char *path_img, char *path_label);
//int bcnn_free_mnist_iterator(bcnn_iterator *iter);

int bcnn_init_iterator(bcnn_net *net, bcnn_iterator *iter, char *path_input, char *path_label, char *type);
int bcnn_advance_iterator(bcnn_net *net, bcnn_iterator *iter);
int bcnn_free_iterator(bcnn_iterator *iter);

//int bcnn_init_list_iterator(bcnn_net *net, bcnn_iterator *iter, char *path_input);
//int bcnn_list_iter(bcnn_net *net, bcnn_iterator *iter);

/* Load / Write model */
int bcnn_load_model(bcnn_net *net, char *filename);
int bcnn_write_model(bcnn_net *net, char *filename);

/* Matrix computation routines */
int bcnn_fill_f32(int n, float a, float *x);
int bcnn_copy_f32(int n, float *x, float *y);
int bcnn_axpy(int n, float a, float *x, float *y);
int bcnn_scal(int n, float a, float *x);
int bcnn_add_scalar(int n, float a, float *x);
int bcnn_pow(int n, float *x, float a, float *y);
float bcnn_dot(int n, float *x, float *y);
int bcnn_vsum(int n, float *x, float *sum);
int bcnn_vadd(int n, float *a, float *b, float *y);
int bcnn_vsub(int n, float *a, float *b, float *y);
int bcnn_vdiv(int n, float *a, float *b, float *y);
int bcnn_vmul(int n, float *a, float *b, float *y);
int bcnn_axpby(int n, float a, float *x, float b, float *y);
int bcnn_gemv(int trans_a, int m, int n, float alpha, float *a, float *x,
	float beta, float *y);
int bcnn_gemm(int trans_a, int trans_b, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc);
int bcnn_xnor_gemm(int trans_a, int trans_b, int M, int N, int K, float ALPHA,
                        uint32_t *A, int lda,
                        uint32_t *B, int ldb,
						float BETA,
                        float *C, int ldc);
float bcnn_l2_distance(float *x, float *y, int n);
float bcnn_sqrdiff_vs(float *x, float a, int n);
float bcnn_shiftdot(int n, float *x, float a, float *y, float b);
int bcnn_varnorm(int n, float *a, float c, float *y);
int bcnn_varmean(int n, float *m, float a, float *var);

int bcnn_init_workload(bcnn_net *net);
int bcnn_free_workload(bcnn_net *net);

/* Network allocation routines */
int bcnn_alloc(bcnn_net *net, int nb_layers);
int bcnn_realloc(bcnn_net *net, int nb_layers);

/* Conv layer */
int bcnn_add_convolutional_layer(bcnn_net *net, int n, int size, int stride, int pad,
	int batch_norm, bcnn_weights_init init, bcnn_activation activation, int quantize, char *id);
int bcnn_forward_conv_layer(bcnn_connection *conn);
int bcnn_backward_conv_layer(bcnn_connection *conn);

/* Deconv layer */
int bcnn_add_deconvolutional_layer(bcnn_net *net, int n, int size, int stride, int pad,
	bcnn_weights_init init, bcnn_activation activation, char *id);
int bcnn_forward_deconv_layer(bcnn_connection *conn);
int bcnn_backward_deconv_layer(bcnn_connection *conn);

/* Batchnorm layer */
int bcnn_add_batchnorm_layer(bcnn_net *net, char *id);
int bcnn_forward_batchnorm_layer(bcnn_connection *conn);
int bcnn_backward_batchnorm_layer(bcnn_connection *conn);

/* Full-connected layer */
int bcnn_add_fullc_layer(bcnn_net *net, int output_size, bcnn_weights_init init, bcnn_activation activation, int quantize, char *id);
int bcnn_forward_fullc_layer(bcnn_connection *conn);
int bcnn_backward_fullc_layer(bcnn_connection *conn);

/* Activation layer */
int bcnn_add_activation_layer(bcnn_net *net, bcnn_activation type, char *id);
int bcnn_forward_activation_cpu(float *x, int sz, bcnn_activation a);
int bcnn_forward_activation_layer(bcnn_connection *conn);
int bcnn_backward_activation_cpu(float *x, float *dx, int sz, bcnn_activation a);
int bcnn_backward_activation_layer(bcnn_connection *conn);

/* Softmax layer */
int bcnn_add_softmax_layer(bcnn_net *net, char *id);
int bcnn_forward_softmax_layer(bcnn_connection *conn);
int bcnn_backward_softmax_layer(bcnn_connection *conn);

/* Pooling layer */
int bcnn_add_maxpool_layer(bcnn_net *net, int size, int stride, char *id);
int bcnn_forward_maxpool_layer(bcnn_connection *conn);
int bcnn_backward_maxpool_layer(bcnn_connection *conn);

/* Concat layer */
int bcnn_add_concat_layer(bcnn_net *net, char *concat, char *id);
int bcnn_forward_concat_layer(bcnn_net *net, bcnn_connection *conn);
int bcnn_backward_concat_layer(bcnn_net *net, bcnn_connection *conn);

/* Dropout layer */
int bcnn_add_dropout_layer(bcnn_net *net, float rate, char *id);
int bcnn_forward_dropout_layer(bcnn_connection *conn);
int bcnn_backward_dropout_layer(bcnn_connection *conn);

/* Cost layer */
int bcnn_add_cost_layer(bcnn_net *net, bcnn_loss_metric loss_metric, float scale);
int bcnn_forward_cost_layer(bcnn_connection *conn);
int bcnn_backward_cost_layer(bcnn_connection *conn);

/* Core network routines */
int bcnn_update(bcnn_net *net);
int bcnn_sgd_optimizer(bcnn_connection *conn, int batch_size, float learning_rate, float momentum, float decay);
int bcnn_visualize_network(bcnn_net *net);
int bcnn_forward(bcnn_net *net);
int bcnn_backward(bcnn_net *net);

/* General routines for training / predict */
int bcnn_train_on_batch(bcnn_net *net, bcnn_iterator *iter, float *loss);
int bcnn_predict_on_batch(bcnn_net *net, bcnn_iterator *iter, float **pred, float *error);

/* Free routines */
int bcnn_free_layer(bcnn_layer **layer);
int bcnn_free_net(bcnn_net *cnn);

/* Helpers */
int bcnn_pack_data(char *list, int label_width, bcnn_label_type type, char *out_pack);
int bcnn_load_image_from_csv(char *str, int w, int h, int c, unsigned char **img);
int bcnn_load_image_from_path(char *path, int w, int h, int c, unsigned char **img, int state, int *x_shift, int *y_shift);
int bcnn_load_image_from_memory(unsigned char *buffer, int buffer_size, int w, int h, int c, unsigned char **img, int state,
	int *x_shift, int *y_shift);
int bcnn_data_augmentation(unsigned char *img, int width, int height, int depth, bcnn_data_augment *param,
	unsigned char *buffer);
unsigned int _read_int(char *v);

void get_binary_row(float *row, uint32_t *bin_row, int size);
void get_binary_col(float *col, uint32_t *bin_col, int n, int k);
void get_binary_col_unrolled(float *col, uint32_t *bin_col, int n, int k);

int bcnn_iter_batch(bcnn_net *net, bcnn_iterator *iter);

int bcnn_convert_img_to_float(unsigned char *src, int w, int h, int c, int swap_to_bgr, 
	float mean_r, float mean_g, float mean_b, float *dst);

typedef struct {
	int state;
	float r;
} bcnn_gauss_gen;
float bcnn_rng_gaussian(bcnn_gauss_gen *g);

/* Cuda kernels routines */
#ifdef BCNN_USE_CUDA
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
#define BCNN_CUDA_THREADS 1024
#else
#define BCNN_CUDA_THREADS 512
#endif

#define bcnn_cuda_check(RET) {													\
	if ((RET) != cudaSuccess) {													\
		fprintf(stderr, "[ERROR] [CUDA] %s\n", cudaGetErrorString((RET)));		\
		exit((RET));															\
	}																			\
}
#define bcnn_cublas_check(RET) {												\
	if ((RET) != CUBLAS_STATUS_SUCCESS) {										\
		fprintf(stderr, "[ERROR] [CUBLAS] %d\n", (int)(RET));					\
		exit((RET));															\
	}																			\
}

#define bcnn_curand_check(RET) {												\
	if ((RET) != CURAND_STATUS_SUCCESS) {										\
		fprintf(stderr, "[ERROR] [CURAND] %d\n", (int)(RET));					\
		exit((RET));															\
	}																			\
}

#ifdef BCNN_USE_CUDNN
#define bcnn_cudnn_check(RET) {													\
	if ((RET) != CUDNN_STATUS_SUCCESS) {										\
		fprintf(stderr, "[ERROR] [CUDNN] %s\n", cudnnGetErrorString((RET)));	\
		exit((RET));															\
	}																			\
}

cudnnHandle_t bcnn_cudnn_handle();
#endif

/* Cuda generic helpers */
cublasHandle_t bcnn_cublas_handle();
dim3 bcnn_cuda_gridsize(unsigned int n);
int *bcnn_cuda_malloc_i32(int n);
float *bcnn_cuda_malloc_f32(int n);
float *bcnn_cuda_memcpy_f32(float *x, int n);
void bcnn_cuda_fill_with_random(float *x_gpu, int n);
void bcnn_cuda_free(void *x_gpu);
void bcnn_cuda_memcpy_host2dev(float *x_gpu, float *x, int n);
void bcnn_cuda_memcpy_dev2host(float *x_gpu, float *x, int n);

void bcnn_cuda_gemm(int trans_a, int trans_b, int m, int n, int k, float alpha,
	float *a, int lda,
	float *b, int ldb,
	float beta,
	float *c, int ldc);
void bcnn_cuda_gemv(int trans_a, const int m,
	const int n, const float alpha, const float *a, const float *x,
	const float beta, float *y);
void bcnn_cuda_fill_f32(int n, float alpha, float *x, int incx);
void bcnn_cuda_copy_f32(int n, float * x, int incx, float * y, int incy);
void bcnn_cuda_axpy(int n, float alpha, float *x, int incx, float *y, int incy);
void bcnn_cuda_scal(int n, float alpha, float *x, int incx);
void bcnn_cuda_pow(int n, float *x, float a, float *y);
void bcnn_cuda_axpby(int n, float a, float *x, float b, float *y);
void bcnn_cuda_add_scalar(int n, float a, float* y);
void bcnn_cuda_vadd(int n, float *a, float *b, float *y);
void bcnn_cuda_vsub(int n, float *a, float *b, float *y);
void bcnn_cuda_vmul(int n, float *a, float *b, float *y);
void bcnn_cuda_vdiv(int n, float *a, float *b, float *y);

void bcnn_cuda_mean_variance_forward(float *x, int b, int c, int wxh, float *mean, float *var);
void bcnn_cuda_norm_forward(float *x, float *mean, float *variance, int b, int c, int wxh);
void bcnn_cuda_mean_variance_backward(float *x, float *grad, float *mean, float *var, int b, int c, int wxh, float *mean_diff, float *var_diff);
void bcnn_cuda_norm_backward(float *x, float *mean, float *var, float *mean_diff, float *var_diff, int b, int c, int wxh, float *grad);

void bcnn_im2col_gpu(float *im,
	int channels, int height, int width,
	int ksize, int stride, int pad, float *data_col);
void bcnn_col2im_gpu(float *data_col,
	int channels, int height, int width,
	int ksize, int stride, int pad, float *data_im);

int bcnn_forward_bias_gpu(float *output, float *biases, int batch_size, int n, int size);
int bcnn_backward_bias_gpu(float *bias_diff, float *diff, int batch_size, int n, int size);

int bcnn_forward_activation_gpu(float *x, int sz, bcnn_activation a);
int bcnn_forward_activation_layer_gpu(bcnn_connection *conn);
int bcnn_backward_activation_gpu(float *x, float *dx, int sz, bcnn_activation a);
int bcnn_backward_activation_layer_gpu(bcnn_connection *conn);
int bcnn_forward_conv_layer_gpu(bcnn_connection *conn);
int bcnn_backward_conv_layer_gpu(bcnn_connection *conn);
int bcnn_forward_deconv_layer_gpu(bcnn_connection *conn);
int bcnn_backward_deconv_layer_gpu(bcnn_connection *conn);
int bcnn_forward_concat_layer_gpu(bcnn_connection *conn);
int bcnn_backward_concat_layer_gpu(bcnn_connection *conn);
int bcnn_forward_fullc_layer_gpu(bcnn_connection *conn);
int bcnn_backward_fullc_layer_gpu(bcnn_connection *conn);
int bcnn_forward_maxpool_layer_gpu(bcnn_connection *conn);
int bcnn_backward_maxpool_layer_gpu(bcnn_connection *conn);
int bcnn_forward_dropout_layer_gpu(bcnn_connection *conn);
int bcnn_backward_dropout_layer_gpu(bcnn_connection *conn);
int bcnn_forward_softmax_layer_gpu(bcnn_connection *conn);
int bcnn_backward_softmax_layer_gpu(bcnn_connection *conn);
int bcnn_forward_batchnorm_layer_gpu(bcnn_connection *conn);
int bcnn_backward_batchnorm_layer_gpu(bcnn_connection *conn);
int bcnn_forward_cost_layer_gpu(bcnn_connection *conn);
int bcnn_backward_cost_layer_gpu(bcnn_connection *conn);
#endif


#ifdef __cplusplus
}
#endif

#endif
