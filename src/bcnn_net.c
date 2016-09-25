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

/* include bh helpers */
#include <bh/bh.h>
#include <bh/bh_error.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>
#include <bh/bh_timer.h>

/* include bip image processing lib */
#include <bip/bip.h>

#include "bcnn/bcnn.h"


int bcnn_init_net(bcnn_net **net)
{
	if (*net == NULL) {
		*net = (bcnn_net *)calloc(1, sizeof(bcnn_net));
	}

	return BCNN_SUCCESS;
}

int bcnn_end_net(bcnn_net **net)
{
	bcnn_free_net(*net);
	bh_free(*net);		

	return BCNN_SUCCESS;
}

int bcnn_free_net(bcnn_net *net)
{
	int i;
    for (i = 0; i < net->nb_layers; ++i) {
        bcnn_free_layer(&net->layers[i]);
    }
    bh_free(net->layers);
	bcnn_free_workload(net);
	return BCNN_SUCCESS;
}

int bcnn_set_param(bcnn_net *net, char *name, char *val)
{
	if (strcmp(name, "input_width") == 0) net->w = atoi(val);
	else if (strcmp(name, "input_height") == 0) net->h = atoi(val); 
	else if (strcmp(name, "input_channels") == 0) net->c = atoi(val); 
	else if (strcmp(name, "batch_size") == 0) net->batch_size = atoi(val);
	else if (strcmp(name, "max_batches") == 0) net->max_batches = atoi(val); 
	else if (strcmp(name, "loss") == 0) {
		if (strcmp(val, "error") == 0) net->loss_metric = COST_ERROR;
		else if (strcmp(val, "logloss") == 0) net->loss_metric = COST_LOGLOSS;
		else if (strcmp(val, "sse") == 0) net->loss_metric = COST_SSE;
		else if (strcmp(val, "mse") == 0) net->loss_metric = COST_MSE;
		else if (strcmp(val, "crps") == 0) net->loss_metric = COST_CRPS;
		else if (strcmp(val, "dice") == 0) net->loss_metric = COST_DICE;
		else {
			fprintf(stderr, "[WARNING] Unknown cost metric %s, going with sse\n", val);
			net->loss_metric = COST_SSE;
		}
	}
	else if (strcmp(name, "learning_policy") == 0) {
		if (strcmp(val, "sigmoid") == 0) net->learner.policy = SIGMOID;
		else if (strcmp(val, "constant") == 0) net->learner.policy = CONSTANT;
		else if (strcmp(val, "exp") == 0) net->learner.policy = EXP;
		else if (strcmp(val, "inv") == 0) net->learner.policy = INV;
		else if (strcmp(val, "step") == 0) net->learner.policy = STEP;
		else if (strcmp(val, "poly") == 0) net->learner.policy = POLY;
		else net->learner.policy = CONSTANT;
	}
	else if (strcmp(name, "step") == 0) net->learner.step = atoi(val);
	else if (strcmp(name, "learning_rate") == 0) net->learner.learning_rate = (float)atof(val);
	else if (strcmp(name, "decay") == 0) net->learner.decay = (float)atof(val);
	else if (strcmp(name, "momentum") == 0) net->learner.momentum = (float)atof(val);
	else if (strcmp(name, "gamma") == 0) net->learner.gamma = (float)atof(val);
	else if (strcmp(name, "range_shift_x") == 0) net->data_aug.range_shift_x = atoi(val);
	else if (strcmp(name, "range_shift_y") == 0) net->data_aug.range_shift_y = atoi(val);
	else if (strcmp(name, "min_scale") == 0) net->data_aug.min_scale = (float)atof(val);
	else if (strcmp(name, "max_scale") == 0) net->data_aug.max_scale = (float)atof(val);
	else if (strcmp(name, "rotation_range") == 0) net->data_aug.rotation_range = (float)atof(val);
	else if (strcmp(name, "min_contrast") == 0) net->data_aug.min_contrast = (float)atof(val);
	else if (strcmp(name, "max_contrast") == 0) net->data_aug.max_contrast = (float)atof(val);
	else if (strcmp(name, "min_brightness") == 0) net->data_aug.min_brightness = atoi(val);
	else if (strcmp(name, "max_brightness") == 0) net->data_aug.max_brightness = atoi(val);
	else if (strcmp(name, "max_distortion") == 0) net->data_aug.max_distortion = (float)atof(val);
	else if (strcmp(name, "prediction_type") == 0) {
		if (strcmp(val, "classif") == 0 || strcmp(val, "classification") == 0) net->prediction_type = CLASSIFICATION;
		else if (strcmp(val, "reg") == 0 || strcmp(val, "regression") == 0) net->prediction_type = REGRESSION;
		else if (strcmp(val, "heatmap") == 0 || strcmp(val, "heatmap_regression") == 0) net->prediction_type = HEATMAP_REGRESSION;
		else if (strcmp(val, "segmentation") == 0) net->prediction_type = SEGMENTATION;
	}
	return BCNN_SUCCESS;
}

int bcnn_alloc(bcnn_net *net, int nb_layers)
{
	net->nb_layers = nb_layers;
	net->layers = (bcnn_layer *)calloc(net->nb_layers, sizeof(bcnn_layer));
	return BCNN_SUCCESS;
}

int bcnn_realloc(bcnn_net *net, int nb_layers)
{
	net->nb_layers = nb_layers;
	net->layers = (bcnn_layer *)realloc(net->layers, net->nb_layers * sizeof(bcnn_layer));
	if (net->layers == NULL)
		bh_error("bcnn_realloc: Allocation failed", BCNN_FAILED_ALLOC);
	return BCNN_SUCCESS;
}

int bcnn_init_workload(bcnn_net *net, int is_train)
{
	int sz = net->w * net->h * net->c;

	net->wrk.input = (float *)calloc(sz * net->batch_size, sizeof(float));
	net->wrk.truth = (float *)calloc(net->output_size * net->batch_size, sizeof(float));
	net->wrk.train = is_train;
	net->wrk.diff = NULL;
	net->wrk.batch_size = net->batch_size;
#ifdef BCNN_USE_CUDA
	net->wrk.input_gpu = bcnn_cuda_malloc_f32(sz * net->batch_size);
	net->wrk.truth_gpu = bcnn_cuda_malloc_f32(net->output_size * net->batch_size);
#endif

	return BCNN_SUCCESS;
}


int bcnn_free_workload(bcnn_net *net)
{
	bh_free(net->wrk.input);
	bh_free(net->wrk.truth);
#ifdef BCNN_USE_CUDA
	bcnn_cuda_free(net->wrk.input_gpu);
	bcnn_cuda_free(net->wrk.truth_gpu);
#endif
	return 0;
}

int bcnn_compile_net(bcnn_net *net, char *phase)
{
	int n = net->nb_layers;
	int k = (net->layers[n - 1].type == COST ? (n - 2) : (n - 1));
	net->output_size = net->layers[k].output_shape[0] * net->layers[k].output_shape[1] *
			net->layers[k].output_shape[2];

	bcnn_free_workload(net);
	if (strcmp(phase, "train") == 0) {
		bcnn_init_workload(net, 1);
	}
	else if (strcmp(phase, "predict") == 0) {
		bcnn_init_workload(net, 0);
	}
	else {
		fprintf(stderr, "[ERROR] bcnn_compile_net: Available option are 'train' and 'predict'");
		return BCNN_INVALID_PARAMETER;
	}

	return BCNN_SUCCESS;
}


int bcnn_forward(bcnn_net *net, bcnn_workload *wrk)
{
	 int i, sz;
	 bcnn_layer layer = { 0 };
#ifdef BCNN_USE_CUDA
	 float *original_input = wrk->input_gpu;
#else
	 float *original_input = wrk->input;
#endif

	 for (i = 0; i < net->nb_layers; ++i) {
        layer = net->layers[i];
		sz = layer.output_shape[0] * layer.output_shape[1] * layer.output_shape[2] *
			wrk->batch_size;
#ifdef BCNN_USE_CUDA
		if (layer.diff_gpu != NULL) bcnn_cuda_fill_f32(sz, 0.0f, layer.diff_gpu, 1);
#else
		if (layer.diff != NULL) memset(layer.diff, 0, sz * sizeof(float));
#endif
		switch (layer.type) {
		case CONVOLUTIONAL:
			bcnn_forward_conv_layer(&layer, wrk);
			break;
		case DECONVOLUTIONAL:
			bcnn_forward_deconv_layer(&layer, wrk);
			break;
		case ACTIVATION:
			bcnn_forward_activation_layer(&layer, wrk);
			break;
		case BATCHNORM:
			bcnn_forward_batchnorm_layer(&layer, wrk);
			break;
		case FULL_CONNECTED:
			bcnn_forward_fullc_layer(&layer, wrk);
			break;
		case MAXPOOL:
			bcnn_forward_maxpool_layer(&layer, wrk);
			break;
		case SOFTMAX:
			bcnn_forward_softmax_layer(&layer, wrk);
			break;
		case DROPOUT:
			bcnn_forward_dropout_layer(&layer, wrk);
			break;
		case CONCAT:
#ifdef BCNN_USE_CUDA
			wrk->input2_gpu = net->layers[layer.concat_index].output_gpu;
#endif
			wrk->input2 = net->layers[layer.concat_index].output;
			bcnn_forward_concat_layer(&layer, wrk);
			break;
		case COST:
			bcnn_forward_cost_layer(&layer, wrk);
			break;
		default:
			break;
		}
		// Update for next layer
#ifdef BCNN_USE_CUDA
		wrk->input_gpu = layer.output_gpu;
#else
		wrk->input = layer.output;
#endif
    }
#ifdef BCNN_USE_CUDA
	wrk->input_gpu = original_input;
#else
	wrk->input = original_input;
#endif
	return BCNN_SUCCESS;
}

int bcnn_forward2(bcnn_net *net)
{
	 int i, sz;
	 bcnn_layer layer = { 0 };
	 bcnn_workload wrk = net->wrk;
#ifdef BCNN_USE_CUDA
	 float *original_input = wrk.input_gpu;
#else
	 float *original_input = wrk.input;
#endif

	 for (i = 0; i < net->nb_layers; ++i) {
        layer = net->layers[i];
		sz = layer.output_shape[0] * layer.output_shape[1] * layer.output_shape[2] *
			wrk.batch_size;
#ifdef BCNN_USE_CUDA
		if (layer.diff_gpu != NULL) bcnn_cuda_fill_f32(sz, 0.0f, layer.diff_gpu, 1);
#else
		if (layer.diff != NULL) memset(layer.diff, 0, sz * sizeof(float));
#endif
		switch (layer.type) {
		case CONVOLUTIONAL:
			bcnn_forward_conv_layer(&layer, &wrk);
			break;
		case DECONVOLUTIONAL:
			bcnn_forward_deconv_layer(&layer, &wrk);
			break;
		case ACTIVATION:
			bcnn_forward_activation_layer(&layer, &wrk);
			break;
		case BATCHNORM:
			bcnn_forward_batchnorm_layer(&layer, &wrk);
			break;
		case FULL_CONNECTED:
			bcnn_forward_fullc_layer(&layer, &wrk);
			break;
		case MAXPOOL:
			bcnn_forward_maxpool_layer(&layer, &wrk);
			break;
		case SOFTMAX:
			bcnn_forward_softmax_layer(&layer, &wrk);
			break;
		case DROPOUT:
			bcnn_forward_dropout_layer(&layer, &wrk);
			break;
		case CONCAT:
#ifdef BCNN_USE_CUDA
			wrk.input2_gpu = net->layers[layer.concat_index].output_gpu;
#endif
			wrk.input2 = net->layers[layer.concat_index].output;
			bcnn_forward_concat_layer(&layer, &wrk);
			break;
		case COST:
			bcnn_forward_cost_layer(&layer, &wrk);
			break;
		default:
			break;
		}
		// Update for next layer
#ifdef BCNN_USE_CUDA
		wrk.input_gpu = layer.output_gpu;
#else
		wrk.input = layer.output;
#endif
    }
#ifdef BCNN_USE_CUDA
	wrk.input_gpu = original_input;
#else
	wrk.input = original_input;
#endif
	return BCNN_SUCCESS;
}

int bcnn_backward(bcnn_net *net, bcnn_workload *wrk)
{
	int i;
	bcnn_layer curr_layer, prev_layer;
    float *original_delta = wrk->diff;
#ifdef BCNN_USE_CUDA
	float *original_input = wrk->input_gpu;
#else
	float *original_input = wrk->input;
#endif

    for (i = net->nb_layers - 1; i >= 0; --i) {
		if (i == 0) {
#ifdef BCNN_USE_CUDA
            wrk->input_gpu = original_input;
#else
			wrk->input = original_input;
#endif
			wrk->diff = original_delta;
        }
		else {
            prev_layer = net->layers[i - 1];
#ifdef BCNN_USE_CUDA
			wrk->input_gpu = prev_layer.output_gpu;
			wrk->diff = prev_layer.diff_gpu;
#else
			wrk->input = prev_layer.output;
			wrk->diff = prev_layer.diff;
#endif
        }
		curr_layer = net->layers[i];
		switch (curr_layer.type) {
		case CONVOLUTIONAL:
			bcnn_backward_conv_layer(&curr_layer, wrk);
			break;
		case DECONVOLUTIONAL:
			bcnn_backward_deconv_layer(&curr_layer, wrk);
			break;
		case ACTIVATION:
			bcnn_backward_activation_layer(&curr_layer, wrk);
			break;
		case BATCHNORM:
			bcnn_backward_batchnorm_layer(&curr_layer, wrk);
			break;
		case FULL_CONNECTED:
			bcnn_backward_fullc_layer(&curr_layer, wrk);
			break;
		case MAXPOOL:
			bcnn_backward_maxpool_layer(&curr_layer, wrk);
			break;
		case SOFTMAX:
			bcnn_backward_softmax_layer(&curr_layer, wrk);
			break;
		case DROPOUT:
			bcnn_backward_dropout_layer(&curr_layer, wrk);
			break;
		case CONCAT:
#ifdef BCNN_USE_CUDA
			wrk->diff2 = net->layers[curr_layer.concat_index].diff_gpu;
#endif
			wrk->diff2 = net->layers[curr_layer.concat_index].diff;
			bcnn_backward_concat_layer(&curr_layer, wrk);
			break;
		case COST:
			bcnn_backward_cost_layer(&curr_layer, wrk);
			break;
		default:
			break;
		}
    }
	return BCNN_SUCCESS;
}

int bcnn_backward2(bcnn_net *net)
{
	int i;
	bcnn_layer curr_layer, prev_layer;
	bcnn_workload wrk = net->wrk;

    float *original_delta = wrk.diff;
#ifdef BCNN_USE_CUDA
	float *original_input = wrk.input_gpu;
#else
	float *original_input = wrk.input;
#endif

    for (i = net->nb_layers - 1; i >= 0; --i) {
		if (i == 0) {
#ifdef BCNN_USE_CUDA
            wrk.input_gpu = original_input;
#else
			wrk.input = original_input;
#endif
			wrk.diff = original_delta;
        }
		else {
            prev_layer = net->layers[i - 1];
#ifdef BCNN_USE_CUDA
			wrk.input_gpu = prev_layer.output_gpu;
			wrk.diff = prev_layer.diff_gpu;
#else
			wrk.input = prev_layer.output;
			wrk.diff = prev_layer.diff;
#endif
        }
		curr_layer = net->layers[i];
		switch (curr_layer.type) {
		case CONVOLUTIONAL:
			bcnn_backward_conv_layer(&curr_layer, &wrk);
			break;
		case DECONVOLUTIONAL:
			bcnn_backward_deconv_layer(&curr_layer, &wrk);
			break;
		case ACTIVATION:
			bcnn_backward_activation_layer(&curr_layer, &wrk);
			break;
		case BATCHNORM:
			bcnn_backward_batchnorm_layer(&curr_layer, &wrk);
			break;
		case FULL_CONNECTED:
			bcnn_backward_fullc_layer(&curr_layer, &wrk);
			break;
		case MAXPOOL:
			bcnn_backward_maxpool_layer(&curr_layer, &wrk);
			break;
		case SOFTMAX:
			bcnn_backward_softmax_layer(&curr_layer, &wrk);
			break;
		case DROPOUT:
			bcnn_backward_dropout_layer(&curr_layer, &wrk);
			break;
		case CONCAT:
#ifdef BCNN_USE_CUDA
			wrk.diff2 = net->layers[curr_layer.concat_index].diff_gpu;
#endif
			wrk.diff2 = net->layers[curr_layer.concat_index].diff;
			bcnn_backward_concat_layer(&curr_layer, &wrk);
			break;
		case COST:
			bcnn_backward_cost_layer(&curr_layer, &wrk);
			break;
		default:
			break;
		}
    }
	return BCNN_SUCCESS;
}


static float bcnn_update_learning_rate(bcnn_net *net)
{
    int iter = net->seen / net->batch_size;
    switch (net->learner.policy) {
        case CONSTANT:
            return net->learner.learning_rate;
        case STEP:
            return net->learner.learning_rate * (float)pow(net->learner.scale, iter / net->learner.step);
		case INV:
			return net->learner.learning_rate * (float)pow(1.0f + net->learner.gamma * iter, -net->learner.power);
        case EXP:
            return net->learner.learning_rate * (float)pow(net->learner.gamma, iter);
        case POLY:
            return net->learner.learning_rate * (float)pow(1 - (float)iter / net->max_batches, net->learner.power);
        case SIGMOID:
            return net->learner.learning_rate * (1.0f / (1.0f + (float)exp(net->learner.gamma * (iter - net->learner.step))));
        default:
            return net->learner.learning_rate;
    }
}


int bcnn_apply_update_to_layer(bcnn_layer *layer, int batch_size, float learning_rate, float momentum, float decay)
{
#ifdef BCNN_USE_CUDA
	bcnn_cuda_axpy(layer->bias_size, -learning_rate / batch_size, layer->bias_diff_gpu, 1, layer->bias_gpu, 1);
	bcnn_cuda_scal(layer->bias_size, momentum, layer->bias_diff_gpu, 1);

	bcnn_cuda_axpy(layer->weights_size, decay * batch_size, layer->weight_gpu, 1, layer->weight_diff_gpu, 1);
	bcnn_cuda_axpy(layer->weights_size, -learning_rate / batch_size, layer->weight_diff_gpu, 1, layer->weight_gpu, 1);
	bcnn_cuda_scal(layer->weights_size, momentum, layer->weight_diff_gpu, 1);

	if (layer->bn_scale_diff_gpu && layer->bn_scale_gpu && layer->bn_shift_gpu && layer->bn_shift_diff_gpu) {
		bcnn_cuda_axpy(layer->output_shape[2], -learning_rate / batch_size, layer->bn_scale_diff_gpu, 1, layer->bn_scale_gpu, 1);
		bcnn_cuda_scal(layer->output_shape[2], momentum, layer->bn_scale_diff_gpu, 1);
		bcnn_cuda_axpy(layer->output_shape[2], -learning_rate / batch_size, layer->bn_shift_diff_gpu, 1, layer->bn_shift_gpu, 1);
		bcnn_cuda_scal(layer->output_shape[2], momentum, layer->bn_shift_diff_gpu, 1);
	}
#else
	bcnn_axpy(layer->bias_size, -learning_rate / batch_size, layer->bias_diff, layer->bias);
	bcnn_scal(layer->bias_size, momentum, layer->bias_diff);

	bcnn_axpy(layer->weights_size, decay * batch_size, layer->weight, layer->weight_diff);
	bcnn_axpy(layer->weights_size, -learning_rate / batch_size, layer->weight_diff, layer->weight);
	bcnn_scal(layer->weights_size, momentum, layer->weight_diff);

	if (layer->bn_scale_diff && layer->bn_scale && layer->bn_shift && layer->bn_shift_diff) {
		bcnn_axpy(layer->output_shape[2], -learning_rate / batch_size, layer->bn_scale_diff, layer->bn_scale);
		bcnn_scal(layer->output_shape[2], momentum, layer->bn_scale_diff);
		bcnn_axpy(layer->output_shape[2], -learning_rate / batch_size, layer->bn_shift_diff, layer->bn_shift);
		bcnn_scal(layer->output_shape[2], momentum, layer->bn_shift_diff);
	}
#endif
	return 0;
}


int bcnn_update(bcnn_net *net)
{
    int i;
    int update_batch = net->batch_size;
    float rate = bcnn_update_learning_rate(net);

	for (i = 0; i < net->nb_layers; ++i) {
		if (net->layers[i].type == CONVOLUTIONAL || net->layers[i].type == FULL_CONNECTED || net->layers[i].type == BATCHNORM)
			bcnn_apply_update_to_layer(&net->layers[i], update_batch, rate, net->learner.momentum, net->learner.decay);
    }
	return BCNN_SUCCESS;
}

int bcnn_train_batch(bcnn_net *net, bcnn_workload *wrk, float *loss)
{
	net->seen += net->batch_size;
	// Forward
    bcnn_forward(net, wrk);
	// Back prop
    bcnn_backward(net, wrk);
	// Update network weight
	bcnn_update(net);
	*loss = net->layers[net->nb_layers - 1].output[0];

    return BCNN_SUCCESS;
}


int bcnn_iter_batch(bcnn_net *net, bcnn_iterator *iter)
{
	int i, j, sz = net->w * net->h * net->c, n, offset;
	int32_t w, h, c, out_w, out_h, out_c;
	char *line = NULL;
	uint8_t *img = NULL, *img_tmp = NULL;
	float *x = net->wrk.input;
	float *y = net->wrk.truth;
	float x_scale, y_scale;
	char **tok = NULL;
	int n_tok = 0, x_pos, y_pos;
	int use_buffer_img = (net->task == TRAIN && net->wrk.train != 0 &&
		(net->data_aug.range_shift_x != 0 || net->data_aug.range_shift_y != 0 ||
		net->data_aug.rotation_range != 0));
	bcnn_data_augment *param = &(net->data_aug);

	memset(x, 0, sz * net->batch_size * sizeof(float));
	if (net->task != PREDICT)
		memset(y, 0, net->output_size * net->batch_size * sizeof(float));

	if (use_buffer_img)
		img_tmp = (uint8_t *)calloc(sz, sizeof(uint8_t));
	
	if (iter->type == ITER_MNIST) {
		for (i = 0; i < net->batch_size; ++i) {
			bcnn_mnist_next_iter(net, iter);
			// Data augmentation
			if (net->task == TRAIN && net->wrk.train)
				bcnn_data_augmentation(iter->input_uchar, net->w, net->h, net->c, param, img_tmp);
			bip_convert_u8_to_f32(iter->input_uchar, net->w, net->h, net->c, net->w * net->c, x);
			x += sz;
			if (net->task != PREDICT) {
				// Load truth
				y[iter->label_int[0]] = 1;
				y += net->output_size;
			}
		}
	}
	else if (net->prediction_type == CLASSIFICATION) { //  Classification
		for (i = 0; i < net->batch_size; ++i) {
			//nb_lines_skipped = (int)((float)rand() / RAND_MAX * net->batch_size);
			//bh_fskipline(f, nb_lines_skipped);
			line = bh_fgetline(iter->f_input);
			if (line == NULL) {
				rewind(iter->f_input);
				line = bh_fgetline(iter->f_input);
			}
			n_tok = bh_strsplit(line, ' ', &tok);
			if (net->task != PREDICT) {
				bh_assert(n_tok == 2,
					"Wrong data format for classification", BCNN_INVALID_DATA);
			}
			if (iter->type == ITER_LIST) {
				bip_load_image(tok[0], &img, &w, &h, &c);
				bh_assert(w == net->w && h == net->h && c == net->c,
					"Network input size and data size are different",
					BCNN_INVALID_DATA);
			}
			else
				bcnn_load_image_from_csv(tok[0], net->w, net->h, net->c, &img);
			// Online data augmentation
			if (net->task == TRAIN && net->wrk.train)
				bcnn_data_augmentation(img, net->w, net->h, net->c, param, img_tmp);
			bip_convert_u8_to_f32(img, net->w, net->h, net->c, net->w * net->c, x);
			bh_free(img);
			x += sz;
			if (net->task != PREDICT) {
				// Load truth
				y[atoi(tok[1])] = 1;
				y += net->output_size;
			}
			bh_free(line);
			for (j = 0; j < n_tok; ++j)
				bh_free(tok[j]);
			bh_free(tok);
		}
	}
	else if (net->prediction_type == REGRESSION) { // Regression
		for (i = 0; i < net->batch_size; ++i) {
			line = bh_fgetline(iter->f_input);
			if (line == NULL) {
				rewind(iter->f_input);
				line = bh_fgetline(iter->f_input);
			}
			j = 0;
			//field = strtok(line, delims);
			n_tok = bh_strsplit(line, ' ', &tok);
			if (net->task != PREDICT) {
				bh_assert(n_tok == net->output_size + 1,
					"Wrong data format for regression", BCNN_INVALID_DATA);
			}
			if (iter->type == IMG) {
				bip_load_image(tok[0], &img, &w, &h, &c);
				bh_assert(w == net->w && h == net->h && c == net->c,
					"Network input size and data size are different",
					BCNN_INVALID_DATA);
			}
			else
				bcnn_load_image_from_csv(tok[0], net->w, net->h, net->c, &img);
			// Online data augmentation
			if (net->task == TRAIN && net->wrk.train)
				bcnn_data_augmentation(img, net->w, net->h, net->c, param, img_tmp);
			bip_convert_u8_to_f32(img, net->w, net->h, net->c, net->w * net->c, x);
			bh_free(img);
			x += sz;
			if (net->task != PREDICT) {
				// Load truth
				for (j = 1; j < n_tok; ++j) {
					y[j - 1] = (float)atof(tok[j]);
				}
				y += net->output_size;
			}
			bh_free(line);
			for (j = 0; j < n_tok; ++j)
				bh_free(tok[j]);
			bh_free(tok);
		}
	}
	else if (net->prediction_type == HEATMAP_REGRESSION) {
		// Format must be this way: train/img.png X1 Y1 ... Xn Yn
		// With (Xk, Yk) being a target position on the heatmap
		// Size of the heatmap is the last (conv) layer size before cost
		for (i = 0; i < net->batch_size; ++i) {
			line = bh_fgetline(iter->f_input);
			if (line == NULL) {
				rewind(iter->f_input);
				line = bh_fgetline(iter->f_input);
			}
			n_tok = bh_strsplit(line, ' ', &tok);
			if (net->task != PREDICT) {
				bh_assert(n_tok > 0 && (n_tok % 2) != 0,
					"Wrong data format for heatmap regression", BCNN_INVALID_DATA);
			}
			if (iter->type == IMG) {
				bip_load_image(tok[0], &img, &w, &h, &c);
				bh_assert(w == net->w && h == net->h && c == net->c,
					"Network input size and data size are different",
					BCNN_INVALID_DATA);
			}
			else
				bcnn_load_image_from_csv(tok[0], net->w, net->h, net->c, &img);
			// Online data augmentation
			if (net->task == TRAIN && net->wrk.train)
				bcnn_data_augmentation(img, w, h, c, param, img_tmp);
			bip_convert_u8_to_f32(img, w, h, c, w * c, x);
			bh_free(img);
			x += sz;
			if (net->task != PREDICT) {
				// Load truth
				w = net->layers[net->nb_layers - 2].output_shape[0];
				h = net->layers[net->nb_layers - 2].output_shape[1];
				c = net->layers[net->nb_layers - 2].output_shape[2];
				x_scale = w / (float)net->w;
				y_scale = h / (float)net->h;
				for (j = 1; j < n_tok; j += 2) {
					x_pos = (int)(atof(tok[j]) * x_scale);
					y_pos = (int)(atof(tok[j + 1]) * y_scale);
					// Set gaussian kernel around (x_pos,y_pos)
					n = j >> 1;
					offset = n * w * h + (y_pos * w + x_pos);
					if (x_pos >= 0 && x_pos < w && y_pos >= 0 && y_pos < h) y[offset] = 1.0f;
					if (x_pos > 0) y[offset - 1] = 0.5f;
					if (x_pos < w - 1) y[offset + 1] = 0.5f;
					if (y_pos > 0) y[offset - w] = 0.5f;
					if (y_pos < h - 1) y[offset + w] = 0.5f;
					if (x_pos > 0 && y_pos > 0) y[offset - w - 1] = 0.25f;
					if (x_pos < w - 1 && y_pos > 0) y[offset - w + 1] = 0.25f;
					if (x_pos > 0 && y_pos < h - 1) y[offset + w - 1] = 0.25f;
					if (x_pos < w - 1 && y_pos < h - 1) y[offset + w + 1] = 0.25f;
				}
				y += net->output_size;
			}
			bh_free(line);
			for (j = 0; j < n_tok; ++j)
				bh_free(tok[j]);
			bh_free(tok);
		}
	}
	else if (net->prediction_type == SEGMENTATION) {
		// Format must be this way: img.png mask.png
		out_w = net->layers[net->nb_layers - 2].output_shape[0];
		out_h = net->layers[net->nb_layers - 2].output_shape[1];
		out_c = net->layers[net->nb_layers - 2].output_shape[2];
		for (i = 0; i < net->batch_size; ++i) {
			line = bh_fgetline(iter->f_input);
			if (line == NULL) {
				rewind(iter->f_input);
				line = bh_fgetline(iter->f_input);
			}
			n_tok = bh_strsplit(line, ' ', &tok);
			if (net->task != PREDICT) {
				bh_assert(n_tok > 0 && (n_tok == 2),
					"Wrong data format for segmentation", BCNN_INVALID_DATA);
			}
			// First load the image
			if (iter->type == IMG) {
				bip_load_image(tok[0], &img, &w, &h, &c);
				bh_assert(w == net->w && h == net->h && c == net->c,
					"Network input size and data size are different",
					BCNN_INVALID_DATA);
			}
			else
				bcnn_load_image_from_csv(tok[0], net->w, net->h, net->c, &img);
			// Online data augmentation
			if (net->task == TRAIN && net->wrk.train) {
				net->data_aug.use_precomputed = 0;
				bcnn_data_augmentation(img, net->w, net->h, net->c, param, img_tmp);
			}
			bip_convert_u8_to_f32(img, net->w, net->h, net->c, net->w * net->c, x);
			bh_free(img);
			x += sz;
			if (net->task != PREDICT) {
				// Load truth i.e the segmentation mask
				if (iter->type == IMG) {
					bip_load_image(tok[1], &img, &w, &h, &c);
					bh_assert(w == out_w &&
					h == out_h &&
					c == out_c,
					"Segmentation mask size and output size of the network must be the same",
					BCNN_INVALID_DATA);
				}
				else
					bcnn_load_image_from_csv(tok[1], out_w, out_h, out_c, &img);
				if (net->wrk.train) {
					// Apply to the mask the same data augmentation parameters applied to the image
					net->data_aug.use_precomputed = 1;
					bcnn_data_augmentation(img, net->w, net->h, net->c, param, img_tmp);
				}
				bip_convert_u8_to_f32(img, out_w, out_h, out_c, out_w * out_c, y);
				bh_free(img);
				y += net->output_size;
			}
			bh_free(line);
			for (j = 0; j < n_tok; ++j)
				bh_free(tok[j]);
			bh_free(tok);
		}
	}
	if (use_buffer_img)
		bh_free(img_tmp);

#ifdef BCNN_USE_CUDA
	bcnn_cuda_memcpy_host2dev(net->wrk.input_gpu, net->wrk.input, sz * net->batch_size);
	if (net->task != PREDICT)
		bcnn_cuda_memcpy_host2dev(net->wrk.truth_gpu, net->wrk.truth, net->output_size * net->batch_size);
#endif
	return BCNN_SUCCESS;
}

int bcnn_load_batch_mnist(bcnn_iterator *iter, bcnn_net *net, bcnn_workload *wrk)
{
	unsigned char *img_tmp = NULL;
	int i, sz = net->w * net->h * net->c;
	float *x = wrk->input;
	float *y = wrk->truth;
	int use_buffer_img = (net->task == TRAIN && wrk->train != 0 &&
		(net->data_aug.range_shift_x != 0 || net->data_aug.range_shift_y != 0 ||
		net->data_aug.rotation_range != 0));

	if (use_buffer_img)
		img_tmp = (uint8_t *)calloc(sz, sizeof(uint8_t));

	memset(x, 0, sz * net->batch_size * sizeof(float));
	if (net->task != PREDICT)
		memset(y, 0, net->output_size * net->batch_size * sizeof(float));

	for (i = 0; i < net->batch_size; ++i) {
		bcnn_mnist_next_iter(net, iter);
		// Data augmentation
		if (net->task == TRAIN && wrk->train)
			bcnn_data_augmentation(iter->input_uchar, net->w, net->h, net->c, &(net->data_aug), img_tmp);
		bip_convert_u8_to_f32(iter->input_uchar, net->w, net->h, net->c, net->w * net->c, x);
		x += sz;
		if (net->task != PREDICT) {
			// Load truth
			y[iter->label_int[0]] = 1;
			y += net->output_size;
		}
	}
	
	if (use_buffer_img)
		bh_free(img_tmp);

#ifdef BCNN_USE_CUDA
	bcnn_cuda_memcpy_host2dev(wrk->input_gpu, wrk->input, sz * net->batch_size);
	if (net->task != PREDICT)
		bcnn_cuda_memcpy_host2dev(wrk->truth_gpu, wrk->truth, net->output_size * net->batch_size);
#endif

	return BCNN_SUCCESS;
}

int bcnn_predict_batch(bcnn_net *net, bcnn_workload *wrk, float **pred, float *error)
{
	int i;

	bcnn_forward(net, wrk);

	for (i = net->nb_layers - 1; i > 0; --i) {
		if (net->layers[i].type != COST)
			break;
	}
#ifdef BCNN_USE_CUDA
	bcnn_cuda_memcpy_dev2host(net->layers[i].output_gpu, net->layers[i].output, 
		net->output_size * net->batch_size);
#endif
	(*pred) = net->layers[i].output;
	
	*error = *(net->layers[net->nb_layers - 1].output);

	return BCNN_SUCCESS;
}


int bcnn_train_on_batch(bcnn_net *net, bcnn_iterator *iter, float *loss)
{
	bcnn_iter_batch(net, iter);

	net->seen += net->batch_size;
	// Forward
	bcnn_forward2(net);
	//bcnn_forward(net, &net->wrk);
	// Back prop
	bcnn_backward2(net);
	//bcnn_backward(net, &net->wrk);
	// Update network weight
	bcnn_update(net);
	*loss = net->layers[net->nb_layers - 1].output[0];

	return BCNN_SUCCESS;
}

int bcnn_predict_on_batch(bcnn_net *net, bcnn_iterator *iter, float **pred, float *error)
{
	int i;

	bcnn_iter_batch(net, iter);

	bcnn_forward2(net);

	for (i = net->nb_layers - 1; i > 0; --i) {
		if (net->layers[i].type != COST)
			break;
	}
#ifdef BCNN_USE_CUDA
	bcnn_cuda_memcpy_dev2host(net->layers[i].output_gpu, net->layers[i].output, 
		net->output_size * net->batch_size);
#endif
	(*pred) = net->layers[i].output;
	
	*error = *(net->layers[net->nb_layers - 1].output);

	return BCNN_SUCCESS;
}


int bcnn_write_model(bcnn_net *net, char *filename)
{
	bcnn_layer *layer = NULL;
	int i;

	FILE *fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "ERROR: can't open file %s\n", filename);
		return -1;
	}

	fwrite(&net->learner.learning_rate, sizeof(float), 1, fp);
	fwrite(&net->learner.momentum, sizeof(float), 1, fp);
	fwrite(&net->learner.decay, sizeof(float), 1, fp);
	fwrite(&net->seen, sizeof(int), 1, fp);

	for (i = 0; i < net->nb_layers; ++i){
		layer = &net->layers[i];
		if (layer->type == CONVOLUTIONAL ||
			layer->type == DECONVOLUTIONAL ||
			layer->type == FULL_CONNECTED) {
#ifdef BCNN_USE_CUDA
			bcnn_cuda_memcpy_dev2host(layer->weight_gpu, layer->weight, layer->weights_size);
			bcnn_cuda_memcpy_dev2host(layer->bias_gpu, layer->bias, layer->bias_size);
#endif
			fwrite(layer->bias, sizeof(float), layer->bias_size, fp);
			fwrite(layer->weight, sizeof(float), layer->weights_size, fp);
		}
	}
	fclose(fp);
	return BCNN_SUCCESS;
}

int bcnn_load_model(bcnn_net *net, char *filename)
{
	FILE *fp = fopen(filename, "rb");
	bcnn_layer *layer = NULL;
	int i;
	size_t nb_read = 0;

	if (!fp) {
		fprintf(stderr, "[ERROR] can't open file %s\n", filename);
		return -1;
	}

	fread(&net->learner.learning_rate, sizeof(float), 1, fp);
	fread(&net->learner.momentum, sizeof(float), 1, fp);
	fread(&net->learner.decay, sizeof(float), 1, fp);
	fread(&net->seen, sizeof(int), 1, fp);


	for (i = 0; i < net->nb_layers; ++i) {
		layer = &net->layers[i];
		if (layer->type == CONVOLUTIONAL ||
			layer->type == DECONVOLUTIONAL ||
			layer->type == FULL_CONNECTED) {
			nb_read = fread(layer->bias, sizeof(float), layer->bias_size, fp);
			nb_read = fread(layer->weight, sizeof(float), layer->weights_size, fp);
#ifdef BCNN_USE_CUDA
			bcnn_cuda_memcpy_host2dev(layer->weight_gpu, layer->weight, layer->weights_size);
			bcnn_cuda_memcpy_host2dev(layer->bias_gpu, layer->bias, layer->bias_size);
#endif
		}
	}
	if (fp != NULL)
		fclose(fp);

	fprintf(stderr, "[INFO] Model %s loaded succesfully\n", filename);
	fflush(stdout);

	return BCNN_SUCCESS;
}




/* CPU only */
#ifndef BCNN_USE_CUDA
int bcnn_visualize_network(bcnn_net *net)
{
	int i, j, k, sz;
	bcnn_layer *layer = NULL;
	char name[256];

	for (i = 0; i < net->batch_size; ++i) {
		for (j = 0; j < net->nb_layers; ++j) {
			if (net->layers[j].type == CONVOLUTIONAL) {
				layer = &net->layers[j];
				sz = layer->output_shape[0] * layer->output_shape[1] * layer->output_shape[2];
				for (k = 0; k < net->layers[j].output_shape[2]; ++k) {
					sprintf(name, "sample%d_layer%d_fmap%d.png", i, j, k);
					bip_write_float_image(name, net->layers[j].output +
						i * sz + k * net->layers[j].output_shape[0] * net->layers[j].output_shape[1],
						net->layers[j].output_shape[0], net->layers[j].output_shape[1],
						1, net->layers[j].output_shape[0] * sizeof(float));
				}
			}
		}
	}

	return BCNN_SUCCESS;
}
#endif


int bcnn_free_layer(bcnn_layer *layer)
{
    if (layer->type == DROPOUT) {
        bh_free(layer->rand);
#ifdef BCNN_USE_CUDA
		if (layer->rand_gpu)             bcnn_cuda_free(layer->rand_gpu);
#endif
        return BCNN_SUCCESS;
    }
    bh_free(layer->indexes);
    bh_free(layer->weight);
	bh_free(layer->weight_diff);
    bh_free(layer->bias);
    bh_free(layer->bias_diff);
    bh_free(layer->conv_workspace);
	if (layer->type != ACTIVATION) {
		bh_free(layer->diff);
		bh_free(layer->output);
	}
	bh_free(layer->mean);
	bh_free(layer->diff_mean);
	bh_free(layer->global_mean);
	bh_free(layer->variance);
	bh_free(layer->diff_variance);
	bh_free(layer->global_variance);
	bh_free(layer->x_norm);
	bh_free(layer->bn_scale);
	bh_free(layer->bn_scale_diff);
	bh_free(layer->bn_workspace);
	bh_free(layer->spatial_stats);
	bh_free(layer->bn_shift);
	bh_free(layer->bn_shift_diff);
	bh_free(layer->spatial_sum_multiplier);
	bh_free(layer->batch_sum_multiplier);

#ifdef BCNN_USE_CUDA
	if (layer->indexes_gpu)          bcnn_cuda_free(layer->indexes_gpu);
	if (layer->weight_gpu)          bcnn_cuda_free(layer->weight_gpu);
	if (layer->weight_diff_gpu)   bcnn_cuda_free(layer->weight_diff_gpu);
	if (layer->conv_workspace_gpu)        bcnn_cuda_free(layer->conv_workspace_gpu);
	if (layer->bias_gpu)           bcnn_cuda_free(layer->bias_gpu);
	if (layer->bias_diff_gpu)     bcnn_cuda_free(layer->bias_diff_gpu);
	if (layer->type != ACTIVATION) {
		if (layer->output_gpu)           bcnn_cuda_free(layer->output_gpu);
		if (layer->diff_gpu)            bcnn_cuda_free(layer->diff_gpu);
	}
	if (layer->mean_gpu)			bcnn_cuda_free(layer->mean_gpu);
	if (layer->diff_mean_gpu)		bcnn_cuda_free(layer->diff_mean_gpu);
	if (layer->global_mean_gpu)		bcnn_cuda_free(layer->global_mean_gpu);
	if (layer->variance_gpu)		bcnn_cuda_free(layer->variance_gpu);
	if (layer->diff_variance_gpu)	bcnn_cuda_free(layer->diff_variance_gpu);
	if (layer->global_variance_gpu)	bcnn_cuda_free(layer->global_variance_gpu);
	if (layer->x_norm_gpu)			bcnn_cuda_free(layer->x_norm_gpu);
	if (layer->bn_scale_gpu)		bcnn_cuda_free(layer->bn_scale_gpu);
	if (layer->bn_scale_diff_gpu)	bcnn_cuda_free(layer->bn_scale_diff_gpu);
	if (layer->spatial_sum_multiplier_gpu)	bcnn_cuda_free(layer->spatial_sum_multiplier_gpu);
	if (layer->batch_sum_multiplier_gpu)	bcnn_cuda_free(layer->batch_sum_multiplier_gpu);
	if (layer->bn_workspace_gpu)	bcnn_cuda_free(layer->bn_workspace_gpu);
	if (layer->spatial_stats_gpu)	bcnn_cuda_free(layer->spatial_stats_gpu);
	if (layer->bn_shift_gpu)	bcnn_cuda_free(layer->bn_shift_gpu);
	if (layer->bn_shift_diff_gpu)	bcnn_cuda_free(layer->bn_shift_diff_gpu);
#ifdef BCNN_USE_CUDNN
	cudnnDestroyTensorDescriptor(layer->src_tensor_desc);
	cudnnDestroyTensorDescriptor(layer->dst_tensor_desc);
	cudnnDestroyTensorDescriptor(layer->bias_desc);
	cudnnDestroyFilterDescriptor(layer->filter_desc);
	cudnnDestroyConvolutionDescriptor(layer->conv_desc);
#endif
#endif
	layer = NULL;
	return BCNN_SUCCESS;
}