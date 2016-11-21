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

int bcnn_free_node(bcnn_node *node)
{
	bh_free(node->data);
	bh_free(node->grad_data);
#ifdef BCNN_USE_CUDA
	bcnn_cuda_free(node->data_gpu);
	bcnn_cuda_free(node->grad_data_gpu);
#endif
	return BCNN_SUCCESS;
}

int bcnn_free_connection(bcnn_connection *conn)
{
	if (conn->layer->type != ACTIVATION && 
		conn->layer->type != DROPOUT) {
		bcnn_free_node(&conn->dst_node);
	}
	bh_free(conn->id);
	bcnn_free_layer(&conn->layer);
	return BCNN_SUCCESS;
}

int bcnn_free_net(bcnn_net *net)
{
	int i;
	bcnn_free_workload(net);
    for (i = 0; i < net->nb_connections; ++i) {
		bcnn_free_connection(&net->connections[i]);
    }
	bh_free(net->connections);
	for (i = 0; i < net->nb_finetune; ++i) {
		bh_free(net->finetune_id[i]);
	}
	bh_free(net->finetune_id);
	return BCNN_SUCCESS;
}

int bcnn_set_param(bcnn_net *net, char *name, char *val)
{
	if (strcmp(name, "input_width") == 0) net->input_node.w = atoi(val);
	else if (strcmp(name, "input_height") == 0) net->input_node.h = atoi(val);
	else if (strcmp(name, "input_channels") == 0) net->input_node.c = atoi(val);
	else if (strcmp(name, "batch_size") == 0) net->input_node.b = atoi(val);
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
	else if (strcmp(name, "mean_r") == 0) net->data_aug.mean_r = (float)atof(val) / 255.0f;
	else if (strcmp(name, "mean_g") == 0) net->data_aug.mean_g = (float)atof(val) / 255.0f;
	else if (strcmp(name, "mean_b") == 0) net->data_aug.mean_b = (float)atof(val) / 255.0f;
	else if (strcmp(name, "swap_to_bgr") == 0) net->data_aug.swap_to_bgr = atoi(val);
	else if (strcmp(name, "prediction_type") == 0) {
		if (strcmp(val, "classif") == 0 || strcmp(val, "classification") == 0) net->prediction_type = CLASSIFICATION;
		else if (strcmp(val, "reg") == 0 || strcmp(val, "regression") == 0) net->prediction_type = REGRESSION;
		else if (strcmp(val, "heatmap") == 0 || strcmp(val, "heatmap_regression") == 0) net->prediction_type = HEATMAP_REGRESSION;
		else if (strcmp(val, "segmentation") == 0) net->prediction_type = SEGMENTATION;
	}
	else if (strcmp(name, "finetune_id") == 0) {
		net->nb_finetune++;
		if (net->nb_finetune == 1)
			net->finetune_id = (char **)calloc(net->nb_finetune, sizeof(char *));
		else
			net->finetune_id = (char **)realloc(net->finetune_id, net->nb_finetune);
		bh_fill_option(&net->finetune_id[net->nb_finetune - 1], val);
	}
	return BCNN_SUCCESS;
}

int bcnn_net_add_connection(bcnn_net *net, bcnn_connection conn)
{
	net->connections = (bcnn_connection *)realloc(net->connections,
		net->nb_connections * sizeof(bcnn_connection));
	net->connections[net->nb_connections - 1] = conn;
	return BCNN_SUCCESS;
}

int bcnn_init_workload(bcnn_net *net)
{
	int i;
	int sz = bcnn_node_size(&net->input_node);
	int n = net->nb_connections;
	int k = (net->connections[n - 1].layer->type == COST ? (n - 2) : (n - 1));
	int output_size = bcnn_node_size(&net->connections[k].dst_node);

	net->input_node.data = (float *)calloc(sz, sizeof(float));
	for (i = 0; i < n; ++i) {
		if (net->connections[i].layer->type == COST) {
			net->connections[i].label = (float *)calloc(output_size, sizeof(float));
		}
	}
	net->input_node.grad_data = NULL;
	net->connections[0].src_node.data = net->input_node.data;
	net->connections[0].src_node.grad_data = net->input_node.grad_data;
#ifdef BCNN_USE_CUDA
	net->input_node.data_gpu = bcnn_cuda_malloc_f32(sz);
	for (i = 0; i < n; ++i) {
		if (net->connections[i].layer->type == COST) {
			net->connections[i].label_gpu = bcnn_cuda_malloc_f32(output_size);
		}
	}
	net->input_node.grad_data_gpu = NULL;
	net->connections[0].src_node.data_gpu = net->input_node.data_gpu;
	net->connections[0].src_node.grad_data_gpu = net->input_node.grad_data_gpu;
#endif

	return BCNN_SUCCESS;
}


int bcnn_free_workload(bcnn_net *net)
{
	int i;
	int n = net->nb_connections;

	bh_free(net->input_node.data);
	for (i = 0; i < n; ++i) {
		if (net->connections[i].layer->type == COST) bh_free(net->connections[i].label);
	}
#ifdef BCNN_USE_CUDA
	bcnn_cuda_free(net->input_node.data_gpu);
	for (i = 0; i < n; ++i) {
		if (net->connections[i].layer->type == COST) bcnn_cuda_free(net->connections[i].label);
	}
#endif
	return 0;
}

int bcnn_compile_net(bcnn_net *net, char *phase)
{
	int i;

	bcnn_free_workload(net);
	bcnn_init_workload(net);
	if (strcmp(phase, "train") == 0) net->state = 1;
	else if (strcmp(phase, "predict") == 0) net->state = 0;
	else {
		fprintf(stderr, "[ERROR] bcnn_compile_net: Available option are 'train' and 'predict'");
		return BCNN_INVALID_PARAMETER;
	}
	// State propagation through connections
	for (i = 0; i < net->nb_connections; ++i)
		net->connections[i].state = net->state;

	return BCNN_SUCCESS;
}


int bcnn_forward(bcnn_net *net)
{
	 int i;
	 int n = net->nb_connections;
	 int k = (net->connections[n - 1].layer->type == COST ? (n - 2) : (n - 1));
	 int output_size = 0;
	 bcnn_connection conn = { 0 };


	 for (i = 0; i < net->nb_connections; ++i) {
		conn = net->connections[i];
		output_size = bcnn_node_size(&conn.dst_node);
#ifdef BCNN_USE_CUDA
		if (conn.dst_node.grad_data_gpu != NULL)
			bcnn_cuda_fill_f32(output_size, 0.0f, conn.dst_node.grad_data_gpu, 1);
#else
		if (conn.dst_node.grad_data != NULL)
			memset(conn.dst_node.grad_data, 0, output_size * sizeof(float));
#endif
		switch (conn.layer->type) {
		case CONVOLUTIONAL:
			bcnn_forward_conv_layer(&conn);
			break;
		case DECONVOLUTIONAL:
			bcnn_forward_deconv_layer(&conn);
			break;
		case ACTIVATION:
			bcnn_forward_activation_layer(&conn);
			break;
		case BATCHNORM:
			bcnn_forward_batchnorm_layer(&conn);
			break;
		case FULL_CONNECTED:
			bcnn_forward_fullc_layer(&conn);
			break;
		case MAXPOOL:
			bcnn_forward_maxpool_layer(&conn);
			break;
		case SOFTMAX:
			bcnn_forward_softmax_layer(&conn);
			break;
		case DROPOUT:
			bcnn_forward_dropout_layer(&conn);
			break;
		case COST:
			bcnn_forward_cost_layer(&conn);
			break;
		default:
			break;
		}
    }

	return BCNN_SUCCESS;
}


int bcnn_backward(bcnn_net *net)
{
	int i;
	bcnn_connection conn = { 0 };

	for (i = net->nb_connections - 1; i >= 0; --i) {
		conn = net->connections[i];
		switch (conn.layer->type) {
		case CONVOLUTIONAL:
			bcnn_backward_conv_layer(&conn);
			break;
		case DECONVOLUTIONAL:
			bcnn_backward_deconv_layer(&conn);
			break;
		case ACTIVATION:
			bcnn_backward_activation_layer(&conn);
			break;
		case BATCHNORM:
			bcnn_backward_batchnorm_layer(&conn);
			break;
		case FULL_CONNECTED:
			bcnn_backward_fullc_layer(&conn);
			break;
		case MAXPOOL:
			bcnn_backward_maxpool_layer(&conn);
			break;
		case SOFTMAX:
			bcnn_backward_softmax_layer(&conn);
			break;
		case DROPOUT:
			bcnn_backward_dropout_layer(&conn);
			break;
		case COST:
			bcnn_backward_cost_layer(&conn);
			break;
		default:
			break;
		}
    }
	return BCNN_SUCCESS;
}


static float bcnn_update_learning_rate(bcnn_net *net)
{
    int iter = net->seen / net->input_node.b;

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


int bcnn_apply_update_to_layer(bcnn_connection *conn, int batch_size, float learning_rate, float momentum, float decay)
{
	bcnn_layer *layer = conn->layer;
	bcnn_node dst = conn->dst_node;

#ifdef BCNN_USE_CUDA
	bcnn_cuda_axpy(layer->bias_size, -learning_rate / batch_size, layer->bias_diff_gpu, 1, layer->bias_gpu, 1);
	bcnn_cuda_scal(layer->bias_size, momentum, layer->bias_diff_gpu, 1);

	bcnn_cuda_axpy(layer->weights_size, decay * batch_size, layer->weight_gpu, 1, layer->weight_diff_gpu, 1);
	bcnn_cuda_axpy(layer->weights_size, -learning_rate / batch_size, layer->weight_diff_gpu, 1, layer->weight_gpu, 1);
	bcnn_cuda_scal(layer->weights_size, momentum, layer->weight_diff_gpu, 1);

	if (layer->bn_scale_diff_gpu && layer->bn_scale_gpu && layer->bn_shift_gpu && layer->bn_shift_diff_gpu) {
		bcnn_cuda_axpy(dst.c, -learning_rate / batch_size, layer->bn_scale_diff_gpu, 1, layer->bn_scale_gpu, 1);
		bcnn_cuda_scal(dst.c, momentum, layer->bn_scale_diff_gpu, 1);
		bcnn_cuda_axpy(dst.c, -learning_rate / batch_size, layer->bn_shift_diff_gpu, 1, layer->bn_shift_gpu, 1);
		bcnn_cuda_scal(dst.c, momentum, layer->bn_shift_diff_gpu, 1);
	}
#else
	bcnn_axpy(layer->bias_size, -learning_rate / batch_size, layer->bias_diff, layer->bias);
	bcnn_scal(layer->bias_size, momentum, layer->bias_diff);

	bcnn_axpy(layer->weights_size, decay * batch_size, layer->weight, layer->weight_diff);
	bcnn_axpy(layer->weights_size, -learning_rate / batch_size, layer->weight_diff, layer->weight);
	bcnn_scal(layer->weights_size, momentum, layer->weight_diff);

	if (layer->bn_scale_diff && layer->bn_scale && layer->bn_shift && layer->bn_shift_diff) {
		bcnn_axpy(dst.c, -learning_rate / batch_size, layer->bn_scale_diff, layer->bn_scale);
		bcnn_scal(dst.c, momentum, layer->bn_scale_diff);
		bcnn_axpy(dst.c, -learning_rate / batch_size, layer->bn_shift_diff, layer->bn_shift);
		bcnn_scal(dst.c, momentum, layer->bn_shift_diff);
	}
#endif
	return 0;
}


int bcnn_update(bcnn_net *net)
{
    int i;
    float lr = bcnn_update_learning_rate(net);
	bcnn_layer_type	type;

	for (i = 0; i < net->nb_connections; ++i) {
		type = net->connections[i].layer->type;
		if ((type == CONVOLUTIONAL || 
			type == DECONVOLUTIONAL || 
			type == FULL_CONNECTED ||
			type == BATCHNORM)) {
			bcnn_apply_update_to_layer(&net->connections[i],
					net->input_node.b, lr, net->learner.momentum, net->learner.decay);
		}
    }
	return BCNN_SUCCESS;
}

int bcnn_convert_img_to_float(unsigned char *src, int w, int h, int c, int swap_to_bgr, 
	float mean_r, float mean_g, float mean_b, float *dst)
{
	int x, y, k;
	float m = 0.0f;
	
	if (swap_to_bgr) {
		for (k = 0; k < c; ++k){
			switch (k) {
			case 0:
				m = mean_r;
				break;
			case 1:
				m = mean_g;
				break;
			case 2:
				m = mean_b;
				break;
			}
			for (y = 0; y < h; ++y){
				for (x = 0; x < w; ++x){
					dst[w * (h * (2 - k) + y) + x] = (float)src[c * (x + w * y) + k] / 255.0f - m;
				}
			}
		}
	}
	else {
		for (k = 0; k < c; ++k){
			for (y = 0; y < h; ++y){
				for (x = 0; x < w; ++x){
					dst[w * (h * k + y) + x] = (float)src[c * (x + w * y) + k] / 255.0f - m;
				}
			}
		}
	}
	return 0;
}


int bcnn_iter_batch(bcnn_net *net, bcnn_iterator *iter)
{
	int i, j, sz = net->input_node.w * net->input_node.h * net->input_node.c, n, offset;
	int ret;
	int nb = net->nb_connections;
	int end_node = (net->connections[nb - 1].layer->type == COST ? (nb - 2) : (nb - 1));
	int w, h, c, out_w, out_h, out_c;
	char *line = NULL;
	unsigned char *img = NULL, *img_tmp = NULL;
	float *x = net->input_node.data;
	float *y = net->connections[nb - 1].label;
	float x_scale, y_scale;
	char **tok = NULL;
	int n_tok = 0, x_pos, y_pos;
	int use_buffer_img = (net->task == TRAIN && net->state != 0 &&
		(net->data_aug.range_shift_x != 0 || net->data_aug.range_shift_y != 0 ||
		net->data_aug.rotation_range != 0));
	bcnn_data_augment *param = &(net->data_aug);
	int input_size = bcnn_node_size(&net->input_node);
	int output_size = net->connections[nb - 2].dst_node.w * 
		net->connections[nb - 2].dst_node.h *
		net->connections[nb - 2].dst_node.c;

	memset(x, 0, sz * net->input_node.b * sizeof(float));
	if (net->task != PREDICT)
		memset(y, 0, output_size * net->input_node.b * sizeof(float));

	if (use_buffer_img)
		img_tmp = (uint8_t *)calloc(sz, sizeof(uint8_t));
	
	if (iter->type == ITER_MNIST) {
		for (i = 0; i < net->input_node.b; ++i) {
			bcnn_mnist_next_iter(net, iter);
			// Data augmentation
			if (net->task == TRAIN && net->state)
				bcnn_data_augmentation(iter->input_uchar, net->input_node.w, net->input_node.h, net->input_node.c, param, img_tmp);
			bcnn_convert_img_to_float(iter->input_uchar, net->input_node.w, net->input_node.h, net->input_node.c, param->swap_to_bgr, 
				param->mean_r, param->mean_g, param->mean_b, x);
			x += sz;
			if (net->task != PREDICT) {
				// Load truth
				y[iter->label_int[0]] = 1;
				y += output_size;
			}
		}
	}
	else if (iter->type == ITER_BIN) {
		for (i = 0; i < net->input_node.b; ++i) {
			bcnn_bin_iter(net, iter);
			// Data augmentation
			if (net->task == TRAIN && net->state)
				bcnn_data_augmentation(iter->input_uchar, net->input_node.w, net->input_node.h, net->input_node.c, param, img_tmp);
			bcnn_convert_img_to_float(iter->input_uchar, net->input_node.w, net->input_node.h, net->input_node.c, param->swap_to_bgr, 
				param->mean_r, param->mean_g, param->mean_b, x);
			x += sz;
			if (net->task != PREDICT) {
				// Load truth
				switch (net->prediction_type) {
				case CLASSIFICATION:
					y[(int)iter->label_float[0]] = 1;
					y += output_size;
					break;
				case REGRESSION:
					for (j = 0; j < iter->label_width; ++j) {
						y[j] = iter->label_float[j];
					}
					y += output_size;
					break;
				case HEATMAP_REGRESSION:
					// Load truth
					w = net->connections[net->nb_connections - 2].dst_node.w;
					h = net->connections[net->nb_connections - 2].dst_node.h;
					c = net->connections[net->nb_connections - 2].dst_node.c;
					x_scale = (float)w / (float)net->input_node.w;
					y_scale = (float)h / (float)net->input_node.h;
					for (j = 0; j < iter->label_width; j += 2) {
						x_pos = (int)((iter->label_float[j] - net->data_aug.shift_x) * x_scale + 0.5f);
						y_pos = (int)((iter->label_float[j + 1] - net->data_aug.shift_y) * y_scale + 0.5f);
						// Set gaussian kernel around (x_pos, y_pos)
						n = j >> 1;
						offset = n * w * h + (y_pos * w + x_pos);
						if (x_pos >= 0 && x_pos < w && y_pos >= 0 && y_pos < h) {
							y[offset] = 1.0f;
							if (x_pos > 0) y[offset - 1] = 0.5f;
							if (x_pos < w - 1) y[offset + 1] = 0.5f;
							if (y_pos > 0) y[offset - w] = 0.5f;
							if (y_pos < h - 1) y[offset + w] = 0.5f;
							if (x_pos > 0 && y_pos > 0) y[offset - w - 1] = 0.25f;
							if (x_pos < w - 1 && y_pos > 0) y[offset - w + 1] = 0.25f;
							if (x_pos > 0 && y_pos < h - 1) y[offset + w - 1] = 0.25f;
							if (x_pos < w - 1 && y_pos < h - 1) y[offset + w + 1] = 0.25f;
						}
					}
					y += output_size;
					break;
				default:
					bh_error("Target type not implemented for this data format. Please use list format instead.", BCNN_INVALID_PARAMETER);
				}
			}
		}
	}
	else if (net->prediction_type == CLASSIFICATION) { //  Classification
		for (i = 0; i < net->input_node.b; ++i) {
			//nb_lines_skipped = (int)((float)rand() / RAND_MAX * net->input_node.b);
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
				bcnn_load_image_from_path(tok[0], net->input_node.w, net->input_node.h, net->input_node.c, &img, net->state);
			}
			else
				bcnn_load_image_from_csv(tok[0], net->input_node.w, net->input_node.h, net->input_node.c, &img);
			if (img) {
				// Online data augmentation
				if (net->task == TRAIN && net->state)
					bcnn_data_augmentation(img, net->input_node.w, net->input_node.h, net->input_node.c, param, img_tmp);
				bcnn_convert_img_to_float(iter->input_uchar, net->input_node.w, net->input_node.h, net->input_node.c, param->swap_to_bgr, 
					param->mean_r, param->mean_g, param->mean_b, x);
				bh_free(img);
			}
			x += sz;
			if (net->task != PREDICT) {
				// Load truth
				y[atoi(tok[1])] = 1;
				y += output_size;
			}
			bh_free(line);
			for (j = 0; j < n_tok; ++j)
				bh_free(tok[j]);
			bh_free(tok);
		}
	}
	else if (net->prediction_type == REGRESSION) { // Regression
		for (i = 0; i < net->input_node.b; ++i) {
			line = bh_fgetline(iter->f_input);
			if (line == NULL) {
				rewind(iter->f_input);
				line = bh_fgetline(iter->f_input);
			}
			j = 0;
			//field = strtok(line, delims);
			n_tok = bh_strsplit(line, ' ', &tok);
			if (net->task != PREDICT) {
				bh_assert(n_tok == output_size + 1,
					"Wrong data format for regression", BCNN_INVALID_DATA);
			}
			if (iter->type == ITER_LIST) {
				bcnn_load_image_from_path(tok[0], net->input_node.w, net->input_node.h, net->input_node.c, &img, net->state);
			}
			else
				bcnn_load_image_from_csv(tok[0], net->input_node.w, net->input_node.h, net->input_node.c, &img);
			if (img) {
				// Online data augmentation
				if (net->task == TRAIN && net->state)
					bcnn_data_augmentation(img, net->input_node.w, net->input_node.h, net->input_node.c, param, img_tmp);
				bcnn_convert_img_to_float(iter->input_uchar, net->input_node.w, net->input_node.h, net->input_node.c, param->swap_to_bgr, 
					param->mean_r, param->mean_g, param->mean_b, x);
				bh_free(img);
			}
			x += sz;
			if (net->task != PREDICT) {
				// Load truth
				for (j = 1; j < n_tok; ++j) {
					y[j - 1] = (float)atof(tok[j]);
				}
				y += output_size;
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
		for (i = 0; i < net->input_node.b; ++i) {
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
			if (iter->type == ITER_LIST) {
				bcnn_load_image_from_path(tok[0], net->input_node.w, net->input_node.h, net->input_node.c, &img, net->state);
			}
			else
				bcnn_load_image_from_csv(tok[0], net->input_node.w, net->input_node.h, net->input_node.c, &img);
			if (img) {
				// Online data augmentation
				if (net->task == TRAIN && net->state)
					bcnn_data_augmentation(img, net->input_node.w, net->input_node.h, net->input_node.c, param, img_tmp);
				bcnn_convert_img_to_float(iter->input_uchar, net->input_node.w, net->input_node.h, net->input_node.c, param->swap_to_bgr, 
					param->mean_r, param->mean_g, param->mean_b, x);
				bh_free(img);
			}
			x += sz;
			if (net->task != PREDICT) {
				// Load truth
				w = net->connections[net->nb_connections - 2].dst_node.w;
				h = net->connections[net->nb_connections - 2].dst_node.h;
				c = net->connections[net->nb_connections - 2].dst_node.c;
				x_scale = w / (float)net->input_node.w;
				y_scale = h / (float)net->input_node.h;
				for (j = 1; j < n_tok; j += 2) {
					x_pos = (int)(atof(tok[j]) * x_scale);
					y_pos = (int)(atof(tok[j + 1]) * y_scale);
					// Set gaussian kernel around (x_pos,y_pos)
					n = j >> 1;
					offset = n * w * h + (y_pos * w + x_pos);
					if (x_pos >= 0 && x_pos < w && y_pos >= 0 && y_pos < h) {
						y[offset] = 1.0f;
						if (x_pos > 0) y[offset - 1] = 0.5f;
						if (x_pos < w - 1) y[offset + 1] = 0.5f;
						if (y_pos > 0) y[offset - w] = 0.5f;
						if (y_pos < h - 1) y[offset + w] = 0.5f;
						if (x_pos > 0 && y_pos > 0) y[offset - w - 1] = 0.25f;
						if (x_pos < w - 1 && y_pos > 0) y[offset - w + 1] = 0.25f;
						if (x_pos > 0 && y_pos < h - 1) y[offset + w - 1] = 0.25f;
						if (x_pos < w - 1 && y_pos < h - 1) y[offset + w + 1] = 0.25f;
					}
				}
				y += output_size;
			}
			bh_free(line);
			for (j = 0; j < n_tok; ++j)
				bh_free(tok[j]);
			bh_free(tok);
		}
	}
	else if (net->prediction_type == SEGMENTATION) {
		// Format must be this way: img.png mask.png
		out_w = net->connections[net->nb_connections - 2].dst_node.w;
		out_h = net->connections[net->nb_connections - 2].dst_node.h;
		out_c = net->connections[net->nb_connections - 2].dst_node.c;
		for (i = 0; i < net->input_node.b; ++i) {
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
			if (iter->type == ITER_LIST) {
				bh_assert(ret = bcnn_load_image_from_path(tok[0], net->input_node.w, net->input_node.h, net->input_node.c, &img, net->state)
					== BCNN_SUCCESS, "Problem while loading image", ret);
			}
			else
				bcnn_load_image_from_csv(tok[0], net->input_node.w, net->input_node.h, net->input_node.c, &img);
			if (img) {
				// Online data augmentation
				if (net->task == TRAIN && net->state) {
					net->data_aug.use_precomputed = 0;
					bcnn_data_augmentation(img, net->input_node.w, net->input_node.h, net->input_node.c, param, img_tmp);
				}
				bcnn_convert_img_to_float(iter->input_uchar, net->input_node.w, net->input_node.h, net->input_node.c, param->swap_to_bgr, 
					param->mean_r, param->mean_g, param->mean_b, x);
				bh_free(img);
			}
			x += sz;
			if (net->task != PREDICT) {
				// Load truth i.e the segmentation mask
				if (iter->type == ITER_LIST) {
					bip_load_image(tok[1], &img, &w, &h, &c);
					bh_assert(w == out_w &&
					h == out_h &&
					c == out_c,
					"Segmentation mask size and output size of the network must be the same",
					BCNN_INVALID_DATA);
				}
				else
					bcnn_load_image_from_csv(tok[1], out_w, out_h, out_c, &img);
				if (net->state) {
					// Apply to the mask the same data augmentation parameters applied to the image
					net->data_aug.use_precomputed = 1;
					bcnn_data_augmentation(img, net->input_node.w, net->input_node.h, net->input_node.c, param, img_tmp);
				}
				bip_convert_u8_to_f32(img, out_w, out_h, out_c, out_w * out_c, y);
				bh_free(img);
				y += output_size;
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
	bcnn_cuda_memcpy_host2dev(net->input_node.data_gpu, net->input_node.data, input_size);
	if (net->task != PREDICT)
		bcnn_cuda_memcpy_host2dev(net->connections[nb - 1].label_gpu, net->connections[nb - 1].label, output_size
		* net->connections[nb - 1].src_node.b);
#endif
	return BCNN_SUCCESS;
}

int bcnn_train_on_batch(bcnn_net *net, bcnn_iterator *iter, float *loss)
{
	bcnn_iter_batch(net, iter);

	net->seen += net->input_node.b;
	// Forward
	bcnn_forward(net);
	// Back prop
	bcnn_backward(net);
	// Update network weight
	bcnn_update(net);
	*loss = net->connections[net->nb_connections - 1].dst_node.data[0];
	
	return BCNN_SUCCESS;
}

int bcnn_predict_on_batch(bcnn_net *net, bcnn_iterator *iter, float **pred, float *error)
{
	int nb = net->nb_connections;
	int en = (net->connections[nb - 1].layer->type == COST ? (nb - 2) : (nb - 1));
	int output_size = bcnn_node_size(&net->connections[en].dst_node);

	bcnn_iter_batch(net, iter);

	bcnn_forward(net);

#ifdef BCNN_USE_CUDA
	bcnn_cuda_memcpy_dev2host(net->connections[en].dst_node.data_gpu, net->connections[en].dst_node.data,
		output_size);
#endif
	(*pred) = net->connections[en].dst_node.data;
	*error = *(net->connections[nb - 1].dst_node.data);

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

	for (i = 0; i < net->nb_connections; ++i){
		layer = net->connections[i].layer;
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
	int i, j, is_ft = 0;
	size_t nb_read = 0;
	float tmp = 0.0f;
	
	if (!fp) {
		fprintf(stderr, "[ERROR] can't open file %s\n", filename);
		return -1;
	}

	fread(&tmp, sizeof(float), 1, fp);
	fread(&tmp, sizeof(float), 1, fp);
	fread(&tmp, sizeof(float), 1, fp);
	fread(&net->seen, sizeof(int), 1, fp);
	fprintf(stderr, "lr= %f ", net->learner.learning_rate);
	fprintf(stderr, "m= %f ", net->learner.momentum);
	fprintf(stderr, "decay= %f ", net->learner.decay);
	fprintf(stderr, "seen= %d\n", net->seen);

	for (i = 0; i < net->nb_connections; ++i) {
		layer = net->connections[i].layer;
		is_ft = 0;
		if (net->connections[i].id != NULL) {
			for (j = 0; j < net->nb_finetune; ++j) {
				if (strcmp(net->connections[i].id, net->finetune_id[j]) == 0)
					is_ft = 1;
			}
		}
		if ((layer->type == CONVOLUTIONAL ||
			layer->type == DECONVOLUTIONAL ||
			layer->type == FULL_CONNECTED) && is_ft == 0) {
			nb_read = fread(layer->bias, sizeof(float), layer->bias_size, fp);
			fprintf(stderr, "layer= %d nbread_bias= %d bias_size_expected= %d\n", i, nb_read, layer->bias_size);
			nb_read = fread(layer->weight, sizeof(float), layer->weights_size, fp);
			fprintf(stderr, "layer= %d nbread_weight= %d weight_size_expected= %d\n", i, nb_read, layer->weights_size);
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



int bcnn_visualize_network(bcnn_net *net)
{
	int i, j, k, sz, w, h, c;
	bcnn_layer *layer = NULL;
	char name[256];
	FILE *ftmp = NULL;
	int nb = net->nb_connections;
	int output_size = net->connections[nb - 2].dst_node.w * 
		net->connections[nb - 2].dst_node.h *
		net->connections[nb - 2].dst_node.c;
		
	for (j = 0; j < net->nb_connections; ++j) {
		if (net->connections[j].layer->type == CONVOLUTIONAL) {
			w = net->connections[j].dst_node.w;
			h = net->connections[j].dst_node.h;
			c = net->connections[j].dst_node.c;
			sz = w * h * c;
#ifdef BCNN_USE_CUDA
			bcnn_cuda_memcpy_dev2host(net->connections[j].dst_node.data_gpu,
					net->connections[j].dst_node.data, sz * net->input_node.b);
#endif
			for (i = 0; i < net->input_node.b / 8; ++i) {	
				layer = net->connections[j].layer;
				for (k = 0; k < net->connections[j].dst_node.c / 16; ++k) {
					sprintf(name, "sample%d_layer%d_fmap%d.png", i, j, k);
					bip_write_float_image_norm(name, net->connections[j].dst_node.data +
						i * sz + k * w * h, w, h, 1, w * sizeof(float));
				}
			}
		}
		else if (net->connections[j].layer->type == FULL_CONNECTED || 
			net->connections[j].layer->type == SOFTMAX) {
			w = net->connections[j].dst_node.w;
			h = net->connections[j].dst_node.h;
			c = net->connections[j].dst_node.c;
			sz = w * h * c;
#ifdef BCNN_USE_CUDA
			bcnn_cuda_memcpy_dev2host(net->connections[j].dst_node.data_gpu,
					net->connections[j].dst_node.data, sz * net->input_node.b);
#endif
			sprintf(name, "ip_%d.txt", j);
			ftmp = fopen(name, "wt");
			for (i = 0; i < net->input_node.b; ++i) {
				layer = net->connections[j].layer;
				for (k = 0; k < sz; ++k) {
					fprintf(ftmp, "%f ", net->connections[j].dst_node.data[i * sz + k]);
				}
				fprintf(ftmp, "\n");
			}
			fclose(ftmp);
			if (sz == 2 && net->connections[j].layer->type == FULL_CONNECTED) {
				sz = sz * net->connections[j].src_node.w * net->connections[j].src_node.h
					*net->connections[j].src_node.c;
#ifdef BCNN_USE_CUDA
				bcnn_cuda_memcpy_dev2host(net->connections[j].layer->weight_gpu,
					net->connections[j].layer->weight, sz);
#endif
				sprintf(name, "wgt_%d.txt", j);
				ftmp = fopen(name, "wt");
				layer = net->connections[j].layer;
				for (k = 0; k < sz; ++k) {
					fprintf(ftmp, "%f ", layer->weight[k]);
				}
				fprintf(ftmp, "\n");
				fclose(ftmp);
				sz = 2;
#ifdef BCNN_USE_CUDA
				bcnn_cuda_memcpy_dev2host(net->connections[j].layer->bias_gpu,
					net->connections[j].layer->bias, sz);
#endif
				sprintf(name, "b_%d.txt", j);
				ftmp = fopen(name, "wt");
				layer = net->connections[j].layer;
				for (k = 0; k < sz; ++k) {
					fprintf(ftmp, "%f ", layer->bias[k]);
				}
				fprintf(ftmp, "\n");
				fclose(ftmp);
			}
		}
	}

	return BCNN_SUCCESS;
}


int bcnn_free_layer(bcnn_layer **layer)
{
	bcnn_layer *p_layer = (*layer);
    bh_free(p_layer->indexes);
    bh_free(p_layer->weight);
	bh_free(p_layer->weight_diff);
    bh_free(p_layer->bias);
    bh_free(p_layer->bias_diff);
    bh_free(p_layer->conv_workspace);
	bh_free(p_layer->mean);
	bh_free(p_layer->diff_mean);
	bh_free(p_layer->global_mean);
	bh_free(p_layer->variance);
	bh_free(p_layer->diff_variance);
	bh_free(p_layer->global_variance);
	bh_free(p_layer->x_norm);
	bh_free(p_layer->bn_scale);
	bh_free(p_layer->bn_scale_diff);
	bh_free(p_layer->bn_workspace);
	bh_free(p_layer->spatial_stats);
	bh_free(p_layer->bn_shift);
	bh_free(p_layer->bn_shift_diff);
	bh_free(p_layer->spatial_sum_multiplier);
	bh_free(p_layer->batch_sum_multiplier);
	bh_free(p_layer->rand);
#ifdef BCNN_USE_CUDA
	if (p_layer->indexes_gpu)          bcnn_cuda_free(p_layer->indexes_gpu);
	if (p_layer->weight_gpu)          bcnn_cuda_free(p_layer->weight_gpu);
	if (p_layer->weight_diff_gpu)   bcnn_cuda_free(p_layer->weight_diff_gpu);
	if (p_layer->conv_workspace_gpu)        bcnn_cuda_free(p_layer->conv_workspace_gpu);
	if (p_layer->bias_gpu)           bcnn_cuda_free(p_layer->bias_gpu);
	if (p_layer->bias_diff_gpu)     bcnn_cuda_free(p_layer->bias_diff_gpu);
	if (p_layer->mean_gpu)			bcnn_cuda_free(p_layer->mean_gpu);
	if (p_layer->diff_mean_gpu)		bcnn_cuda_free(p_layer->diff_mean_gpu);
	if (p_layer->global_mean_gpu)		bcnn_cuda_free(p_layer->global_mean_gpu);
	if (p_layer->variance_gpu)		bcnn_cuda_free(p_layer->variance_gpu);
	if (p_layer->diff_variance_gpu)	bcnn_cuda_free(p_layer->diff_variance_gpu);
	if (p_layer->global_variance_gpu)	bcnn_cuda_free(p_layer->global_variance_gpu);
	if (p_layer->x_norm_gpu)			bcnn_cuda_free(p_layer->x_norm_gpu);
	if (p_layer->bn_scale_gpu)		bcnn_cuda_free(p_layer->bn_scale_gpu);
	if (p_layer->bn_scale_diff_gpu)	bcnn_cuda_free(p_layer->bn_scale_diff_gpu);
	if (p_layer->spatial_sum_multiplier_gpu)	bcnn_cuda_free(p_layer->spatial_sum_multiplier_gpu);
	if (p_layer->batch_sum_multiplier_gpu)	bcnn_cuda_free(p_layer->batch_sum_multiplier_gpu);
	if (p_layer->bn_workspace_gpu)	bcnn_cuda_free(p_layer->bn_workspace_gpu);
	if (p_layer->spatial_stats_gpu)	bcnn_cuda_free(p_layer->spatial_stats_gpu);
	if (p_layer->bn_shift_gpu)	bcnn_cuda_free(p_layer->bn_shift_gpu);
	if (p_layer->bn_shift_diff_gpu)	bcnn_cuda_free(p_layer->bn_shift_diff_gpu);
	if (p_layer->rand_gpu)             bcnn_cuda_free(p_layer->rand_gpu);
#ifdef BCNN_USE_CUDNN
	cudnnDestroyTensorDescriptor(p_layer->src_tensor_desc);
	cudnnDestroyTensorDescriptor(p_layer->dst_tensor_desc);
	cudnnDestroyTensorDescriptor(p_layer->bias_desc);
	cudnnDestroyFilterDescriptor(p_layer->filter_desc);
	cudnnDestroyConvolutionDescriptor(p_layer->conv_desc);
#endif
#endif
	bh_free(*layer);
	return BCNN_SUCCESS;
}