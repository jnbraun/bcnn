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

#include <bh/bh.h>
#include <bh/bh_timer.h>
#include <bh/bh_error.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>

#include <bip/bip.h>

#include "bcnn/bcnn.h"
#include "bcnn/bcnn_cl.h"


int bcnncl_init_from_config(bcnn_net *net, char *config_file, bcnncl_param *param)
{
	FILE *file = NULL;
	char *line = NULL, *curr_layer = NULL;
	char **tok = NULL;
	int nb_lines = 0, nb_layers = 0;
	int w = 0, h = 0, c = 0, stride = 1, pad = 0, n_filts = 1, batch_norm = 0, input_size = 0, size = 3, outputs = 0;
	bcnn_activation a = NONE;
	bcnn_weights_init init = XAVIER;
	bcnn_loss_metric cost = COST_SSE;
	float scale = 1.0f, rate = 1.0f;
	int input_shape[3] = { 0 };
	int n_tok;
	int concat_index = 0;
	int nb_connections;
	char *layer_id = NULL;
	char *finetune_id = NULL;

	file = fopen(config_file, "rt");
	if (file == 0) {
		fprintf(stderr, "Couldn't open file: %s\n", config_file);
		exit(-1);
	}

	bh_info("Network architecture");
	while ((line = bh_fgetline(file)) != 0) {
		nb_lines++;
		bh_strstrip(line);
		switch (line[0]) {
		case '{':
			if (nb_layers > 0) {
				if (nb_layers == 1) {
					bh_assert(net->input_node.w > 0 &&
						net->input_node.h > 0 && net->input_node.c > 0,
						"Input's width, height and channels must be > 0", BCNN_INVALID_PARAMETER);
					bh_assert(net->input_node.b > 0, "Batch size must be > 0", BCNN_INVALID_PARAMETER);
				}
				if (strcmp(curr_layer, "{conv}") == 0 ||
					strcmp(curr_layer, "{convolutional}") == 0) {
					bcnn_add_convolutional_layer(net, n_filts, size, stride, pad, 0, init, a, layer_id);
				}
				else if (strcmp(curr_layer, "{deconv}") == 0 ||
					strcmp(curr_layer, "{deconvolutional}") == 0) {
					bcnn_add_deconvolutional_layer(net, n_filts, size, stride, pad, init, a, layer_id);
				}
				else if (strcmp(curr_layer, "{activation}") == 0 ||
					strcmp(curr_layer, "{nl}") == 0) {
					bcnn_add_activation_layer(net, a, layer_id);
				}
				else if (strcmp(curr_layer, "{batchnorm}") == 0 ||
					strcmp(curr_layer, "{bn}") == 0) {
					bcnn_add_batchnorm_layer(net, layer_id);
				}
				else if (strcmp(curr_layer, "{connected}") == 0 ||
					strcmp(curr_layer, "{fullconnected}") == 0 ||
					strcmp(curr_layer, "{fc}") == 0 ||
					strcmp(curr_layer, "{ip}") == 0) {
					bcnn_add_fullc_layer(net, outputs, init, a, layer_id);
				}
				else if (strcmp(curr_layer, "{softmax}") == 0) {
					bcnn_add_softmax_layer(net, layer_id);
				}
				else if (strcmp(curr_layer, "{max}") == 0 ||
					strcmp(curr_layer, "{maxpool}") == 0) {
					bcnn_add_maxpool_layer(net, size, stride, layer_id);
				}
				else if (strcmp(curr_layer, "{dropout}") == 0) {
					bcnn_add_dropout_layer(net, rate, layer_id);
				}
				else {
					fprintf(stderr, "[ERROR] Unknown Layer %s\n", curr_layer);
					return BCNN_INVALID_PARAMETER;
				}
				bh_free(curr_layer);
				bh_free(layer_id);
				a = NONE;
			}
			curr_layer = line;
			nb_layers++;
			break;
		case '!':
		case '\0':
		case '#':
			bh_free(line);
			break;
		default:
			n_tok = bh_strsplit(line, '=', &tok);
			bh_assert(n_tok == 2, "Wrong format option in config file", BCNN_INVALID_PARAMETER);
			if (strcmp(tok[0], "task") == 0) {
				if (strcmp(tok[1], "train") == 0) param->task = TRAIN;
				else if (strcmp(tok[1], "predict") == 0)  param->task = PREDICT;
				else bh_error("Invalid parameter for task, available parameters: TRAIN, PREDICT", BCNN_INVALID_PARAMETER);
			}
			else if (strcmp(tok[0], "data_format") == 0) bh_fill_option(&param->data_format, tok[1]);
			else if (strcmp(tok[0], "input_model") == 0) bh_fill_option(&param->input_model, tok[1]);
			else if (strcmp(tok[0], "output_model") == 0) bh_fill_option(&param->output_model, tok[1]);
			else if(strcmp(tok[0], "out_pred") == 0) bh_fill_option(&param->pred_out, tok[1]);
			else if(strcmp(tok[0], "eval_test") == 0) param->eval_test = atoi(tok[1]);
			else if(strcmp(tok[0], "eval_period") == 0) param->eval_period = atoi(tok[1]);
			else if(strcmp(tok[0], "save_model") == 0) param->save_model = atoi(tok[1]);
			else if(strcmp(tok[0], "nb_pred") == 0) param->nb_pred = atoi(tok[1]);
			else if (strcmp(tok[0], "source_train") == 0) bh_fill_option(&param->train_input, tok[1]);
			else if (strcmp(tok[0], "source_test") == 0) bh_fill_option(&param->test_input, tok[1]);
			else if (strcmp(tok[0], "dropout_rate") == 0 || strcmp(tok[0], "rate") == 0) rate = (float)atof(tok[1]);
			else if (strcmp(tok[0], "with") == 0) concat_index = atoi(tok[1]);
			else if (strcmp(tok[0], "batch_norm") == 0) batch_norm = atoi(tok[1]);
			else if (strcmp(tok[0], "filters") == 0) n_filts = atoi(tok[1]);
			else if (strcmp(tok[0], "size") == 0) size = atoi(tok[1]);
			else if (strcmp(tok[0], "stride") == 0) stride = atoi(tok[1]);
			else if (strcmp(tok[0], "pad") == 0) pad = atoi(tok[1]);
			else if (strcmp(tok[0], "id") == 0) bh_fill_option(&layer_id, tok[1]);
			else if (strcmp(tok[0], "output") == 0) outputs = atoi(tok[1]);
			else if (strcmp(tok[0], "function") == 0) {
				if (strcmp(tok[1], "relu") == 0) a = RELU;
				else if (strcmp(tok[1], "tanh") == 0) a = TANH;
				else if (strcmp(tok[1], "ramp") == 0) a = RAMP;
				else if (strcmp(tok[1], "clamp") == 0) a = CLAMP;
				else if (strcmp(tok[1], "softplus") == 0) a = SOFTPLUS;
				else if (strcmp(tok[1], "leaky_relu") == 0 || strcmp(tok[1], "lrelu") == 0) a = LRELU;
				else if (strcmp(tok[1], "abs") == 0) a = ABS;
				else if (strcmp(tok[1], "none") == 0) a = NONE;
				else {
					fprintf(stderr, "[WARNING] Unknown activation type %s, going with ReLU\n", tok[1]);
					a = RELU;
				}
			}
			else if (strcmp(tok[0], "init") == 0) {
				if (strcmp(tok[1], "xavier") == 0) init = XAVIER;
				else if (strcmp(tok[1], "msra") == 0) init = MSRA;
				else {
					fprintf(stderr, "[WARNING] Unknown init type %s, going with xavier init\n", tok[1]);
					init = XAVIER;
				}
			}
			else if (strcmp(tok[0], "metric") == 0) {
				if (strcmp(tok[1], "error") == 0) cost = COST_ERROR;
				else if (strcmp(tok[1], "logloss") == 0) cost = COST_LOGLOSS;
				else if (strcmp(tok[1], "sse") == 0) cost = COST_SSE;
				else if (strcmp(tok[1], "mse") == 0) cost = COST_MSE;
				else if (strcmp(tok[1], "crps") == 0) cost = COST_CRPS;
				else if (strcmp(tok[1], "dice") == 0) cost = COST_DICE;
				else {
					fprintf(stderr, "[WARNING] Unknown cost metric %s, going with sse\n", tok[1]);
					cost = COST_SSE;
				}
			}
			else
				bcnn_set_param(net, tok[0], tok[1]);
			
			bh_free(tok[0]);
			bh_free(tok[1]);
			bh_free(tok);
			bh_free(line);
			break;
		}
	}
	// Add cost layer
	if (strcmp(curr_layer, "{cost}") == 0) {
		bcnn_add_cost_layer(net, cost, 1.0f);
	}
	else
		bh_error("Error in config file: last layer must be a cost layer", BCNN_INVALID_PARAMETER);
	bh_free(curr_layer);
	fclose(file);
	nb_connections = net->nb_connections;

	param->eval_period = (param->eval_period > 0 ? param->eval_period : 100);

	fflush(stderr);
	return 0;
}


int bcnncl_train(bcnn_net *net, bcnncl_param *param, float *error)
{
	float error_batch = 0.0f, sum_error = 0.0f, error_valid = 0.0f;
	int i = 0, nb_iter = net->max_batches;
	int batch_size = net->input_node.b;
	bh_timer t = { 0 };
	bcnn_iterator iter_data = { 0 };

	if (bcnn_init_iterator(net, &iter_data, param->train_input, NULL, param->data_format) != 0)
		return -1;

	bcnn_compile_net(net, "train");

	bh_timer_start(&t);
	for (i = 0; i < nb_iter; ++i) {
		bcnn_train_on_batch(net, &iter_data, &error_batch);
		sum_error += error_batch;
		if (param->eval_test) {
			if (i % param->eval_period == 0 && i > 0) {
				bh_timer_stop(&t);
				bcnncl_predict(net, param, &error_valid, 1);
				fprintf(stderr, "iter= %d train-error= %f test-error= %f training-time= %lf sec\n", i,
					sum_error / (param->eval_period * batch_size), error_valid, bh_timer_get_msec(&t) / 1000);
				fflush(stderr);
				bh_timer_start(&t);
				sum_error = 0;
				bcnn_compile_net(net, "train");
			}
		}
		else {
			if (i % param->eval_period == 0 && i > 0) {
				bh_timer_stop(&t);
				fprintf(stderr, "iter= %d train-error= %f training-time= %lf sec\n", i,
					sum_error / (param->eval_period * batch_size), bh_timer_get_msec(&t) / 1000);
				fflush(stderr);
				bh_timer_start(&t);
				sum_error = 0;
			}
		}
	}

	bcnn_free_iterator(&iter_data);
	*error = (float)sum_error / (param->eval_period * batch_size);

	return 0;
}


int bcnncl_predict(bcnn_net *net, bcnncl_param *param, float *error, int dump_pred)
{
	int i = 0, j = 0, n = 0, k = 0;
	float *out = NULL;
	float err = 0.0f, error_batch = 0.0f;
	FILE *f = NULL;
	int batch_size = net->input_node.b;
	unsigned char *img_pred = NULL;
	char out_pred_name[128] = { 0 };
	bcnn_iterator iter_data = { 0 };
	int out_w = net->connections[net->nb_connections - 2].dst_node.w;
	int out_h = net->connections[net->nb_connections - 2].dst_node.h;
	int out_c = net->connections[net->nb_connections - 2].dst_node.c;
	int output_size = out_w * out_h * out_c;

	bcnn_init_iterator(net, &iter_data, param->test_input, NULL, param->data_format);

	if (dump_pred) {
		if (net->prediction_type == HEATMAP_REGRESSION ||
			net->prediction_type == SEGMENTATION) {
			img_pred = (unsigned char *)calloc(out_w * out_h, sizeof(unsigned char));
		}
		else {
			f = fopen(param->pred_out, "wt");
			if (f == NULL) {
				fprintf(stderr, "[ERROR] bcnn_predict: Can't open file %s", param->pred_out);
				return -1;
			}
		}
	}

	bcnn_compile_net(net, "predict");

	n = param->nb_pred / batch_size;
	for (i = 0; i < n; ++i) {
		bcnn_predict_on_batch(net, &iter_data, &out, &error_batch);
		err += error_batch;
		// Dump predictions
		if (dump_pred) {
			if (net->prediction_type == HEATMAP_REGRESSION ||
				net->prediction_type == SEGMENTATION) {
				for (j = 0; j < net->input_node.b; ++j) {
					for (k = 0; k < out_c; ++k) {
						sprintf(out_pred_name, "%d_%d.png", i * net->input_node.b + j, k);
						bip_write_float_image(out_pred_name,
							out + j * out_w * out_h * out_c + k * out_w * out_h,
							out_w, out_h, 1, out_w * sizeof(float));
					}
				}
			}
			else {
				for (j = 0; j < net->input_node.b; ++j) {
					for (k = 0; k < output_size; ++k)
						fprintf(f, "%f ", out[j * output_size + k]);
					fprintf(f, "\n");
				}
			}
		}
	}
	// Process last instances
	n = param->nb_pred % net->input_node.b;
	if (n > 0) {
		for (i = 0; i < n; ++i) {
			bcnn_predict_on_batch(net, &iter_data, &out, &error_batch);
			err += error_batch;
			// Dump predictions
			if (dump_pred) {
				if (net->prediction_type == HEATMAP_REGRESSION ||
					net->prediction_type == SEGMENTATION) {
					for (k = 0; k < out_c; ++k) {
						sprintf(out_pred_name, "%d_%d.png", i, k);
						bip_write_float_image(out_pred_name,
							out + k * out_w * out_h,
							out_w, out_h, 1, out_w * sizeof(float));
					}
				}
				else {
					for (k = 0; k < output_size; ++k)
						fprintf(f, "%f ", out[k]);
					fprintf(f, "\n");
				}
			}
		}
	}
	*error = err / param->nb_pred;

	if (f != NULL)
		fclose(f);
	bcnn_free_iterator(&iter_data);
	return 0;
}

int bcnncl_free_param(bcnncl_param *param)
{
	bh_free(param->input_model);
	bh_free(param->output_model);
	bh_free(param->pred_out);
	bh_free(param->train_input);
	bh_free(param->test_input);
	bh_free(param->data_format);
	return 0;
}

int run(char *config_file)
{
	bcnn_net *net = NULL;
	bcnncl_param param = { 0 };
	float error_train = 0.0f, error_valid = 0.0f, error_test = 0.0f;
	
	bcnn_init_net(&net);
	// Initialize network from config file
	if (bcnncl_init_from_config(net, config_file, &param) != BCNN_SUCCESS)
		bh_error("Error in config file", -1);

	if (param.task == TRAIN) {
		if (param.input_model != NULL) {
			fprintf(stderr, "[INFO] Loading pre-trained model %s\n", param.input_model);
			bcnn_load_model(net, param.input_model);
		}
		bh_info("Start training...");
		if (bcnncl_train(net, &param, &error_train) != 0)
			bh_error("Can not perform training", -1);
		if (param.pred_out != NULL)
			bcnncl_predict(net, &param, &error_valid, 1);
		if (param.output_model != NULL)
			bcnn_write_model(net, param.output_model);
		bh_info("Training ended successfully");
	}
	else if (param.task == PREDICT) {
		if (param.input_model != NULL)
			bcnn_load_model(net, param.input_model);
		else
			bh_error("No model in input. Inform which model to use in config file with field 'input_model'", BCNN_INVALID_PARAMETER);
		bh_info("Start prediction...");
		if (bcnncl_predict(net, &param, &error_test, 1) != 0)
			bh_error("Can not perform prediction", -1);
		bh_info("Prediction ended successfully");
	}
	bcnn_end_net(&net);
	bcnncl_free_param(&param);
	return 0;
}


int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <config>\n", argv[0]);
		return -1;
	}
	run(argv[1]);
	return 0;
}

