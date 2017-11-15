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
#include <bh/bh_string.h>
#include <bh/bh_error.h>

/* include bip image processing lib */
#include <bip/bip.h>

#include "bcnn/bcnn.h"


int bcnn_pack_data(char *list, int label_width, bcnn_label_type type, char *out_pack)
{
	FILE *f_lst = NULL, *f_out = NULL, *f_outlst = NULL;
	char *line = NULL;
	int n = 0, n_tok = 0;
	char **tok = NULL;
	int i, w, h, c, buf_sz, part = 0;
	float lf;
	unsigned char *img = NULL;
	unsigned char *buf = NULL;
	char name[256];
	size_t cnt = 0, max_part_sz = 256000000;

	f_lst = fopen(list, "rt");
	if (f_lst == NULL) {
		fprintf(stderr, "[ERROR] Can not open %s\n", list);
		return -1;
	}
	f_outlst = fopen(out_pack, "wt");
	if (f_outlst == NULL) {
		fprintf(stderr, "[ERROR] Can not open %s\n", out_pack);
		return -1;
	}

	while ((line = bh_fgetline(f_lst)) != NULL) {
		n++;
		bh_free(line);
	}
	rewind(f_lst);

	sprintf(name, "%s_%d.bin", out_pack, part);
	f_out = fopen(name, "wb");
	cnt += fwrite(&n, 1, sizeof(int), f_out);
	cnt += fwrite(&label_width, 1, sizeof(int), f_out);
	cnt += fwrite(&type, 1, sizeof(int), f_out);
	
	while ((line = bh_fgetline(f_lst)) != NULL) {
		if (cnt > max_part_sz) {
			if (f_out != NULL)
				fclose(f_out);
			cnt = 0;
			part++;
			fprintf(f_outlst, "%s\n", name);
			sprintf(name, "%s_%d.bin", out_pack, part);
			f_out = fopen(name, "wb");
			cnt += fwrite(&n, 1, sizeof(int), f_out);
			cnt += fwrite(&label_width, 1, sizeof(int), f_out);
			cnt += fwrite(&type, 1, sizeof(int), f_out);
		}
		n_tok = bh_strsplit(line, ' ', &tok);
		bh_assert((n_tok - 1 == label_width),
			"Data and label_width are not consistent", BCNN_INVALID_DATA);
		bip_load_image(tok[0], &img, &w, &h, &c);
		bip_write_image_to_memory(&buf, &buf_sz, img, w, h, c, w * c);
		// Write img
		cnt += fwrite(&buf_sz, 1, sizeof(int), f_out);
		cnt += fwrite(buf, 1, buf_sz, f_out);
		bh_free(buf);
		bh_free(img);
		// Write label(s)
		switch (type) {
		case LABEL_INT:
			for (i = 1; i < n_tok; ++i) {
				lf = (float)atoi(tok[i]);
				cnt += fwrite(&lf, 1, sizeof(float), f_out);
			}
			break;
		case LABEL_FLOAT:
			for (i = 1; i < n_tok; ++i) {
				lf = (float)atof(tok[i]);
				cnt += fwrite(&lf, 1, sizeof(float), f_out);
			}
			break;
		}
		
		bh_free(line);
		for (i = 0; i < n_tok; ++i) bh_free(tok[i]);
		bh_free(tok);
	}
	fprintf(f_outlst, "%s\n", name);
	if (f_out != NULL)
		fclose(f_out);
	if (f_lst != NULL)
		fclose(f_lst);
	if (f_outlst != NULL)
		fclose(f_outlst);

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
					dst[w * (h * (2 - k) + y) + x] = ((float)src[c * (x + w * y) + k] / 255.0f - m) * 2.0f;
				}
			}
		}
	}
	else {
		for (k = 0; k < c; ++k){
			for (y = 0; y < h; ++y){
				for (x = 0; x < w; ++x){
					dst[w * (h * k + y) + x] = ((float)src[c * (x + w * y) + k] / 255.0f - 0.5f) * 2.0f;
				}
			}
		}
	}
	return 0;
}


/* IO */
int bcnn_load_image_from_csv(char *str, int w, int h, int c, unsigned char **img)
{
	int i, n_tok, sz = w * h * c;
	char **tok = NULL;
	unsigned char *ptr_img = NULL;

	n_tok = bh_strsplit(str, ',', &tok);

	bh_assert(n_tok == sz, "Incorrect data size in csv", BCNN_INVALID_DATA);

	ptr_img = (unsigned char *)calloc(sz, sizeof(unsigned char));
	for (i = 0; i < n_tok; ++i) {
		ptr_img[i] = (unsigned char)atoi(tok[i]);
	}
	*img = ptr_img;

	for (i = 0; i < n_tok; ++i)
		bh_free(tok[i]);
	bh_free(tok);

	return BCNN_SUCCESS;
}


int bcnn_load_image_from_path(char *path, int w, int h, int c, unsigned char **img, int state,
	int *x_shift, int *y_shift)
{
	int w_img, h_img, c_img, x_ul = 0, y_ul = 0;
	unsigned char *buf = NULL, *pimg = NULL;

	bip_load_image(path, &buf, &w_img, &h_img, &c_img);
	bh_assert(w_img > 0 && h_img > 0 && buf,
		"Invalid image",
		BCNN_INVALID_DATA);
	if (c != c_img) {
		fprintf(stderr, "Unexpected number of channels of image %s\n", path);
		bh_free(buf);
		return BCNN_INVALID_DATA;
	}
	
	if (w_img != w || h_img != h) {
		if (state == 0) { // state predict, always center crop
			x_ul = (w_img - w) / 2;
			y_ul = (h_img - h) / 2;
		}
		else { // state train, random crop
			x_ul = (int)((float)rand() / RAND_MAX * (w_img - w));
			y_ul = (int)((float)rand() / RAND_MAX * (h_img - h));
		}
		pimg = (unsigned char*)calloc(w * h * c, sizeof(unsigned char));
		bip_crop_image(buf, w_img, h_img, w_img * c_img, x_ul, y_ul,
			pimg, w, h, w * c, c);
		*img = pimg;
		bh_free(buf);
	}
	else
		*img = buf;
	*x_shift = x_ul;
	*y_shift = y_ul;

	return BCNN_SUCCESS;
}


int bcnn_load_image_from_memory(unsigned char *buffer, int buffer_size, int w, int h, int c, unsigned char **img, int state,
	int *x_shift, int *y_shift)
{
	int w_img, h_img, c_img, x_ul = 0, y_ul = 0;
	unsigned char *tmp = NULL, *pimg = NULL;

	bip_load_image_from_memory(buffer, buffer_size, &tmp, &w_img, &h_img, &c_img);
	bh_assert(w_img > 0 && h_img > 0 && tmp,
		"Invalid image",
		BCNN_INVALID_DATA);
	if (c != c_img) {
		//fprintf(stderr, "Unexpected number of channels\n");
		bh_free(tmp);
		return BCNN_INVALID_DATA;
	}
	
	if (w_img != w || h_img != h) {
		if (state == 0) { // state predict, always center crop
			x_ul = (w_img - w) / 2;
			y_ul = (h_img - h) / 2;
		}
		else { // state train, random crop
			x_ul = (int)((float)rand() / RAND_MAX * (w_img - w));
			y_ul = (int)((float)rand() / RAND_MAX * (h_img - h));
		}
		pimg = (unsigned char*)calloc(w * h * c, sizeof(unsigned char));
		bip_crop_image(tmp, w_img, h_img, w_img * c_img, x_ul, y_ul,
			pimg, w, h, w * c, c);
		memcpy(*img, pimg, w * h * c);
		bh_free(pimg);
	}
	else {
		memcpy(*img, tmp, w * h * c);
	}
	bh_free(tmp);
	*x_shift = x_ul;
	*y_shift = y_ul;

	return BCNN_SUCCESS;
}

/* Mnist iter */
unsigned int _read_int(char *v)
{
	int i;
	unsigned int ret = 0;

	for (i = 0; i < 4; ++i) {
		ret <<= 8;
		ret |= (unsigned char)v[i];
	}

	return ret;
}

static int bcnn_mnist_next_iter(bcnn_net *net, bcnn_iterator *iter)
{
	char tmp[16];
	unsigned char l;
	unsigned int n_img = 0, n_labels = 0;
	size_t n = 0;
	
	if (fread((char *)&l, 1, sizeof(char), iter->f_input) == 0)
		rewind(iter->f_input);
	else
		fseek(iter->f_input, -1, SEEK_CUR);
	if (fread((char *)&l, 1, sizeof(char), iter->f_label) == 0)
		rewind(iter->f_label);
	else
		fseek(iter->f_label, -1, SEEK_CUR);

	if (ftell(iter->f_input) == 0 && ftell(iter->f_label) == 0) {
		fread(tmp, 1, 16, iter->f_input);
		n_img = _read_int(tmp + 4);
		iter->input_height = _read_int(tmp + 8);
		iter->input_width = _read_int(tmp + 12);
		fread(tmp, 1, 8, iter->f_label);
		n_labels = _read_int(tmp + 4);
		bh_assert(n_img == n_labels, "MNIST data: number of images and labels must be the same", 
			BCNN_INVALID_DATA);
		bh_assert(net->input_node.h == iter->input_height && net->input_node.w == iter->input_width,
			"MNIST data: incoherent image width and height",
			BCNN_INVALID_DATA);
		iter->n_samples = n_img;
	}

	// Read label
	n = fread((char *)&l, 1, sizeof(char), iter->f_label);
	iter->label_int[0] = (int)l;
	// Read img
	n = fread(iter->input_uchar, 1, iter->input_width * iter->input_height, iter->f_input);

	return BCNN_SUCCESS;
}

static int bcnn_init_bin_iterator(bcnn_net *net, bcnn_iterator *iter, char *path_input)
{
	FILE *f_bin = NULL, *f_lst = NULL;
	char *line = NULL;
	bcnn_label_type type;

	iter->type = ITER_BIN;

	f_lst = fopen(path_input, "rt");
	if (f_lst == NULL) {
		fprintf(stderr, "[ERROR] Can not open file %s\n", path_input);
		return BCNN_INVALID_PARAMETER;
	}
	// Open first binary file
	line = bh_fgetline(f_lst);
	if (line == NULL)
		bh_error("Empty data list", BCNN_INVALID_DATA);

	//bh_strstrip(line);
	f_bin = fopen(line, "rb");
	if (f_bin == NULL) {
		fprintf(stderr, "[ERROR] Can not open file %s\n", line);
		return BCNN_INVALID_PARAMETER;
	}

	fread(&iter->n_samples, 1, sizeof(int), f_bin);
	fread(&iter->label_width, 1, sizeof(int), f_bin);
	fread(&type, 1, sizeof(int), f_bin);
	iter->input_width = net->input_node.w;
	iter->input_height = net->input_node.h;
	iter->input_depth = net->input_node.c;
	iter->input_uchar = (unsigned char *)calloc(iter->input_width * iter->input_height * iter->input_depth,
		sizeof(unsigned char));
	iter->label_float = (float *)calloc(iter->label_width, sizeof(float));

	iter->f_input = f_bin;
	iter->f_list = f_lst;

	bh_free(line);

	return BCNN_SUCCESS;
}

static int bcnn_bin_iter(bcnn_net *net, bcnn_iterator *iter)
{
	unsigned char l;
	size_t n = 0;
	int i, buf_sz = 0, label_width, type;
	float lf;
	unsigned char *buf = NULL;
	char *line = NULL;

	if (fread((char *)&l, 1, sizeof(char), iter->f_input) == 0) {
		// Jump to next binary part file
		fclose(iter->f_input);
		line = bh_fgetline(iter->f_list);
		if (line == NULL) {
			rewind(iter->f_list);
			line = bh_fgetline(iter->f_list);
		}
		iter->f_input = fopen(line, "rb");
		if (iter->f_input == NULL) {
			fprintf(stderr, "[ERROR] Can not open file %s\n", line);
			return BCNN_INVALID_PARAMETER;
		}
		bh_free(line);
	}
	else
		fseek(iter->f_input, -1, SEEK_CUR);

	if (ftell(iter->f_input) == 0) {
		fread(&n, 1, sizeof(int), iter->f_input);
		fread(&label_width, 1, sizeof(int), iter->f_input);
		fread(&type, 1, sizeof(int), iter->f_input);
	}

	// Read image
	fread(&buf_sz, 1, sizeof(int), iter->f_input);
	buf = (unsigned char *)calloc(buf_sz, sizeof(unsigned char));
	fread(buf, 1, buf_sz, iter->f_input);
	bcnn_load_image_from_memory(buf, buf_sz, net->input_node.w, net->input_node.h, net->input_node.c,
		&iter->input_uchar, net->state, &net->data_aug.shift_x, &net->data_aug.shift_y);
	bh_free(buf);

	// Read label
	for (i = 0; i < iter->label_width; ++i) {
		n = fread(&lf, 1, sizeof(float), iter->f_input);
		iter->label_float[i] = lf;
	}

	return BCNN_SUCCESS;
}

/* Handles cifar10 binary format */
static int bcnn_init_cifar10_iterator(bcnn_net *net, bcnn_iterator *iter, char *path_input)
{
	FILE *f_bin = NULL;

	iter->type = ITER_CIFAR10;

	f_bin = fopen(path_input, "rb");
	if (f_bin == NULL) {
		fprintf(stderr, "[ERROR] Can not open file %s\n", path_input);
		return BCNN_INVALID_PARAMETER;
	}

	iter->n_samples = 0; // not used
	iter->label_width = 1;

	iter->input_uchar = (unsigned char *)calloc(iter->input_width * iter->input_height, sizeof(unsigned char));
	iter->label_int = (int *)calloc(1, sizeof(int));
	iter->input_width = 32;
	iter->input_height = 32;
	iter->input_depth = 3;
	iter->input_uchar = (unsigned char *)calloc(iter->input_width * iter->input_height * iter->input_depth,
		sizeof(unsigned char));
	iter->f_input = f_bin;

	return BCNN_SUCCESS;
}

static int bcnn_cifar10_iter(bcnn_net *net, bcnn_iterator *iter)
{
	unsigned char l;
	unsigned int n_img = 0, n_labels = 0;
	size_t n = 0;
	int x, y, k, i, rand_skip = /*(int)(10.0f * (float)rand() / RAND_MAX) + 1*/0;
	char tmp[3072];

	if (net->state == TRAIN) {
		for (i = 0; i < rand_skip; ++i) {
			if (fread((char *)&l, 1, sizeof(char), iter->f_input) == 0)
				rewind(iter->f_input);
			else
				fseek(iter->f_input, -1, SEEK_CUR);
			fseek(iter->f_input, iter->input_width * iter->input_height * iter->input_depth + 1, SEEK_CUR);
		}
	}
	if (fread((char *)&l, 1, sizeof(char), iter->f_input) == 0)
			rewind(iter->f_input);
	else
		fseek(iter->f_input, -1, SEEK_CUR);

	// Read label
	n = fread((char *)&l, 1, sizeof(char), iter->f_input);
	iter->label_int[0] = (int)l;
	// Read img
	n = fread(tmp, 1, iter->input_width * iter->input_height * iter->input_depth, iter->f_input);
	// Swap depth <-> spatial dim arrangement
	for (k = 0; k < iter->input_depth; ++k) {
		for (y = 0; y < iter->input_height; ++y) {
			for (x = 0; x < iter->input_width; ++x) {
				iter->input_uchar[(x + iter->input_width * y) * iter->input_depth + k] =
					tmp[iter->input_width * (iter->input_height * k + y) + x];
			}
		}
	}
	/*bip_write_image("test00.png", iter->input_uchar, iter->input_width, iter->input_height, iter->input_depth, 
		iter->input_width * iter->input_depth);*/

	return BCNN_SUCCESS;
}


static int bcnn_init_list_iterator(bcnn_net *net, bcnn_iterator *iter, char *path_input)
{
	int i;
	FILE *f_list = NULL;
	char *line = NULL;
	char **tok = NULL;
	int n_tok = 0;
	unsigned char *img = NULL;
	int out_w = net->connections[net->nb_connections - 2].dst_tensor.w;
	int	out_h = net->connections[net->nb_connections - 2].dst_tensor.h;
	int	out_c = net->connections[net->nb_connections - 2].dst_tensor.c;

	iter->type = ITER_LIST;

	f_list = fopen(path_input, "rb");
	if (f_list == NULL) {
		fprintf(stderr, "[ERROR] Can not open file %s\n", path_input);
		return BCNN_INVALID_PARAMETER;
	}

	line = bh_fgetline(f_list);
	n_tok = bh_strsplit(line, ' ', &tok);
	bcnn_load_image_from_path(tok[0], net->input_node.w, net->input_node.h, net->input_node.c, &img, net->state,
		&net->data_aug.shift_x, &net->data_aug.shift_y);
	iter->input_width = net->input_node.w;
	iter->input_height = net->input_node.w;
	iter->input_depth = net->input_node.c;
	iter->input_uchar = (unsigned char *)calloc(iter->input_width * iter->input_height * iter->input_depth,
		sizeof(unsigned char));
	
	if (net->prediction_type != SEGMENTATION)
		iter->label_width = n_tok - 1;
	else
		iter->label_width = out_w * out_h * out_c;
	iter->label_float = (float *)calloc(iter->label_width, sizeof(float));
	
	rewind(f_list);
	iter->f_input = f_list;
	bh_free(line);
	bh_free(img);
	for (i = 0; i < n_tok; ++i) bh_free(tok[i]);
	bh_free(tok);
	
	return BCNN_SUCCESS;
}


static int bcnn_list_iter(bcnn_net *net, bcnn_iterator *iter)
{
	char *line = NULL;
	char **tok = NULL;
	int i, n_tok = 0, tmp_x, tmp_y;
	int out_w = net->connections[net->nb_connections - 2].dst_tensor.w;
	int	out_h = net->connections[net->nb_connections - 2].dst_tensor.h;
	int	out_c = net->connections[net->nb_connections - 2].dst_tensor.c;
	unsigned char *img = NULL;
	//nb_lines_skipped = (int)((float)rand() / RAND_MAX * net->input_node.b);
	//bh_fskipline(f, nb_lines_skipped);
	line = bh_fgetline(iter->f_input);
	if (line == NULL) {
		rewind(iter->f_input);
		line = bh_fgetline(iter->f_input);
	}
	n_tok = bh_strsplit(line, ' ', &tok);
	if (net->task != PREDICT && net->prediction_type == CLASSIFICATION) {
		bh_assert(n_tok == 2,
			"Wrong data format for classification", BCNN_INVALID_DATA);
	}
	if (iter->type == ITER_LIST) {
		bcnn_load_image_from_path(tok[0], net->input_node.w, net->input_node.h, net->input_node.c, &iter->input_uchar, net->state,
			&net->data_aug.shift_x, &net->data_aug.shift_y);
	}
	else
		bcnn_load_image_from_csv(tok[0], net->input_node.w, net->input_node.h, net->input_node.c, &iter->input_uchar);

	// Label
	if (net->prediction_type != SEGMENTATION) {
		for (i = 0; i < iter->label_width; ++i) {
			iter->label_float[i] = (float)atof(tok[i + 1]);
		}
	}
	else {
		for (i = 0; i < iter->label_width; ++i) {
			if (iter->type == ITER_LIST) {
				bcnn_load_image_from_path(tok[i], out_w, out_h, out_c, &img, net->state,
					&tmp_x, &tmp_y);
			}
			else
				bcnn_load_image_from_csv(tok[i], out_w, out_h, out_c, &img);
			bcnn_convert_img_to_float(img, out_w, out_h, out_c, 0, 0, 0, 0, iter->label_float);
			bh_free(img);
		}
	}
	

	bh_free(line);
	for (i = 0; i < n_tok; ++i)
		bh_free(tok[i]);
	bh_free(tok);

	return BCNN_SUCCESS;
}

/* Data augmentation */
int bcnn_data_augmentation(unsigned char *img, int width, int height, int depth, bcnn_data_augment *param,
	unsigned char *buffer)
{
	int sz = width * height * depth;
	unsigned char *img_scale = NULL;
	int x_ul = 0, y_ul = 0, w_scale, h_scale;
	float scale = 1.0f, theta = 0.0f, contrast = 1.0f, kx, ky, distortion;
	int brightness = 0;

	if (param->random_fliph) {
		if ((float)rand() / RAND_MAX > 0.5f) {
			bip_fliph_image(img, width, height, depth, width * depth, buffer, width * depth);
			memcpy(img, buffer, sz * sizeof(unsigned char));
		}
	}
	if (param->range_shift_x || param->range_shift_y) {
		memset(buffer, 128, sz);
		if (param->use_precomputed) {
			x_ul = param->shift_x;
			y_ul = param->shift_y;
		}
		else {
			x_ul = (int)((float)(rand() - RAND_MAX / 2) / RAND_MAX * param->range_shift_x);
			y_ul = (int)((float)(rand() - RAND_MAX / 2) / RAND_MAX * param->range_shift_y);
			param->shift_x = x_ul;
			param->shift_y = y_ul;
		}
		bip_crop_image(img, width, height, width * depth, x_ul, y_ul, buffer, width, height, width * depth, depth);
		memcpy(img, buffer, sz * sizeof(unsigned char));
	}
	if (param->max_scale > 0.0f || param->min_scale > 0.0f) {
		if (param->use_precomputed) {
			scale = param->scale;
		}
		else {
			scale = (((float)rand() / RAND_MAX) * (param->max_scale - param->min_scale) +
				param->min_scale);
			param->scale = scale;
		}
		w_scale = (int)(width * scale); 
		h_scale = (int)(height * scale);
		img_scale = (unsigned char *)calloc(w_scale * h_scale * depth, sizeof(unsigned char));
		bip_resize_bilinear(img, width, height, width * depth, img_scale, w_scale, h_scale, w_scale * depth, depth);
		bip_crop_image(img_scale, w_scale, h_scale, w_scale * depth, x_ul, y_ul, img, width, height, width * depth, depth);
		bh_free(img_scale);
	}
	if (param->rotation_range > 0.0f) {
		if (param->use_precomputed) {
			theta = param->rotation;
		}
		else {
			theta = bip_deg2rad((float)(rand() - RAND_MAX / 2) / RAND_MAX  * param->rotation_range);
			param->rotation = theta;
		}
		memset(buffer, 128, sz);
		bip_rotate_image(img, width, height, width * depth,
			buffer, width, height, width * depth, depth, theta, width / 2, height / 2, BILINEAR);
		memcpy(img, buffer, width * height * depth * sizeof(unsigned char));
	}
	if (param->min_contrast > 0.0f || param->max_contrast > 0.0f) {
		if (param->use_precomputed) {
			contrast = param->contrast;
		}
		else {
			contrast = (((float)rand() / RAND_MAX) * (param->max_contrast - param->min_contrast) +
				param->min_contrast);
			param->contrast = contrast;
		}
		bip_contrast_stretch(img, width * depth, width, height, depth, img, width * depth, contrast);	
	}
	if (param->min_brightness != 0 || param->max_brightness != 0) {
		if (param->use_precomputed) {
			brightness = param->brightness;
		}
		else {
			brightness = (int)(((float)rand() / RAND_MAX) * (param->max_brightness - param->min_brightness) +
				param->min_brightness);
			param->brightness = brightness;
		}
		bip_image_brightness(img, width * depth, width, height, depth, img, width * depth, brightness);
	}
	if (param->max_distortion > 0.0f) {
		if (param->use_precomputed) {
			kx = param->distortion_kx;
			ky = param->distortion_ky;
			distortion = param->distortion;
		}
		else {
			kx = (((float)rand() - RAND_MAX / 2) / RAND_MAX);
			ky = (((float)rand() - RAND_MAX / 2) / RAND_MAX);
			distortion = ((float)rand() / RAND_MAX) * (param->max_distortion);
			param->distortion_kx = kx;
			param->distortion_ky = ky;
			param->distortion = distortion;
		}
		bip_image_perlin_distortion(img, width * depth, width, height, depth, buffer, width * depth,
			param->distortion, kx, ky);
		memcpy(img, buffer, width * height * depth * sizeof(unsigned char));
	}

	return BCNN_SUCCESS;
}


static int bcnn_init_mnist_iterator(bcnn_iterator *iter, char *path_img, char *path_label)
{
	FILE *f_img = NULL, *f_label = NULL;
	char tmp[16] = { 0 };
	int n_img = 0, n_lab = 0;

	iter->type = ITER_MNIST;
	f_img = fopen(path_img, "rb");
	if (f_img == NULL) {
		fprintf(stderr, "[ERROR] Cound not open file %s\n", path_img);
		return -1;
	}
	f_label = fopen(path_label, "rb");
	if (f_label == NULL) {
		fprintf(stderr, "[ERROR] Cound not open file %s\n", path_label);
		return -1;
	}

	iter->f_input = f_img;
	iter->f_label = f_label;
	iter->n_iter = 0;
	// Read header
	fread(tmp, 1, 16, iter->f_input);
	n_img = _read_int(tmp + 4);
	iter->input_height = _read_int(tmp + 8);
	iter->input_width = _read_int(tmp + 12);
	fread(tmp, 1, 8, iter->f_label);
	n_lab = _read_int(tmp + 4);
	bh_assert(n_img == n_lab, "Inconsistent MNIST data: number of images and labels must be the same",
		BCNN_INVALID_DATA);

	iter->input_uchar = (unsigned char *)calloc(iter->input_width * iter->input_height, sizeof(unsigned char));
	iter->label_int = (int *)calloc(1, sizeof(int));
	rewind(iter->f_input);
	rewind(iter->f_label);

	return 0;
}


int bcnn_init_iterator(bcnn_net *net, bcnn_iterator *iter, char *path_input, char *path_label, char *type)
{
	
	if (strcmp(type, "mnist") == 0) {
		return bcnn_init_mnist_iterator(iter, path_input, path_label);
	}
	else if (strcmp(type, "bin") == 0) {
		return bcnn_init_bin_iterator(net, iter, path_input);
	}
	else if (strcmp(type, "list") == 0) {
		return bcnn_init_list_iterator(net, iter, path_input);
	}
	else if (strcmp(type, "cifar10") == 0) {
		return bcnn_init_cifar10_iterator(net, iter, path_input);
	}
	else {
		bh_error("Unknown data_format. Available are 'mnist' 'bin' 'list' 'cifar10'",
			BCNN_INVALID_PARAMETER);
	}
	
	return BCNN_SUCCESS;
}

int bcnn_advance_iterator(bcnn_net *net, bcnn_iterator *iter)
{
	switch (iter->type) {
	case ITER_MNIST:
		bcnn_mnist_next_iter(net, iter);
		break;
	case ITER_CIFAR10:
		bcnn_cifar10_iter(net, iter);
		break;
	case ITER_BIN:
		bcnn_bin_iter(net, iter);
		break;
	case ITER_LIST:
		bcnn_list_iter(net, iter);
		break;
	default: break;
	}
	return 0;
}

int bcnn_free_iterator(bcnn_iterator *iter)
{
	if (iter->f_input != NULL)
		fclose(iter->f_input);
	if (iter->f_label != NULL)
		fclose(iter->f_label);
	if (iter->f_list != NULL)
		fclose(iter->f_list);
	bh_free(iter->input_uchar);
	bh_free(iter->label_float);
	bh_free(iter->label_uchar);
	bh_free(iter->label_int);

	return BCNN_SUCCESS;
}