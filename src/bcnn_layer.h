/*
* Copyright (c) 2016-2018 Jean-Noel Braun.
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

#ifndef BH_LAYER_H
#define BH_LAYER_H

#ifdef __cplusplus
extern "C" {
#endif
/* Experimental API... WIP
typedef struct bcnn_layer {
    bcnn_layer_type layer_type;
    int data_size;
    void* data;
    void (*initialize)(bcnn_layer* layer);
    void (*terminate)(bcnn_layer* layer);
    void (*forward)(bcnn_layer* layer);
    void (*backward)(bcnn_layer* layer);
    void (*update)(bcnn_layer* layer);
} bcnn_layer;

bcnn_layer* bcnn_layer_create(bcnn_layer_type type, bcnn_layer_param* param);

bcnn_layer bcnn_layer_fullc = {
    .layer_type = FULL_CONNECTED,
    .data_size = sizeof(struct bcnn_layer_fullc_data),
    .initialize = bcnn_layer_fullc_init,
    .terminate = bcnn_layer_fullc_terminate,
    .func_update = bcnn_layer_fullc_update,
    .forward = bcnn_forward_fullc_layer,
    .backward = bcnn_backward_fullc_layer};

typedef struct bcnn_layer_fullc_data {
    bcnn_tensor weights;
    bcnn_tensor bias;
    bcnn_activation activation;
} bcnn_layer_fullc_data;

void bcnn_layer_fullc_init(bcnn_layer* layer) {
    bcnn_layer_fullc_data* data = (bcnn_layer_fullc_data*)layer->data;
    data->weights;
}*/

#ifdef __cplusplus
}
#endif

#endif  // BH_LAYER_H
