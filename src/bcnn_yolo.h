#ifndef BCNN_YOLO_H
#define BCNN_YOLO_H

#include <bcnn/bcnn.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bcnn_yolo_param {
    int num;      // num bboxes per cell
    int classes;  // # classes
    int coords;   // # coords
    int truths;   // labels
    int max_boxes;
    int total;
    bcnn_tensor biases;
    int *mask;
    float *cost;
} bcnn_yolo_param;

void bcnn_forward_yolo_layer(bcnn_net *net, bcnn_node *node);
void bcnn_backward_yolo_layer(bcnn_net *net, bcnn_node *node);
void bcnn_release_param_yolo_layer(bcnn_node *node);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_YOLO_H