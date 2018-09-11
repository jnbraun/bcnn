#ifndef BCNN_YOLO_H
#define BCNN_YOLO_H

#include <bcnn/bcnn.h>

#ifdef __cplusplus
extern "C" {
#endif

void bcnn_forward_yolo_layer(bcnn_net *net, bcnn_node *node);
void bcnn_backward_yolo_layer(bcnn_net *net, bcnn_node *node);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_YOLO_H