#include "bcnn_yolo.h"

#include <bh/bh_string.h>

#include "bcnn_activation_layer.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"

/** From yolo darknet */
bcnn_status bcnn_add_yolo_layer(bcnn_net *net, int n, int classes, int coords,
                                int total, int *mask, float *anchors,
                                char *src_id, char *dst_id) {
    // layer l = {0};
    bcnn_node node = {0};

    BCNN_CHECK_AND_LOG(net->log_ctx, net->num_nodes >= 1,
                       BCNN_INVALID_PARAMETER,
                       "Yolo layer can't be the first layer of the network");
    int is_src_node_found = 0;
    for (int i = net->num_tensors - 1; i >= 0; --i) {
        if (strcmp(net->tensors[i].name, src_id) == 0) {
            bcnn_node_add_input(net, &node, i);
            is_src_node_found = 1;
            break;
        }
    }
    BCNN_CHECK_AND_LOG(net->log_ctx, is_src_node_found, BCNN_INVALID_PARAMETER,
                       "Yolo layer: invalid input node name %s", src_id);

    node.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    node.layer->type = YOLO;
    node.layer->num = n;  // num bboxes per cell
    node.layer->total = total;
    node.layer->mask = (int *)calloc(n, sizeof(int));
    memcpy(node.layer->mask, mask, 3 * sizeof(int));
    // Setup output tensor
    bcnn_tensor dst_tensor = {0};
    bcnn_tensor_set_shape(&dst_tensor,
                          net->tensors[node.src[0]].n,  // batch size
                          n * (classes + coords + 1),   // depth
                          net->tensors[node.src[0]].h,  // height
                          net->tensors[node.src[0]].w,  // width
                          1);
    bcnn_tensor_allocate(&dst_tensor);
    bh_strfill(&dst_tensor.name, dst_id);
    // Add tensor to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(net, &node, net->num_tensors - 1);

    node.layer->classes = classes;
    node.layer->coords = coords;
    node.layer->cost = (float *)calloc(1, sizeof(float));
    // Setup layer biases
    char biases_name[256];
    sprintf(biases_name, "%s_b", src_id);
    bcnn_tensor_create(&node.layer->biases, 1, 1, 1, total * 2, 1, biases_name);
    bcnn_tensor_filler w_filler = {.value = 0.5f, .type = FIXED};
    bcnn_tensor_fill(&node.layer->biases, w_filler);
    if (anchors != NULL) {
        memcpy(node.layer->biases.data, anchors, total * 2 * sizeof(float));
    }
    node.layer->max_boxes = 90;
    node.layer->truths = node.layer->max_boxes * (coords + 1);

    // Add connection to net
    bcnn_net_add_node(net, node);
    BCNN_INFO(net->log_ctx,
              "[Yolo] input_shape= %dx%dx%d num_classes= %d num_coords= %d "
              "output_shape= %dx%dx%d",
              net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
              net->tensors[node.src[0]].c, classes, coords,
              net->tensors[node.dst[0]].w, net->tensors[node.dst[0]].h,
              net->tensors[node.dst[0]].c);

    return 0;
}

float overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(yolo_box a, yolo_box b) {
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w * h;
    return area;
}

float box_union(yolo_box a, yolo_box b) {
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

float box_iou(yolo_box a, yolo_box b) {
    return box_intersection(a, b) / box_union(a, b);
}

static yolo_box get_yolo_box(float *x, float *biases, int n, int index, int i,
                             int j, int lw, int lh, int w, int h, int stride) {
    yolo_box b;
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = expf(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = expf(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

static float delta_yolo_box(yolo_box truth, float *x, float *biases, int n,
                            int index, int i, int j, int lw, int lh, int w,
                            int h, float *delta, float scale, int stride) {
    yolo_box pred =
        get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x * lw - i);
    float ty = (truth.y * lh - j);
    float tw = log(truth.w * w / biases[2 * n]);
    float th = log(truth.h * h / biases[2 * n + 1]);

    delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
    delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
    delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
    delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);
    return iou;
}

void delta_region_mask(float *truth, float *x, int n, int index, float *delta,
                       int stride, int scale) {
    int i;
    for (i = 0; i < n; ++i) {
        delta[index + i * stride] = scale * (truth[i] - x[index + i * stride]);
    }
}

void delta_yolo_class(float *output, float *delta, int index, int class,
                      int classes, int stride, float *avg_cat) {
    int n;
    if (delta[index]) {
        delta[index + stride * class] = 1 - output[index + stride * class];
        if (avg_cat) {
            *avg_cat += output[index + stride * class];
        }
        return;
    }
    for (n = 0; n < classes; ++n) {
        delta[index + stride * n] =
            ((n == class) ? 1 : 0) - output[index + stride * n];
        if (n == class && avg_cat) {
            *avg_cat += output[index + stride * n];
        }
    }
}

int entry_index(bcnn_layer *layer, bcnn_tensor *dst_tensor, int batch,
                int location, int entry) {
    int n = location / (dst_tensor->w * dst_tensor->h);
    int loc = location % (dst_tensor->w * dst_tensor->h);
    return batch * bcnn_tensor_size3d(dst_tensor) +
           n * (dst_tensor->w * dst_tensor->h) *
               (layer->coords + layer->classes + 1) +
           entry * (dst_tensor->w * dst_tensor->h) + loc;
}

static yolo_box float_to_box(float *f, int stride) {
    yolo_box b = {0};
    b.x = f[0];
    b.y = f[1 * stride];
    b.w = f[2 * stride];
    b.h = f[3 * stride];
    return b;
}

void bcnn_forward_yolo_layer_cpu(bcnn_net *net, bcnn_layer *layer,
                                 bcnn_tensor *src_tensor, bcnn_tensor *label,
                                 bcnn_tensor *dst_tensor) {
    int i, j, b, t, n;
    memcpy(dst_tensor->data, src_tensor->data,
           bcnn_tensor_size(dst_tensor) * sizeof(float));
    for (b = 0; b < dst_tensor->n; ++b) {
        for (n = 0; n < layer->num; ++n) {
            int index = entry_index(layer, dst_tensor, b,
                                    n * src_tensor->w * src_tensor->h, 0);
            bcnn_forward_activation_cpu(dst_tensor->data + index,
                                        2 * src_tensor->w * src_tensor->h,
                                        LOGISTIC);
            index =
                entry_index(layer, dst_tensor, b,
                            n * src_tensor->w * src_tensor->h, layer->coords);
            bcnn_forward_activation_cpu(
                dst_tensor->data + index,
                (1 + layer->classes) * src_tensor->w * src_tensor->h, LOGISTIC);
        }
    }

    memset(dst_tensor->grad_data, 0,
           bcnn_tensor_size(dst_tensor) * sizeof(float));
    if (!layer->net_state) {  // state != train
        return;
    }
    // This part is for training
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(layer->cost) = 0;
    for (b = 0; b < dst_tensor->n; ++b) {
        for (j = 0; j < src_tensor->h; ++j) {
            for (i = 0; i < src_tensor->w; ++i) {
                for (n = 0; n < layer->num; ++n) {
                    int box_index =
                        entry_index(layer, dst_tensor, b,
                                    n * src_tensor->w * src_tensor->h +
                                        j * src_tensor->w + i,
                                    0);
                    yolo_box pred = get_yolo_box(
                        dst_tensor->data, layer->biases.data, layer->mask[n],
                        box_index, i, j, src_tensor->w, src_tensor->h,
                        net->input_width, net->input_height,
                        src_tensor->w * src_tensor->h);
                    float best_iou = 0;
                    int best_box = 0;
                    for (t = 0; t < layer->max_boxes; ++t) {
                        yolo_box truth =
                            float_to_box(label->data + t * (layer->coords + 1) +
                                             b * layer->truths,
                                         1);
                        if (!truth.x) {
                            break;
                        }
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_box = t;
                        }
                    }
                    int obj_index =
                        entry_index(layer, dst_tensor, b,
                                    n * src_tensor->w * src_tensor->h +
                                        j * src_tensor->w + i,
                                    layer->coords);
                    avg_anyobj += dst_tensor->data[obj_index];
                    dst_tensor->grad_data[obj_index] =
                        (0 - dst_tensor->data[obj_index]);
                    if (best_iou >
                        /*ignore_thresh*/ 0.7) {  // thresh set to default 0.7
                        dst_tensor->grad_data[obj_index] = 0;
                    }
                }
            }
        }
        for (t = 0; t < layer->max_boxes; ++t) {
            yolo_box truth = float_to_box(
                label->data + t * (layer->coords + 1) + b * layer->truths, 1);

            if (!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * src_tensor->w);
            j = (truth.y * src_tensor->h);
            yolo_box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            for (n = 0; n < layer->total; ++n) {
                yolo_box pred = {0};
                pred.w = layer->biases.data[2 * n] / net->input_width;
                pred.h = layer->biases.data[2 * n + 1] / net->input_height;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_n = n;
                }
            }
            int mask_n = -1;
            for (int k = 0; k < layer->num; ++k) {
                if (layer->mask[k] == best_n) {
                    mask_n = k;
                    break;
                }
            }
            if (mask_n >= 0) {
                int box_index =
                    entry_index(layer, dst_tensor, b,
                                mask_n * src_tensor->w * src_tensor->h +
                                    j * src_tensor->w + i,
                                0);
                float iou = delta_yolo_box(
                    truth, dst_tensor->data, layer->biases.data, best_n,
                    box_index, i, j, src_tensor->w, src_tensor->h,
                    net->input_width, net->input_height, dst_tensor->grad_data,
                    (2 - truth.w * truth.h), src_tensor->w * src_tensor->h);
                int obj_index =
                    entry_index(layer, dst_tensor, b,
                                mask_n * src_tensor->w * src_tensor->h +
                                    j * src_tensor->w + i,
                                layer->coords);
                avg_obj += dst_tensor->data[obj_index];
                dst_tensor->grad_data[obj_index] =
                    (1 - dst_tensor->data[obj_index]);

                int class = label->data[t * (layer->coords + 1) +
                                        b * layer->truths + layer->coords];
                int class_index =
                    entry_index(layer, dst_tensor, b,
                                mask_n * src_tensor->w * src_tensor->h +
                                    j * src_tensor->w + i,
                                layer->coords + 1);
                delta_yolo_class(dst_tensor->data, dst_tensor->grad_data,
                                 class_index, class, layer->classes,
                                 src_tensor->w * src_tensor->h, &avg_cat);
                ++count;
                ++class_count;
                if (iou > 0.5f) {
                    recall += 1;
                }
                if (iou > 0.75f) {
                    recall75 += 1;
                }
                avg_iou += iou;
            }
        }
    }
    *(layer->cost) =
        powf(sqrtf(bcnn_dot(bcnn_tensor_size(dst_tensor), dst_tensor->grad_data,
                            dst_tensor->grad_data)),
             2);
}

#ifdef BCNN_USE_CUDA
void bcnn_forward_yolo_layer_gpu(bcnn_net *net, bcnn_layer *layer,
                                 bcnn_tensor *src_tensor, bcnn_tensor *label,
                                 bcnn_tensor *dst_tensor) {
    int sz = bcnn_tensor_size(src_tensor);
    bcnn_cuda_memcpy_dev2host(src_tensor->data_gpu, src_tensor->data, sz);
    bcnn_forward_yolo_layer_cpu(net, layer, src_tensor, label, dst_tensor);
    if (layer->net_state) {  // state == train
        sz = bcnn_tensor_size(dst_tensor);
        bcnn_cuda_memcpy_host2dev(dst_tensor->grad_data_gpu,
                                  dst_tensor->grad_data, sz);
    }
    return;
}

void bcnn_backward_yolo_layer_gpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                  bcnn_tensor *dst_tensor) {
    bcnn_cuda_axpy(bcnn_tensor_size(src_tensor), 1, dst_tensor->grad_data_gpu,
                   1, src_tensor->grad_data_gpu, 1);
    return;
}
#endif

void bcnn_backward_yolo_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                  bcnn_tensor *dst_tensor) {
    bcnn_axpy(bcnn_tensor_size(src_tensor), 1, dst_tensor->grad_data,
              src_tensor->grad_data);
    return;
}

void bcnn_forward_yolo_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *label = &net->tensors[1];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_yolo_layer_gpu(net, node->layer, src, label, dst);
#else
    return bcnn_forward_yolo_layer_cpu(net, node->layer, src, label, dst);
#endif
}

void bcnn_backward_yolo_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_yolo_layer_gpu(node->layer, src, dst);
#else
    return bcnn_backward_yolo_layer_cpu(node->layer, src, dst);
#endif
}

static void correct_region_boxes(yolo_detection *dets, int n, int w, int h,
                                 int netw, int neth, int relative) {
    int i;
    int new_w = 0;
    int new_h = 0;
    if (((float)netw / w) < ((float)neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    } else {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i) {
        yolo_box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

// w = 640 h = 480 netw = 416 neth = 416
yolo_detection *bcnn_yolo_get_detections(bcnn_net *net, int w, int h, int netw,
                                         int neth, float thresh, int relative,
                                         int *num_dets) {
    int i, j, n, z;
    int count = 0;
    // First, compute the total number of detections
    for (int k = 0; k < net->num_nodes; ++k) {
        if (net->nodes[k].layer->type == YOLO) {
            bcnn_layer *layer = net->nodes[k].layer;
            bcnn_tensor *dst = &net->tensors[net->nodes[k].dst[0]];
            for (i = 0; i < dst->w * dst->h; ++i) {
                for (n = 0; n < layer->num; ++n) {
                    int obj_index = entry_index(
                        layer, dst, 0, n * dst->w * dst->h + i, layer->coords);
                    if (dst->data[obj_index] > thresh) {
                        ++count;
                    }
                }
            }
        }
    }
    yolo_detection *dets = NULL;
    if (count > 0) {
        dets = (yolo_detection *)calloc(count, sizeof(yolo_detection));
        bcnn_layer *layer = net->nodes[net->num_nodes - 1].layer;
        for (i = 0; i < count; ++i) {
            dets[i].prob = (float *)calloc(layer->classes, sizeof(float));
            if (layer->coords > 4) {
                dets[i].mask =
                    (float *)calloc(layer->coords - 4, sizeof(float));
            }
        }
    }

    // Fill the detected boxes
    count = 0;
    for (int k = 0; k < net->num_nodes; ++k) {
        if (net->nodes[k].layer->type == YOLO) {
            bcnn_layer *layer = net->nodes[k].layer;
            bcnn_tensor *dst = &net->tensors[net->nodes[k].dst[0]];
            for (i = 0; i < dst->w * dst->h; ++i) {
                int row = i / dst->w;
                int col = i % dst->w;
                for (n = 0; n < layer->num; ++n) {
                    int obj_index = entry_index(
                        layer, dst, 0, n * dst->w * dst->h + i, layer->coords);
                    float objectness = dst->data[obj_index];
                    if (objectness <= thresh) {
                        continue;
                    }
                    int box_index =
                        entry_index(layer, dst, 0, n * dst->w * dst->h + i, 0);
                    dets[count].bbox = get_yolo_box(
                        dst->data, layer->biases.data, layer->mask[n],
                        box_index, col, row, dst->w, dst->h, net->input_width,
                        net->input_height, dst->w * dst->h);
                    dets[count].objectness = objectness;
                    dets[count].classes = layer->classes;
                    for (j = 0; j < layer->classes; ++j) {
                        int class_index =
                            entry_index(layer, dst, 0, n * dst->w * dst->h + i,
                                        layer->coords + 1 + j);
                        float prob = objectness * dst->data[class_index];
                        dets[count].prob[j] = (prob > thresh) ? prob : 0;
                    }
                    ++count;
                }
            }
        }
    }
    *num_dets = count;
    correct_region_boxes(dets, count, w, h, netw, neth, relative);
    return dets;
}