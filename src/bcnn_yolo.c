#include "bcnn_yolo.h"

#include "bcnn_activation_layer.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"
#include "bh_log.h"
/** From yolo darknet */

int bcnn_add_yolo_layer(bcnn_net *net, int n, int classes, int coords,
                        float *anchors, char *src_id, char *dst_id) {
    // layer l = {0};
    bcnn_node node = {0};

    bh_check(net->num_nodes >= 1,
             "Yolo layer can't be the first layer of the network");
    int is_src_node_found = 0;
    for (int i = net->num_tensors - 1; i >= 0; --i) {
        if (strcmp(net->tensors[i].name, src_id) == 0) {
            bcnn_node_add_input(&node, i);
            is_src_node_found = 1;
            break;
        }
    }
    bh_check(is_src_node_found, "Yolo layer: invalid input node name %s",
             src_id);

    node.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    node.layer->type = YOLO;
    node.layer->num = n;  // num bboxes per cell
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
    bcnn_node_add_output(&node, net->num_tensors - 1);

    node.layer->classes = classes;
    node.layer->coords = coords;
    node.layer->cost = (float *)calloc(1, sizeof(float));
    // Setup layer biases
    char biases_name[256];
    sprintf(biases_name, "%s_b", src_id);
    bcnn_tensor_create(&node.layer->biases, 1, 1, 1, n * 2, 1, biases_name);
    bcnn_tensor_filler w_filler = {.value = 0.5f, .type = FIXED};
    bcnn_tensor_fill(&node.layer->biases, w_filler);
    if (anchors != NULL) {
        memcpy(node.layer->biases.data, anchors, n * 2 * sizeof(float));
    }
    // layer->biases.data = calloc(n * 2, sizeof(float));
    node.layer->truths = 30 * (coords + 1);
    // dst_tensor->grad_data = calloc(batch * l.outputs, sizeof(float));
    // dst_tensor->data = calloc(batch * l.outputs, sizeof(float));

    // Add connection to net
    bcnn_net_add_node(net, node);
    bh_log_info(
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

static yolo_box get_region_box(float *x, float *biases, int n, int index, int i,
                               int j, int w, int h, int stride) {
    yolo_box b;
    b.x = (i + x[index + 0 * stride]) / w;
    b.y = (j + x[index + 1 * stride]) / h;
    b.w = expf(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = expf(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    /*fprintf(stderr, "w %f expw %f lw %d bw %f\n", x[index + 2 * stride],
            biases[2 * n], w, b.w);
    fprintf(stderr, "h %f exph %f lh %d bh %f\n", x[index + 3 * stride],
            biases[2 * n + 1], h, b.h);*/
    return b;
}

static float delta_region_box(yolo_box truth, float *x, float *biases, int n,
                              int index, int i, int j, int w, int h,
                              float *delta, float scale, int stride) {
    yolo_box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x * w - i);
    float ty = (truth.y * h - j);
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

void delta_region_class(float *output, float *delta, int index, int class,
                        int classes, float scale, int stride, float *avg_cat,
                        int tag) {
    int i, n;
    if (delta[index] && tag) {
        delta[index + stride * class] =
            scale * (1 - output[index + stride * class]);
        return;
    }
    for (n = 0; n < classes; ++n) {
        delta[index + stride * n] =
            scale * (((n == class) ? 1 : 0) - output[index + stride * n]);
        if (n == class) *avg_cat += output[index + stride * n];
    }
}

static float logit(float x) { return log(x / (1. - x)); }

static float tisnan(float x) { return (x != x); }

int entry_index(bcnn_layer *layer, bcnn_tensor *dst_tensor, int batch,
                int location, int entry) {
    int n = location / (dst_tensor->w * dst_tensor->h);
    int loc = location % (dst_tensor->w * dst_tensor->h);
    return batch * bcnn_tensor_get_size3d(dst_tensor) +
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

static void softmax(float *input, int n, float temp, int stride,
                    float *output) {
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for (i = 0; i < n; ++i) {
        if (input[i * stride] > largest) largest = input[i * stride];
    }
    for (i = 0; i < n; ++i) {
        float e = exp(input[i * stride] / temp - largest / temp);
        sum += e;
        output[i * stride] = e;
    }
    for (i = 0; i < n; ++i) {
        output[i * stride] /= sum;
    }
}

static void softmax_cpu(float *input, int n, int batch, int batch_offset,
                        int groups, int group_offset, int stride, float temp,
                        float *output) {
    int g, b;
    for (b = 0; b < batch; ++b) {
        for (g = 0; g < groups; ++g) {
            softmax(input + b * batch_offset + g * group_offset, n, temp,
                    stride, output + b * batch_offset + g * group_offset);
        }
    }
}

void bcnn_forward_yolo_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                 bcnn_tensor *label, bcnn_tensor *dst_tensor) {
    int i, j, b, t, n;
    memcpy(dst_tensor->data, src_tensor->data,
           bcnn_tensor_size(dst_tensor) * sizeof(float));
    FILE *flog = fopen("yolo_in.txt", "wt");
    for (i = 0; i < dst_tensor->w * dst_tensor->h * dst_tensor->c; ++i) {
        fprintf(flog, "%d %f\n", i, dst_tensor->data[i]);
    }
    fclose(flog);
    //#ifndef BCNN_USE_CUDA
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
            bcnn_forward_activation_cpu(dst_tensor->data + index,
                                        src_tensor->w * src_tensor->h,
                                        LOGISTIC);
            index = entry_index(layer, dst_tensor, 0, 0, layer->coords + 1);
            softmax_cpu(src_tensor->data + index, layer->classes,
                        src_tensor->n * layer->num,
                        bcnn_tensor_size(src_tensor) / layer->num,
                        src_tensor->w * src_tensor->h, 1,
                        src_tensor->w * src_tensor->h, 1,
                        dst_tensor->data + index);
        }
    }
    //#endif

    memset(dst_tensor->grad_data, 0,
           bcnn_tensor_size(dst_tensor) * sizeof(float));
    if (!layer->net_state) {  // state != train
        return;
    }
    // This part is for training
    float avg_iou = 0;
    float recall = 0;
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
                    yolo_box pred = get_region_box(
                        dst_tensor->data, layer->biases.data, n, box_index, i,
                        j, src_tensor->w, src_tensor->h,
                        src_tensor->w * src_tensor->h);
                    float best_iou = 0;
                    for (t = 0; t < 30; ++t) {
                        yolo_box truth =
                            float_to_box(label->data + t * (layer->coords + 1) +
                                             b * layer->truths,
                                         1);
                        if (!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    int obj_index =
                        entry_index(layer, dst_tensor, b,
                                    n * src_tensor->w * src_tensor->h +
                                        j * src_tensor->w + i,
                                    layer->coords);
                    avg_anyobj += dst_tensor->data[obj_index];
                    dst_tensor->grad_data[obj_index] =
                        /*l.noobject_scale **/ (0 -
                                                dst_tensor->data[obj_index]);
                    if (best_iou > 0.5) {  // thresh set to default 0.5
                        dst_tensor->grad_data[obj_index] = 0;
                    }

// Comment this part for now
#if 0
                    if (*(net.seen) <
                        12800) {  // TODO make it a parameter for training
                        yolo_box truth = {0};
                        truth.x = (i + .5) / src_tensor->w;
                        truth.y = (j + .5) / src_tensor->h;
                        truth.w = layer->biases.data[2 * n] / src_tensor->w;
                        truth.h = layer->biases.data[2 * n + 1] / src_tensor->h;
                        delta_region_box(truth, dst_tensor->data,
                                         layer->biases.data, n, box_index, i, j,
                                         src_tensor->w, src_tensor->h,
                                         dst_tensor->grad_data, .01,
                                         src_tensor->w * src_tensor->h);
                    }
#endif
                }
            }
        }
        for (t = 0; t < 30; ++t) {
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
            for (n = 0; n < layer->num; ++n) {
                int box_index = entry_index(
                    layer, dst_tensor, b,
                    n * src_tensor->w * src_tensor->h + j * src_tensor->w + i,
                    0);
                yolo_box pred = get_region_box(
                    dst_tensor->data, layer->biases.data, n, box_index, i, j,
                    src_tensor->w, src_tensor->h,
                    src_tensor->w * src_tensor->h);
                // if (l.bias_match) { // bias match set to 1?
                pred.w = layer->biases.data[2 * n] / src_tensor->w;
                pred.h = layer->biases.data[2 * n + 1] / src_tensor->h;
                //}
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_n = n;
                }
            }

            int box_index = entry_index(
                layer, dst_tensor, b,
                best_n * src_tensor->w * src_tensor->h + j * src_tensor->w + i,
                0);
            float iou = delta_region_box(
                truth, dst_tensor->data, layer->biases.data, best_n, box_index,
                i, j, src_tensor->w, src_tensor->h, dst_tensor->grad_data,
                (2 - truth.w * truth.h), src_tensor->w * src_tensor->h);
            if (layer->coords > 4) {
                int mask_index =
                    entry_index(layer, dst_tensor, b,
                                best_n * src_tensor->w * src_tensor->h +
                                    j * src_tensor->w + i,
                                4);
                delta_region_mask(label->data + t * (layer->coords + 1) +
                                      b * layer->truths + 5,
                                  dst_tensor->data, layer->coords - 4,
                                  mask_index, dst_tensor->grad_data,
                                  src_tensor->w * src_tensor->h,
                                  /*l.mask_scale*/ 1);
            }
            if (iou > .5) {
                recall += 1;
            }
            avg_iou += iou;

            int obj_index = entry_index(
                layer, dst_tensor, b,
                best_n * src_tensor->w * src_tensor->h + j * src_tensor->w + i,
                layer->coords);
            avg_obj += dst_tensor->data[obj_index];
            dst_tensor->grad_data[obj_index] =
                /*l.object_scale*/ 1 * (1 - dst_tensor->data[obj_index]);

            int class = label->data[t * (layer->coords + 1) +
                                    b * layer->truths + layer->coords];
            /*if (l.map) {
                class = l.map[class];
            }*/
            int class_index = entry_index(
                layer, dst_tensor, b,
                best_n * src_tensor->w * src_tensor->h + j * src_tensor->w + i,
                layer->coords + 1);
            delta_region_class(dst_tensor->data, dst_tensor->grad_data,
                               class_index, class, layer->classes,
                               /*l.class_scale*/ 1,
                               src_tensor->w * src_tensor->h, &avg_cat, 1);
            ++count;
            ++class_count;
        }
    }
    *(layer->cost) =
        powf(sqrtf(bcnn_dot(bcnn_tensor_size(dst_tensor),
                            dst_tensor->grad_data, dst_tensor->grad_data)),
             2);
    fprintf(
        stderr,
        "Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  "
        "count: %d\n",
        avg_iou / count, avg_cat / class_count, avg_obj / count,
        avg_anyobj /
            (src_tensor->w * src_tensor->h * layer->num * dst_tensor->n),
        recall / count, count);
}

#ifdef BCNN_USE_CUDA
void bcnn_forward_yolo_layer_gpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                 bcnn_tensor *label, bcnn_tensor *dst_tensor) {
    int sz = bcnn_tensor_size(src_tensor);
    bcnn_cuda_memcpy_dev2host(src_tensor->data_gpu, src_tensor->data, sz);
    bcnn_forward_yolo_layer_cpu(layer, src_tensor, label, dst_tensor);
    return;
}

void bcnn_backward_yolo_layer_gpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                  bcnn_tensor *dst_tensor) {
    return;
}
#endif

void bcnn_backward_yolo_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                  bcnn_tensor *dst_tensor) {
    return;
}

void bcnn_forward_yolo_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *label = &net->tensors[1];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_yolo_layer_gpu(node->layer, src, label, dst);
#else
    return bcnn_forward_yolo_layer_cpu(node->layer, src, label, dst);
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
    fprintf(stderr, "netw %d neth %d new_w %d new_h %d\n", netw, neth, new_w,
            new_h);
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
        // fprintf(stderr, "idet %d x %f y %f w %f h %f\n", i, b.x, b.y, b.w,
        // b.h);
    }
}

// w = 640 h = 480 netw = 416 neth = 416
void bcnn_yolo_get_detections(bcnn_net *net, bcnn_node *node, int w, int h,
                              int netw, int neth, float thresh, int relative,
                              yolo_detection *dets) {
    int i, j, n, z;
    bcnn_layer *layer = node->layer;
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
    float *predictions = dst->data;
    FILE *flog = fopen("lolo.txt", "wt");
    for (i = 0; i < dst->w * dst->h * dst->c; ++i) {
        fprintf(flog, "%d %f\n", i, dst->data[i]);
    }
    fclose(flog);
    float max_objectness = 0.0f;
    for (i = 0; i < dst->w * dst->h; ++i) {
        int row = i / dst->w;
        int col = i % dst->w;
        for (n = 0; n < layer->num; ++n) {
            int index = n * dst->w * dst->h + i;
            for (j = 0; j < layer->classes; ++j) {
                dets[index].prob[j] = 0;
            }
            int obj_index = entry_index(layer, dst, 0, n * dst->w * dst->h + i,
                                        layer->coords);
            // fprintf(stderr, "obj_index %d\n", obj_index);
            int box_index =
                entry_index(layer, dst, 0, n * dst->w * dst->h + i, 0);
            int mask_index =
                entry_index(layer, dst, 0, n * dst->w * dst->h + i, 4);
            float scale = predictions[obj_index];
            dets[index].bbox =
                get_region_box(predictions, layer->biases.data, n, box_index,
                               col, row, dst->w, dst->h, dst->w * dst->h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if (max_objectness < dets[index].objectness) {
                max_objectness = dets[index].objectness;
            }
            if (dets[index].mask) {
                for (j = 0; j < layer->coords - 4; ++j) {
                    dets[index].mask[j] =
                        dst->data[mask_index + j * dst->w * dst->h];
                }
            }

            int class_index = entry_index(
                layer, dst, 0, n * dst->w * dst->h + i, layer->coords + 1);
            if (dets[index].objectness) {
                for (j = 0; j < layer->classes; ++j) {
                    int class_index =
                        entry_index(layer, dst, 0, n * dst->w * dst->h + i,
                                    layer->coords + 1 + j);
                    float prob = scale * predictions[class_index];
                    dets[index].prob[j] = (prob > thresh) ? prob : 0;
                }
            }
        }
    }
    fprintf(stderr, "max_objectness %f\n", max_objectness);
    correct_region_boxes(dets, dst->w * dst->h * layer->num, w, h, netw, neth,
                         relative);
}