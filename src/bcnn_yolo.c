

/** From yolo darknet */

int bcnn_add_yolo_layer(bcnn_net *net, int n, int classes, int coords,
                        char *src_id, char *dst_id) {
    // layer l = {0};
    bcnn_node node = {0};

    bh_check(net->num_nodes >= 1,
             "Yolo layer can't be the first layer of the network");
    int is_src_node_found = 0;
    for (i = net->num_tensors - 1; i >= 0; --i) {
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

    // l.type = REGION;

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
    // layer->biases.data = calloc(n * 2, sizeof(float));
    // l.bias_updates = calloc(n * 2, sizeof(float));
    // l.outputs = h * w * n * (classes + coords + 1);
    // l.inputs = l.outputs;
    node.layer->truths = 30 * (coords + 1);
// dst_tensor->grad_data = calloc(batch * l.outputs, sizeof(float));
// dst_tensor->data = calloc(batch * l.outputs, sizeof(float));
/*int i;
for (i = 0; i < n * 2; ++i) {
    layer->biases.data[i] = .5;
}*/

// l.forward = forward_region_layer;
// l.backward = backward_region_layer;
#ifdef GPU
// l.forward_gpu = forward_region_layer_gpu;
// l.backward_gpu = backward_region_layer_gpu;
// l.output_gpu = cuda_make_array(dst_tensor->data, batch * l.outputs);
// l.delta_gpu = cuda_make_array(dst_tensor->grad_data, batch * l.outputs);
#endif

    bh_log_info(
        "[Yolo] input_shape= %dx%dx%d num_classes= %d num_coords= %d "
        "output_shape= %dx%dx%d",
        net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
        net->tensors[node.src[0]].c, classes, coords,
        net->tensors[node.dst[0]].w, net->tensors[node.dst[0]].h,
        net->tensors[node.dst[0]].c);

    return 0;
}

typedef struct { float x, y, w, h; } box;

static box get_region_box(float *x, float *biases, int n, int index, int i,
                          int j, int w, int h, int stride) {
    box b;
    b.x = (i + x[index + 0 * stride]) / w;
    b.y = (j + x[index + 1 * stride]) / h;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

static float delta_region_box(box truth, float *x, float *biases, int n,
                              int index, int i, int j, int w, int h,
                              float *delta, float scale, int stride) {
    box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
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

int entry_index(bcnn_layer *layer, bcnn_tensor *dst_tensor, int location,
                int entry) {
    int n = location / (dst_tensor->w * dst_tensor->h);
    int loc = location % (dst_tensor->w * dst_tensor->h);
    return dst_tensor->n * bcnn_tensor_get_size3d(dst_tensor) +
           n * (dst_tensor->w * dst_tensor->h) * (l->coords + l->classes + 1) +
           entry * (dst_tensor->w * dst_tensor->h) + loc;
}

static box float_to_box(float *f, int stride) {
    box b = {0};
    b.x = f[0];
    b.y = f[1 * stride];
    b.w = f[2 * stride];
    b.h = f[3 * stride];
    return b;
}

int bcnn_forward_yolo_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                bcnn_tensor *dst_tensor) {
    int i, j, b, t, n;
    memcpy(dst_tensor->data, src_tensor->data,
           bcnn_tensor_get_size(dst_tensor) * sizeof(float));

    //#ifndef BCNN_USE_CUDA
    for (b = 0; b < dst_tensor->n; ++b) {
        for (n = 0; n < layer->num; ++n) {
            int index = entry_index(layer, dst_tensor,
                                    n * src_tensor->w * src_tensor->h, 0);
            bcnn_forward_activation_cpu(dst_tensor->data + index,
                                        2 * src_tensor->w * src_tensor->h,
                                        LOGISTIC);
            index =
                entry_index(layer, dst_tensor,
                            n * src_tensor->w * src_tensor->h, layer->coords);
            bcnn_forward_activation_cpu(dst_tensor->data + index,
                                        src_tensor->w * src_tensor->h,
                                        LOGISTIC);

            index = entry_index(layer, dst_tensor,
                                n * src_tensor->w * src_tensor->h,
                                layer->coords + 1);
            bcnn_forward_activation_cpu(
                dst_tensor->data + index,
                layer->classes * src_tensor->w * src_tensor->h, LOGISTIC);
        }
    }
    //#endif

    memset(dst_tensor->grad_data, 0,
           bcnn_tensor_get_size(dst_tensor) * sizeof(float));
    if (!layer->net_state) {  // state != train
        return 0;
    }
    // This part is for training
    bcnn_tensor *label = &net->tensors[1];
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
                    int box_index = entry_index(
                        layer, b, n * src_tensor->w * src_tensor->h +
                                      j * src_tensor->w + i,
                        0);
                    box pred = get_region_box(
                        dst_tensor->data, layer->biases.data, n, box_index, i,
                        j, src_tensor->w, src_tensor->h,
                        src_tensor->w * src_tensor->h);
                    float best_iou = 0;
                    for (t = 0; t < 30; ++t) {
                        box truth =
                            float_to_box(label->data + t * (layer->coords + 1) +
                                             b * layer->truths,
                                         1);
                        if (!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    int obj_index = entry_index(
                        layer, b, n * src_tensor->w * src_tensor->h +
                                      j * src_tensor->w + i,
                        layer->coords);
                    avg_anyobj += dst_tensor->data[obj_index];
                    dst_tensor->grad_data[obj_index] =
                        /*l.noobject_scale **/ (0 -
                                                dst_tensor->data[obj_index]);
                    if (best_iou > 0.5) {  // thresh set to default 0.5
                        dst_tensor->grad_data[obj_index] = 0;
                    }

                    if (*(net.seen) <
                        12800) {  // TODO make it a parameter for training
                        box truth = {0};
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
                }
            }
        }
        for (t = 0; t < 30; ++t) {
            box truth = float_to_box(
                label->data + t * (layer->coords + 1) + b * layer->truths, 1);

            if (!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * src_tensor->w);
            j = (truth.y * src_tensor->h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            for (n = 0; n < layer->num; ++n) {
                int box_index = entry_index(
                    layer, b,
                    n * src_tensor->w * src_tensor->h + j * src_tensor->w + i,
                    0);
                box pred = get_region_box(dst_tensor->data, layer->biases.data,
                                          n, box_index, i, j, src_tensor->w,
                                          src_tensor->h,
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
                layer, b,
                best_n * src_tensor->w * src_tensor->h + j * src_tensor->w + i,
                0);
            float iou = delta_region_box(
                truth, dst_tensor->data, layer->biases.data, best_n, box_index,
                i, j, src_tensor->w, src_tensor->h, dst_tensor->grad_data,
                (2 - truth.w * truth.h), src_tensor->w * src_tensor->h);
            if (layer->coords > 4) {
                int mask_index = entry_index(
                    layer, b, best_n * src_tensor->w * src_tensor->h +
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
                layer, b,
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
                layer, b,
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
    *(layer->cost) = pow(
        mag_array(dst_tensor->grad_data, bcnn_tensor_get_size(dst_tensor)), 2);
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
void bcnn_forward_yolo_layer_gpu(bcnn_net *net, bcnn_tensor *src_tensor,
                                 bcnn_tensor *dst_tensor) {
    return;
}

void bcnn_backward_yolo_layer_gpu(bcnn_net *net, bcnn_tensor *src_tensor,
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
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_yolo_layer_gpu(node->layer, src, dst);
#else
    return bcnn_forward_yolo_layer_cpu(node->layer, src, dst);
#endif
}

int bcnn_backward_yolo_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_yolo_layer_gpu(node->layer, src, dst);
#else
    return bcnn_backward_yolo_layer_cpu(node->layer, src, dst);
#endif
}