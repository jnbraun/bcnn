#include <bip/bip.h>

#include <bh/bh_macros.h>
#include <bh/bh_string.h>

typedef struct {
    float x_tl;
    float y_tl;
    float x_br;
    float y_br;
    float x;  // center
    float y;  // center
    float w;
    float h;
} box_t;

static float rand_between(float min, float max) {
    if (min > max) {
        return 0.f;
    }
    return ((float)rand() / RAND_MAX * (max - min)) + min;
}

static float overlap(float x1, float w1, float x2, float w2) {
    float left = x1 > x2 ? x1 : x2;
    float r1 = x1 + w1;
    float r2 = x2 + w2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static float box_intersection(box_t a, box_t b) {
    // fprintf(stderr, "overlap c %f %f %f %f\n", a.x_tl, a.w, b.x_tl, b.w);
    float w = overlap(a.x_tl, a.w, b.x_tl, b.w);
    float h = overlap(a.y_tl, a.h, b.y_tl, b.h);
    // fprintf(stderr, "overlap %f %f\n", w, h);
    if (w < 0 || h < 0) return 0;
    float area = w * h;
    return area;
}

static float box_union(box_t a, box_t b) {
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

static float box_iou(box_t a, box_t b) {
    return box_intersection(a, b) / box_union(a, b);
}

static void points2heatmap(float x, float y, float x_min, float x_max,
                           float y_min, float y_max, unsigned char *heatmap,
                           int width, int height, float sigma) {
    float mu_x = (x - x_min) / (x_max - x_min) * width;
    float mu_y = (y - y_min) / (y_max - y_min) * height;
    float norm = 1 /*/ (2 * BIP_PI * sigma * sigma)*/;
    float inv_sigma2 = 1 / (sigma * sigma);
    // fprintf(stderr, "norm %f mu_x %f mu_y %f\n", norm, mu_x, mu_y);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float val =
                norm * exp(-0.5f * (inv_sigma2 * (j - mu_x) * (j - mu_x) +
                                    inv_sigma2 * (i - mu_y) * (i - mu_y)));
            // fprintf(stderr, "(%d %d) %f\n", i, j, val);
            heatmap[i * width + j] =
                (unsigned char)bh_clamp(255.0f * val, 0, 255);
        }
    }
    return;
}

#if 0
int main(int argc, char **argv) {
    if (argc != 11) {
        fprintf(stderr,
                "Usage: %s <x> <y> <x_min> <x_max> <y_min> <y_max> <w_heatmap> "
                "<h_heatmap> <sigma> <out_path>\n",
                argv[0]);
        return -1;
    }
    float x = atof(argv[1]);
    float y = atof(argv[2]);
    float x_min = atof(argv[3]);
    float x_max = atof(argv[4]);
    float y_min = atof(argv[5]);
    float y_max = atof(argv[6]);
    int w = atoi(argv[7]);
    int h = atoi(argv[8]);
    float sigma = atof(argv[9]);
    unsigned char *heatmap =
        (unsigned char *)calloc(w * h, sizeof(unsigned char));
    points2heatmap(x, y, x_min, x_max, y_min, y_max, heatmap, w, h, sigma);
    bip_write_image(argv[10], heatmap, w, h, 1, w);
    bh_free(heatmap);
    return 0;
}
#endif
#if 0
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <annotations> <path_to_img>\n", argv[0]);
        return -1;
    }
    FILE *f_in = NULL;
    f_in = fopen(argv[1], "rt");
    if (f_in == NULL) {
        fprintf(stderr, "Could not open %s\n", argv[1]);
    }
    char *line = NULL;
    char **tok = NULL;
    int w = 0, h = 0, c = 0;
    int valid = 0;
    unsigned char *img = NULL;
    int num_boxes = 0;
    box_t *boxes = NULL;
    int cnt = 0;
    unsigned char *patch =
        (unsigned char *)calloc(12 * 12, sizeof(unsigned char));
    int total_neg = 0, total_pos = 0, total_part = 0;
    FILE *f_pos = fopen("pos.txt", "wt");
    FILE *f_part = fopen("part.txt", "wt");
    while ((line = bh_fgetline(f_in)) != 0) {
        int n_tok = bh_strsplit(line, '-', &tok);
        fprintf(stderr, "%s\n", line);
        if (n_tok == 3) {
            // Finish processing previous image
            if (img) {
                // generate negative
                fprintf(stderr, "Process negatives\n");
                int num_neg = 0;
                while (num_neg < 50) {
                    int sz = rand_between(12, bh_min(w, h) / 2);
                    int xtl = rand_between(0, w - sz);
                    int ytl = rand_between(0, h - sz);
                    box_t crop_box = {.x_tl = xtl,
                                      .y_tl = ytl,
                                      .x_br = xtl + sz,
                                      .y_br = ytl + sz,
                                      .w = sz,
                                      .h = sz};
                    // Check the IOU
                    float max_iou = 0.f;
                    for (int i = 0; i < num_boxes; ++i) {
                        float iou = box_iou(boxes[i], crop_box);
                        if (iou > max_iou) {
                            max_iou = iou;
                        }
                    }
                    if (max_iou < 0.3f) {
                        unsigned char *crop = (unsigned char *)calloc(
                            3 * sz * sz, sizeof(unsigned char));
                        unsigned char *gray = (unsigned char *)calloc(
                            sz * sz, sizeof(unsigned char));
                        bip_crop_image(img, w, h, w * c, (int)crop_box.x_tl,
                                       (int)crop_box.y_tl, crop, sz, sz, sz * c,
                                       c);
                        bip_rgb2gray(crop, sz, sz, sz * 3, gray, sz);
                        bip_resize_bilinear(gray, sz, sz, sz, patch, 12, 12, 12,
                                            1);
                        char outname[256];
                        sprintf(outname, "train12/neg/%d.png", total_neg);
                        bip_write_image(outname, patch, 12, 12, 1, 12);
                        fprintf(stderr, "Wrote negative patch %s\n", outname);
                        ++num_neg;
                        ++total_neg;
                        bh_free(crop);
                        bh_free(gray);
                    }
                }
                fprintf(stderr, "Process positives and partial\n");
                // Generate positive
                for (int i = 0; i < num_boxes; ++i) {
                    if (bh_max(boxes[i].w, boxes[i].h) < 50 ||
                        boxes[i].x_tl < 0 || boxes[i].y_tl < 0) {
                        fprintf(stderr, "Skipping box...\n");
                        continue;
                    }
                    int cnt = 0;
                    while (cnt < 20) {
                        int sz =
                            rand_between(0.8 * bh_min(boxes[i].w, boxes[i].h),
                                         1.25 * bh_max(boxes[i].w, boxes[i].h));
                        int delta_x =
                            rand_between(-boxes[i].w * 0.2, boxes[i].w * 0.2);
                        int delta_y =
                            rand_between(-boxes[i].h * 0.2, boxes[i].h * 0.2);
                        int nx1 = bh_max(
                            boxes[i].x_tl + boxes[i].w / 2 + delta_x - sz / 2,
                            0);
                        int ny1 = bh_max(
                            boxes[i].y_tl + boxes[i].h / 2 + delta_y - sz / 2,
                            0);
                        int nx2 = nx1 + sz;
                        int ny2 = ny1 + sz;
                        if (nx2 > w || ny2 > h) {
                            continue;
                        }
                        box_t crop_box = {.x_tl = nx1,
                                          .y_tl = ny1,
                                          .x_br = nx2,
                                          .y_br = ny2,
                                          .w = sz,
                                          .h = sz};
                        /*fprintf(stderr, "%f %f %f %f vs %f %f %f %f\n",
                                crop_box.x_tl, crop_box.y_tl, crop_box.x_br,
                                crop_box.y_br, boxes[i].x_tl, boxes[i].y_tl,
                                boxes[i].x_br, boxes[i].y_br);*/
                        float offset_x1 =
                            (float)(boxes[i].x_tl - nx1) / (float)sz;
                        float offset_y1 =
                            (float)(boxes[i].y_tl - ny1) / (float)sz;
                        float offset_x2 =
                            (float)(boxes[i].x_br - nx2) / (float)sz;
                        float offset_y2 =
                            (float)(boxes[i].y_br - ny2) / (float)sz;
                        // Check IOU
                        float iou = box_iou(crop_box, boxes[i]);
                        fprintf(stderr, "IOU %f\n", iou);
                        unsigned char *crop = (unsigned char *)calloc(
                            3 * sz * sz, sizeof(unsigned char));
                        unsigned char *gray = (unsigned char *)calloc(
                            sz * sz, sizeof(unsigned char));
                        bip_crop_image(img, w, h, w * c, (int)crop_box.x_tl,
                                       (int)crop_box.y_tl, crop, sz, sz, sz * c,
                                       c);
                        bip_rgb2gray(crop, sz, sz, sz * 3, gray, sz);
                        bip_resize_bilinear(gray, sz, sz, sz, patch, 12, 12, 12,
                                            1);
                        char outname[256];
                        if (iou > 0.7) {
                            sprintf(outname, "train12/pos/%d.png", total_pos);
                            ++total_pos;
                            fprintf(f_pos, "%s 1 %f %f %f %f\n", outname,
                                    offset_x1, offset_y1, offset_x2, offset_y2);
                            bip_write_image(outname, patch, 12, 12, 1, 12);
                            fprintf(stderr, "Wrote positive patch %s\n",
                                    outname);
                        } else if (iou > 0.45) {
                            sprintf(outname, "train12/part/%d.png", total_part);
                            ++total_part;
                            fprintf(f_part, "%s 1 %f %f %f %f\n", outname,
                                    offset_x1, offset_y1, offset_x2, offset_y2);
                            bip_write_image(outname, patch, 12, 12, 1, 12);
                            fprintf(stderr, "Wrote partial patch %s\n",
                                    outname);
                        }
                        ++cnt;
                        bh_free(crop);
                        bh_free(gray);
                    }
                }
                bh_free(img);
            }
            // Load next image
            char imgpath[256];
            sprintf(imgpath, "%s/%s", argv[2], line);
            if (bip_load_image(imgpath, &img, &w, &h, &c) != BIP_SUCCESS) {
                fprintf(stderr, "Could not open image %s\n", imgpath);
                valid = 0;
                bh_free(img);
                continue;
            } else {
                valid = 1;
            }
        } else if (valid) {
            fprintf(stderr, "valid...\n");
            char **labels = NULL;
            int n = bh_strsplit(line, ' ', &labels);
            if (n >= 4) {
                float x = (float)atoi(labels[0]);
                float y = (float)atoi(labels[1]);
                float wbox = (float)atoi(labels[2]);
                float hbox = (float)atoi(labels[3]);
                fprintf(stderr, "0 %f %f %f %f\n", x, y, wbox, hbox);
                boxes[cnt].x_tl = x;
                boxes[cnt].y_tl = y;
                boxes[cnt].x_br = x + wbox;
                boxes[cnt].y_br = y + hbox;
                boxes[cnt].w = wbox;
                boxes[cnt].h = hbox;
                cnt++;
            } else {
                bh_free(boxes);
                num_boxes = atoi(labels[0]);
                boxes = (box_t *)calloc(num_boxes, sizeof(box_t));
                cnt = 0;
            }
            for (int i = 0; i < n; ++i) {
                bh_free(labels[i]);
            }
            bh_free(labels);
        }
        for (int i = 0; i < n_tok; ++i) {
            bh_free(tok[i]);
        }
        bh_free(tok);
        bh_free(line);
    }
    bh_free(patch);
    fclose(f_in);
    fclose(f_pos);
    fclose(f_part);
    return 0;
}
#endif

#if 0
int main_widerface(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "Usage: %s <annotations> <path_to_img> <out_annot> <out_dir>\n",
                argv[0]);
        return -1;
    }

    FILE *f_in = NULL;
    f_in = fopen(argv[1], "rt");
    if (f_in == NULL) {
        fprintf(stderr, "Could not open %s\n", argv[1]);
    }
    char *line = NULL;
    char **tok = NULL;
    int w = 0, h = 0, c = 0;
    int valid_img = 0;
    int valid_boxes = 0;
    int first_box = 0;
    FILE *f_out = fopen(argv[3], "wt");
    box_t *boxes = NULL;
    int ibox = 0;
    int w_rz = 128;
    int h_rz = 128;
    char out_name[256], out_name2[256];
    unsigned char *img = NULL;
    unsigned char *rz_img =
        (unsigned char *)calloc(128 * 128, sizeof(unsigned char));
    unsigned char *crop640 =
        (unsigned char *)calloc(640 * 640, sizeof(unsigned char));
    while ((line = bh_fgetline(f_in)) != 0) {
        int n_tok = bh_strsplit(line, '-', &tok);
        fprintf(stderr, "%s\n", line);
        if (n_tok >= 3) {
            if (valid_img != 0) {  // Write previous img if any valid
                fprintf(stderr, "valid_img...\n");
                // Convert to gray
                unsigned char *gray = (unsigned char *)calloc(w * h, 1);
                if (c > 1) {
                    bip_rgb2gray(img, w, h, w * c, gray, w);
                } else {
                    memcpy(gray, img, w * h);
                }
                for (int k = 0; k < 2; ++k) {
                    // Random crop 640x640
                    int x_tl_crop = rand_between(0, w - 640);
                    int y_tl_crop = rand_between(0, h - 640);
                    fprintf(stderr, "x_crop %d y_crop %d\n", x_tl_crop,
                            y_tl_crop);
                    memset(crop640, 128, 640 * 640);
                    bip_crop_image(gray, w, h, w, x_tl_crop, y_tl_crop, crop640,
                                   640, 640, 640, 1);
                    // Resize to 128x128
                    sprintf(out_name2, "%s_%d.png", out_name, k);
                    bip_resize_bilinear(crop640, 640, 640, 640, rz_img, 128,
                                        128, 128, 1);
                    bip_write_image(out_name2, rz_img, 128, 128, 1, 128);
                    fprintf(f_out, "%s", out_name2);
                    for (int i = 0; i < ibox; ++i) {
                        if (boxes[i].x - x_tl_crop > 0 &&
                            boxes[i].x - x_tl_crop < 640 &&
                            boxes[i].y - y_tl_crop > 0 &&
                            boxes[i].y - y_tl_crop < 640) {
                            float x = (float)(boxes[i].x - x_tl_crop) / 640;
                            float y = (float)(boxes[i].y - y_tl_crop) / 640;
                            float wbox = boxes[i].w / 640;
                            float hbox = boxes[i].h / 640;
                            fprintf(f_out, " 0 %f %f %f %f", x, y, wbox, hbox);
                        }
                    }
                    fprintf(f_out, "\n");
                }
                bh_free(gray);
            }
            char imgpath[256];
            sprintf(imgpath, "%s/%s", argv[2], line);
            bh_free(img);
            if (bip_load_image(imgpath, &img, &w, &h, &c) != BIP_SUCCESS) {
                fprintf(stderr, "Could not open image %s\n", imgpath);
                valid_img = 0;
                bh_free(img);
                continue;
            } else {
                if (c == 3 || c == 1) {
                    valid_img = 1;
                    char **strid = NULL;
                    int np = bh_strsplit(line, '/', &strid);
                    sprintf(out_name, "%s/%s", argv[4], strid[np - 1]);
                    for (int i = 0; i < np; ++i) {
                        bh_free(strid[i]);
                    }
                    bh_free(strid);
                } else {
                    valid_img = 0;
                    continue;
                }
            }
        } else if (valid_img) {
            char **labels = NULL;
            int n = bh_strsplit(line, ' ', &labels);
            if (n >= 4) {
                if (bh_max((float)atoi(labels[2]), (float)atoi(labels[3])) <
                    90) {
                    valid_img = 0;
                    bh_free(boxes);
                    ibox = 0;
                } else {
                    boxes[ibox].h = (float)atoi(labels[3]);
                    boxes[ibox].w = (float)atoi(labels[2]);
                    boxes[ibox].x = (float)atoi(labels[0]) + boxes[ibox].w / 2;
                    boxes[ibox].y =
                        (float)atoi(labels[1]) /*-
                    ((float)atoi(labels[2]) - (float)atoi(labels[3])) /
                    2*/ + boxes[ibox].h / 2;
                    ibox++;
                    fprintf(stderr, "box %f %f %f %f\n", boxes[ibox - 1].x,
                            boxes[ibox - 1].y, boxes[ibox - 1].w,
                            boxes[ibox - 1].h);
                }
            } else if (n == 1) {  // number of boxes
                int num_boxes = atoi(labels[0]);
                bh_free(boxes);
                ibox = 0;
                boxes = (box_t *)calloc(num_boxes, sizeof(box_t));
            }
            for (int i = 0; i < n; ++i) {
                bh_free(labels[i]);
            }
            bh_free(labels);
        }
        for (int i = 0; i < n_tok; ++i) {
            bh_free(tok[i]);
        }
        bh_free(tok);
        bh_free(line);
    }
    bh_free(boxes);
    bh_free(crop640);
    bh_free(rz_img);
    fclose(f_out);
    fclose(f_in);
    return 0;
}
#endif

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "Usage: %s <annotations> <path_to_img> <out_annot> <out_dir>\n",
                argv[0]);
        return -1;
    }

    FILE *f_in = NULL;
    f_in = fopen(argv[1], "rt");
    if (f_in == NULL) {
        fprintf(stderr, "Could not open %s\n", argv[1]);
    }
    char *line = NULL;
    char **tok = NULL;
    int w = 0, h = 0, c = 0;
    int valid_img = 0;
    FILE *f_out = fopen(argv[3], "wt");
    char out_name[256], out_name2[256];
    unsigned char *img = NULL;
    unsigned char *rz_img =
        (unsigned char *)calloc(128 * 128, sizeof(unsigned char));
    unsigned char *canvas = NULL, *gray = NULL;
    while ((line = bh_fgetline(f_in)) != 0) {
        valid_img = 0;
        int n_tok = bh_strsplit(line, ' ', &tok);
        fprintf(stderr, "%s\n", line);
        if (atoi(tok[1]) == 1) {
            char imgpath[256];
            sprintf(imgpath, "%s/%s", argv[2], tok[0]);
            fprintf(stderr, "%s\n", imgpath);
            bh_free(img);
            if (bip_load_image(imgpath, &img, &w, &h, &c) != BIP_SUCCESS) {
                fprintf(stderr, "Could not open image %s\n", imgpath);
                valid_img = 0;
                bh_free(img);
            } else if (c == 1 || c == 3) {
                bh_free(gray);
                gray = (unsigned char *)calloc(w * h, 1);
                if (c > 1) {
                    bip_rgb2gray(img, w, h, w * c, gray, w);
                } else {
                    memcpy(gray, img, w * h);
                }
                valid_img = 1;
                char **strid = NULL;
                int np = bh_strsplit(tok[0], '/', &strid);
                sprintf(out_name, "%s/%s", argv[4], strid[np - 1]);
                for (int i = 0; i < np; ++i) {
                    bh_free(strid[i]);
                }
                bh_free(strid);
            } else {
                fprintf(stderr,
                        "Warning: %s has incorrect number of channels %d\n",
                        tok[0], c);

                valid_img = 0;
            }
        }
        if (valid_img != 0) {
            fprintf(stderr, "Loaded image %s\n", tok[0]);
            // Fill in square canvas
            int x_tl_crop, y_tl_crop;
            if (w > h) {
                x_tl_crop = 0;
                y_tl_crop = (h - w) / 2;
            } else {
                x_tl_crop = (w - h) / 2;
                y_tl_crop = 0;
            }
            bh_free(canvas);
            int sz_canvas = bh_max(w, h);
            canvas = (unsigned char *)calloc(sz_canvas * sz_canvas, 1);
            fprintf(stderr, "x_crop %d y_crop %d\n", x_tl_crop, y_tl_crop);
            memset(canvas, 128, sz_canvas * sz_canvas);
            bip_crop_image(gray, w, h, w, x_tl_crop, y_tl_crop, canvas,
                           sz_canvas, sz_canvas, sz_canvas, 1);
            // Resize to 128x128
            sprintf(out_name2, "%s.png", out_name);
            bip_resize_bilinear(canvas, sz_canvas, sz_canvas, sz_canvas, rz_img,
                                128, 128, 128, 1);
            bip_write_image(out_name2, rz_img, 128, 128, 1, 128);
            fprintf(f_out, "%s", out_name2);

            // TODO
            float box_h =
                ((float)atoi(tok[5]) - (float)atoi(tok[3])) / sz_canvas;
            float box_w =
                ((float)atoi(tok[4]) - (float)atoi(tok[2])) / sz_canvas;
            float box_x =
                ((float)atoi(tok[2]) - x_tl_crop) / sz_canvas + box_w / 2;
            float box_y =
                ((float)atoi(tok[3]) - y_tl_crop) / sz_canvas + box_h / 2;
            //
            fprintf(f_out, " 0 %f %f %f %f\n", box_x, box_y, box_w, box_h);
        }
        for (int i = 0; i < n_tok; ++i) {
            bh_free(tok[i]);
        }
        bh_free(tok);
        bh_free(line);
    }
    bh_free(gray);
    bh_free(img);
    bh_free(rz_img);
    fclose(f_out);
    fclose(f_in);
    return 0;
}