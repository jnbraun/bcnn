#include "bcnn_cost_layer.h"

#include <math.h>

#include "bcnn/bcnn.h"
#include "bcnn_mat.h"
#include "bcnn_tensor.h"
#include "bcnn_utils.h"

#include <bh/bh_mem.h>

#ifdef BCNN_USE_BLAS
#include "cblas.h"
#endif

void bcnn_lifted_struct_loss_forward(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *label = &net->tensors[1];
    bcnn_cost_param *param = (bcnn_cost_param *)node->param;
    /*
        1. D^2 = x1_transpose + 1x_transpose - 2XX_transpose
        2. Construct pairwise label matrix
        3. Compute lose function J = 1/(2p) SUM( max(0, J_ij)^2)
            J_ji = log(SUM(exp{margin-D_ik}) + SUM(exp{margin - D_jjl})) + D_ij
        4. Compute gradients
            dJ_dD_{ij} = 1/p J_ij indicat
    */

    // bottom[0]
    // top[0]
    // previous layer channel = num of Feature vector
    int channels = src_tensor->c;
    int input_size = src_tensor->w * src_tensor->h * src_tensor->c;
    int batch_size = src_tensor->n;
    int sz = src_tensor->n * input_size;

    const int M_ = batch_size;
    const int N_ = batch_size;
    const int K_ = channels;

#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_dev2host(src_tensor->data_gpu, src_tensor->data, sz);
    bcnn_cuda_memcpy_dev2host(src_tensor->grad_data_gpu, src_tensor->grad_data,
                              sz);

    bcnn_cuda_memcpy_dev2host(dst_tensor->data_gpu, dst_tensor->data, sz);
    bcnn_cuda_memcpy_dev2host(dst_tensor->grad_data_gpu, dst_tensor->grad_data,
                              sz);

    bcnn_cuda_memcpy_dev2host(label->data_gpu, label->data, sz);
#endif

    size_t align_offset = 32;
    /*********************************************************************
        Step 1: Compute D^2 = x1_transpose + 1x_transpose - 2XX_transpose
    **********************************************************************/
    // Dist square = D^2
    float *dist_sq = (float *)bh_align_calloc(M_ * sizeof(float), align_offset);
    for (int i = 0; i < M_; ++i) {
        dist_sq[i] = bcnn_dot(channels, src_tensor->data + (i * channels),
                              src_tensor->data + (i * channels));
    }

    // dot =-2 XX_transpose
    float *dot_ =
        (float *)bh_align_calloc(M_ * M_ * sizeof(float), align_offset);
#if BCNN_USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M_, N_, K_, -2.0,
                src_tensor->data, K_, src_tensor->data, K_, 0, dot_, N_);
#else
    bcnn_gemm(net->gemm_ctx, 0, 1, M_, N_, K_, -2.0, src_tensor->data, K_,
              src_tensor->data, K_, 0, dot_, N_, net->num_threads);
#endif

    // one array
    float *one =
        (float *)bh_align_malloc(batch_size * sizeof(float), align_offset);
    for (int i = 0; i < batch_size; ++i) {
        one[i] = 1.0f;
    }

    // dot_ = x1_transpose - 2XX_transpose
    for (int i = 0; i < batch_size; ++i) {
        bcnn_axpy(N_, dist_sq[i], one, dot_ + i * batch_size);
    }

    // dot_ = x1_transpose + 1x_transpose - 2XX_transpose
    for (int i = 0; i < batch_size; ++i) {
        bcnn_axpy(batch_size, 1.0f, dist_sq, dot_ + i * batch_size);
    }

    /*******************************************
        Step 2: Construct pairwise label matrix
    ********************************************/
    // array for indicating sample data are same class or not
    int *label_mat =
        (int *)bh_align_calloc(N_ * N_ * sizeof(int), align_offset);

    // each label is a One-Hot array
    int length = bcnn_tensor_size3d(label);
    for (int i = 0; i < batch_size; ++i) {
        // find out which element in the One Hot label array
        // is 1 and the index is the label
        double label_i = -1;
        for (int l = 0; l < length; ++l) {
            if (label->data[i * length + l] > 0.0f) {
                label_i = l;
                break;
            }
        }
        double label_j = -1;
        for (int j = 0; j < batch_size; ++j) {
            for (int l = 0; l < length; ++l) {
                if (label->data[j * length + l] > 0.0f) {
                    label_j = l;
                    break;
                }
            }
            // label_mat[i][j] = (int)(label_i == label_j);
            label_mat[i * N_ + j] = (int)(label_i == label_j);
        }
    }

    /*********************************
        Step 3: Compute lose function
    **********************************/
    float loss = 0;
    float margin = 1.0;
    param->num_constraints = 0;
    float *bin = src_tensor->data;
    float *bout = src_tensor->grad_data;
    memset(bout, 0, sz);  // initialize grad_data

    float *blob_pos_diff =
        (float *)bh_align_calloc(channels * sizeof(float), align_offset);
    float *blob_neg_diff =
        (float *)bh_align_calloc(channels * sizeof(float), align_offset);

    // dynamic array according to num_negatives
    float *loss_aug_inference = NULL;
    float *summer_vec = NULL;

    // Compute the loss of each sample to others
    for (int i = 0; i < batch_size; ++i) {
        for (int j = i + 1; j < batch_size; ++j) {
            if (label_mat[i * batch_size + j]) {
                // dist_pos = D_ij
                float dist_pos = sqrt(dot_[i * batch_size + j]);
                bcnn_vsub(K_, bin + (i * K_), bin + (j * K_), blob_pos_diff);

                // 1. count the number of negatives sample
                int num_negatives = 0;
                for (int k = 0; k < N_; ++k) {
                    if (!label_mat[i * N_ + k]) {
                        num_negatives += 1;
                    }
                }

                for (int k = 0; k < N_; ++k) {
                    if (!label_mat[j * N_ + k]) {
                        num_negatives += 1;
                    }
                }

                loss_aug_inference = (float *)bh_align_calloc(
                    num_negatives * sizeof(float), align_offset);
                summer_vec = (float *)bh_align_calloc(
                    num_negatives * sizeof(float), align_offset);

                for (int ss = 0; ss < num_negatives; ++ss) {
                    summer_vec[ss] = 1.0f;
                }

                int neg_idx = 0;
                // mine negative (anchor i, neg k)
                for (int k = 0; k < N_; ++k) {
                    if (!label_mat[i * N_ + k]) {
                        loss_aug_inference[neg_idx] =
                            margin - sqrt(dot_[i * N_ + k]);
                        neg_idx++;
                    }
                }

                // mine negative (anchor j, neg k)
                for (int k = 0; k < N_; ++k) {
                    if (!label_mat[j * N_ + k]) {
                        loss_aug_inference[neg_idx] =
                            margin - sqrt(dot_[j * N_ + k]);
                        neg_idx++;
                    }
                }

                // compute softmax of loss aug inference vector
                // Get the max element in loss_aug_inference
                float max_elem = 0;
                for (int e = 0; e < num_negatives; ++e) {
                    if (loss_aug_inference[e] > max_elem) {
                        max_elem = loss_aug_inference[e];
                    }
                }

                // loss_aug_inference[i] - max_elem
                bcnn_add_scalar(num_negatives, -max_elem, loss_aug_inference);

                // exp(loss_aug_inference[i])
                for (int f = 0; f < num_negatives; ++f) {
                    loss_aug_inference[f] = exp(loss_aug_inference[f]);
                }

                // log(SUM(loss_aug_inference))
                float sum_exp =
                    bcnn_dot(num_negatives, summer_vec, loss_aug_inference);
                float soft_maximum = log(sum_exp) + max_elem;

                float this_loss = (soft_maximum + dist_pos) > 0
                                      ? (soft_maximum + dist_pos)
                                      : 0;

                // squared hinge
                loss += this_loss * this_loss;
                param->num_constraints += 1.0;

                /****************************
                    Step 4: Compute gradient
                *****************************/

                // Update from positive distance: d_J_dD_{ij}
                float scaler = 2.0f * this_loss / (dist_pos + 1e-10);
                // update x_i
                bcnn_axpy(K_, scaler, blob_pos_diff, bout + i * K_);
                // update x_j
                bcnn_axpy(K_, -scaler, blob_pos_diff, bout + j * K_);

                // Update from negative distance: dJ_dD_{ik}; update x_i, x_k
                neg_idx = 0;
                float dJ_dDik = 0;
                for (int k = 0; k < N_; ++k) {
                    if (!label_mat[i * N_ + k]) {
                        bcnn_vsub(K_, bin + (i * K_), bin + (k * K_),
                                  blob_neg_diff);

                        dJ_dDik = 2.0f * this_loss * -1.0f *
                                  loss_aug_inference[neg_idx] / sum_exp;
                        neg_idx++;

                        scaler = dJ_dDik / sqrt(dot_[i * N_ + k]);

                        // update x_i
                        bcnn_axpy(K_, scaler, blob_neg_diff, bout + i * K_);
                        // update x_k
                        bcnn_axpy(K_, -scaler, blob_neg_diff, bout + k * K_);
                    }
                }

                // Update from negative distance: dJ_dD_{jk}; update x_j, x_k
                float dJ_dDjk = 0;
                for (int k = 0; k < N_; ++k) {
                    if (!label_mat[j * N_ + k]) {
                        bcnn_vsub(K_, bin + (j * K_), bin + (k * K_),
                                  blob_neg_diff);

                        dJ_dDjk = 2.0f * this_loss * -1.0f *
                                  loss_aug_inference[neg_idx] / sum_exp;
                        neg_idx++;

                        scaler = dJ_dDjk / sqrt(dot_[j * N_ + k]);

                        // update x_i
                        bcnn_axpy(K_, scaler, blob_neg_diff, bout + j * K_);
                        // update x_k
                        bcnn_axpy(K_, -scaler, blob_neg_diff, bout + k * K_);
                    }
                }
                bh_align_free(loss_aug_inference);
                bh_align_free(summer_vec);
            }
        }
    }
    loss = loss / param->num_constraints;
    dst_tensor->data[0] = loss;

    bh_align_free(dist_sq);
    bh_align_free(dot_);
    bh_align_free(one);
    bh_align_free(label_mat);
    bh_align_free(blob_pos_diff);
    bh_align_free(blob_neg_diff);

#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_host2dev(src_tensor->data_gpu, src_tensor->data, sz);
    bcnn_cuda_memcpy_host2dev(src_tensor->grad_data_gpu, src_tensor->grad_data,
                              sz);
    // bcnn_cuda_memcpy_host2dev(dst_tensor->data_gpu, dst_tensor->data, sz);
    bcnn_cuda_memcpy_host2dev(dst_tensor->grad_data_gpu, dst_tensor->grad_data,
                              sz);
#endif
}

void bcnn_lifted_struct_loss_backward(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_cost_param *param = (bcnn_cost_param *)node->param;
    int batch_size = src_tensor->n;
    int channels = src_tensor->c;
    float alpha = param->scale / param->num_constraints;

#ifdef BCNN_USE_CUDA
    float *bout = src_tensor->grad_data_gpu;
    for (int i = 0; i < batch_size; ++i) {
        bcnn_cuda_scal(channels, alpha, bout + (i * channels), 1);
    }
#else
    float *bout = src_tensor->grad_data;
    for (int i = 0; i < batch_size; ++i) {
        bcnn_scal(channels, alpha, (bout + (i * channels)));
    }

#endif
}