
#include "bcnn/bcnn.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"

void bcnn_LiftedStructSimilaritySoftmax_loss_forward(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *label_node, bcnn_node *dst_node) 
{
    /*
        1. D^2 = x1_transpose + 1x_transpose - 2XX_transpose
        2. Construct pairwise label matrix
        3. Compute lose function J = 1/(2p) SUM( max(0, J_ij)^2)
            J_ji = log(SUM(exp{margin-D_ik}) + SUM(exp{margin - D_jjl})) + D_ij
        4. Compute gradients
            dJ_dD_{ij} = 1/p J_ij indicat
    */

    bcnn_tensor src = src_node->tensor; // bottom[0]
    bcnn_tensor dst = dst_node->tensor; // top[0]
    bcnn_tensor label = label_node->tensor; // bottom[1]

    // previous layer channel = num of Feature vector
    int channels = src.c;
    int input_size = src.w * src.h * src.c;
    int batch_size = src.n;
    int sz = src.n * input_size;

    const int M_ = batch_size;
    const int N_ = batch_size;
    const int K_ = channels;

#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_dev2host(src.data_gpu, src.data, sz);
    bcnn_cuda_memcpy_dev2host(src.grad_data_gpu, src.grad_data, sz);

    bcnn_cuda_memcpy_dev2host(dst.data_gpu, dst.data, sz);
    bcnn_cuda_memcpy_dev2host(dst.grad_data_gpu, dst.grad_data, sz);

    bcnn_cuda_memcpy_dev2host(label.data_gpu, label.data, sz);
#endif

    /*********************************************************************
        Step 1: Compute D^2 = x1_transpose + 1x_transpose - 2XX_transpose
    **********************************************************************/
    // Dist square = D^2
    float dist_sq[M_];
    memset(dist_sq, 0, M_*sizeof(float));
    for (int i = 0; i < M_; ++i)
    {
        dist_sq[i] = bcnn_dot(channels, src.data+(i*channels), src.data+(i*channels));
    }

    // dot =-2 XX_transpose
    float dot_[batch_size * batch_size];
    memset(dot_, 0, batch_size * batch_size*sizeof(float));

    bcnn_gemm(0, 1, M_, N_, K_, -2.0, src.data, K_, src.data, K_, 0, dot_, N_);

    
    // one array
    float one[batch_size];
    for (int i = 0; i < batch_size; ++i)
    {
        one[i] = 1.0f;
    }

    // dot_ = x1_transpose - 2XX_transpose
    for (int i = 0; i < batch_size; ++i)
    {
        bcnn_axpy(N_, dist_sq[i], one, dot_+i*batch_size);
    }

    // dot_ = x1_transpose + 1x_transpose - 2XX_transpose
    for (int i = 0; i < batch_size; ++i)
    {
        bcnn_axpy(batch_size, 1.0f, dist_sq, dot_+i*batch_size);
    }

    /*******************************************
        Step 2: Construct pairwise label matrix
    ********************************************/
    // array for indicating sample data are same class or not
    int label_mat[batch_size][batch_size];

    // each label is a One-Hot array
    int length = bcnn_tensor_get_size3d(&label);
    for (int i = 0; i < batch_size; ++i)
    {
        // find out which element in the One Hot label array
        // is 1 and the index is the label
        double label_i;
        for (int l = 0; l <length; ++l) {
            if (label.data[i * length + l] > 0.0f) {
                label_i = l;
                break;
            }
        }
        double label_j;
        for (int j = 0; j < batch_size; ++j)
        {
            for (int l = 0; l <length; ++l) {
                if (label.data[j * length + l] > 0.0f) {
                    label_j = l;
                    break;
                }
            }
            label_mat[i][j] = (int)(label_i == label_j);
        }
    }

    /*********************************
        Step 3: Compute lose function
    **********************************/
    float loss = 0;
    float margin = 1.0;
    layer->num_constraints = 0;
    float* bin = src.data;
    float* bout = src.grad_data;
    memset(bout, 0, sz); // initialize grad_data

    float blob_pos_diff[channels];
    memset(blob_pos_diff, 0, channels*sizeof(float));
    float blob_neg_diff[channels];
    memset(blob_neg_diff, 0, channels*sizeof(float));


    // dynamic array according to num_negatives
    float* loss_aug_inference = NULL;
    float* summer_vec = NULL;

    // Compute the loss of each sample to others
    for (int i = 0; i < batch_size; ++i)
    {
        for (int j = i+1; j < batch_size; ++j)
        {
            if (label_mat[i][j])
            {
                // dist_pos = D_ij
                float dist_pos = sqrt(dot_[i*batch_size + j]);
                bcnn_vsub(K_, bin+(i*K_), bin+(j*K_), blob_pos_diff);

                // 1. count the number of negatives sample
                int num_negatives = 0;
                for (int k = 0; k < N_; ++k)
                {
                    if (!label_mat[i][k])
                    {
                        num_negatives +=1;
                    }
                }

                for (int k = 0; k < N_; ++k)
                {
                    if (!label_mat[j][k])
                    {
                        num_negatives +=1;
                    }
                }

                free(loss_aug_inference);
                loss_aug_inference = (float *)calloc(num_negatives, sizeof(float));
                free(summer_vec);
                summer_vec = (float *)calloc(num_negatives, sizeof(float));
                
                for (int ss = 0; ss < num_negatives; ++ss)
                {
                    summer_vec[ss] = 1.0f;
                }

                int neg_idx = 0;
                // mine negative (anchor i, neg k)
                for (int k = 0; k < N_; ++k)
                {

                    if (!label_mat[i][k])
                    {
                        loss_aug_inference[neg_idx] = margin - sqrt(dot_[i*N_ + k]);
                        neg_idx ++;
                    }
                }

                // mine negative (anchor j, neg k)
                for (int k = 0; k < N_; ++k)
                {
                    if (!label_mat[j][k])
                    {
                        loss_aug_inference[neg_idx] = margin - sqrt(dot_[j*N_ + k]);
                        neg_idx++;
                    }
                }

                // compute softmax of loss aug inference vector
                // Get the max element in loss_aug_inference
                float max_elem = 0;
                for (int e = 0; e < num_negatives; ++e)
                {
                    if (loss_aug_inference[e] > max_elem)
                    {
                        max_elem = loss_aug_inference[e];
                    }
                }

                // loss_aug_inference[i] - max_elem
                bcnn_add_scalar(num_negatives, -max_elem, loss_aug_inference);

                // exp(loss_aug_inference[i])
                for (int f = 0; f < num_negatives; ++f)
                {
                    loss_aug_inference[f] = exp(loss_aug_inference[f]);
                }

                // log(SUM(loss_aug_inference))
                float sum_exp = bcnn_dot(num_negatives, summer_vec, loss_aug_inference);
                float soft_maximum = log(sum_exp) + max_elem;
                
                float this_loss = (soft_maximum + dist_pos) > 0 ? (soft_maximum + dist_pos) : 0;

                // squared hinge
                loss += this_loss * this_loss;
                layer->num_constraints += 1.0;

                /****************************
                    Step 4: Compute gradient
                *****************************/
                
                // Update from positive distance: d_J_dD_{ij}
                float scaler = 2.0f * this_loss / (dist_pos + 1e-10);
                // update x_i
                bcnn_axpy(K_, scaler, blob_pos_diff, bout + i*K_);
                // update x_j
                bcnn_axpy(K_, -scaler, blob_pos_diff, bout + j*K_);

                // Update from negative distance: dJ_dD_{ik}; update x_i, x_k
                neg_idx = 0;
                float dJ_dDik = 0;
                for (int k = 0; k < N_; ++k)
                {
                    if (!label_mat[i][k])
                    {
                        bcnn_vsub(K_, bin+(i*K_), bin+(k*K_), blob_neg_diff);

                        dJ_dDik = 2.0f * this_loss * -1.0f * loss_aug_inference[neg_idx] / sum_exp;
                        neg_idx++;

                        scaler = dJ_dDik / sqrt(dot_[i*N_ + k]);

                        // update x_i
                        bcnn_axpy(K_, scaler, blob_neg_diff, bout + i*K_);
                        // update x_k
                        bcnn_axpy(K_, -scaler, blob_neg_diff, bout + k*K_);
                    }
                }

                // Update from negative distance: dJ_dD_{jk}; update x_j, x_k
                float dJ_dDjk = 0;
                for (int k = 0; k < N_; ++k)
                {
                    if (!label_mat[j][k])
                    {
                        bcnn_vsub(K_, bin+(j*K_), bin+(k*K_), blob_neg_diff);

                        dJ_dDjk = 2.0f * this_loss * -1.0f * loss_aug_inference[neg_idx] / sum_exp;
                        neg_idx++;

                        scaler = dJ_dDjk / sqrt(dot_[j*N_ + k]);

                        // update x_i
                        bcnn_axpy(K_, scaler, blob_neg_diff, bout + j*K_);
                        // update x_k
                        bcnn_axpy(K_, -scaler, blob_neg_diff, bout + k*K_);
                    }
                }
            }
        }
    }
    loss = loss / layer->num_constraints;
    dst.data[0] = loss;

#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_host2dev(src.data_gpu, src.data, sz);
    bcnn_cuda_memcpy_host2dev(src.grad_data_gpu, src.grad_data, sz);
    // bcnn_cuda_memcpy_host2dev(dst.data_gpu, dst.data, sz);
    bcnn_cuda_memcpy_host2dev(dst.grad_data_gpu, dst.grad_data, sz);
#endif
}

void bcnn_LiftedStructSimilaritySoftmax_loss_backward(
    bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node) {
    bcnn_tensor src = src_node->tensor;
    int batch_size = src.n;
    int channels = src.c;
    float alpha = layer->scale / layer->num_constraints;

#ifdef BCNN_USE_CUDA
        float* bout = src.grad_data_gpu;
        for (int i = 0; i < batch_size; ++i)
        {
            bcnn_cuda_scal(channels, alpha, bout+(i*channels), 1);
        }
#else
        float *bout = src.grad_data;
        for (int i = 0; i < batch_size; ++i) {
            bcnn_scal(channels, alpha, (bout + (i * channels)) );
        }
    
#endif
}