#ifndef BCNN_CL_H
#define BCNN_CL_H

#include "bcnn/bcnn.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Command-line tool parameters.
 */
typedef struct {
    int save_model;  /* Periodicity of model saving. */
    int num_pred;    /* Number of samples to be predicted in test file. */
    int eval_period; /* Periodicity of evaluating the train/test error. */
    int eval_test;   /* Set to 1 if evaluation of test database is asked. */
    bcnn_loader_type data_format;
    char *train_input;      /* Path to train file. */
    char *test_input;       /* Path to test/validation file. */
    char *path_train_label; /* Path to label train file (used for mnist format
                           only). */
    char *path_test_label;  /* Path to label test file (used for mnist format
                        only). */
    char *input_model;      /* Path to input model. */
    char *output_model;     /* Path to output model. */
    char *pred_out;         /* Path to output prediction file. */
} bcnn_cl_param;

bcnn_status bcnn_cl_load_param(const char *config_path, bcnn_cl_param *param);
bcnn_status bcnn_cl_train(bcnn_net *net, bcnn_cl_param *param, float *error);
bcnn_status bcnn_cl_predict(bcnn_net *net, bcnn_cl_param *param, float *error);
void bcnn_cl_free_param(bcnn_cl_param *param);

#ifdef __cplusplus
}
#endif

#endif /*BCNN_CL_H */