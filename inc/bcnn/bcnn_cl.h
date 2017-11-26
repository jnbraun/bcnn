#ifndef BCNN_CL_H
#define BCNN_CL_H


#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>


/**
 * \brief Structure for general parameters.
 */
typedef struct {
    char                        *train_input;       /**< Path to train file. */
    char                        *test_input;        /**< Path to test/validation file. */
    char                        *path_train_label;  /**< Path to label train file (used for mnist format only). */
    char                        *path_test_label;   /**< Path to label test file (used for mnist format only). */
    char                        *input_model;       /**< Path to input model. */
    char                        *output_model;      /**< Path to output model. */
    char                        *pred_out;          /**< Path to output prediction file. */
    bcnn_task                   task;               /**< Task to process. */
    bcnn_target                 prediction_type;    /**< Type of prediction to make. */
    char                        *data_format;       /**< Data format. */
    int                         save_model;         /**< Periodicity of model saving. */
    int                         nb_pred;            /**< Number of samples to be predicted in test file. */
    int                         eval_period;        /**< Periodicity of evaluating the train/test error. */
    int                         eval_test;          /**< Set to 1 if evaluation of test database is asked. */
} bcnncl_param;


int bcnncl_init_from_config(bcnn_net *handle_net, char *config_file, bcnncl_param *param);

int bcnncl_train(bcnn_net *handle_net, bcnncl_param *param, float *error);

int bcnncl_predict(bcnn_net *handle_net, bcnncl_param *param, float *error, int dump_pred);

int bcnncl_free_param(bcnncl_param *param);

#ifdef __cplusplus
}
#endif

#endif /*BCNN_CL_H */