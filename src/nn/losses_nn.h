#ifndef NN_LOSSES_H
#define NN_LOSSES_H

#include "nn/nn.h"

extern nn_t nn_loss_mse_append(nn_t network, int o, const char* oname, unsigned long mean_dims);
extern nn_t nn_loss_cce_append(nn_t network, int o, const char* oname, unsigned long scaling_flag);
extern nn_t nn_loss_dice_append(nn_t network, int o, const char* oname, unsigned long label_flag, unsigned long mean_flag, float weighting_exponent, _Bool square_denominator);

#endif