#ifndef NN_ACTIVATIONS_H
#define NN_ACTIVATIONS_H

#include "nn/nn.h"
#include "nn/activation.h"

extern nn_t nn_append_activation(nn_t network, int o, const char* oname, enum ACTIVATION activation);
extern nn_t nn_append_activation_bias(nn_t network, int o, const char* oname, const char* bname, enum ACTIVATION activation, unsigned long bflag);

#endif