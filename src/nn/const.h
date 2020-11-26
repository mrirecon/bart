
#ifndef NN_CONST_H
#define NN_CONST_H

#include "nn/nn.h"

extern nn_t nn_set_input_const_F(nn_t op, int i, const char* iname, int N, const long dims[N], _Bool copy, const _Complex float* in);
extern nn_t nn_set_input_const_F2(nn_t op, int i, const char* iname, int N, const long dims[N], const long strs[N], _Bool copy, const _Complex float* in);
extern nn_t nn_del_out_F(nn_t op, int o, const char* oname);
extern nn_t nn_del_out_bn_F(nn_t op);
extern nn_t nn_ignore_input_F(nn_t op, int i, const char* iname, int N, const long dims[N], _Bool copy, const _Complex float* in);

#endif