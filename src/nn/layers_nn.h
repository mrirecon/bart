#ifndef NN_LAYERS_H
#define NN_LAYERS_H

#include "nn/layers.h"
#include "nn/nn.h"
#include "nn/init.h"

extern nn_t nn_append_convcorr_layer_generic(nn_t network, int o, const char* oname, const char* ker_name, unsigned long conv_flag, unsigned long channel_flag, unsigned long group_flag, unsigned int N, long const kernel_dims[__VLA(N)], const long strides[__VLA(N)], const long dilations[__VLA(N)], _Bool conv, enum PADDING conv_pad, const struct initializer_s* init);
extern nn_t nn_append_transposed_convcorr_layer_generic(nn_t network, int o, const char* oname, const char* ker_name, unsigned long conv_flag, unsigned long channel_flag, unsigned long group_flag, unsigned int N, long const kernel_dims[__VLA(N)], const long strides[__VLA(N)], const long dilations[__VLA(N)], _Bool conv, enum PADDING conv_pad, _Bool adjoint, const struct initializer_s* init);

extern nn_t nn_append_maxpool_layer_generic(nn_t network, int o, const char* oname, unsigned int N, const long pool_size[__VLA(N)], enum PADDING conv_pad);

extern nn_t nn_append_convcorr_layer(nn_t network, int o, const char* oname, const char* ker_name, int filters, long const kernel_size[3], _Bool conv, enum PADDING conv_pad, _Bool channel_first, const long strides[3], const long dilations[3], const struct initializer_s* init);
extern nn_t nn_append_transposed_convcorr_layer(nn_t network, int o, const char* oname, const char* ker_name, int channels, long const kernel_size[3], _Bool conv, _Bool adjoint, enum PADDING conv_pad, _Bool channel_first, const long strides[3], const long dilations[3], const struct initializer_s* init);
extern nn_t nn_append_dense_layer(nn_t network, int o, const char* oname, const char* weights_name, int out_neurons, const struct initializer_s* init);
extern nn_t nn_append_batchnorm_layer(nn_t network, int o, const char* oname, const char* stat_name, unsigned long norm_flags, enum NETWORK_STATUS status, const struct initializer_s* init);

extern nn_t nn_append_maxpool_layer(nn_t network, int o, const char* oname, const long pool_size[3], enum PADDING conv_pad, _Bool channel_first);
extern nn_t nn_append_blurpool_layer(nn_t network, int o, const char* oname, const long pool_size[3], enum PADDING conv_pad, _Bool channel_first);
extern nn_t nn_append_avgpool_layer(nn_t network, int o, const char* oname, const long pool_size[3], enum PADDING conv_pad, _Bool channel_first);
extern nn_t nn_append_upsampl_layer(nn_t network, int o, const char* oname, const long pool_size[3], _Bool channel_first);
extern nn_t nn_append_dropout_layer(nn_t network, int o, const char* oname, float p, enum NETWORK_STATUS status);
extern nn_t nn_append_flatten_layer(nn_t network, int o, const char* oname);
extern nn_t nn_append_padding_layer(nn_t network, int o, const char* oname, long N, long pad_for[__VLA(N)], long pad_after[__VLA(N)], enum PADDING pad_type);

#endif