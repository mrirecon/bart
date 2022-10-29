#ifndef LAYERS_H
#define LAYERS_H

#include "nlops/conv.h"
#include "misc/cppwrap.h"

enum NETWORK_STATUS {STAT_TRAIN, STAT_TEST};

extern const struct nlop_s* append_convcorr_layer_generic(const struct nlop_s* network, int o, unsigned long conv_flag, unsigned long channel_flag, unsigned long group_flag, int N, long const kernel_dims[__VLA(N)], const long strides[__VLA(N)], const long dilations[__VLA(N)], _Bool conv, enum PADDING conv_pad);
extern const struct nlop_s* append_transposed_convcorr_layer_generic(const struct nlop_s* network, int o, unsigned long conv_flag, unsigned long channel_flag, unsigned long group_flag, int N, long const kernel_dims[__VLA(N)], const long strides[__VLA(N)], const long dilations[__VLA(N)], _Bool conv, enum PADDING conv_pad, _Bool adjoint);

extern const struct nlop_s* append_maxpool_layer_generic(const struct nlop_s* network, int o, int N, const long pool_size[__VLA(N)], enum PADDING conv_pad);

extern const struct nlop_s* append_dense_layer(const struct nlop_s* network, int o, int out_neurons);

extern const struct nlop_s* append_convcorr_layer(const struct nlop_s* network, int o, int filters, const long kernel_size[3], _Bool conv, enum PADDING conv_pad, _Bool channel_first, const long strides[3], const long dilations[3]);
extern const struct nlop_s* append_transposed_convcorr_layer(const struct nlop_s* network, int o, int channels, long const kernel_size[3], _Bool conv, _Bool adjoint, enum PADDING conv_pad, _Bool channel_first, const long strides[3], const long dilations[3]);
extern const struct nlop_s* append_maxpool_layer(const struct nlop_s* network, int o, const long pool_size[3], enum PADDING conv_pad, _Bool channel_first);

extern const struct nlop_s* append_padding_layer(const struct nlop_s* network, int o, long N, long pad_for[__VLA(N)], long pad_after[__VLA(N)], enum PADDING pad_type);

extern const struct nlop_s* append_dropout_layer(const struct nlop_s* network, int o, float p, enum NETWORK_STATUS status);
extern const struct nlop_s* append_flatten_layer(const struct nlop_s* network, int o);

extern const struct nlop_s* append_batchnorm_layer(const struct nlop_s* network, int o, unsigned long norm_flags, enum NETWORK_STATUS status);

#endif
