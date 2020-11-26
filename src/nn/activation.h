#pragma once
enum ACTIVATION {ACT_LIN, ACT_RELU, ACT_SOFTMAX, ACT_SIGMOID};

extern const struct nlop_s* append_activation(const struct nlop_s* network, int o, enum ACTIVATION activation);
extern const struct nlop_s* append_activation_bias(const struct nlop_s* network, int o, enum ACTIVATION activation, unsigned long bflags);

extern const struct nlop_s* nlop_bias_create(unsigned int N, const long dims[__VLA(N)], const long bdims[__VLA(N)]);

extern const struct nlop_s* nlop_relu_create2(unsigned int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], const long istrs[__VLA(N)]);
extern const struct nlop_s* nlop_relu_create(unsigned int N, const long dims[__VLA(N)]);
extern const struct nlop_s* nlop_relu_bias_create(unsigned int N, const long dims[__VLA(N)], const long bdims[__VLA(N)]);

extern const struct nlop_s* nlop_softmax_create2(unsigned int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], const long istrs[__VLA(N)], unsigned int flag);
extern const struct nlop_s* nlop_softmax_create(unsigned int N, const long dims[__VLA(N)], unsigned int flag);
extern const struct nlop_s* nlop_softmax_bias_create(unsigned int N, const long dims[__VLA(N)], unsigned int batch_flag, const long bdims[__VLA(N)]);

extern const struct nlop_s* nlop_sigmoid_create2(unsigned int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], const long istrs[__VLA(N)]);
extern const struct nlop_s* nlop_sigmoid_create(unsigned int N, const long dims[__VLA(N)]);
extern const struct nlop_s* nlop_sigmoid_bias_create(unsigned int N, const long dims[__VLA(N)], const long bdims[__VLA(N)]);
