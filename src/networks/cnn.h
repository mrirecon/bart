
#ifndef NN_CNN_H
#define NN_CNN_H

#include "misc/mri.h"
#include "misc/types.h"
#include "nn/layers.h"
#include "nn/activation.h"
#include "nn/nn.h"
#include "nn/nn_ops.h"
#include "nn/init.h"

struct network_s;

typedef nn_t (*network_create_t)(const struct network_s* config, unsigned int NO, const long odims[NO], unsigned int NI, const long idims[NI], enum NETWORK_STATUS status);

typedef struct network_s {

	TYPEID* TYPEID;

	network_create_t create;
	_Bool low_mem;

	enum norm norm;
	unsigned long norm_batch_flag;

	_Bool debug;
	_Bool residual;
	_Bool bart_to_channel_first;

	const char* prefix;

} network_t;

extern nn_t network_create(const struct network_s* config, unsigned int NO, const long odims[NO], unsigned int NI, const long idims[NI], enum NETWORK_STATUS status);

struct network_resnet_s {

	INTERFACE(network_t);

	unsigned int N;

	long kdims[DIMS];
	long dilations[DIMS];

	long Nl; // number of blocks

	long Nf; // number of filters
	long Kx; // filter size
	long Ky; // filter size
	long Kz; // filter size
	long Ng; // number groups

	unsigned long conv_flag;
	unsigned long channel_flag;
	unsigned long group_flag;
	unsigned long batch_flag;

	_Bool batch_norm;
	_Bool batch_norm_lf;
	_Bool bias;

	enum ACTIVATION activation;
	enum ACTIVATION last_activation;

	_Bool zero_init;
};
extern struct network_resnet_s network_resnet_default;

struct network_varnet_s {

	INTERFACE(network_t);

	long Kx;
	long Ky;
	long Kz;

	long Nf;
	long Nw;

	float Imax;
	float Imin;

	float init_scale_mu;
};

extern struct network_varnet_s network_varnet_default;

extern struct network_s network_mnist_default;

#endif
