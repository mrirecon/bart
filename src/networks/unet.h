#include "networks/cnn.h"


enum UNET_DOWNSAMPLING_METHOD {
	UNET_DS_STRIDED_CONV,
	UNET_DS_FFT,
	//UNET_DS_MAX_POOL,
	//UNET_DS_AVERAGE_POOL
	};

enum UNET_UPSAMPLING_METHOD {
	UNET_US_STRIDED_CONV,
	UNET_US_FFT,
	//UNET_US_AVERAGE_POOL,
	};

enum UNET_COMBINE_METHOD {
	UNET_COMBINE_ADD,
	UNET_COMBINE_ATTENTION_SIGMOID,
};

struct network_unet_s {

	INTERFACE(network_t);

	unsigned int N;
	long kdims[DIMS];
	long dilations[DIMS];

	long Nf; // number of filters (top-level)
	long Kx; // filter size
	long Ky; // filter size
	long Kz; // filter size
	long Ng; // number groups

	unsigned long conv_flag;
	unsigned long channel_flag;
	unsigned long group_flag;
	unsigned long batch_flag;

	long N_level;

	float channel_factor; //number channels on lower level
	float reduce_factor; //reduce resolution of lower level

	long Nl_before; //number of layers per level
	long Nl_after; //number of layers per level
	long Nl_lowest; //number of layers per level

	_Bool real_constraint;

	_Bool init_real;		//initialize weights with real numbers
	_Bool init_zeros_residual;	//initialize weights such that output of each level is initialized with zeros

	_Bool use_bn;
	_Bool use_bias;

	enum ACTIVATION activation;
	enum ACTIVATION activation_output;	//output of unet

	enum PADDING padding;

	enum UNET_DOWNSAMPLING_METHOD ds_method;
	enum UNET_UPSAMPLING_METHOD us_method;
	enum UNET_COMBINE_METHOD combine_method;

	_Bool residual;

	_Bool adjoint;
};

extern struct network_unet_s network_unet_default_reco;
extern struct network_unet_s network_unet_default_segm;

extern nn_t network_unet_create(const struct network_s* config, unsigned int NO, const long odims[NO], unsigned int NI, const long idims[NI], enum NETWORK_STATUS status);
