/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>

#include "misc/opts.h"

#include "networks/unet.h"
#include "networks/cnn.h"
#include "networks/tf.h"

#include "network.h"
#include "nn/activation.h"

struct network_s* get_default_network(enum NETWORK_SELECT net)
{
	switch (net) {

		case NETWORK_NONE:
			return NULL;
		case NETWORK_MNIST:
			return &network_mnist_default;
		case NETWORK_RESBLOCK:
			return CAST_UP(&network_resnet_default);
		case NETWORK_VARNET:
			return CAST_UP(&network_varnet_default);
		case NETWORK_TENSORFLOW:
			return CAST_UP(&network_tensorflow_default);
	}

	assert(0);
}

struct opt_s res_block_opts[] = {

	OPTL_LONG('L', "layers", &(network_resnet_default.Nl), "d", "number of layers in residual block (default: 5)"),
	OPTL_LONG('F', "filters", &(network_resnet_default.Nf), "d", "number of filters in residual block (default: 32)"),

	OPTL_LONG('X', "filter-x", &(network_resnet_default.Kx), "d", "filter size in x-dimension (default: 3)"),
	OPTL_LONG('Y', "filter-y", &(network_resnet_default.Ky), "d", "filter size in y-dimension (default: 3)"),
	OPTL_LONG('Z', "filter-z", &(network_resnet_default.Kz), "d", "filter size in z-dimension (default: 1)"),

	OPTL_CLEAR(0, "no-batch-normalization", &(network_resnet_default.batch_norm), "do not use batch normalization"),
	OPTL_CLEAR(0, "no-bias", &(network_resnet_default.bias), "do not use bias"),

	OPTL_SELECT_DEF(0, "cardioid", enum ACTIVATION, &(network_resnet_default.activation), ACT_CARDIOID, ACT_RELU, "use cardioid as activation"),
};
const int N_res_block_opts = ARRAY_SIZE(res_block_opts);

struct opt_s variational_block_opts[] = {

	OPTL_LONG('W', "basis", &(network_varnet_default.Nw), "d", "number of basis functions (default: 31)"),
	OPTL_LONG('F', "filters", &(network_varnet_default.Nf), "d", "number of filters in residual block (default: 24)"),

	OPTL_LONG('X', "filter-x", &(network_varnet_default.Kx), "d", "filter size in x-dimension (default: 11)"),
	OPTL_LONG('Y', "filter-y", &(network_varnet_default.Ky), "d", "filter size in y-dimension (default: 11)"),
	OPTL_LONG('Z', "filter-z", &(network_varnet_default.Kz), "d", "filter size in z-dimension (default: 1)"),

};
const int N_variational_block_opts = ARRAY_SIZE(variational_block_opts);

struct opt_s unet_reco_opts[] = {

	OPTL_LONG('F', "filters", &(network_unet_default_reco.Nf), "int", "number of filters in first level (default: 32)"),
	OPTL_LONG('L', "levels", &(network_unet_default_reco.N_level), "int", "number of levels in U-Net (default: 4)"),

	OPTL_FLOAT('f', "filter-factor", &(network_unet_default_reco.channel_factor), "float", "factor to increase amount of filters in lower levels (default: 1.)"),
	OPTL_FLOAT('r', "resolution-factor", &(network_unet_default_reco.reduce_factor), "float", "factor to reduce spatial resolution in lower levels (default: 2.)"),

	OPTL_LONG('b', "layers-before", &(network_unet_default_reco.Nl_before), "int", "number of layers before down-sampling (default: 2)"),
	OPTL_LONG('a', "layers-after", &(network_unet_default_reco.Nl_after), "int", "number of layers after down-sampling (default: 2)"),
	OPTL_LONG('l', "layers-lowest", &(network_unet_default_reco.Nl_lowest), "int", "number of layers in lowest level (default: 4)"),

	OPTL_LONG('X', "filter-x", &(network_unet_default_reco.Kx), "int", "filter sze in x-dimension (default: 3)"),
	OPTL_LONG('Y', "filter-y", &(network_unet_default_reco.Ky), "int", "filter sze in y-dimension (default: 3)"),
	OPTL_LONG('Z', "filter-z", &(network_unet_default_reco.Kz), "int", "filter sze in z-dimension (default: 1)"),

	OPTL_SET(0, "init-real", &(network_unet_default_reco.init_real), "initialize weights with real values"),
	OPTL_SET(0, "init-zeros", &(network_unet_default_reco.init_zeros_residual), "initialize weights such that the output of each level is zero"),

	OPTL_SELECT_DEF(0, "ds-fft", enum UNET_DOWNSAMPLING_METHOD ,&(network_unet_default_reco.ds_method), UNET_DS_FFT, UNET_DS_STRIDED_CONV, "use high frequency cropping for down-sampling"),
	OPTL_SELECT_DEF(0, "us-fft", enum UNET_UPSAMPLING_METHOD ,&(network_unet_default_reco.us_method), UNET_US_FFT, UNET_US_STRIDED_CONV, "use high frequency zero-filling for up-sampling"),

	OPTL_SET(0, "batch-normalization", &(network_unet_default_reco.use_bn), "use batch normalization"),
	OPTL_CLEAR(0, "no-bias", &(network_unet_default_reco.use_bias), "do not use bias"),
};
const int N_unet_reco_opts = ARRAY_SIZE(unet_reco_opts);

struct opt_s unet_segm_opts[] = {

	OPTL_LONG('F', "filters", &(network_unet_default_segm.Nf), "int", "number of filters in first level (default: 64)"),
	OPTL_LONG('L', "levels", &(network_unet_default_segm.N_level), "int", "number of levels in U-Net (default: 6)"),

	OPTL_FLOAT('f', "filter-factor", &(network_unet_default_segm.channel_factor), "float", "factor to increase amount of filters in lower levels (default: 1.)"),
	OPTL_FLOAT('r', "resolution-factor", &(network_unet_default_segm.reduce_factor), "float", "factor to reduce spatial resolution in lower levels (default: 2.)"),

	OPTL_LONG('b', "layers-before", &(network_unet_default_segm.Nl_before), "int", "number of layers before down-sampling (default: 1)"),
	OPTL_LONG('a', "layers-after", &(network_unet_default_segm.Nl_after), "int", "number of layers after down-sampling (default: 1)"),
	OPTL_LONG('l', "layers-lowest", &(network_unet_default_segm.Nl_lowest), "int", "number of layers in lowest level (default: 2)"),

	OPTL_LONG('X', "filter-x", &(network_unet_default_segm.Kx), "int", "filter sze in x-dimension (default: 3)"),
	OPTL_LONG('Y', "filter-y", &(network_unet_default_segm.Ky), "int", "filter sze in y-dimension (default: 3)"),
	OPTL_LONG('Z', "filter-z", &(network_unet_default_segm.Kz), "int", "filter sze in z-dimension (default: 1)"),

	OPTL_CLEAR(0, "no-real-constraint", &(network_unet_default_segm.real_constraint), "allow complex numbers in network"),
	OPTL_SET(0, "init-real", &(network_unet_default_segm.init_real), "initialize weights with real values (if no real constraint)"),
	OPTL_SET(0, "init-zeros", &(network_unet_default_segm.init_zeros_residual), "initialize weights such that the output of each level is zero"),

	OPTL_SELECT_DEF(0, "ds-fft", enum UNET_DOWNSAMPLING_METHOD ,&(network_unet_default_segm.ds_method), UNET_DS_FFT, UNET_DS_STRIDED_CONV, "use high frequency cropping for down-sampling"),
	OPTL_SELECT_DEF(0, "us-fft", enum UNET_UPSAMPLING_METHOD ,&(network_unet_default_segm.us_method), UNET_US_FFT, UNET_US_STRIDED_CONV, "use high frequency zero-filling for up-sampling"),

	OPTL_SELECT_DEF(0, "combine-attention", enum UNET_COMBINE_METHOD ,&(network_unet_default_segm.combine_method), UNET_COMBINE_ATTENTION_SIGMOID, UNET_COMBINE_ADD, ""),

	OPTL_SET(0, "batch-normalization", &(network_unet_default_segm.use_bn), "use batch normalization"),
	OPTL_CLEAR(0, "no-bias", &(network_unet_default_segm.use_bias), "do not use bias"),
};
const int N_unet_segm_opts = ARRAY_SIZE(unet_segm_opts);

struct opt_s network_tensorflow_opts[] = {

	OPTL_STRING('p', "tf1-path", &(network_tensorflow_default.model_path), "path", "path to TensorFlow v1 graph"),
};
const int N_tensorflow_opts = ARRAY_SIZE(network_tensorflow_opts);
