/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include "iter/italgos.h"
#include "nlops/nlop.h"

#include "nn/layers.h"
#include "nn/nn.h"

#include "layers_nn.h"

/**
 * Append convolution/correlation layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param oname
 * @param ker_name name for the kernel input
 * @param conv_flag spatial dimensions which are convolved over
 * @param channel_flag dimensions holding channels
 * @param group_flag group dimensions, i.e. batch dimension with non-shared kernel
 * @param kernel_dims kernel size (if conv_flag); out_channel(if channel_flag); idims (if group_flag); 1 else
 * @param strides of output (only for spatial dims)
 * @param dilations of kernel (only for spatial dims)
 * @param conv convolution if true, correlation else
 * @param conv_pad padding for the convolution
 */
nn_t nn_append_convcorr_layer_generic(
				nn_t network, int o, const char* oname, const char* ker_name,
				unsigned long conv_flag, unsigned long channel_flag, unsigned long group_flag,
				unsigned int N, long const kernel_dims[N], const long strides[N], const long dilations[N],
				bool conv, enum PADDING conv_pad, const struct initializer_s* init)
{
	o = nn_get_out_arg_index(network, o, oname);

	auto nlop = append_convcorr_layer_generic(nlop_clone(nn_get_nlop(network)), o, conv_flag, channel_flag, group_flag, N, kernel_dims, strides, dilations, conv, conv_pad);
	auto result = nn_from_nlop_F(nlop);

	nn_clone_args(result, network);

	result = nn_set_in_type_F(result, -1, NULL, IN_OPTIMIZE);

	unsigned long in_flag = in_flag_conv_generic(N, conv_flag, channel_flag, group_flag);

	result = nn_set_initializer_F(result, -1, NULL, (NULL != init) ? init : init_kaiming_create(in_flag, true, false, 0));

	if (NULL != ker_name)
		result = nn_set_input_name_F(result, -1, ker_name);

	nn_free(network);

	return result;
}


/**
 * Append transposed convolution/correlation layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param oname
 * @param ker_name name for the kernel input
 * @param conv_flag spatial dimensions which are convolved over
 * @param channel_flag dimensions holding channels
 * @param group_flag group dimensions, i.e. batch dimension with non-shared kernel
 * @param kernel_dims kernel size (if conv_flag); out_channel(if channel_flag); idims (if group_flag); 1 else
 * @param strides of input (only for spatial dims)
 * @param dilations of kernel (only for spatial dims)
 * @param conv convolution if true, correlation else
 * @param conv_pad padding for the convolution
 * @param adjoint adjoint convolution if true, transposed else
 */
nn_t nn_append_transposed_convcorr_layer_generic(
				nn_t network, int o, const char* oname, const char* ker_name,
				unsigned long conv_flag, unsigned long channel_flag, unsigned long group_flag,
				unsigned int N, long const kernel_dims[N], const long strides[N], const long dilations[N],
				bool conv, enum PADDING conv_pad, bool adjoint, const struct initializer_s* init)
{
	o = nn_get_out_arg_index(network, o, oname);
	auto nlop = append_transposed_convcorr_layer_generic(nlop_clone(nn_get_nlop(network)), o, conv_flag, channel_flag, group_flag, N, kernel_dims, strides, dilations, conv, conv_pad, adjoint);
	auto result = nn_from_nlop_F(nlop);
	nn_clone_args(result, network);

	result = nn_set_in_type_F(result, -1, NULL, IN_OPTIMIZE);

	unsigned long in_flag = out_flag_conv_generic(N, conv_flag, channel_flag, group_flag); //input of conv is output of transposed conv

	result = nn_set_initializer_F(result, -1, NULL, (NULL != init) ? init : init_kaiming_create(in_flag, true, false, 0));

	if (NULL != ker_name)
		result = nn_set_input_name_F(result, -1, ker_name);

	nn_free(network);

	return result;
}

/**
 * Append max-pooling layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param oname
 * @param N
 * @param pool_size size of pooling
 * @param conv_pad must be PAD_VALID/PAD_SAME if image size is not a multiple of padding size, the image is shrinked/expanded to a multiple
 */
nn_t nn_append_maxpool_layer_generic(nn_t network, int o, const char* oname, unsigned int N, const long pool_size[N], enum PADDING conv_pad)
{
	o = nn_get_out_arg_index(network, o, oname);
	auto result = nn_from_nlop_F(append_maxpool_layer_generic(nlop_clone(nn_get_nlop(network)), o, N, pool_size, conv_pad));
	nn_clone_args(result, network);
	nn_free(network);
	return result;
}

/**
 * Append convolution/correlation layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param oname
 * @param filters number of output channels
 * @param kernel_size {kx, ky, kz} size of convolution kernel
 * @param conv convolution if true, correlation else
 * @param conv_pad padding for the convolution
 * @param channel_first data layout is {c, x, y, z}/{filter, channel, kx, ky, kz} if true, {x, y, z, c} {kx, ky, kz, channel, filter} else
 * @param strides (not supported, must be NULL)
 * @param dilations (not supported, must be NULL)
 * @param initializer (NULL falls back to default)
 */
nn_t nn_append_convcorr_layer(nn_t network, int o, const char* oname, const char* ker_name, int filters, long const kernel_size[3], bool conv, enum PADDING conv_pad, bool channel_first, const long strides[3], const long dilations[3], const struct initializer_s* init)
{
	o = nn_get_out_arg_index(network, o, oname);

	auto nlop = append_convcorr_layer(nlop_clone(nn_get_nlop(network)), o, filters, kernel_size, conv, conv_pad, channel_first, strides, dilations);
	auto result = nn_from_nlop_F(nlop);

	nn_clone_args(result, network);

	result = nn_set_in_type_F(result, -1, NULL, IN_OPTIMIZE);
	result = nn_set_initializer_F(result, -1, NULL, (NULL != init) ? init : init_kaiming_create(in_flag_conv(channel_first), true, false, 0));

	if (NULL != ker_name)
		result = nn_set_input_name_F(result, -1, ker_name);

	nn_free(network);

	return result;
}

/**
 * Append transposed convolution/correlation layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param oname
 * @param filters number of output channels
 * @param kernel_size {kx, ky, kz} size of convolution kernel
 * @param conv convolution if true, correlation else
 * @param adjoint if true, the operator is a adjoint convolution, else it's a transposed one
 * @param conv_pad padding for the convolution
 * @param channel_first data layout is {c, x, y, z}/{filter, channel, kx, ky, kz} if true, {x, y, z, c} {kx, ky, kz, channel, filter} else
 * @param strides (not supported, must be NULL)
 * @param dilations (not supported, must be NULL)
 * @param initializer (NULL falls back to default)
 */
nn_t nn_append_transposed_convcorr_layer(nn_t network, int o, const char* oname, const char* ker_name, int channels, long const kernel_size[3], bool conv, bool adjoint, enum PADDING conv_pad, bool channel_first, const long strides[3], const long dilations[3], const struct initializer_s* init)
{
	o = nn_get_out_arg_index(network, o, oname);
	auto nlop = append_transposed_convcorr_layer(nlop_clone(nn_get_nlop(network)), o, channels, kernel_size, conv, conv_pad, adjoint, channel_first, strides, dilations);
	auto result = nn_from_nlop_F(nlop);
	nn_clone_args(result, network);

	result = nn_set_in_type_F(result, -1, NULL, IN_OPTIMIZE);
	result = nn_set_initializer_F(result, -1, NULL, (NULL != init) ? init : init_kaiming_create(in_flag_conv(channel_first), true, false, 0));

	if (NULL != ker_name)
		result = nn_set_input_name_F(result, -1, ker_name);

	nn_free(network);

	return result;
}


/**
 * Append dense layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param oname
 * @param neurons number of output neurons
 * @param initializer (NULL falls back to default)
 */
nn_t nn_append_dense_layer(nn_t network, int o, const char* oname, const char* weights_name, int out_neurons, const struct initializer_s* init)
{
	o = nn_get_out_arg_index(network, o, oname);
	auto nlop = append_dense_layer(nlop_clone(nn_get_nlop(network)), o, out_neurons);
	auto result = nn_from_nlop_F(nlop);
	nn_clone_args(result, network);

	result = nn_set_in_type_F(result, -1, NULL, IN_OPTIMIZE);
	result = nn_set_initializer_F(result, -1, NULL, (NULL != init) ? init : init_kaiming_create(MD_BIT(1), true, false, 0));

	if (NULL != weights_name)
		result = nn_set_input_name_F(result, -1, weights_name);

	nn_free(network);

	return result;
}


/**
 * Append batch normalization
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param oname
 * @param norm_flags select dimension over which we normalize
 * @param initializer (NULL falls back to default)
 */
nn_t nn_append_batchnorm_layer(nn_t network, int o, const char* oname, const char* stat_name, unsigned long norm_flags, enum NETWORK_STATUS status, const struct initializer_s* init)
{
	o = nn_get_out_arg_index(network, o, oname);

	auto nlop = append_batchnorm_layer(nlop_clone(nn_get_nlop(network)), o, norm_flags, status);
	auto result = nn_from_nlop_F(nlop);
	nn_clone_args(result, network);

	result = nn_set_in_type_F(result, -1, NULL, IN_BATCHNORM);
	result = nn_set_dup_F(result, -1, NULL, false);
	result = nn_set_out_type_F(result, -1, NULL, OUT_BATCHNORM);
	result = nn_set_initializer_F(result, -1, NULL, (NULL != init) ? init : init_const_create(0.));

	if (NULL != stat_name) {

		result = nn_set_input_name_F(result, -1, stat_name);
		result = nn_set_output_name_F(result, -1, stat_name);
	}

	nn_free(network);

	return result;
}

/**
 * Append max-pooling layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param oname
 * @param pool_size {px, py, pz} size of pooling
 * @param conv_pad must be PAD_VALID/PAD_SAME if image size is not a multiple of padding size, the image is shrinked/expanded to a multiple
 * @param channel_first data layout is {c, x, y, z} if true, {x, y, z, c} else
 */
nn_t nn_append_maxpool_layer(nn_t network, int o, const char* oname, const long pool_size[3], enum PADDING conv_pad, bool channel_first)
{
	o = nn_get_out_arg_index(network, o, oname);
	auto result = nn_from_nlop_F(append_maxpool_layer(nlop_clone(nn_get_nlop(network)), o, pool_size, conv_pad, channel_first));
	nn_clone_args(result, network);
	nn_free(network);
	return result;
}

/**
 * Append dropout layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param oname
 * @param p procentage of outputs dropt out
 */
nn_t nn_append_dropout_layer(nn_t network, int o, const char* oname, float p, enum NETWORK_STATUS status)
{
	o = nn_get_out_arg_index(network, o, oname);
	auto result = nn_from_nlop_F(append_dropout_layer(nlop_clone(nn_get_nlop(network)), o, p, status));
	nn_clone_args(result, network);
	nn_free(network);
	return result;
}

/**
 * Append flatten layer
 * flattens all dimensions except the last one (batch dim)
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param oname
 */
nn_t nn_append_flatten_layer(nn_t network, int o, const char* oname)
{
	o = nn_get_out_arg_index(network, o, oname);
	auto result = nn_from_nlop_F(append_flatten_layer(nlop_clone(nn_get_nlop(network)), o));
	nn_clone_args(result, network);
	nn_free(network);
	return result;
}

/**
 * Append padding layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param oname
 * @param N number of dimensions
 * @param pad_for
 * @param pad_after
 * @param pad_type
 */
nn_t nn_append_padding_layer(nn_t network, int o, const char* oname, long N, long pad_for[N], long pad_after[N], enum PADDING pad_type)
{
	o = nn_get_out_arg_index(network, o, oname);
	auto result = nn_from_nlop_F(append_padding_layer(nlop_clone(nn_get_nlop(network)), o, N, pad_for, pad_after, pad_type));
	nn_clone_args(result, network);
	nn_free(network);
	return result;
}