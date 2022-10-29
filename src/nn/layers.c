/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/conv.h"

#include "nn/batchnorm.h"
#include "nn/nn_ops.h"
#include "layers.h"

/**
 * Append convolution/correlation layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param conv_flag spatial dimensions which are convolved over
 * @param channel_flag dimensions holding channels
 * @param group_flag group dimensions, i.e. batch dimension with non-shared kernel
 * @param kernel_dims kernel size (if conv_flag); out_channel(if channel_flag); idims (if group_flag); 1 else
 * @param strides of output (only for spatial dims)
 * @param dilations of kernel (only for spatial dims)
 * @param conv convolution if true, correlation else
 * @param conv_pad padding for the convolution
 * @param swap_kernel_matrix if true, the kernel has dims (IC, OC) instead of (OC, IC)
 */
const struct nlop_s* append_convcorr_layer_generic(const struct nlop_s* network, int o,
						unsigned long conv_flag, unsigned long channel_flag, unsigned long group_flag,
						int N, long const kernel_dims[N], const long strides[N], const long dilations[N],
						bool conv, enum PADDING conv_pad)
{
	int NO = nlop_get_nr_out_args(network);
	int NI = nlop_get_nr_in_args(network);

	assert(o < NO);
	assert((nlop_generic_codomain(network, o))->N == N);

	assert(0 == (conv_flag & channel_flag));
	assert(0 == (conv_flag & group_flag));
	assert(0 == (channel_flag & group_flag));

	//set default dilation/strides
	long ones[N];
	for (int i = 0; i < N; i++)
		ones[i] = 1;

	if (NULL == strides)
		strides = ones;

	if (NULL == dilations)
		dilations = ones;

	long idims[N];
	long kdims[2 * N];
	long odims[N];

	md_copy_dims(N, idims, nlop_generic_codomain(network, o)->dims);
	md_singleton_dims(2 * N, kdims);
	md_singleton_dims(N, odims);

	long idims_op[2 * N];
	long odims_op[2 * N];
	long kdims_op[2 * N];

	md_singleton_dims(2 * N, idims_op);
	md_singleton_dims(2 * N, odims_op);
	md_singleton_dims(2 * N, kdims_op);

	long dil_op[2 * N];
	long str_op[2 * N];

	md_singleton_dims(2 * N, dil_op);
	md_singleton_dims(2 * N, str_op);

	unsigned long conv_op_flags = 0;

	int ip = 0;
	int ik = 0;

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(conv_flag, i)) {

			if (conv_pad == PAD_VALID) {

				assert(0 == (idims[i] - dilations[i] * (kernel_dims[i] - 1) - 1) % strides[i]);
				odims[i] = (idims[i] - dilations[i] * (kernel_dims[i] - 1) - 1) / strides[i] + 1;
				odims_op[ip] = odims[i];

			} else {

				assert(0 == idims[i] % strides[i]);
				odims[i] = idims[i] / strides[i];
				odims_op[ip] = odims[i];
			}

			kdims_op[ip] = kernel_dims[i];
			idims_op[ip] = idims[i];

			str_op[ip] = strides[i];
			dil_op[ip] = dilations[i];

			kdims[ik] = kernel_dims[i];

			conv_op_flags = MD_SET(conv_op_flags, ip);

			ip++;
			ik++;

			continue;
		}

		if ((1 != dilations[i]) || (1 != strides[i])) {

			debug_printf(DP_INFO, "convcorr strides: ");
			debug_print_dims(DP_INFO, N, strides);
			debug_printf(DP_INFO, "convcorr dilations: ");
			debug_print_dims(DP_INFO, N, dilations);
			error("Dilations and Strides are only allowed in spatial dimensions!\n");
		}

		if (MD_IS_SET(channel_flag, i)) {

			kdims_op[ip] = kernel_dims[i];
			odims_op[ip] = kernel_dims[i];
			ip++;
			kdims_op[ip] = idims[i];
			idims_op[ip] = idims[i];

			kdims[ik] = kernel_dims[i];
			ik++;
			kdims[ik] = idims[i];
			odims[i] = kernel_dims[i];

			ip++;
			ik++;

			continue;
		}

		if (MD_IS_SET(group_flag, i)) {

			// batch dimensions
			assert(kernel_dims[i] == idims[i]);

			kdims_op[ip] = kernel_dims[i];
			odims_op[ip] = idims[i];
			idims_op[ip] = idims[i];

			kdims[ik] = kernel_dims[i];
			odims[i] = idims[i];

			ip++;
			ik++;

			continue;
		}

		// batch dimensions
		assert(1 == kernel_dims[i]);
		kdims_op[ip] = kernel_dims[i];
		odims_op[ip] = idims[i];
		idims_op[ip] = idims[i];

		odims[i] = idims[i];

		ip++;
	}

	const struct nlop_s* nlop_conv = nlop_convcorr_geom_create(	ip, conv_op_flags,
									odims_op, idims_op, kdims_op,
									conv_pad, conv,
									str_op, dil_op,
									'N'
								);

	nlop_conv = nlop_reshape_out_F(nlop_conv, 0, N, odims);
	nlop_conv = nlop_reshape_in_F(nlop_conv, 0, N, idims);
	nlop_conv = nlop_reshape_in_F(nlop_conv, 1, ik, kdims);

	network = nlop_chain2_FF(network, o, nlop_conv, 0);

	network = nlop_shift_input_F(network, NI, 0);
	network = nlop_shift_output_F(network, o, 0);

	return network;
}



/**
 * Append transposed convolution/correlation layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param conv_flag spatial dimensions which are convolved over
 * @param channel_flag dimensions holding channels
 * @param group_flag group dimensions, i.e. batch dimension with non-shared kernel
 * @param kernel_dims kernel size (if conv_flag); out_channel(if channel_flag); idims (if group_flag); 1 else
 * @param strides of input (only for spatial dims)
 * @param dilations of kernel (only for spatial dims)
 * @param conv convolution if true, correlation else
 * @param conv_pad padding for the convolution
 * @param adjoint adjoint convolution if true, transposed else
 * @param swap_kernel_matrix if true, the kernel has dims (IC, OC) instead of (OC, IC)
 */
const struct nlop_s* append_transposed_convcorr_layer_generic(const struct nlop_s* network, int o,
						unsigned long conv_flag, unsigned long channel_flag, unsigned long group_flag,
						int N, long const kernel_dims[N], const long strides[N], const long dilations[N],
						bool conv, enum PADDING conv_pad, bool adjoint)
{
	int NO = nlop_get_nr_out_args(network);
	int NI = nlop_get_nr_in_args(network);

	assert(o < NO);
	assert((nlop_generic_codomain(network, o))->N == N);

	assert(0 == (conv_flag & channel_flag));
	assert(0 == (conv_flag & group_flag));
	assert(0 == (channel_flag & group_flag));

	//set default dilation/strides
	long ones[N];
	for (int i = 0; i < N; i++)
		ones[i] = 1;

	if (NULL == strides)
		strides = ones;

	if (NULL == dilations)
		dilations = ones;

	long idims[N];
	long kdims[2 * N];
	long odims[N];

	md_copy_dims(N, odims, nlop_generic_codomain(network, o)->dims);
	md_singleton_dims(2 * N, kdims);
	md_singleton_dims(N, idims);

	long idims_op[2 * N];
	long odims_op[2 * N];
	long kdims_op[2 * N];

	md_singleton_dims(2 * N, idims_op);
	md_singleton_dims(2 * N, odims_op);
	md_singleton_dims(2 * N, kdims_op);

	long dil_op[2 * N];
	long str_op[2 * N];

	md_singleton_dims(2 * N, dil_op);
	md_singleton_dims(2 * N, str_op);

	unsigned long conv_op_flags = 0;

	int ip = 0;
	int ik = 0;

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(conv_flag, i)) {

			if (conv_pad == PAD_VALID) {

				idims[i] = (odims[i] - 1) * strides[i] + 1 + dilations[i] * (kernel_dims[i] - 1);
				idims_op[ip] = idims[i];

			} else {

				idims[i] = odims[i] * strides[i];
				idims_op[ip] = idims[i];
			}

			kdims_op[ip] = kernel_dims[i];
			odims_op[ip] = odims[i];

			str_op[ip] = strides[i];
			dil_op[ip] = dilations[i];

			kdims[ik] = kernel_dims[i];

			conv_op_flags = MD_SET(conv_op_flags, ip);

			ip++;
			ik++;

			continue;
		}

		if ((1 != dilations[i]) || (1 != strides[i])) {

			debug_printf(DP_INFO, "convcorr strides: ");
			debug_print_dims(DP_INFO, N, strides);
			debug_printf(DP_INFO, "convcorr dilations: ");
			debug_print_dims(DP_INFO, N, dilations);
			error("Dilations and Strides are only allowed in spatial dimensions!\n");
		}

		if (MD_IS_SET(channel_flag, i)) {

			kdims_op[ip] = odims[i];
			odims_op[ip] = odims[i];
			ip++;
			kdims_op[ip] = kernel_dims[i];
			idims_op[ip] = kernel_dims[i];

			kdims[ik] = odims[i];
			ik++;
			idims[i] = kernel_dims[i];
			kdims[ik] = idims[i];

			ip++;
			ik++;

			continue;
		}

		if (MD_IS_SET(group_flag, i)) {

			// batch dimensions
			assert(kernel_dims[i] == idims[i]);

			kdims_op[ip] = kernel_dims[i];
			odims_op[ip] = idims[i];
			idims_op[ip] = idims[i];

			kdims[ik] = kernel_dims[i];
			odims[i] = idims[i];

			ip++;
			ik++;

			continue;
		}

		// batch dimensions
		assert(1 == kernel_dims[i]);
		kdims_op[ip] = kernel_dims[i];
		odims_op[ip] = odims[i];
		idims_op[ip] = odims[i];

		idims[i] = odims[i];

		ip++;
	}

	const struct nlop_s* nlop_conv = nlop_convcorr_geom_create(	ip, conv_op_flags,
									odims_op, idims_op, kdims_op,
									conv_pad, conv,
									str_op, dil_op,
									adjoint ? 'C' : 'T'
								);

	nlop_conv = nlop_reshape_out_F(nlop_conv, 0, N, idims);
	nlop_conv = nlop_reshape_in_F(nlop_conv, 0, N, odims);
	nlop_conv = nlop_reshape_in_F(nlop_conv, 1, ik, kdims);

	network = nlop_chain2_FF(network, o, nlop_conv, 0);

	network = nlop_shift_input_F(network, NI, 0);
	network = nlop_shift_output_F(network, o, 0);

	return network;
}

/**
 * Append convolution/correlation layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param filters number of output channels
 * @param kernel_size {kx, ky, kz} size of convolution kernel
 * @param conv convolution if true, correlation else
 * @param conv_pad padding for the convolution
 * @param channel_first data layout is {c, x, y, z}/{filter, channel, kx, ky, kz} if true, {x, y, z, c} {kx, ky, kz, filter, channel} else
 * @param strides only take into account convolutions separated by strides {sx, sy, sz} (0 == (idims_xyz[i] - dilations[i] * (kernel_size[i] - 1) - 1) % strides[i]))
 * @param dilations elements of kernel dilated by {dx, dy, dz}
 */
const struct nlop_s* append_convcorr_layer(const struct nlop_s* network, int o, int filters, long const kernel_size[3], bool conv, enum PADDING conv_pad, bool channel_first, const long strides[3], const long dilations[3])
{
	if (channel_first) {

		long kernel[5] = {filters, kernel_size[0], kernel_size[1], kernel_size[2], 1};
		long dil_tmp[5] = { 1, 1, 1, 1, 1};
		long str_tmp[5] = { 1, 1, 1, 1, 1};

		if (NULL != dilations)
			md_copy_dims(3, dil_tmp + 1, dilations);
		if (NULL != strides)
			md_copy_dims(3, str_tmp + 1, strides);

		return append_convcorr_layer_generic(
						network, o,
						14, 1, 0,
						5, kernel, str_tmp, dil_tmp,
						conv, conv_pad
					);
	} else {

		long kernel[5] = {kernel_size[0], kernel_size[1], kernel_size[2], filters, 1};
		long dil_tmp[5] = { 1, 1, 1, 1, 1};
		long str_tmp[5] = { 1, 1, 1, 1, 1};

		if (NULL != dilations)
			md_copy_dims(3, dil_tmp, dilations);
		if (NULL != strides)
			md_copy_dims(3, str_tmp, strides);

		return append_convcorr_layer_generic(
						network, o,
						7, 8, 0,
						5, kernel, str_tmp, dil_tmp,
						conv, conv_pad
					);

	}
}

/**
 * Append transposed convolution/correlation layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param channels number of output channels (input channels of forward convolution)
 * @param kernel_size {kx, ky, kz} size of convolution kernel
 * @param conv convolution if true, correlation else
 * @param adjoint if true, the operator is a adjoint convolution, else it's a transposed one
 * @param conv_pad padding for the convolution
 * @param channel_first data layout is {c, x, y, z}/{filter, channel, kx, ky, kz} if true, {x, y, z, c} {kx, ky, kz, filter, channel} else
 * @param strides only take into account convolutions seperated by strides {sx, sy, sz}
 * @param dilations elements of kernel dilated by {dx, dy, dz}
 */
const struct nlop_s* append_transposed_convcorr_layer(const struct nlop_s* network, int o, int channels, long const kernel_size[3], bool conv, bool adjoint, enum PADDING conv_pad, bool channel_first, const long strides[3], const long dilations[3])
{
	if (channel_first) {

		long kernel[5] = {channels, kernel_size[0], kernel_size[1], kernel_size[2], 1};
		long dil_tmp[5] = { 1, 1, 1, 1, 1};
		long str_tmp[5] = { 1, 1, 1, 1, 1};

		if (NULL != dilations)
			md_copy_dims(3, dil_tmp + 1, dilations);
		if (NULL != strides)
			md_copy_dims(3, str_tmp + 1, strides);

		return append_transposed_convcorr_layer_generic(
						network, o,
						14, 1, 0,
						5, kernel, str_tmp, dil_tmp,
						conv, conv_pad, adjoint
					);
	} else {

		long kernel[5] = {kernel_size[0], kernel_size[1], kernel_size[2], channels, 1};
		long dil_tmp[5] = { 1, 1, 1, 1, 1};
		long str_tmp[5] = { 1, 1, 1, 1, 1};

		if (NULL != dilations)
			md_copy_dims(3, dil_tmp, dilations);
		if (NULL != strides)
			md_copy_dims(3, str_tmp, strides);

		return append_transposed_convcorr_layer_generic(
						network, o,
						7, 8, 0,
						5, kernel, str_tmp, dil_tmp,
						conv, conv_pad, adjoint
					);
	}
}

static bool calc_pooling_working_dims(unsigned int N, long dims_working[N], const long dims[N], const long pool_size[N], enum PADDING conv_pad)
{
	md_copy_dims(N, dims_working, dims);
	bool resize_needed = false;

	for (unsigned int i = 0; i < N; i++){

		if (dims_working[i] % pool_size[i] == 0)
			continue;

		resize_needed = true;

		if (conv_pad == PAD_VALID){

			dims_working[i] -= (dims_working[i] % pool_size[i]);
			continue;
		}

		if (conv_pad == PAD_SAME){

			dims_working[i] += pool_size[i] - (dims_working[i] % pool_size[i]);
			continue;
		}

		assert(0);
	}

	return resize_needed;
}

/**
 * Append maxpooling layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param N
 * @param pool_size {px, py, pz} size of pooling
 * @param conv_pad must be PAD_VALID/PAD_SAME if image size is not a multiple of padding size, the image is shrinked/expanded to a multiple
 */
const struct nlop_s* append_maxpool_layer_generic(const struct nlop_s* network, int o, int N, const long pool_size[N], enum PADDING conv_pad)
{
	//Fixme: we should adapt to tf convention (include strides)

	int NO = nlop_get_nr_out_args(network);
	assert(o < NO);

	assert((PAD_VALID == conv_pad) || (PAD_SAME == conv_pad));

	assert((nlop_generic_codomain(network, o))->N == N);

	long idims_layer[N];
	long idims_working[N];
	md_copy_dims(N, idims_layer, nlop_generic_codomain(network, o)->dims);

	bool resize_needed = calc_pooling_working_dims(N, idims_working, idims_layer, pool_size, conv_pad);

	const struct nlop_s* pool_op = nlop_maxpool_create(N, idims_working, pool_size);

	if (resize_needed)
		pool_op = nlop_chain_FF(nlop_from_linop_F(linop_expand_create(N, idims_layer, idims_working)), pool_op);

	network = nlop_chain2_FF(network, o, pool_op, 0);
	network = nlop_shift_output_F(network, o, 0);

	return network;
}

/**
 * Append maxpooling layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param pool_size {px, py, pz} size of pooling
 * @param conv_pad must be PAD_VALID/PAD_SAME if image size is not a multiple of padding size, the image is shrinked/expanded to a multiple
 * @param channel_first data layout is {c, x, y, z} if true, {x, y, z, c}else
 */
const struct nlop_s* append_maxpool_layer(const struct nlop_s* network, int o, const long pool_size[3], enum PADDING conv_pad, bool channel_first)
{
	//Fixme: we should adapt to tf convention (include strides)
	long npool_size[5];
	md_singleton_dims(5, npool_size);
	md_copy_dims(3, channel_first ? npool_size + 1 : npool_size, pool_size);

	return append_maxpool_layer_generic(network, o, 5, npool_size, conv_pad);
}



/**
 * Append dense layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param out_neurons number of output neurons
 */
const struct nlop_s* append_dense_layer(const struct nlop_s* network, int o, int out_neurons)
{

	int NO = nlop_get_nr_out_args(network);
	int NI = nlop_get_nr_in_args(network);

	assert(o < NO);
	assert((nlop_generic_codomain(network, o))->N == 2);

	long batch = (nlop_generic_codomain(network, o)->dims)[1];
	long in_neurons = (nlop_generic_codomain(network, o)->dims)[0];

	long idims_layer[] = {in_neurons, batch};       //in neurons, batch
	long odims_layer[] = {out_neurons, batch};      //out neurons, batch
	long wdims_layer[] = {out_neurons, in_neurons}; //out neurons, in neurons

	long istrs_layer[2];
	md_copy_strides(2, istrs_layer, nlop_generic_codomain(network, o)->strs);

	long idims_working[] = {1, in_neurons, batch};       //in neurons, batch
	long odims_working[] = {out_neurons, 1, batch};      //out neurons, batch
	long wdims_working[] = {out_neurons, in_neurons, 1}; //out neurons, in neurons

	const struct nlop_s* matmul = nlop_tenmul_create(3, odims_working, idims_working, wdims_working);
	matmul = nlop_reshape_out_F(matmul, 0, 2, odims_layer);
	matmul = nlop_reshape_in_F(matmul, 0, 2, idims_layer);
	matmul = nlop_reshape_in_F(matmul, 1, 2, wdims_layer);

	network = nlop_chain2_FF(network, o, matmul, 0);
	network = nlop_shift_input_F(network, NI, 0);
	network = nlop_shift_output_F(network, o, 0);

	return network;
}


/**
 * Append dropout layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param p procentage of outputs dropt out
 */
const struct nlop_s* append_dropout_layer(const struct nlop_s* network, int o, float p, enum NETWORK_STATUS status)
{
	int NO = nlop_get_nr_out_args(network);
	//int NI = nlop_get_nr_in_args(network);

	assert(o < NO);

	unsigned int N = nlop_generic_codomain(network, o)->N;
	long idims[N];
	md_copy_dims(N, idims, nlop_generic_codomain(network, o)->dims);

	const struct nlop_s* dropout_op = NULL;
	if (status == STAT_TRAIN)
		dropout_op = nlop_dropout_create(N, idims, p, 0);
	else
		dropout_op = nlop_from_linop_F(linop_scale_create(N, idims, 1. - p));

	network = nlop_chain2_FF(network, o, dropout_op, 0);
	network = nlop_shift_output_F(network, 0, o);

	return network;
}

/**
 * Append flatten layer
 * flattens all dimensions except the last one (batch dim)
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 */
const struct nlop_s* append_flatten_layer(const struct nlop_s* network, int o)
{
	int NO = nlop_get_nr_out_args(network);
	//int NI = nlop_get_nr_in_args(network);
	assert(o < NO);

	unsigned int N = nlop_generic_codomain(network, o)->N;

	long idims[N];
	md_copy_dims(N, idims, nlop_generic_codomain(network, o)->dims);

	long size = md_calc_size(N - 1, idims);
	long odims[] = {size, idims[N - 1]};

	return nlop_reshape_out_F(network, o, 2, odims);
}

/**
 * Append padding layer
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param N number of dimensions
 * @param pad_for
 * @param pad_after
 * @param pad_type
 */
const struct nlop_s* append_padding_layer(const struct nlop_s* network, int o, long N, long pad_for[N], long pad_after[N], enum PADDING pad_type)
{
	int NO = nlop_get_nr_out_args(network);
	assert(o < NO);

	auto io = nlop_generic_codomain(network, o);
	assert(io->N == N);
	auto pad_op = nlop_from_linop_F(linop_padding_create(io->N, io->dims, pad_type, pad_for, pad_after));

	network = nlop_chain2_FF(network, o, pad_op, 0);
	network = nlop_shift_output_F(network, o, 0);

	return network;
}

/**
 * Append batch normalization
 *
 * @param network operator to append the layer (the operator is freed)
 * @param o output index of network, the layer is appended
 * @param norm_flags select dimension over which we normalize
 */
const struct nlop_s* append_batchnorm_layer(const struct nlop_s* network, int o, unsigned long norm_flags, enum NETWORK_STATUS status)
{
	int NO = nlop_get_nr_out_args(network);
	int NI = nlop_get_nr_in_args(network);

	assert(o < NO);

	auto batchnorm = nlop_batchnorm_create(nlop_generic_codomain(network, o)->N, nlop_generic_codomain(network, o)->dims, norm_flags, 1.e-3, status);

	network = nlop_chain2_FF(network, o, batchnorm , 0);

	network = nlop_shift_input_F(network, NI, 0);
	network = nlop_shift_output_F(network, NO, 1);
	network = nlop_shift_output_F(network, o, 0);

	return network;
}
