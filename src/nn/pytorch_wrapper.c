/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Moritz Blumenthal
 */


#include "misc/misc.h"
#include "misc/types.h"

#include "num/multind.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "nlops/nlop.h"

#include "nn/pytorch_cpp_wrapper.h"

#include "pytorch_wrapper.h"

#define UNUSED(x) (void)x

#ifdef PYTORCH

struct pytorch_wrapper_s {

	INTERFACE(nlop_data_t);

	struct pytorch_wrapper_s* data;
};

DEF_TYPEID(pytorch_wrapper_s);



static void pytorch_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(pytorch_wrapper_s, _data);

	int device = -1;

#ifdef USE_CUDA
	for (int i = 0; i < N; i++)
		device = cuda_ondevice(args[i]) ? 0 : -1;
#endif

	pytorch_wrapper_apply_unchecked(data->data, N, args, device);
}


static void pytorch_der(const nlop_data_t* _data, int o, int i, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(pytorch_wrapper_s, _data);

	pytorch_wrapper_derivative_unchecked(data->data, o, i, dst, src);
}

static void pytorch_adj(const nlop_data_t* _data, int o, int i, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(pytorch_wrapper_s, _data);

	pytorch_wrapper_adjoint_unchecked(data->data, o, i, dst, src);
}


static void pytorch_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(pytorch_wrapper_s, _data);

	pytorch_wrapper_free(data->data);

	xfree(data);
}

#endif

const struct nlop_s* nlop_pytorch_create(const char* path, int II, const int DI[II], const long* idims[II], bool init_gpu)
{
#ifdef PYTORCH
	int D = 0;

	for (int i = 0; i < II; i++)
		D = MAX(D, DI[i]);

	PTR_ALLOC(struct pytorch_wrapper_s, data);
	SET_TYPEID(pytorch_wrapper_s, data);

	int device = -1;

#ifdef USE_CUDA
	if (init_gpu && -1 < cuda_get_device_id())
		device = cuda_get_device_id();
#else

	UNUSED(idims);
#endif

	data->data = pytorch_wrapper_create(path, II, DI, idims, device);

	int OO = pytorch_wrapper_number_outputs(data->data);

	for (int i = 0; i < OO; i++)
		D = MAX(D, pytorch_wrapper_rank_output(data->data, i));

	long nl_odims[OO][D];
	long nl_idims[II][D];

	for (int i = 0; i < II; i++) {

		md_singleton_dims(D, nl_idims[i]);
		md_copy_dims(DI[i], nl_idims[i], idims[i]);
	}

	for (int i = 0; i < OO; i++) {

		md_singleton_dims(D, nl_odims[i]);
		pytorch_wrapper_dims_output(data->data, i, pytorch_wrapper_rank_output(data->data, i), nl_odims[i]);
	}

	auto tmp = data->data;

	nlop_der_fun_t der[II][OO];
	nlop_der_fun_t adj[II][OO];

	for (int i = 0; i < II; i++) {

		for (int o = 0; o < OO; o++) {

			der[i][o] = pytorch_der;
			adj[i][o] = pytorch_adj;
		}
	}

	const struct nlop_s* ret = nlop_generic_managed_create(OO, D, nl_odims, II, D, nl_idims, CAST_UP(PTR_PASS(data)), pytorch_fun, der, adj, NULL, NULL, pytorch_del, NULL, NULL);

	for (int i = 0; i < II; i++)
		ret = nlop_reshape_in_F(ret, i, DI[i], nl_idims[i]);

	for (int i = 0; i < OO; i++)
		ret = nlop_reshape_out_F(ret, i, pytorch_wrapper_rank_output(tmp, i), nl_odims[i]);

	return ret;
#else
	error("Not compiled with Pytorch support!\n");

	UNUSED(path);
	UNUSED(DI);
	UNUSED(idims);
	UNUSED(init_gpu);

	return NULL;
#endif
}

