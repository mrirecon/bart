/* Copyright 2021-2022. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include <complex.h>

#include "misc/debug.h"
#include "misc/list.h"
#include "misc/misc.h"
#include "misc/types.h"

#include "nn/const.h"
#include "num/multind.h"
#include "num/iovec.h"

#include "iter/italgos.h"
#include "iter/batch_gen.h"

#include "nn/nn.h"

#include "data_list.h"

struct named_tensor_s {

	int N;
	long* dims;
	complex float* data;
	const char* name;
};

static void debug_print_named_tensor(int level, const struct named_tensor_s* ten)
{
	debug_printf(level, "%s at %p: ", ten->name, ten->data);
	debug_print_dims(level, ten->N, ten->dims);
}


static const struct named_tensor_s* named_tensor_create(int N, const long dims[N], complex float* data, const char* name)
{
	auto result = TYPE_ALLOC(struct named_tensor_s);

	result->N = N;
	result->dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, result->dims, dims);

	result->data = data;
	result->name = strdup(name);

	return result;
}


static void named_tensor_free(const struct named_tensor_s* tensor)
{
	xfree(tensor->dims);
	xfree(tensor->name);

	xfree(tensor);
}

struct named_data_list_s* named_data_list_create(void)
{
	return (struct named_data_list_s*)list_create();
}

void debug_print_named_data_list(int level, const struct named_data_list_s* list)
{
	for (int i = 0; i < list_count((list_t)list); i++)
		debug_print_named_tensor(level, list_get_item((list_t)list, i));
}

void named_data_list_free(struct named_data_list_s* data_list)
{
	const struct named_tensor_s* data = list_pop((list_t)data_list);

	while(NULL != data) {

		named_tensor_free(data);
		data = list_pop((list_t)data_list);
	}

	list_free((list_t)data_list);
}

void named_data_list_append(struct named_data_list_s* data_list, int N, const long dims[N], complex float* data, const char* name)
{
	assert(NULL != data);
	list_append((list_t)data_list, (void*)named_tensor_create(N, dims, data, name));
}

static bool cmp_name(const void* _data, const void* _ref)
{
	const struct named_tensor_s* data = _data;
	const char* ref = _ref;

	return (0 == strcmp(data->name, ref));
}

static const struct named_tensor_s* get_tensor_by_name(struct named_data_list_s* data_list, const char* name)
{
	if (-1 == list_get_first_index((list_t)data_list, name, cmp_name))
		error("\"%s\" not found in data list!", name);

	const struct named_tensor_s* tensor = list_get_item((list_t)data_list, list_get_first_index((list_t)data_list, name, cmp_name));
	return tensor;
}

const struct iovec_s* named_data_list_get_iovec(struct named_data_list_s* data_list, const char* name)
{
	auto tensor = get_tensor_by_name(data_list, name);
	return iovec_create(tensor->N, tensor->dims, sizeof(complex float));
}

void* named_data_list_get_data(struct named_data_list_s* data_list, const char* name)
{
	auto tensor = get_tensor_by_name(data_list, name);
	return tensor->data;
}

extern const struct nlop_s* nn_batchgen_create(struct bat_gen_conf_s* config, nn_t network, struct named_data_list_s* train_data)
{
	int II = nn_get_nr_in_args(network);

	const char* names[II];
	enum IN_TYPE in_types[II];

	nn_get_in_args_names(network, II, names, false);
	nn_get_in_types(network, II, in_types);

	int D = 0;
	int N = 0;

	for (int i = 0; i < II; i ++)
		if (IN_BATCH_GENERATOR == in_types[i]) {

			assert(0 != names[i]);

			N = MAX(N, (int)nn_generic_domain(network, 0, names[i])->N);
			D++;
		}

	long bat_dims[D][N];
	long tot_dims[D][N];
	const complex float* data[D];

	D = 0;

	for (int i = 0; i < II; i ++)
		if (IN_BATCH_GENERATOR == in_types[i]) {

			const struct named_tensor_s* tensor = get_tensor_by_name(train_data, names[i]);
			auto iov = nn_generic_domain(network, 0, names[i]);

			assert(tensor->N == (int)(iov->N));

			md_singleton_dims(N, bat_dims[D]);
			md_singleton_dims(N, tot_dims[D]);
			
			md_copy_dims(iov->N, bat_dims[D], iov->dims);
			md_copy_dims(tensor->N, tot_dims[D], tensor->dims);
			data[D] = tensor->data;

			D++;
		}

	return batch_generator_create(config, D, N, bat_dims, tot_dims, data);
}


nn_t nn_valid_create(nn_t network, struct named_data_list_s* valid_data)
{
	int II = nn_get_nr_in_args(network);

	const char* names[II];
	enum IN_TYPE in_types[II];

	nn_get_in_args_names(network, II, names, true);
	nn_get_in_types(network, II, in_types);

	for (int i = 0; i < II; i ++)
		if (IN_BATCH_GENERATOR == in_types[i]) {

			const struct named_tensor_s* tensor = get_tensor_by_name(valid_data, names[i]);
			auto iov = nn_generic_domain(network, 0, names[i]);

			network = nn_ignore_input_F(network, 0, names[i], tensor->N, tensor->dims, false, tensor->data);

			assert(tensor->N == (int)(iov->N));
		}

	for(int i = 0; i < II; i++)
		if (NULL != names[i])
			xfree(names[i]);

	return network;
}



void nn_apply_named_list(nn_t nn_apply, struct named_data_list_s* data, const void* reference)
{
	int OO = nn_get_nr_out_args(nn_apply);
	int II = nn_get_nr_in_args(nn_apply);

	const struct iovec_s* cod[OO];
	const struct iovec_s* dom[II];

	int DO[OO];
	int DI[II];
	
	const long* odims[OO];
	const long* idims[II];
	
	complex float* dst[OO];
	const complex float* src[II];

	unsigned long loop_flags = 0;

	for (int i = 0; i < OO; i++) {

		const char* oname = nn_get_out_names(nn_apply)[i];
		
		assert(NULL != oname);

		cod[i] = named_data_list_get_iovec(data, oname);
		DO[i] = cod[i]->N;
		odims[i] = cod[i]->dims;

		assert(cod[i]->N == nn_generic_codomain(nn_apply, 0, oname)->N);
		loop_flags |= (md_nontriv_dims(DO[i], odims[i]) & (~md_nontriv_dims(DO[i], nn_generic_codomain(nn_apply, 0, oname)->dims)));

		dst[i] = get_tensor_by_name(data, oname)->data;
	}

	for (int i = 0; i < II; i++) {

		const char* iname = nn_get_in_names(nn_apply)[i];
		
		assert(NULL != iname);

		dom[i] = named_data_list_get_iovec(data, iname);
		DI[i] = dom[i]->N;
		idims[i] = dom[i]->dims;

		assert(dom[i]->N == nn_generic_domain(nn_apply, 0, iname)->N);
		loop_flags |= (md_nontriv_dims(DI[i], idims[i]) & (~md_nontriv_dims(DI[i], nn_generic_domain(nn_apply, 0, iname)->dims)));
	
		src[i] = get_tensor_by_name(data, iname)->data;
	}

	nlop_generic_apply_loop_sameplace(nn_get_nlop(nn_apply), loop_flags, OO, DO, odims, dst, II, DI, idims, src, reference);

	for (int i = 0; i < OO; i++)
		iovec_free(cod[i]);

	for (int i = 0; i < II; i++)
		iovec_free(dom[i]);
}
