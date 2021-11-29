
#include <complex.h>

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
		error("\"%s\" not found in data list!");

	const struct named_tensor_s* tensor = list_get_item((list_t)data_list, list_get_first_index((list_t)data_list, name, cmp_name));
	return tensor;
}

const struct iovec_s* named_data_list_get_iovec(struct named_data_list_s* data_list, const char* name)
{
	auto tensor = get_tensor_by_name(data_list, name);
	return iovec_create(tensor->N, tensor->dims, sizeof(complex float));
}


const struct nlop_s* nn_batchgen_create(nn_t network, struct named_data_list_s* train_data, enum BATCH_GEN_TYPE type, unsigned int seed)
{
	int II = nn_get_nr_in_args(network);

	const char* names[II];
	enum IN_TYPE in_types[II];

	nn_get_in_args_names(network, II, names, false);
	nn_get_in_types(network, II, in_types);

	int D = 0;
	for (int i = 0; i < II; i ++)
		if (IN_BATCH_GENERATOR == in_types[i])
			D++;

	int N[D];
	const long* bat_dims[D];
	const long* tot_dims[D];
	const complex float* data[D];

	D = 0;

	for (int i = 0; i < II; i ++)
		if (IN_BATCH_GENERATOR == in_types[i]) {

			const struct named_tensor_s* tensor = get_tensor_by_name(train_data, names[i]);
			auto iov = nn_generic_domain(network, 0, names[i]);

			assert(tensor->N == (int)(iov->N));

			N[D] = tensor->N;
			bat_dims[D] = iov->dims;
			tot_dims[D] = tensor->dims;
			data[D] = tensor->data;

			D++;
		}

	return batch_gen_create(D, N, bat_dims, tot_dims, data, 0, type, seed);
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