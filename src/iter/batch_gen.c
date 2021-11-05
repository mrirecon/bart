/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "iter/iter6.h"

#include "batch_gen.h"


static void rand_draw_data(unsigned int* rand_seed, long N, long perm[N], long Nb)
{
	UNUSED(Nb);

	for (int i = 0; i < N; i++) {

		#pragma omp critical
		perm[i] = rand_r(rand_seed) % N;
	}
}

static void rand_perm_data(unsigned int* rand_seed, long N, long perm[N], long Nb)
{
	UNUSED(Nb);

	bool drawn[N];
	for (int i = 0; i < N; i++)
		drawn[i] = false;

	for (int i = 0; i < N; i++) {

		#pragma omp critical
		perm[i] = rand_r(rand_seed) % (N - i);

		for (int j = 0; j < N; j++)
			if (drawn[j] && perm[i] >= j)
				perm[i] ++;

		drawn[perm[i]] = true;
	}
}

static void rand_perm_batches(unsigned int* rand_seed, long N, long perm[N], long Nb)
{
	long perm_batch[N / Nb];

	rand_perm_data(rand_seed, N / Nb, perm_batch, 0);

	for (int i = 0; i < N / Nb; i++)
		for (int j = 0; j < Nb; j++)
			perm[Nb * i + j] = perm_batch[i] * Nb + j;

	for (int i = (N / Nb) * Nb; i < N; i++)
		perm[i] = i;
}

struct batch_gen_data_s {

	INTERFACE(nlop_data_t);

	long D; //number of arrays

	long N;
	const long** dims;

	const long** ostrs;
	const long** istrs;

	const int* bat_idx;

	long Nb;
	long Nt;
	const complex float** data;

	long start;

	long* perm;

	enum BATCH_GEN_TYPE type;

	unsigned int rand_seed;
};

DEF_TYPEID(batch_gen_data_s);

static void get_indices(struct batch_gen_data_s* data)
{
	if (data->start + data->Nb <= data->Nt)
		return;

	switch (data->type) {

	case BATCH_GEN_SAME:

		for (long i = 0; i < data->Nt; i++)
			data->perm[i] = i;
		break;

	case BATCH_GEN_SHUFFLE_BATCHES:

		rand_perm_batches(&(data->rand_seed), data->Nt, data->perm, data->Nb);
		break;

	case BATCH_GEN_SHUFFLE_DATA:

		rand_perm_data(&(data->rand_seed), data->Nt, data->perm, data->Nb);
		break;

	case BATCH_GEN_RANDOM_DATA:

		rand_draw_data(&(data->rand_seed), data->Nt, data->perm, data->Nb);
		break;

	default:
		assert(0);
	}

	data->start = 0;
}


static void batch_gen_fun(const struct nlop_data_s* _data, int N_args, complex float* args[N_args])
{
	const auto data = CAST_DOWN(batch_gen_data_s, _data);

	assert(data->D == N_args);

	get_indices(data);

	int N = data->N;

	for (long j = 0; j < data->D; j++) {

		if (-1 == data->bat_idx[j]) {

			md_copy(N, data->dims[j], args[j], data->data[j], CFL_SIZE);
			continue;
		}

		long ipos[N];
		long opos[N];

		for (int i = 0; i < N; i++) {

			ipos[i] = 0;
			opos[i] = 0;
		}

		for (int i = 0; i < data->Nb; i++) {

			ipos[data->bat_idx[j]] = data->perm[(data->start + i)];
			opos[data->bat_idx[j]] = i;

			md_copy2(	N, data->dims[j],
					data->ostrs[j], &MD_ACCESS(N, data->ostrs[j], opos, args[j]),
					data->istrs[j], &MD_ACCESS(N, data->istrs[j], ipos, data->data[j]),
					CFL_SIZE);
		}
	}

	data->start += data->Nb;
}

static void batch_gen_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(batch_gen_data_s, _data);

	for(long i = 0; i < data->D; i ++) {

		xfree(data->dims[i]);
		xfree(data->ostrs[i]);
		xfree(data->istrs[i]);
	}

	xfree(data->dims);
	xfree(data->ostrs);
	xfree(data->istrs);
	xfree(data->bat_idx);
	xfree(data->data);
	xfree(data->perm);
	xfree(data);
}



/**
 * Create an operator copying Nb random (not necessarily distinct) datasets to the output
 *
 * @param D number of tensores
 * @param Ns number of dimensions for each tensor
 * @param bat_dims output_dims
 * @param tot_dims total dims of dataset
 * @param data pointers to data
 * @param Nb batch size
 * @param Nc number of calls (initializes the nlop as it had been applied Nc times) -> reproducible warm start
 * @param type methode to compose new batches
 * @param seed seed for random reshuffeling of batches
 */
const struct nlop_s* batch_gen_create(int D, const int Ns[D], const long* bat_dims[D], const long* tot_dims[D], const _Complex float* data[D], long Nc, enum BATCH_GEN_TYPE type, unsigned int seed)
{
	PTR_ALLOC(struct batch_gen_data_s, d);
	SET_TYPEID(batch_gen_data_s, d);

	d->D = D;
	d->Nb = 1;
	d->Nt = 1;

	int N = 0;

	int bat_idx[D];

	for (int j = 0; j < D; j++) {

		bat_idx[j] = -1;

		for (int i = 0; i < Ns[j]; i++) {

			if (bat_dims[j][i] != tot_dims[j][i]) {

				assert(-1 == bat_idx[j]);
				bat_idx[j] = i;

				assert((d->Nt == tot_dims[j][i]) || (1 == d->Nt) || (1 == tot_dims[j][i]));
				d->Nt = MAX(d->Nt, tot_dims[j][i]);

				assert((d->Nb == bat_dims[j][i]) || (1 == d->Nb));
				d->Nb = MAX(d->Nb, bat_dims[j][i]);
			}
		}

		N = MAX(N, Ns[j]);
	}

	long nl_odims[D][N];
	for(long i = 0; i < D; i++)
		md_singleton_dims(N, nl_odims[i]);

	PTR_ALLOC(const long*[D], sdims);
	PTR_ALLOC(const long*[D], ostrs);
	PTR_ALLOC(const long*[D], istrs);

	PTR_ALLOC(int[D], n_bat_idx);

	PTR_ALLOC(const complex float*[D], ndata);

	for (long j = 0; j < D; j++) {

		md_copy_dims(Ns[j], nl_odims[j], bat_dims[j]);

		PTR_ALLOC(long [N], slice_dims);
		md_singleton_dims(N, *slice_dims);
		md_copy_dims(Ns[j], *slice_dims, bat_dims[j]);

		if (-1 != bat_idx[j])
			(*slice_dims)[bat_idx[j]] = 1;

		PTR_ALLOC(long [N], ostr);
		PTR_ALLOC(long [N], istr);

		md_calc_strides(N, *ostr, nl_odims[j], CFL_SIZE);

		md_singleton_strides(N, *istr);
		md_calc_strides(Ns[j], *istr, tot_dims[j], CFL_SIZE);

		(*sdims)[j] = *PTR_PASS(slice_dims);
		(*ostrs)[j] = *PTR_PASS(ostr);
		(*istrs)[j] = *PTR_PASS(istr);

		(*ndata)[j] = data[j];

		(*n_bat_idx)[j] = bat_idx[j];
	}

	d->N = N;
	d->data = *PTR_PASS(ndata);
	d->dims = *PTR_PASS(sdims);
	d->ostrs = *PTR_PASS(ostrs);
	d->istrs = *PTR_PASS(istrs);

	d->bat_idx = *PTR_PASS(n_bat_idx);

	d->rand_seed = seed;
	d->type = type;

	PTR_ALLOC(long[d->Nt], perm);
	d->perm = *PTR_PASS(perm);

	d->start = d->Nt + 1; //enforce drwaing new permutation
	get_indices(d);

	for (int i = 0; i < Nc; i++) { //initializing the state after Nc calls to batchnorm

		get_indices(d);
		d->start = (d->start + d->Nb);
	}

	assert(d->Nb <= d->Nt);

	const struct nlop_s* result = nlop_generic_create(D, N, nl_odims, 0, 0, NULL, CAST_UP(PTR_PASS(d)), batch_gen_fun, NULL, NULL, NULL, NULL, batch_gen_del);

	for (int i = 0; i < D; i ++)
		result = nlop_reshape_out_F(result, i, Ns[i], nl_odims[i]);

	return result;

}

const struct nlop_s* batch_gen_create_from_iter(struct iter6_conf_s* iter_conf, int D, const int Ns[D], const long* bat_dims[D], const long* tot_dims[D], const _Complex float* data[D], long Nc)
{
	return batch_gen_create(D, Ns, bat_dims, tot_dims, data, Nc, iter_conf->batchgen_type, iter_conf->batch_seed);
}

