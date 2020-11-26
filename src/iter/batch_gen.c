/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
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


static void batch_gen_fun(const struct nlop_data_s* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(batch_gen_data_s, _data);

	assert(data->D == N);

	get_indices(data);

	for (long j = 0; j < data->D; j++){

		if (1 == data->dims[j][data->N-1]){ //not batched data

			md_copy(data->N, data->dims[j], args[j], data->data[j], CFL_SIZE);
			continue;
		}

		size_t size_dataset =  md_calc_size(data->N-1, data->dims[j]);
		for (long i = 0; i < data->Nb; i++){

			long offset_src = (data->perm[(data->start + i)]) * size_dataset;
			long offset_dst = i * size_dataset;
			md_copy(data->N - 1, data->dims[j], args[j] + offset_dst, (complex float*)data->data[j] + offset_src, CFL_SIZE);
		}
	}

	data->start += data->Nb;
}

static void batch_gen_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(batch_gen_data_s, _data);

	for(long i = 0; i < data->D; i ++)
		xfree(data->dims[i]);
	xfree(data->dims);
	xfree(data->data);
	xfree(data->perm);
	xfree(data);
}



/**
 * Create an operator copying Nb random (not necessarily distinct) datasets to the output
 *
 * @param D number of tensores
 * @param N number of dimensions
 * @param dims pointers to dims (batchsize or 1 in last dimension)
 * @param data pointers to data
 * @param Nt total number of available datasets
 * @param Nc number of calls (initializes the nlop as it had been applied Nc times) -> reproducible warm start
 * @param type methode to compose new batches
 * @param seed seed for random reshuffeling of batches
 */
const struct nlop_s* batch_gen_create(long D, long N, const long* dims[D], const _Complex float* data[__VLA(D)], long Nt, long Nc, enum BATCH_GEN_TYPE type, unsigned int seed)
{
	PTR_ALLOC(struct batch_gen_data_s, d);
	SET_TYPEID(batch_gen_data_s, d);

	d->D = D;
	d->N = N;
	d->Nb = 0; //(dims[0][N-1]);
	d->Nt = Nt;
	PTR_ALLOC(const long*[D], ndims);
	PTR_ALLOC(const complex float*[D], ndata);

	for (long j = 0; j < D; j++) {

		PTR_ALLOC(long[N], dimsj);
		md_copy_dims(N, *dimsj, dims[j]);
		(*ndims)[j] = *PTR_PASS(dimsj);
		assert(	(0 == d->Nb) //d-Nt not set
			|| (d->Nb == dims[j][N-1]) // last dim equals
			|| (1 == dims[j][N-1]) // not batched
			);
		if (1 != dims[j][N-1])
			d->Nb = dims[j][N-1];

		(*ndata)[j] = data[j];
	}


	d->data = *PTR_PASS(ndata);
	d->dims = *PTR_PASS(ndims);

	d->rand_seed = seed;
	d->type = type;

	PTR_ALLOC(long[Nt], perm);
	d->perm = *PTR_PASS(perm);
	
	d->start = d->Nt + 1; //enforce drwaing new permutation
	get_indices(d);

	for (int i = 0; i < Nc; i++) { //initializing the state after Nc calls to batchnorm

		get_indices(d);
		d->start = (d->start + d->Nb);
	}

	assert(d->Nb <= d->Nt);

	long nl_odims[D][N];
	for(long i = 0; i < D; i++)
		md_copy_dims(N, nl_odims[i], dims[i]);

	return nlop_generic_create(D, N, nl_odims, 0, 0, NULL, CAST_UP(PTR_PASS(d)), batch_gen_fun, NULL, NULL, NULL, NULL, batch_gen_del);
}

const struct nlop_s* batch_gen_create_from_iter(struct iter6_conf_s* iter_conf, long D, long N, const long* dims[D], const _Complex float* data[__VLA(D)], long Nt, long Nc)
{
	return batch_gen_create(D, N, dims, data, Nt, Nc, iter_conf->batchgen_type, iter_conf->batch_seed);
}