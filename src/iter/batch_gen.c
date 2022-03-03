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

struct bat_gen_conf_s bat_gen_conf_default = {

	.Nc = 0,
	.type = BATCH_GEN_SHUFFLE_DATA,
	.seed = 123,
	.bat_flags = 0,
};

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
	long N;	//rank of arrays

	const long* bat_dims_bat;	// used to unravel pos of current batch
	const long* bat_dims_tot;	// used to unravel pos of current batch

	long Nb;
	long Nt;

	const long** dims;		// dims to copy batch (not containing the batch dimensions)
	const long** bat_strs;		// strides
	const long** tot_strs;		// strides

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

/**
 * Convert flat index to pos
 *
 */
static void unravel_index(int N, long pos[N], const long dims[N], long index)
{
	for (int d = 0; d < N; ++d) {

		pos[d] = 0;
		if (1 == dims[d])
			continue;

		pos[d] = index % dims[d];
		index /= dims[d];
	}

	assert(0 == index);
}


static void batch_gen_fun(const struct nlop_data_s* _data, int N_args, complex float* args[N_args])
{
	const auto data = CAST_DOWN(batch_gen_data_s, _data);

	assert(data->D == N_args);

	get_indices(data);

	int N = data->N;

	for (long j = 0; j < data->D; j++) {

		long ipos[N];
		long opos[N];

		for (int i = 0; i < data->Nb; i++) {

			unravel_index(N, ipos, data->bat_dims_tot, data->perm[(data->start + i)]);
			unravel_index(N, opos, data->bat_dims_bat, i);

			md_copy2(	N, data->dims[j],
					data->bat_strs[j], &MD_ACCESS(N, data->bat_strs[j], opos, args[j]),
					data->tot_strs[j], &MD_ACCESS(N, data->tot_strs[j], ipos, data->data[j]),
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
		xfree(data->tot_strs[i]);
		xfree(data->bat_strs[i]);
	}

	xfree(data->dims);
	xfree(data->tot_strs);
	xfree(data->bat_strs);
	xfree(data->bat_dims_bat);
	xfree(data->bat_dims_tot);
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
 * @param type methode to compose new batches
 * @param seed seed for random reshuffeling of batches
 */
const struct nlop_s* batch_gen_create(int D, const int Ns[D], const long* bat_dims[D], const long* tot_dims[D], const _Complex float* data[D], long Nc, enum BATCH_GEN_TYPE type, unsigned int seed)
{

	int N = 0;

	long Nt = 1;
	long Nb = 1;

	int bat_idx[D];

	for (int j = 0; j < D; j++) {

		bat_idx[j] = -1;

		for (int i = 0; i < Ns[j]; i++) {

			if (bat_dims[j][i] != tot_dims[j][i]) {

				assert(-1 == bat_idx[j]);
				bat_idx[j] = i;

				assert((Nt == tot_dims[j][i]) || (1 == Nt) || (1 == tot_dims[j][i]));
				Nt = MAX(Nt, tot_dims[j][i]);

				assert((Nb == bat_dims[j][i]) || (1 == Nb));
				Nb = MAX(Nb, bat_dims[j][i]);
			}
		}

		N = MAX(N, Ns[j]);
	}

	N += 1;

	long nbat_dims[D][N];
	long ntot_dims[D][N];

	for(long i = 0; i < D; i++) {

		md_singleton_dims(N, nbat_dims[i]);
		md_singleton_dims(N, ntot_dims[i]);

		md_copy_dims(Ns[i], nbat_dims[i], bat_dims[i]);
		md_copy_dims(Ns[i], ntot_dims[i], tot_dims[i]);

		if (-1 < bat_idx[i]) {

			assert(1 == md_calc_size(N - bat_idx[i] - 1, nbat_dims[i] + bat_idx[i] + 1));
			assert(1 == md_calc_size(N - bat_idx[i] - 1, ntot_dims[i] + bat_idx[i] + 1));

			nbat_dims[i][N - 1] = nbat_dims[i][bat_idx[i]];
			ntot_dims[i][N - 1] = ntot_dims[i][bat_idx[i]];

			nbat_dims[i][bat_idx[i]] = 1;
			ntot_dims[i][bat_idx[i]] = 1;
		}
	}

	struct bat_gen_conf_s conf = bat_gen_conf_default;
	conf.bat_flags = MD_BIT(N - 1);
	conf.type = type;
	conf.seed = seed;
	conf.Nc = Nc;


	const struct nlop_s* result = batch_generator_create(&conf, D, N, nbat_dims, ntot_dims, data);

	for (int i = 0; i < D; i ++)
		result = nlop_reshape_out_F(result, i, Ns[i], bat_dims[i]);

	return result;

}

const struct nlop_s* batch_gen_create_from_iter(struct iter6_conf_s* iter_conf, int D, const int Ns[D], const long* bat_dims[D], const long* tot_dims[D], const _Complex float* data[D], long Nc)
{
	return batch_gen_create(D, Ns, bat_dims, tot_dims, data, Nc, iter_conf->batchgen_type, iter_conf->batch_seed);
}


const struct nlop_s* batch_generator_create2(struct bat_gen_conf_s* config, int D, int N, const long bat_dims[D][N], const long tot_dims[D][N], const long tot_strs[D][N], const complex float* data[D])
{
	PTR_ALLOC(struct batch_gen_data_s, d);
	SET_TYPEID(batch_gen_data_s, d);

	unsigned long bat_flags = config->bat_flags;
	if (0 == bat_flags) {

		for (int i = 0; i < N; i++)
			for (int j = 0; (j < D) && !MD_IS_SET(bat_flags, i) ; j++)
				if (tot_dims[j][i] > bat_dims[j][i]) {

					debug_printf(DP_INFO, "Batch dimension detected: %d\n", i);
					bat_flags = MD_SET(bat_flags, i);
				}
	}

	d->D = D;
	d->N = N;

	PTR_ALLOC(const long*[D], sdims);
	PTR_ALLOC(const long*[D], ostrs);
	PTR_ALLOC(const long*[D], istrs);

	long bat_dims_bat[N];
	long bat_dims_tot[N];

	md_singleton_dims(N, bat_dims_bat);
	md_singleton_dims(N, bat_dims_tot);

	for (int i = 0; i < D; i++) {

		(*istrs)[i] = ARR_CLONE(long[N], tot_strs[i]);

		long tmp1[N];
		long tmp2[N];

		md_calc_strides(N, tmp1, bat_dims[i], CFL_SIZE);
		(*ostrs)[i] = ARR_CLONE(long[N], tmp1);

		md_select_dims(N, bat_flags, tmp1, bat_dims[i]);
		assert(md_check_compat(N, bat_flags, tmp1, bat_dims_bat));
		md_max_dims(N, bat_flags, bat_dims_bat, bat_dims_bat, tmp1);

		md_select_dims(N, bat_flags, tmp1, tot_dims[i]);
		assert(md_check_compat(N, bat_flags, tmp1, bat_dims_tot));
		md_max_dims(N, bat_flags, bat_dims_tot, bat_dims_tot, tmp1);

		md_select_dims(N, ~bat_flags, tmp1, bat_dims[i]);
		md_select_dims(N, ~bat_flags, tmp2, tot_dims[i]);

		assert(md_check_compat(N, md_nontriv_dims(N, tmp1), tmp1, tmp2));

		(*sdims)[i] = ARR_CLONE(long[N], tmp1);
	}

	d->bat_dims_bat = ARR_CLONE(long[N], bat_dims_bat);
	d->bat_dims_tot = ARR_CLONE(long[N], bat_dims_tot);
	
	d->Nb = md_calc_size(N, bat_dims_bat);
	d->Nt = md_calc_size(N, bat_dims_tot);

	d->dims = *PTR_PASS(sdims);
	d->bat_strs = *PTR_PASS(ostrs);
	d->tot_strs = *PTR_PASS(istrs);

	d->data = ARR_CLONE(const complex float*[D], data);

	d->rand_seed = config->seed;
	d->type = config->type;

	d->perm = *TYPE_ALLOC(long[d->Nt]);

	d->start = d->Nt + 1; //enforce drawing new permutation
	get_indices(d);

	for (int i = 0; i < config->Nc; i++) { //initializing the state after Nc calls to batchnorm

		get_indices(d);
		d->start = (d->start + d->Nb);
	}

	assert(d->Nb <= d->Nt);

	const struct nlop_s* result = nlop_generic_create(D, N, bat_dims, 0, 0, NULL, CAST_UP(PTR_PASS(d)), batch_gen_fun, NULL, NULL, NULL, NULL, batch_gen_del);

	return result;
}

const struct nlop_s* batch_generator_create(struct bat_gen_conf_s* config, int D, int N, const long bat_dims[D][N], const long tot_dims[D][N], const complex float* data[D])
{
	long tot_strs[D][N];
	for (int i = 0; i < D; i++)
		md_calc_strides(N, tot_strs[i], tot_dims[i], CFL_SIZE);

	return batch_generator_create2(config, D, N, bat_dims, tot_dims, tot_strs, data);
}
