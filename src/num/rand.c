/* Copyright 2013. The Regents of the University of California.
 * Copyright 2021. Uecker Lab. University Center Göttingen.
 * Copyright 2024. Institute of Biomedical Imaging. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Martin Uecker, Dara Bahri, Moritz Blumenthal, Christian Holme
 */

#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <limits.h>
#include <stdint.h>
#include <assert.h>

#ifdef _WIN32
#include "win/rand_r.h"
#endif

#include "misc/version.h"
#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mmio.h"

#include "num/multind.h"
#include "num/vptr.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#include "num/gpurand.h"
#endif

#include "rand.h"


static bool use_obsolete_rng()
{
	return use_compat_to_version("v0.9.00");
}



struct bart_rand_state {

	unsigned int num_rand_seed;
	// for philox 4x32: 64bits of state, and 2 64bit counters:
	uint64_t state;
	uint64_t ctr1;
	uint64_t ctr2;
};

static void philox_4x32(const uint64_t state, const uint64_t ctr1, const uint64_t ctr2, uint64_t out[2]);

// same as calling rand_state_update(0):
#define MAX_WORKER 128
static struct bart_rand_state global_rand_state[MAX_WORKER] = { [0 ... MAX_WORKER - 1] = { .num_rand_seed = 123, .state = 0x7012082D361B3B31, .ctr1 = 0, .ctr2 = 0 } };

void rand_state_update(struct bart_rand_state* state, unsigned long long seed)
{
	static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "unsigned long long is not 64 bits!\n");

	// for the old PRNG
	if (0 == seed) // special case to preserve old behavior
		state->num_rand_seed = 123;
	else
		state->num_rand_seed = (unsigned int)seed; // truncate here, as rand_r cannot use 64 bits of state

	// For Philox:
	// Idea: use seed as philox state with random counters, then run philox to generate a random state from that
	uint64_t init_state = seed;
	uint64_t init_ctr1 = 0x210b521ad19cdd42;
	uint64_t init_ctr2 = 0x5df75c80f3a19dd3;

	uint64_t out[2];
	philox_4x32(init_state, init_ctr1, init_ctr2, out);

	state->state = out[0];
	state->ctr1 = 0;
	state->ctr2 = 0;
}

struct bart_rand_state* rand_state_create(unsigned long long seed)
{
	if (cfl_loop_desc_active()) {

		static bool warned = false;

#pragma		omp critical
		if (!warned) {

			warned = true;

			if (0 != (cfl_loop_rand_flags & cfl_loop_get_flags()))			
				debug_printf(DP_WARN, "rand_state_create provides identical random numbers for each cfl loop iteration.\n");
		}
	}

	struct bart_rand_state* state = xmalloc(sizeof *state);

	rand_state_update(state, seed);

	return state;
}

static struct bart_rand_state get_worker_state(void)
{ 
	struct bart_rand_state worker_state;

#pragma omp critical(global_rand_state)
	{
		worker_state = global_rand_state[cfl_loop_worker_id()];
		global_rand_state[cfl_loop_worker_id()].ctr1++;
	}

	return worker_state;
}

static struct bart_rand_state get_worker_state_cfl_loop(void)
{ 
	struct bart_rand_state worker_state = get_worker_state();

	if (cfl_loop_desc_active()) {

		int D = cfl_loop_get_rank();

		long dims[D ?:1];
		long pos[D ?:1];

		cfl_loop_get_dims(D, dims);
		cfl_loop_get_pos(D, pos);

		long ind = md_ravel_index(D, pos, cfl_loop_rand_flags, dims);

		assert(0 <= ind);
		worker_state.ctr2 = (uint64_t) ind;
	}

	return worker_state;
}


void num_rand_init(unsigned long long seed)
{
	if (use_obsolete_rng())
		assert(seed <= UINT_MAX);

#pragma omp critical(global_rand_state)
	rand_state_update(&global_rand_state[cfl_loop_worker_id()], seed);
}


static uint64_t rand64_state(struct bart_rand_state* state)
{
	uint64_t out[2];

	philox_4x32(state->state, state->ctr1, state->ctr2, out);
	state->ctr1++;

	return out[0];
}

// shift by 11 to get a 53-bit integer, as double has 53 bits of precision in the mantissa
static inline double ull2double(uint64_t x)
{
	return (double)(x >> 11) * 0x1.0p-53;
}


static double uniform_rand_state(struct bart_rand_state* state)
{
	return ull2double(rand64_state(state));
}

static double uniform_rand_offset(struct bart_rand_state state, long offset)
{
	struct bart_rand_state state_loc = state;
	state_loc.ctr2 = (uint64_t)offset;

	return uniform_rand_state(&state_loc);
}


static double uniform_rand_obsolete()
{
	double r;

#pragma omp critical(global_rand_state)
	r = rand_r(&global_rand_state[cfl_loop_worker_id()].num_rand_seed);

	return r / (double)RAND_MAX;
}



double uniform_rand(void)
{
	double r;

	if (use_obsolete_rng()) {

		r = uniform_rand_obsolete();

	} else {

		struct bart_rand_state worker_state = get_worker_state_cfl_loop();
		r = uniform_rand_state(&worker_state);
	}

	return r;
}


unsigned long long rand_ull_state(struct bart_rand_state* state)
{
	if (use_obsolete_rng())
		return (unsigned long long)rand_r(&state->num_rand_seed);
	else
		return rand64_state(state);
}



unsigned int rand_range_state(struct bart_rand_state* state, unsigned int range)
{
	static_assert(sizeof(unsigned int) == sizeof(uint32_t), "unsigned int is not 32 bits!\n");

	if (1 >= range)
		return 0;

	if (!use_obsolete_rng()) {

		// Lemire's Method, see https://arxiv.org/abs/1805.10941
		// Adapted for 32-bit integers, and written as do { ... } while ();
		// Generates random number in range [0,range)

		uint32_t t = (-range) % range;
		uint64_t m;
		uint32_t l;

		do {
			uint32_t x = rand64_state(state);
			m = (uint64_t)x * (uint64_t)range;
			l = (uint32_t)m;

		} while (l < t);

		return m >> 32;

	} else {

		// Division with rejection, for use with rand_r, as Lemire's method needs a 32-bit PRNG
		uint32_t divisor = RAND_MAX / (range);
		uint32_t retval;

		do {
			retval = (uint32_t)rand_r(&state->num_rand_seed) / divisor;

		} while (retval >= range);

		return retval;
	}
}



unsigned int rand_range(unsigned int range)
{
	unsigned int r;

	if (use_obsolete_rng()) {

#pragma 	omp critical(global_rand_state)
		r = rand_range_state(&global_rand_state[cfl_loop_worker_id()], range);

	} else {

		struct bart_rand_state worker_state = get_worker_state_cfl_loop();
		r = rand_range_state(&worker_state, range);
	}

	return r;
}


/**
 * Box-Muller
 */
static complex double gaussian_rand_obsolete(void)
{
	double u1, u2, s;

	do {
		u1 = 2. * uniform_rand_obsolete() - 1.;
		u2 = 2. * uniform_rand_obsolete() - 1.;
		s = u1 * u1 + u2 * u2;

	} while (s > 1.);

	double re = sqrt(-2. * log(s) / s) * u1;
	double im = sqrt(-2. * log(s) / s) * u2;

	return re + 1.i * im;
}

static complex double gaussian_rand_state(struct bart_rand_state* state)
{
	double u1, u2, s;
	uint64_t out[2];

	// We initialize a new PRNG here, so that we never repeat random numbers
	// Without this, it might happen that two calls of md_gaussian_rand() might repeat random numbers (as the counter states are the same)
	philox_4x32(state->state, state->ctr1, state->ctr2, out);
	state->ctr1++;

	struct bart_rand_state gauss_state = { .num_rand_seed = 0, .state = out[0], .ctr1 = 0, .ctr2 = 0 };

	do {
		philox_4x32(gauss_state.state, gauss_state.ctr1, gauss_state.ctr2, out);
		gauss_state.ctr1++;

		u1 = 2. * ull2double(out[0]) - 1.;
		u2 = 2. * ull2double(out[1]) - 1.;
		s = u1 * u1 + u2 * u2;

	} while (s > 1.);

	double re = sqrt(-2. * log(s) / s) * u1;
	double im = sqrt(-2. * log(s) / s) * u2;

	return re + 1.i * im;
}

complex double gaussian_rand(void)
{
	complex double r;

	if (use_obsolete_rng()) {

		r = gaussian_rand_obsolete();

	} else {

		struct bart_rand_state worker_state = get_worker_state_cfl_loop();
		r = gaussian_rand_state(&worker_state);
	}

	return r;
}




void gaussian_rand_vec(long N, float* dst)
{
	complex float* tmp = md_alloc_sameplace(1, MD_DIMS((N + 1) / 2), sizeof(complex float), dst);

	md_gaussian_rand(1, MD_DIMS((N + 1) / 2), tmp);
	md_copy(1, MD_DIMS(N), dst, tmp, sizeof(float));

	md_free(tmp);
	//This does not need to be scaled as md_gaussian_rand has (complex) variance 2!
}


static void md_gaussian_obsolete_rand(int D, const long dims[D], complex float* dst)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		complex float* tmp = md_alloc(D, dims, sizeof(complex float));

		md_gaussian_obsolete_rand(D, dims, tmp);
		md_copy(D, dims, dst, tmp, sizeof(complex float));

		md_free(tmp);
		return;
	}
#endif
	long N = md_calc_size(D, dims);

	for (long i = 0; i < N; i++)
		dst[i] = (complex float)gaussian_rand_obsolete();
}

static void vec_gaussian_philox_rand(struct bart_rand_state state, long offset, long N, complex float* dst)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		cuda_gaussian_rand(N, dst, state.state, state.ctr1, (uint64_t)offset);

	} else
#endif
	{
#pragma 	omp parallel for
		for (long i = 0; i < N; i++) {

			struct bart_rand_state state_loc = state;
			state_loc.ctr2 = (uint64_t) (i + offset);

			dst[i] = gaussian_rand_state(&state_loc);
		}
	}
}

static long cfl_loop_offset_and_strides(int D, long strs_offset[D], const long dims[D])
{
	md_calc_strides(D, strs_offset, dims, 1);

	if (0 == (cfl_loop_rand_flags & cfl_loop_get_flags()))
		return 0;

	int C = cfl_loop_get_rank();
	long cdims[C ?: 1];

	cfl_loop_get_dims(C, cdims);
	md_select_dims(C, cfl_loop_rand_flags, cdims, cdims);

	bool mergeable = true;

	for (int i = 0; i < MIN(D, C); i++)
		if ((1 != dims[i]) && (1 != cdims[i]))
			mergeable = false;

	if (mergeable) {

		long mdims[MAX(D, C) ?: 1];
		md_singleton_dims(MAX(D, C), mdims);
		md_copy_dims(D, mdims, dims);

		md_max_dims(C, ~0ul, mdims, mdims, cdims);

		long strs_offset_merged[MAX(D, C) ?: 1];
		md_calc_strides(MAX(D, C), strs_offset_merged, mdims, 1);

		long cpos[C ?: 1];
		cfl_loop_get_pos(C, cpos);

		md_copy_strides(D, strs_offset, strs_offset_merged);

		return md_calc_offset(C, strs_offset_merged, cpos);

	} else {

		if (strided_cfl_loop) {

			// CAVEAT: strided_cfl_loop may be set true after random numbers are drawn
			static bool warned_about_random_numbers = false;

#pragma 		omp single
			if (!warned_about_random_numbers) {

				warned_about_random_numbers = true;

				debug_printf(DP_WARN, "Loop dimensions are not the last dimensions, and varying random numbers in those dimensions are selected!\n");
				debug_printf(DP_WARN, "Cannot guarantee consistent random numbers in this case!\n");
			}
		}

		long cpos[C ?: 1];
		cfl_loop_get_pos(C, cpos);

		long ind = md_ravel_index(C, cpos, cfl_loop_rand_flags & cfl_loop_get_flags(), cdims);
		return ind * md_calc_size(D, dims);
	}
}


typedef void CLOSURE_TYPE(md_sample_fun_t)(long offset_rand, long N, complex float* dst);

static void md_sample_mpi(int D, const long dims[D], complex float* dst, md_sample_fun_t vec_fun)
{
	long strs[D];
	md_calc_strides(D, strs, dims, sizeof(complex float));

	long strs_offset[D]; // one based
	long offset_cfl = cfl_loop_offset_and_strides(D, strs_offset, dims);

	unsigned long loop_flags = vptr_block_loop_flags(D, dims, strs, dst, sizeof(complex float));

	if (D != md_calc_blockdim(D, dims, strs_offset, 1))
		loop_flags |= ~(MD_BIT(md_calc_blockdim(D, dims, strs_offset, 1)) - 1);

	long ldims[D];
	long bdims[D];

	md_select_dims(D,  loop_flags, ldims, dims);
	md_select_dims(D, ~loop_flags, bdims, dims);

	long N = md_calc_size(D, bdims);
	long* strs_p = strs;
	long* strs_offset_p = strs_offset;

	NESTED(void, rand_loop, (const long pos[]))
	{
		void* dst_offset = &MD_ACCESS(D, strs_p, pos, dst);

		if (!mpi_accessible(dst_offset))
			return;

		long offset_rand = md_calc_offset(D, strs_offset_p, pos) + offset_cfl;

		vec_fun(offset_rand, N, vptr_resolve(dst_offset));
	};

	md_loop(D, ldims, rand_loop);
}




static void md_gaussian_philox_rand(int D, const long dims[D], complex float* dst)
{
	struct bart_rand_state worker_state = get_worker_state();

	NESTED(void, vec_fun, (long offset, long N, complex float* dst))
	{
		 vec_gaussian_philox_rand(worker_state, offset, N, dst);
	};

	md_sample_mpi(D, dims, dst, vec_fun);
}


void md_gaussian_rand(int D, const long dims[D], complex float* dst)
{
	if (use_obsolete_rng())
		md_gaussian_obsolete_rand(D, dims, dst);
	else
		md_gaussian_philox_rand(D, dims, dst);
}


static void md_uniform_obsolete_rand(int D, const long dims[D], complex float* dst)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		complex float* tmp = md_alloc(D, dims, sizeof(complex float));

		md_uniform_obsolete_rand(D, dims, tmp);
		md_copy(D, dims, dst, tmp, sizeof(complex float));

		md_free(tmp);
		return;
	}
#endif
	long N = md_calc_size(D, dims);

	for (long i = 0; i < N; i++)
		dst[i] = (complex float)uniform_rand_obsolete();
}

static void vec_uniform_philox_rand(struct bart_rand_state state, long offset, long N, complex float* dst)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		cuda_uniform_rand(N, dst, state.state, state.ctr1, (uint64_t)offset);

	} else
#endif
	{
#pragma 	omp parallel for
		for (long i = 0; i < N; i++)
			dst[i] = uniform_rand_offset(state, i + offset);
	}
}

static void md_uniform_philox_rand(int D, const long dims[D], complex float* dst)
{
	struct bart_rand_state worker_state = get_worker_state();

	NESTED(void, vec_fun, (long offset, long N, complex float* dst))
	{
		vec_uniform_philox_rand(worker_state, offset, N, dst);
	};

	md_sample_mpi(D, dims, dst, vec_fun);
}

void md_uniform_rand(int D, const long dims[D], complex float* dst)
{
	if (use_obsolete_rng())
		md_uniform_obsolete_rand(D, dims, dst);
	else
		md_uniform_philox_rand(D, dims, dst);
}

static void md_obsolete_rand_one(int D, const long dims[D], complex float* dst, double p)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		complex float* tmp = md_alloc(D, dims, sizeof(complex float));

		md_obsolete_rand_one(D, dims, tmp, p);
		md_copy(D, dims, dst, tmp, sizeof(complex float));

		md_free(tmp);
		return;
	}
#endif
	long N = md_calc_size(D, dims);

	for (long i = 0; i < N; i++)
		dst[i] = (uniform_rand_obsolete() < p) ? 1. : 0.;
}

static void vec_philox_rand_one(struct bart_rand_state state, long offset, long N, complex float* dst, double p)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		cuda_rand_one(N, dst, p, state.state, state.ctr1, (uint64_t) offset);

	} else
#endif
	{
#pragma 	omp parallel for
		for (long i = 0; i < N; i++)
			dst[i] = (uniform_rand_offset(state, i + offset) < p) ? 1. : 0.;
	}
}

static void md_philox_rand_one(int D, const long dims[D], complex float* dst, double p)
{
	struct bart_rand_state worker_state = get_worker_state();

	NESTED(void, vec_fun, (long offset, long N, complex float* dst))
	{
		vec_philox_rand_one(worker_state, offset, N, dst, p);
	};

	md_sample_mpi(D, dims, dst, vec_fun);
}

void md_rand_one(int D, const long dims[D], complex float* dst, double p)
{
	if (use_obsolete_rng())
		md_obsolete_rand_one(D, dims, dst, p);
	else
		md_philox_rand_one(D, dims, dst, p);
}

#include "num/philox.inc"


