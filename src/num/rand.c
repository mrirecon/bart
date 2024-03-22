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

#ifdef _WIN32
#include "win/rand_r.h"
#endif

#include "misc/version.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "num/multind.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#include "num/gpurand.h"
#endif

#include "rand.h"


static bool warned_about_parallel_rng = false;


static bool use_obsolete_rng()
{
	if (!warned_about_parallel_rng) {

		if (cfl_loop_desc_active())
			debug_printf(DP_WARN, "Random numbers will be same in each parallel process / thread (exactly as if calling the tools sequentially)!\n");

		warned_about_parallel_rng = true;
	}

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
		state->num_rand_seed = (unsigned int) seed; // truncate here, as rand_r cannot use 64 bits of state


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
	struct bart_rand_state* state = malloc(sizeof *state);
	rand_state_update(state, seed);
	return state;
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

static double uniform_rand_obsolete()
{
	struct bart_rand_state* state = &global_rand_state[cfl_loop_worker_id()];
	return rand_r(&state->num_rand_seed) / (double)RAND_MAX;
}



double uniform_rand(void)
{
	double r;

	if (use_obsolete_rng()) {

#pragma 	omp critical(global_rand_state)
		r = uniform_rand_obsolete();

	} else {

#pragma 	omp critical(global_rand_state)
		r = uniform_rand_state(&global_rand_state[cfl_loop_worker_id()]);
	}
	return r;
}




unsigned int rand_range_state(struct bart_rand_state* state, unsigned int range)
{
	static_assert(sizeof(unsigned int) == sizeof(uint32_t), "unsigned int is not 32 bits!\n");

	if (!use_obsolete_rng()) {

		// Lemire's Method, see https://arxiv.org/abs/1805.10941
		// Adapted for 32-bit integers, and written as do { ... } while();
		// Generates random number in range [0,range)

		uint32_t t = (-range) % range;
		uint64_t m;
		uint32_t l;

		do {

			uint32_t x = rand64_state(state);
			m = (uint64_t) x * (uint64_t) range;
			l = (uint32_t) m;
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

#pragma omp critical(global_rand_state)
	r = rand_range_state(&global_rand_state[cfl_loop_worker_id()], range);

	return r;
}


/**
 * Box-Muller
 */

typedef double CLOSURE_TYPE(uniform_rand_t)(void);

static complex double gaussian_rand_func(uniform_rand_t func)
{
	double u1, u2, s;

	do {
		u1 = 2. * NESTED_CALL(func, ()) - 1.;
		u2 = 2. * NESTED_CALL(func, ()) - 1.;
		s = u1 * u1 + u2 * u2;

	} while (s > 1.);

	double re = sqrt(-2. * log(s) / s) * u1;
	double im = sqrt(-2. * log(s) / s) * u2;

	return re + 1.i * im;
}

static complex double gaussian_rand_obsolete(void)
{

	NESTED(double, uniform_rand_obsolete_wrapper, (void))
	{
		double r;
#pragma 	omp critical(global_rand_state)
		r = uniform_rand_obsolete();
		return r;
	};

	return gaussian_rand_func(uniform_rand_obsolete_wrapper);
}

static complex double gaussian_rand_state(struct bart_rand_state* state)
{
#if 0
	NESTED(double, uniform_rand_state_closure, (void))
	{
		return uniform_rand_state(state);
	};

	return gaussian_rand_func(uniform_rand_state_closure);
#else

	double u1, u2, s;
	uint64_t out[2];

	do {
		philox_4x32(state->state, state->ctr1, state->ctr2, out);
		state->ctr1++;

		u1 = 2. * ull2double(out[0]) - 1.;
		u2 = 2. * ull2double(out[1]) - 1.;
		s = u1 * u1 + u2 * u2;

	} while (s > 1.);

	double re = sqrt(-2. * log(s) / s) * u1;
	double im = sqrt(-2. * log(s) / s) * u2;

	return re + 1.i * im;


#endif
}

complex double gaussian_rand(void)
{
	complex double r;

	if (use_obsolete_rng()) {

		r = gaussian_rand_obsolete();
	} else {

#pragma 	omp critical(global_rand_state)
		r = gaussian_rand_state(&global_rand_state[cfl_loop_worker_id()]);
	}
	return r;

}




void gaussian_rand_vec(long N, float* dst)
{
	complex float* tmp = md_alloc(1, MD_DIMS(N / 2 + 1), sizeof(complex float));
	md_gaussian_rand(1, MD_DIMS(N / 2 + 1), tmp);
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
	for (long i = 0; i < md_calc_size(D, dims); i++)
		dst[i] = (complex float)gaussian_rand_obsolete();
}

static void md_gaussian_philox_rand(int D, const long dims[D], complex float* dst)
{

	struct bart_rand_state worker_state;
	#pragma omp critical(global_rand_state)
	{
		worker_state = global_rand_state[cfl_loop_worker_id()];
		global_rand_state[cfl_loop_worker_id()].ctr1++;
	}


#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		cuda_gaussian_rand(md_calc_size(D, dims), dst, worker_state.state, worker_state.ctr1);
	} else
#endif
	{

#pragma 	omp parallel for
		for (long i = 0; i < md_calc_size(D, dims); i++) {

			struct bart_rand_state state = worker_state;
			state.ctr2 = (uint64_t) i;
			dst[i] = gaussian_rand_state(&state);
		}
	}
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
	for (long i = 0; i < md_calc_size(D, dims); i++)
		dst[i] = (complex float)uniform_rand_obsolete();
}

static void md_uniform_philox_rand(int D, const long dims[D], complex float* dst)
{

	struct bart_rand_state worker_state;
	#pragma omp critical(global_rand_state)
	{
		worker_state = global_rand_state[cfl_loop_worker_id()];
		global_rand_state[cfl_loop_worker_id()].ctr1++;
	}

#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		cuda_uniform_rand(md_calc_size(D, dims), dst, worker_state.state, worker_state.ctr1);
	} else
#endif
	{

#pragma 	omp parallel for
		for (long i = 0; i < md_calc_size(D, dims); i++) {

			struct bart_rand_state state = worker_state;
			state.ctr2 = (uint64_t) i;
			dst[i] = (complex float)(uniform_rand_state(&state));
		}
	}
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
	for (long i = 0; i < md_calc_size(D, dims); i++)
		dst[i] = (complex float)(uniform_rand_obsolete() < p);
}

static void md_philox_rand_one(int D, const long dims[D], complex float* dst, double p)
{

	struct bart_rand_state worker_state;
	#pragma omp critical(global_rand_state)
	{
		worker_state = global_rand_state[cfl_loop_worker_id()];
		global_rand_state[cfl_loop_worker_id()].ctr1++;
	}

#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		cuda_rand_one(md_calc_size(D, dims), dst, p, worker_state.state, worker_state.ctr1);
	} else
#endif
	{

#pragma 	omp parallel for
		for (long i = 0; i < md_calc_size(D, dims); i++) {

			struct bart_rand_state state = worker_state;
			state.ctr2 = (uint64_t) i;
			dst[i] = (complex float)(uniform_rand_state(&state) < p);
		}
	}
}

void md_rand_one(int D, const long dims[D], complex float* dst, double p)
{
	if (use_obsolete_rng())
		md_obsolete_rand_one(D, dims, dst, p);
	else
		md_philox_rand_one(D, dims, dst, p);
}

#include "num/philox.inc"


