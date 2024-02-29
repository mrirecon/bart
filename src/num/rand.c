/* Copyright 2013. The Regents of the University of California.
 * Copyright 2021. Uecker Lab. University Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Martin Uecker, Dara Bahri, Moritz Blumenthal
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <limits.h>
#include <stdint.h>

#ifdef _WIN32
#include "win/rand_r.h"
#endif

#include "num/multind.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "rand.h"

static bool use_obsolete_rng()
{
	return true;
}



struct bart_rand_state {
	unsigned int num_rand_seed;
	uint64_t state[4];
};

static struct bart_rand_state global_rand_state =  { .num_rand_seed = 123 };

void rand_state_update(struct bart_rand_state* state, unsigned long long seed)
{
	if (use_obsolete_rng()) {

		if (0 == seed) // special case to preserve old behavior
			state->num_rand_seed = 123;
		else
			state->num_rand_seed = seed;
	} else {

		error("Not yet implemented!\n");
	}
}

struct bart_rand_state* rand_state_create(unsigned long long seed)
{
	struct bart_rand_state* state = malloc(sizeof *state);
	memset(state, 0, sizeof *state);
	rand_state_update(state, seed);
	return state;
}



void num_rand_init(unsigned long long seed)
{
	assert(seed <= UINT_MAX);
	rand_state_update(&global_rand_state, seed);
}

static uint32_t rand32_state(struct bart_rand_state* state)
{
	uint32_t r;
#pragma omp critical
	r = rand_r(&state->num_rand_seed);
	return r;
}

static uint32_t rand32(void)
{
	return rand32_state(&global_rand_state);
}

double uniform_rand(void)
{
	return rand32() / (double)RAND_MAX;
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

			uint32_t x = rand32_state(state);
			m = (uint64_t) x * (uint64_t) range;
			l = (uint32_t) m;
		} while (l < t);

		return m >> 32;
	} else {

		// Division with rejection, for use with rand_r, as Lemire's method needs a 32-bit PRNG
		uint32_t divisor = RAND_MAX / (range);
		uint32_t retval;

		do {
			retval = rand32_state(state) / divisor;

		} while (retval >= range);

		return retval;
	}
}



unsigned int rand_range(unsigned int range)
{
	return rand_range_state(&global_rand_state, range);
}


/**
 * Box-Muller
 */
complex double gaussian_rand(void)
{
	double u1, u2, s;

 	do {

		u1 = 2. * uniform_rand() - 1.;
		u2 = 2. * uniform_rand() - 1.;
   		s = u1 * u1 + u2 * u2;

   	} while (s > 1.);

	double re = sqrt(-2. * log(s) / s) * u1;
	double im = sqrt(-2. * log(s) / s) * u2;

	return re + 1.i * im;
}

void gaussian_rand_vec(long N, float* dst)
{
	complex float* tmp = md_alloc(1, MD_DIMS(N / 2 + 1), sizeof(complex float));
	md_gaussian_rand(1, MD_DIMS(N / 2 + 1), tmp);
	md_copy(1, MD_DIMS(N), dst, tmp, sizeof(float));
	md_free(tmp);
	//This does not need to be scaled as md_gaussian_rand has (complex) variance 2!
}

void md_gaussian_rand(int D, const long dims[D], complex float* dst)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		complex float* tmp = md_alloc(D, dims, sizeof(complex float));
		md_gaussian_rand(D, dims, tmp);
		md_copy(D, dims, dst, tmp, sizeof(complex float));
		md_free(tmp);
		return;
	}
#endif
//#pragma omp parallel for
	for (long i = 0; i < md_calc_size(D, dims); i++)
		dst[i] = (complex float)gaussian_rand();
}

void md_uniform_rand(int D, const long dims[D], complex float* dst)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		complex float* tmp = md_alloc(D, dims, sizeof(complex float));
		md_uniform_rand(D, dims, tmp);
		md_copy(D, dims, dst, tmp, sizeof(complex float));
		md_free(tmp);
		return;
	}
#endif
	for (long i = 0; i < md_calc_size(D, dims); i++)
		dst[i] = (complex float)uniform_rand();
}

void md_rand_one(int D, const long dims[D], complex float* dst, double p)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		complex float* tmp = md_alloc(D, dims, sizeof(complex float));
		md_rand_one(D, dims, tmp, p);
		md_copy(D, dims, dst, tmp, sizeof(complex float));
		md_free(tmp);
		return;
	}
#endif
	for (long i = 0; i < md_calc_size(D, dims); i++)
		dst[i] = (complex float)(uniform_rand() < p);
}
