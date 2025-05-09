/* Copyright 2014-2015 The Regents of the University of California.
 * Copyright 2015-2019 Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2011-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Frank Ong <frankong@berkeley.edu>
 */

#include <math.h>
#include <complex.h>
#include <assert.h>
#include <string.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/specfun.h"
#include "num/multiplace.h"

#include "misc/nested.h"
#include "misc/misc.h"
#include "misc/version.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#include "noncart/gpu_grid.h"
#endif

#include "num/vptr.h"

#include "grid.h"


enum { kb_size_max = 1000 };
int kb_size = -1;
double bessel_kb_beta = -1.; // = bessel_i0(beta);

static float kb_table[kb_size_max + 1];
static double kb_beta = -1.;

void kb_init(double beta)
{
#pragma	omp critical
	if (-1 == kb_beta) {

		bessel_kb_beta = (use_compat_to_version("v0.8.00") ? bessel_i0_compat : bessel_i0)(beta);

		kb_size = use_compat_to_version("v0.8.00") ? 100 : kb_size_max;

		kb_precompute(beta, kb_size, kb_table);
		kb_beta = beta;
	}

	if (fabs(kb_beta - beta) / fabs(kb_beta) >= 1.E-6)
		error("Kaiser-Bessel window initialized with different beta (%e != %e)!\n", kb_beta, beta);
}

const struct multiplace_array_s* kb_get_table(double beta)
{
	kb_init(beta);

	return multiplace_move(1, MD_DIMS(kb_size + 1), FL_SIZE, kb_table);
}


static double kb(double beta, double x)
{
	if (fabs(x) >= 0.5)
		return 0.;

	return bessel_i0(beta * sqrt(1. - pow(2. * x, 2.))) / bessel_kb_beta;
}

static double kb_compat(double beta, double x)
{
	if (fabs(x) >= 0.5)
		return 0.;

	return bessel_i0_compat(beta * sqrt(1. - pow(2. * x, 2.))) / bessel_kb_beta;
}

void kb_precompute(double beta, int n, float table[n + 1])
{
	if (use_compat_to_version("v0.8.00")) {

		for (int i = 0; i < n + 1; i++)
			table[i] = kb_compat(beta, (double)(i) / (double)(n - 1) / 2.);
	} else  {

		for (int i = 0; i < n + 1; i++)
			table[i] = kb(beta, (double)(i) / (double)(n - 1) / 2.);
	}
}


static double ftkb(double beta, double x)
{
	double a = pow(beta, 2.) - pow(M_PI * x, 2.);

	if (0. == a)
		return 1. / bessel_kb_beta;

	if (a > 0)
		return (sinh(sqrt(a)) / sqrt(a)) / bessel_kb_beta;
	else
		return (sin(sqrt(-a)) / sqrt(-a)) / bessel_kb_beta;
}


static double rolloff(double x, double beta, double width)
{
	return 1. / ftkb(beta, x * width) / width;
}

// Linear interpolation
static float lerp(float a, float b, float c)
{
	return (1. - c) * a + c * b;
}

// Linear interpolation look up
static float intlookup(int n, const float table[n + 1], float x)
{
	float fpart;

//	fpart = modff(x * n, &ipart);
//	int index = ipart;

	x *= 2; // to evaluate kb(x) since table has kb(0.5 * n)

	int index = (int)(x * (n - 1));
	fpart = x * (n - 1) - (float)index;
#if 1
	assert(index >= 0);
	assert(index <= n);
	assert(fpart >= 0.);
	assert(fpart <= 1.);
#endif
	float l = lerp(table[index], table[index + 1], fpart);
#if 1
	assert(l <= 1.);
	assert(0 >= 0.);
#endif
	return l;
}



void gridH(const struct grid_conf_s* conf, const long ksp_dims[4], const long trj_strs[4], const complex float* traj, const long ksp_strs[4], complex float* dst, const long grid_dims[4], const long grid_strs[4], const complex float* grid)
{
	if (grid_dims[3] != ksp_dims[3])
		error("Adjoint gridding: ksp and grid are incompatible in dim 3 (%d != %d)!\n", ksp_dims[3], grid_dims[3]);
	
	assert(3 == ksp_dims[0]);
	assert(0 == ksp_strs[0]);
	assert(CFL_SIZE == trj_strs[0]);
	assert(0 == trj_strs[3]);

#ifdef USE_CUDA
	if (cuda_ondevice(traj))
		return cuda_gridH(conf, ksp_dims, trj_strs, traj, ksp_strs, dst, grid_dims, grid_strs, grid);
#endif

	long C = ksp_dims[3];

	// precompute kaiser bessel table
	kb_init(conf->beta);

#pragma omp parallel for collapse(2)
	for (int ir = 0; ir < ksp_dims[1]; ir++) {
		for (int ip = 0; ip < ksp_dims[2]; ip++) {

			long it = (ir * trj_strs[1] + ip * trj_strs[2]) / (long)CFL_SIZE;
			long ik = (ir * ksp_strs[1] + ip * ksp_strs[2]) / (long)CFL_SIZE;

			float pos[3];
			pos[0] = conf->os * (creal(traj[it + 0]) + conf->shift[0]);
			pos[1] = conf->os * (creal(traj[it + 1]) + conf->shift[1]);
			pos[2] = conf->os * (creal(traj[it + 2]) + conf->shift[2]);

			pos[0] += (grid_dims[0] > 1) ? ((float) grid_dims[0] / 2.) : 0.;
			pos[1] += (grid_dims[1] > 1) ? ((float) grid_dims[1] / 2.) : 0.;
			pos[2] += (grid_dims[2] > 1) ? ((float) grid_dims[2] / 2.) : 0.;

			complex float val[C];
			for (int j = 0; j < C; j++)
				val[j] = 0.0;
		
			grid_pointH(C, 3, grid_dims, grid_strs, pos, val, grid, conf->periodic, conf->width, kb_size, kb_table);

			for (int j = 0; j < ksp_dims[3]; j++)
				dst[j * ksp_strs[3] / (long)CFL_SIZE + ik] += val[j];
		}
	}
}


void grid(const struct grid_conf_s* conf, const long ksp_dims[4], const long trj_strs[4], const complex float* traj, const long grid_dims[4], const long grid_strs[4], complex float* grid, const long ksp_strs[4], const complex float* src)
{
	if (grid_dims[3] != ksp_dims[3])
		error("Gridding: ksp and grid are incompatible in dim 3 (%d != %d)!\n", ksp_dims[3], grid_dims[3]);

	assert(3 == ksp_dims[0]);
	assert(0 == ksp_strs[0]);
	assert(CFL_SIZE == trj_strs[0]);
	assert(0 == trj_strs[3]);

#ifdef USE_CUDA
	if (cuda_ondevice(traj))
		return cuda_grid(conf, ksp_dims, trj_strs, traj, grid_dims, grid_strs, grid, ksp_strs, src);
#endif

	long C = ksp_dims[3];

	// precompute kaiser bessel table
	kb_init(conf->beta);

	// grid
#pragma omp parallel for collapse(2)
	for (int ir = 0; ir < ksp_dims[1]; ir++) {
		for (int ip = 0; ip < ksp_dims[2]; ip++) {

			long it = (ir * trj_strs[1] + ip * trj_strs[2]) / (long)CFL_SIZE;
			long ik = (ir * ksp_strs[1] + ip * ksp_strs[2]) / (long)CFL_SIZE;

			float pos[3];
			pos[0] = conf->os * (creal(traj[it + 0]) + conf->shift[0]);
			pos[1] = conf->os * (creal(traj[it + 1]) + conf->shift[1]);
			pos[2] = conf->os * (creal(traj[it + 2]) + conf->shift[2]);

			pos[0] += (grid_dims[0] > 1) ? ((float) grid_dims[0] / 2.) : 0.;
			pos[1] += (grid_dims[1] > 1) ? ((float) grid_dims[1] / 2.) : 0.;
			pos[2] += (grid_dims[2] > 1) ? ((float) grid_dims[2] / 2.) : 0.;

			complex float val[C];
		
			bool skip = true;

			for (int j = 0; j < C; j++) {

				val[j] = src[j * ksp_strs[3] / (long)CFL_SIZE + ik];
				skip = skip && (0. == val[j]);
			}
			
			if (!skip)
				grid_point(C, 3, grid_dims, grid_strs, pos, grid, val, conf->periodic, conf->width, kb_size, kb_table);			
		}
	}
}


static void grid2_dims(int D, const long trj_dims[D], const long ksp_dims[D], const long grid_dims[D])
{
	assert(D >= 4);
	assert(md_check_compat(D - 3, ~0UL, grid_dims + 3, ksp_dims + 3));
//	assert(md_check_compat(D - 3, ~(MD_BIT(0) | MD_BIT(1)), trj_dims + 3, ksp_dims + 3));
	assert(md_check_compat(D - 3, ~0UL, trj_dims + 3, ksp_dims + 3));

	assert(3 == trj_dims[0]);
	assert(1 == trj_dims[3]);
	assert(1 == ksp_dims[0]);
}


void grid2(const struct grid_conf_s* conf, int D, const long trj_dims[D], const complex float* traj, const long grid_dims[D], complex float* dst, const long ksp_dims[D], const complex float* src)
{
	grid2_dims(D, trj_dims, ksp_dims, grid_dims);

	long ksp_strs[D];
	md_calc_strides(D, ksp_strs, ksp_dims, CFL_SIZE);

	long trj_strs[D];
	md_calc_strides(D, trj_strs, trj_dims, CFL_SIZE);

	long grid_strs[D];
	md_calc_strides(D, grid_strs, grid_dims, CFL_SIZE);

	long max_dims[D];
	md_max_dims(D, ~0UL, max_dims, ksp_dims, trj_dims);
	md_max_dims(D - 3, ~0UL, max_dims + 3, max_dims + 3, grid_dims + 3);

	unsigned long mpi_flags = vptr_block_loop_flags(D - 3, max_dims + 3, trj_strs + 3, traj, (size_t)(md_calc_size(3, trj_dims) * (long)CFL_SIZE))
				| vptr_block_loop_flags(D - 3, max_dims + 3, ksp_strs + 3, src, (size_t)(md_calc_size(3, ksp_dims) * (long)CFL_SIZE))
				| vptr_block_loop_flags(D - 3, max_dims + 3, grid_strs + 3, dst, (size_t)(md_calc_size(3, grid_dims) * (long)CFL_SIZE));
	mpi_flags <<= 3;

	if ((trj_strs[2] == trj_strs[1] * max_dims[1]) && (ksp_strs[2] == ksp_strs[1] * max_dims[1])) {

		max_dims[1] *= max_dims[2];
		max_dims[2] = 1;
	}

	for (int i = 4; i < D; i++) {

		if (MD_IS_SET(mpi_flags, i))
			continue;

		if (0 != grid_strs[i])
			continue;
		
		if ((trj_strs[i] == trj_strs[1] * max_dims[1]) && (ksp_strs[i] == ksp_strs[1] * max_dims[1])) {

			max_dims[1] *= max_dims[i];
			max_dims[i] = 1;
		}

		if (1 == max_dims[2]) {

			max_dims[2] = max_dims[i];
			trj_strs[2] = trj_strs[i];
			ksp_strs[2] = ksp_strs[i];
			max_dims[i] = 1;
		}

		if ((trj_strs[i] == trj_strs[2] * max_dims[2]) && (ksp_strs[i] == ksp_strs[2] * max_dims[2])) {

			max_dims[2] *= max_dims[i];
			max_dims[i] = 1;
		}
	}

	long tksp_dims[4];
	long tgrd_dims[4];
	md_select_dims(4, ~mpi_flags, tksp_dims, max_dims);
	md_select_dims(4, ~mpi_flags, tgrd_dims, grid_dims);
	if (!MD_IS_SET(mpi_flags, 3))
		max_dims[3] = 1;

	const long* ptr_grid_dims = &(tgrd_dims[0]);
	const long* ptr_ksp_dims = &(tksp_dims[0]);

	const long* ptr_ksp_strs = &(ksp_strs[0]);
	const long* ptr_trj_strs = &(trj_strs[0]);
	const long* ptr_grid_strs = &(grid_strs[0]);

	NESTED(void, nary_grid, (void* ptr[]))
	{
		const complex float* _trj = ptr[0];
		complex float* _grid = ptr[1];
		const complex float* _ksp = ptr[2];

		grid(conf, ptr_ksp_dims, ptr_trj_strs, _trj, ptr_grid_dims, ptr_grid_strs, _grid, ptr_ksp_strs, _ksp);
	};

	const long* strs[3] = { trj_strs + 3, grid_strs + 3, ksp_strs + 3 };
	void* ptr[3] = { (void*)traj, (void*)dst, (void*)src };
	unsigned long pflags = md_nontriv_dims(D - 3, grid_dims + 3);

#ifdef USE_CUDA
	if (cuda_ondevice(traj))
		pflags = 0;
#endif

	md_parallel_nary(3, D - 3, max_dims + 3, pflags, strs, ptr, nary_grid);
}


void grid2H(const struct grid_conf_s* conf, int D, const long trj_dims[D], const complex float* traj, const long ksp_dims[D], complex float* dst, const long grid_dims[D], const complex float* src)
{
	grid2_dims(D, trj_dims, ksp_dims, grid_dims);

	long ksp_strs[D];
	md_calc_strides(D, ksp_strs, ksp_dims, CFL_SIZE);

	long trj_strs[D];
	md_calc_strides(D, trj_strs, trj_dims, CFL_SIZE);

	long grid_strs[D];
	md_calc_strides(D, grid_strs, grid_dims, CFL_SIZE);

	long max_dims[D];
	md_max_dims(D, ~0UL, max_dims, ksp_dims, trj_dims);
	md_max_dims(D - 3, ~0UL, max_dims + 3, max_dims + 3, grid_dims + 3);

	unsigned long mpi_flags = vptr_block_loop_flags(D - 3, max_dims + 3, trj_strs + 3, traj, (size_t)(md_calc_size(3, trj_dims) * (long)CFL_SIZE))
				| vptr_block_loop_flags(D - 3, max_dims + 3, ksp_strs + 3, dst, (size_t)(md_calc_size(3, ksp_dims) * (long)CFL_SIZE))
				| vptr_block_loop_flags(D - 3, max_dims + 3, grid_strs + 3, src, (size_t)(md_calc_size(3, grid_dims) * (long)CFL_SIZE));

	mpi_flags <<= 3;

	if ((trj_strs[2] == trj_strs[1] * max_dims[1]) && (ksp_strs[2] == ksp_strs[1] * max_dims[1])) {

		max_dims[1] *= max_dims[2];
		max_dims[2] = 1;
	}

	for (int i = 4; i < D; i++) {

		if (MD_IS_SET(mpi_flags, i))
			continue;

		if (0 != grid_strs[i])
			continue;
		
		if ((trj_strs[i] == trj_strs[1] * max_dims[1]) && (ksp_strs[i] == ksp_strs[1] * max_dims[1])) {

			max_dims[1] *= max_dims[i];
			max_dims[i] = 1;
		}

		if (1 == max_dims[2]) {

			max_dims[2] = max_dims[i];
			trj_strs[2] = trj_strs[i];
			ksp_strs[2] = ksp_strs[i];
			max_dims[i] = 1;
		}

		if ((trj_strs[i] == trj_strs[2] * max_dims[2]) && (ksp_strs[i] == ksp_strs[2] * max_dims[2])) {

			max_dims[2] *= max_dims[i];
			max_dims[i] = 1;
		}
	}

	long tksp_dims[4];
	long tgrd_dims[4];
	md_select_dims(4, ~mpi_flags, tksp_dims, max_dims);
	md_select_dims(4, ~mpi_flags, tgrd_dims, grid_dims);
	if (!MD_IS_SET(mpi_flags, 3))
		max_dims[3] = 1;

	const long* ptr_grid_dims = &(tgrd_dims[0]);
	const long* ptr_ksp_dims = &(tksp_dims[0]);

	const long* ptr_ksp_strs = &(ksp_strs[0]);
	const long* ptr_trj_strs = &(trj_strs[0]);
	const long* ptr_grid_strs = &(grid_strs[0]);


	NESTED(void, nary_gridH, (void* ptr[]))
	{
		const complex float* _trj = ptr[0];
		const complex float* _grid = ptr[1];
		complex float* _ksp = ptr[2];

		gridH(conf, ptr_ksp_dims, ptr_trj_strs, _trj, ptr_ksp_strs, _ksp, ptr_grid_dims, ptr_grid_strs, _grid);
	};

	const long* strs[3] = { trj_strs + 3, grid_strs + 3, ksp_strs + 3 };
	void* ptr[3] = { (void*)traj, (void*)src, (void*)dst };
	unsigned long pflags = md_nontriv_dims(D - 3, grid_dims + 3);

#ifdef USE_CUDA
	if (cuda_ondevice(traj))
		pflags = 0;
#endif

	md_parallel_nary(3, D - 3, max_dims + 3, pflags, strs, ptr, nary_gridH);
}


typedef void CLOSURE_TYPE(grid_update_t)(long ind, float d);

#ifndef __clang__
#define VLA(x) x
#else
// blocks extension does not play well even with arguments which
// just look like variably-modified types
#define VLA(x)
#endif

static void grid_point_gen(int N, const long dims[VLA(N)], const long strs[VLA(N)], const float pos[VLA(N)], bool periodic, float width, int kb_size, const float kb_table[VLA(kb_size + 1)], grid_update_t update)
{
#ifndef __clang__
	int sti[N];
	int eni[N];
	int off[N];
#else
	// blocks extension does not play well with variably-modified types
	int* sti = alloca(sizeof(int[N]));
	int* eni = alloca(sizeof(int[N]));
	int* off = alloca(sizeof(int[N]));
#endif
	for (int j = 0; j < N; j++) {

		sti[j] = (int)ceil(pos[j] - 0.5 * width);
		eni[j] = (int)floor(pos[j] + 0.5 * width);
		off[j] = 0;

		if (sti[j] > eni[j])
			return;

		if (!periodic) {

			sti[j] = MAX(sti[j], 0);
			eni[j] = MIN(eni[j], dims[j] - 1);

		} else {

			while (sti[j] + off[j] < 0)
				off[j] += dims[j];
		}

		if (1 == dims[j]) {

			assert(0. == pos[j]); // ==0. fails nondeterministically for test_nufft_forward bbdec08cb
			sti[j] = 0;
			eni[j] = 0;
		}
	}

	__block NESTED(void, grid_point_r, (int N, long ind, float d))	// __block for recursion
	{
		if (0 == N) {

			NESTED_CALL(update, (ind, d));

		} else {

			N--;

			for (int w = sti[N]; w <= eni[N]; w++) {

				float frac = fabs(((float)w - pos[N]));
				float d2 = d * intlookup(kb_size, kb_table, frac / width);
				long ind2 = ind + ((w + off[N]) % dims[N]) * strs[N] / (long)CFL_SIZE;

				grid_point_r(N, ind2, d2);
			}
		}
	};

	grid_point_r(N, 0, 1.);
}



void grid_point(int ch, int N, const long dims[N], const long strs[N], const float pos[N], complex float* dst, const complex float val[ch], bool periodic, float width, int kb_size, const float kb_table[kb_size + 1])
{
	const long *_strs = strs;	// clang workaround
	const complex float *_val = val;

	NESTED(void, update, (long ind, float d))
	{
		const long *strs = _strs;
		const complex float *val = _val;

		for (int c = 0; c < ch; c++) {

			// we are allowed to update real and imaginary part independently which works atomically
#pragma 		omp atomic
			__real(dst[ind + c * strs[3] / (long)CFL_SIZE]) += __real(val[c]) * d;
#pragma 		omp atomic
			__imag(dst[ind + c * strs[3] / (long)CFL_SIZE]) += __imag(val[c]) * d;
		}
	};

	grid_point_gen(N, dims, strs, pos, periodic, width, kb_size, kb_table, update);
}



void grid_pointH(int ch, int N, const long dims[N], const long strs[N], const float pos[N], complex float val[ch], const complex float* src, bool periodic, float width, int kb_size, const float kb_table[kb_size + 1])
{
	const long *_strs = strs;	// clang workaround
	complex float *_val = val;

	NESTED(void, update, (long ind, float d))
	{
		const long *strs = _strs;
		complex float *val = _val;

		for (int c = 0; c < ch; c++) {

			__real(val[c]) += __real(src[ind + c * strs[3] / (long)CFL_SIZE]) * d;
			__imag(val[c]) += __imag(src[ind + c * strs[3] / (long)CFL_SIZE]) * d;
		}
	};

	grid_point_gen(N, dims, strs, pos, periodic, width, kb_size, kb_table, update);
}



double calc_beta(float os, float width)
{
	return M_PI * sqrt(pow((width / os) * (os - 0.5), 2.) - 0.8);
}


static float pos(int d, int i)
{
	return (1 == d) ? 0. : (((float)i - (float)d / 2.) / (float)d);
}


// width is defined in units of the oversampled grid
// use os=1  if dims correspond to the oversampled grid
// use os=os if dims correspond to the not oversampled grid
void rolloff_correction(float os, float width, float beta, const long dimensions[3], complex float* dst)
{
	// precompute kaiser bessel table
	kb_init(beta);

	double scale = 1.;
	
	if (use_compat_to_version("v0.8.00"))
		scale = pow(ftkb(beta, 0.) * width / 2, bitcount(md_nontriv_dims(3, dimensions)));

#pragma omp parallel for collapse(3)
	for (int z = 0; z < dimensions[2]; z++) 
		for (int y = 0; y < dimensions[1]; y++) 
			for (int x = 0; x < dimensions[0]; x++)
				dst[x + dimensions[0] * (y + z * dimensions[1])] 
					= (dimensions[0] > 1 ? rolloff(pos(dimensions[0], x) / os, beta, width) : 1.)
					* (dimensions[1] > 1 ? rolloff(pos(dimensions[1], y) / os, beta, width) : 1.)
					* (dimensions[2] > 1 ? rolloff(pos(dimensions[2], z) / os, beta, width) : 1.)
					* scale;
}

void apply_rolloff_correction2(float os, float width, float beta, int N, const long dims[N], const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	// precompute kaiser bessel table
	kb_init(beta);
	
	long size_bat = 1;
	long obstr = -1;	// batch stride, we support three dims with strides and one batch dim
	long ibstr = -1;	// batch stride, we support three dims with strides and one batch dim

	for (int i = 3; i < N; i++) {

		if (1 == dims[i])
			continue;

		assert((-1 == obstr) || ( (ostrs[i] == obstr * size_bat) && (istrs[i] == ibstr * size_bat)));

		if (-1 == obstr) {
			
			obstr = ostrs[i];
			ibstr = istrs[i];
		}
		
		size_bat *= dims[i];
	}

	obstr /= (long)CFL_SIZE;
	ibstr /= (long)CFL_SIZE;

#ifdef USE_CUDA

	assert(cuda_ondevice(dst) == cuda_ondevice(src));

	if (cuda_ondevice(dst)) {

		long dims_cuda[4] = { dims[0], dims[1], dims[2], md_calc_size(N - 3, dims + 3) };
		long ostrs_cuda[4] = { ostrs[0] / (long)CFL_SIZE, ostrs[1] / (long)CFL_SIZE, ostrs[2] / (long)CFL_SIZE, obstr };
		long istrs_cuda[4] = { istrs[0] / (long)CFL_SIZE, istrs[1] / (long)CFL_SIZE, istrs[2] / (long)CFL_SIZE, ibstr };

		cuda_apply_rolloff_correction2(os, width, beta, N, dims_cuda, ostrs_cuda, dst, istrs_cuda, src);

		if (use_compat_to_version("v0.8.00")) {

			float scale = powf(ftkb(beta, 0) * width / 2, bitcount(md_nontriv_dims(3, dims)));
			md_zsmul2(N, dims, ostrs, dst, ostrs, dst, scale);
		}

		return;
	}
#endif

#pragma omp parallel for collapse(3)
	for (int z = 0; z < dims[2]; z++) {
		for (int y = 0; y < dims[1]; y++) {
			for (int x = 0; x < dims[0]; x++) {

				long oidx = (x * ostrs[0] + y * ostrs[1] + z * ostrs[2]) / (long)CFL_SIZE;
				long iidx = (x * istrs[0] + y * istrs[1] + z * istrs[2]) / (long)CFL_SIZE;

				float val = (dims[0] > 1 ? rolloff(pos(dims[0], x) / os, beta, width) : 1)
					  * (dims[1] > 1 ? rolloff(pos(dims[1], y) / os, beta, width) : 1)
					  * (dims[2] > 1 ? rolloff(pos(dims[2], z) / os, beta, width) : 1);

				for (long i = 0; i < size_bat; i++)
					dst[oidx + i * obstr] = val * src[iidx + i * ibstr];
			}
		}
	}

	if (use_compat_to_version("v0.8.00")) {

		float scale = powf(ftkb(beta, 0) * width / 2, bitcount(md_nontriv_dims(3, dims)));
		md_zsmul2(N, dims, ostrs, dst, ostrs, dst, scale);
	}
}

void apply_rolloff_correction(float os, float width, float beta, int N, const long dims[N], complex float* dst, const complex float* src)
{
	apply_rolloff_correction2(os, width, beta, N, dims, MD_STRIDES(N, dims, CFL_SIZE), dst, MD_STRIDES(N, dims, CFL_SIZE), src);
}




