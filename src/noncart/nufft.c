/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2022-2023. Institute of Biomedical Imaging. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2017 Frank Ong
 * 2014-2022 Martin Uecker
 * 2018      Sebastian Rosenzweig
 *
 */

#include <math.h>
#include <complex.h>
#include <assert.h>
#include <stdbool.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/types.h"
#include "misc/version.h"
#include "misc/opts.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"
#include "num/fft.h"
#include "num/shuffle.h"
#include "num/ops.h"
#include "num/multiplace.h"
#include "num/vptr.h"
#include "num/triagmat.h"
#include "num/compress.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#include "noncart/gpu_grid.h"
#endif

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/fmac.h"

#include "noncart/grid.h"
#include "noncart/nufft_chain.h"

#include "nufft.h"

#define FFT_FLAGS (MD_BIT(0)|MD_BIT(1)|MD_BIT(2))

struct nufft_conf_s nufft_conf_defaults = {

	.toeplitz = true,
	.pcycle = false,
	.periodic = false,
	.lowmem = false,
	.loopdim = -1,
	.flags = FFT_FLAGS,
	.cfft = 0u,
	.decomp = true,
	.nopsf = false,
	.upper_triag = false,
	.real = false,
	.compress_psf = false,
	.precomp = true,
	.precomp_linphase = true,
	.precomp_fftmod = true,
	.precomp_roll = true,
	.zero_overhead = false,
	.width = 6,
	.os = 2.,

};

struct nufft_conf_s nufft_conf_options = {

	.toeplitz = true,
	.pcycle = false,
	.periodic = false,
	.lowmem = false,
	.loopdim = -1,
	.flags = FFT_FLAGS,
	.cfft = 0u,
	.decomp = true,
	.nopsf = false,
	.upper_triag = false,
	.real = false,
	.compress_psf = false,
	.precomp = true,
	.precomp_linphase = true,
	.precomp_fftmod = true,
	.precomp_roll = true,
	.zero_overhead = false,
	.width = 6,
	.os = 2.,

};

struct opt_s nufft_conf_opts[] = {

	OPTL_CLEAR('r', "no-toeplitz", &(nufft_conf_options.toeplitz), "turn-off Toeplitz embedding for inverse NUFFT"),
	OPTL_SET('P', "periodic", &(nufft_conf_options.periodic), "periodic k-space"),
	OPT_CLEAR('1', &(nufft_conf_options.decomp), "(use/return oversampled grid)"),
	OPTL_SET(0, "lowmem", &(nufft_conf_options.lowmem), "(use low-mem mode of the nuFFT)"),
	OPTL_SET(0, "zero-mem", &(nufft_conf_options.zero_overhead), "(use zero-overhead mode of the nuFFT)"),
	OPTL_CLEAR(0, "no-precomp", &(nufft_conf_options.precomp), "don't precompute linphases, rolloff, and fftmod"),
	OPTL_FLOAT('o', "oversampling", &(nufft_conf_options.os), "o", "(oversample grid by factor (default: o=2; required for Toeplitz))"),
	OPTL_FLOAT('w', "width", &(nufft_conf_options.width), "w", "(width of Kaiser-Bessel window on oversampled grid (default: w=6))"),
	OPTL_SET(0, "real-psf", &(nufft_conf_options.real), "only store real part of PSF (lower memory usage and faster in some cases)"),
	OPTL_SET(0, "compress-psf", &(nufft_conf_options.compress_psf), "only store non-zero entries of PSF (lower memory usage and faster in some cases)"),
	OPTL_SET(0, "upper-triag-psf", &(nufft_conf_options.upper_triag), "store only upper triangular part of PSF for subspace (lower memory usage and faster in some cases)"),
};

int N_nufft_conf_opts = ARRAY_SIZE(nufft_conf_opts);

#include "nufft_priv.h"

DEF_TYPEID(nufft_data);

static void compute_factors(int N, unsigned long flags, long factors[N], const long dims[N])
{
	flags = flags & md_nontriv_dims(N, dims);

	for (int i = 0; i < N; i++)
		factors[i] = (MD_IS_SET(flags, i)) ? 2 : 1;
}

static void compute_shift(int NS, float shift[NS], int N, const long factors[N], int idx)
{
	assert(NS <=N);

	for (int i = 0; i < NS; i++) {

		shift[i] = -(float)(idx % factors[i]) / factors[i];
		idx /= factors[i];
	}

	assert(0 == idx);

	for (int i = NS; i < N; i++)
		assert(1 == factors[i]);
}

static struct grid_conf_s compute_grid_conf_decomp(int N, const long factors[N], struct grid_conf_s grid, int idx)
{
	struct grid_conf_s ret = grid;
	ret.width /= 2.;
	ret.os = 1.;

	compute_shift(3, ret.shift, N, factors, idx);
	return ret;
}


static void grid2_decomp(struct grid_conf_s* _conf, int idx, int N, const long factors[N],
			const long trj_dims[N], const complex float* traj,
			const long cim_dims[N], complex float* grid,
			const long ksp_dims[N],  const complex float* ksp)
{

	struct grid_conf_s conf = compute_grid_conf_decomp(N, factors, *_conf, idx);

	for (int i = 0; i < 3; i++)
		if (1 < factors[i])
			conf.shift[i] += (cim_dims[i] / 2.0 - cim_dims[i] / 2) / conf.os;

	grid2(&conf, N, trj_dims, traj, cim_dims, grid, ksp_dims, ksp);

}

static void grid2H_decomp(struct grid_conf_s* _conf, int idx, int N, const long factors[N],
			const long trj_dims[N], const complex float* traj,
			const long ksp_dims[N], complex float* ksp,
			const long cim_dims[N], const complex float* grid)
{

	struct grid_conf_s conf = compute_grid_conf_decomp(N, factors, *_conf, idx);

	for (int i = 0; i < 3; i++)
		if (1 < factors[i])
			conf.shift[i] += (cim_dims[i] / 2.0 - cim_dims[i] / 2) / conf.os;

	grid2H(&conf, N, trj_dims, traj, ksp_dims, ksp, cim_dims, grid);

}







static complex float* compute_linphases(int N, long lph_dims[N + 1], unsigned long flags, const long img_dims[N + 1])
{
	int T = bitcount(flags);
	assert(0 <= T && T < 31);
	float shifts[1 << T][T];

	int s = 0;
	for (unsigned long i = 0; i < (1ul << T); i++) {

		bool skip = false;

		for (int j = 0; j < T; j++) {

			shifts[s][j] = 0.;

			if (MD_IS_SET(i, j)) {

				skip = skip || (1 == img_dims[j]);
				shifts[s][j] = -0.5;
			}
		}

		if (!skip)
			s++;
	}

	int ND = N + 1;
	md_select_dims(ND, flags, lph_dims, img_dims);
	lph_dims[N] = s;

	complex float* linphase = md_alloc(ND, lph_dims, CFL_SIZE);

	//#pragma omp parallel for shared(linphase)
	for (int i = 0; i < s; i++) {

		float shifts2[ND];
		for (int j = 0; j < ND; j++)
			shifts2[j] = 0.;

		for (int j = 0, t = 0; j < N; j++)
			if (MD_IS_SET(flags, j))
				shifts2[j] = shifts[i][t++];

		for (int j = 0; j < N; j++)
			if (MD_IS_SET(flags, j) && 1 < img_dims[j])
				shifts2[j] += (img_dims[j] / 2. - img_dims[j] / 2);

		linear_phase(ND, img_dims, shifts2,
				linphase + i * md_calc_size(ND, img_dims));
	}

	return linphase;
}

static void apply_linphases_3D(int N, const long img_dims[N], const float _shifts[3], complex float* dst, const complex float* src, bool conj, bool fmac, bool fftm, float scale)
{
#ifdef USE_CUDA
	assert(cuda_ondevice(dst) == cuda_ondevice(src));

	if (cuda_ondevice(dst)) {

		cuda_apply_linphases_3D(N, img_dims, _shifts, dst, src, conj, fmac, fftm, scale);
		return;
	}
#endif

	float shifts[3];

	for (int i = 0; i < 3; i++) {

		shifts[i] = _shifts[i];

		if (1 < img_dims[i])
			shifts[i] += (img_dims[i] / 2. - img_dims[i] / 2);
	}

	double shifts2[3];

	for (int n = 0; n < 3; n++)
		shifts2[n] = 2. * M_PI * (double)(shifts[n]) / ((double)img_dims[n]);

	double cn = 0.;

	for (int n = 0; n < 3; n++)
		cn -= shifts2[n] * (double)img_dims[n] / 2.;

	for (int n = 0; fftm && (n < 3); n++) {

		long c = img_dims[n] / 2;
		double shift = (double)c / (double)img_dims[n];

		cn -= 2. * M_PI * (double)c / 2. * shift;
		shifts2[n] += 2. * M_PI * shift;
	}

	long tot = md_calc_size(N - 3, img_dims + 3);

#pragma omp parallel for collapse(3)
	for (long z = 0; z < img_dims[2]; z++) {
		for (long y = 0; y < img_dims[1]; y++) {
			for (long x = 0; x < img_dims[0]; x++) {

				long offset = x + y * img_dims[0] + z * img_dims[0] * img_dims[1];
				long pos[3] = {x, y, z};

				double val = cn;

				for (int n = 0; n < 3; n++)
					val += pos[n] * shifts2[n];

				complex float val2 = scale * cexp(1.I * val);

				if (conj)
					val2 = conjf(val2);

				if (fmac) {

					for (long i = 0; i < tot; i++)
						dst[offset + i * img_dims[0] * img_dims[1] * img_dims[2]] += val2 * src[offset + i * img_dims[0] * img_dims[1] * img_dims[2]];
				} else {

					for (long i = 0; i < tot; i++)
						dst[offset + i * img_dims[0] * img_dims[1] * img_dims[2]] = val2 * src[offset + i * img_dims[0] * img_dims[1] * img_dims[2]];
				}
			}
		}
	}
}


static void linphase_decomp(int N, const long factors[N],
			    const long cml_dims[N],
			    const long cml_strs[N], complex float* grid,
			    const long cim_strs[N], const complex float* cim)
{
	long lph_dims[N];
	md_select_dims(N, md_nontriv_dims(N, factors), lph_dims, cml_dims);

	float scale = 1. / sqrtf(md_calc_size(N, lph_dims));
	long pos[N];
	md_set_dims(N, pos, 0);

	do {
		long idx = md_ravel_index(N, pos, ~0UL, factors);
		float shift[3];
		for (int i = 0; i < 3; i++)
			shift[i] = -(float)(pos[i]) / (float)factors[i];

		long cim_dims[N];
		md_select_dims(N, ~MD_BIT(N - 1), cim_dims, cml_dims);

		long pos_acc[N];
		md_set_dims(N, pos_acc, 0);
		pos_acc[N - 1] = idx;

		apply_linphases_3D(N, cim_dims, shift, &MD_ACCESS(N, cml_strs, pos_acc, grid), &MD_ACCESS(N, cim_strs, pos_acc, cim), false, false, true, scale);

	} while (md_next(N, factors, ~0UL, pos));
}

static void linphaseH_decomp(int N, const long factors[N],
			     const long cml_dims[N],
			     const long cim_strs[N], complex float* cim,
			     const long cml_strs[N], const complex float* grid)
{
	long lph_dims[N];
	md_select_dims(N, md_nontriv_dims(N, factors), lph_dims, cml_dims);

	float scale = 1. / sqrtf(md_calc_size(N, lph_dims));

	long pos[N];
	md_set_dims(N, pos, 0);

	md_clear2(N, cml_dims, cim_strs, cim, CFL_SIZE);

	do {
		long idx = md_ravel_index(N, pos, ~0UL, factors);
		float shift[3];
		for (int i = 0; i < 3; i++)
			shift[i] = -(float)(pos[i]) / (float)factors[i];

		long cim_dims[N];
		md_select_dims(N, ~MD_BIT(N - 1), cim_dims, cml_dims);

		long pos_acc[N];
		md_set_dims(N, pos_acc, 0);
		pos_acc[N - 1] = idx;

		apply_linphases_3D(N, cim_dims, shift, &MD_ACCESS(N, cim_strs, pos_acc, cim), &MD_ACCESS(N, cml_strs, pos_acc, grid), true, true, true, scale);

	} while (md_next(N, factors, ~0UL, pos));
}




static complex float* compute_square_basis(bool upper_triag, int N, long sqr_bas_dims[N], const long bas_dims[N], const complex float* basis, const long ksp_dims[N])
{
	if (NULL == basis) {

		md_singleton_dims(N, sqr_bas_dims);
		return NULL;
	}

	assert(1 == bas_dims[7]);
	long bas_dimsT[N];

	md_transpose_dims(N, 6, 7, bas_dimsT, bas_dims);
	md_max_dims(N, ~0UL, sqr_bas_dims, bas_dims, bas_dimsT);
	sqr_bas_dims[5] = ksp_dims[5];

	complex float* sqr_basis = md_alloc_sameplace(N, sqr_bas_dims, CFL_SIZE, basis);
	md_ztenmulc(N, sqr_bas_dims, sqr_basis, bas_dims, basis, bas_dimsT, basis);

	sqr_bas_dims[6] *= sqr_bas_dims[6];
	sqr_bas_dims[7] = 1;

	if (use_compat_to_version("v0.7.00"))
		md_zsmul(N, sqr_bas_dims, sqr_basis, sqr_basis, (double)bas_dims[6]);

	if (upper_triag) {

		long sqr_bas_dims2[N];
		complex float* sqr_basis2 = hermite_to_uppertriag(6, 6, 6, N, sqr_bas_dims2, sqr_bas_dims, sqr_basis);

		md_free(sqr_basis);
		sqr_basis = sqr_basis2;

		md_copy_dims(N, sqr_bas_dims, sqr_bas_dims2);
	}

	return sqr_basis;
}

static complex float* compute_square_weights(int N, const long wgh_dims[N], const complex float* weights)
{
	if (NULL == weights)
		return NULL;

	complex float* sqr_weights = md_alloc_sameplace(N, wgh_dims, CFL_SIZE, weights);
	md_zmulc(N, wgh_dims, sqr_weights, weights, weights);

	return sqr_weights;
}

static struct nufft_conf_s compute_psf_nufft_conf(bool periodic, bool lowmem, bool vptr)
{
	struct nufft_conf_s conf = nufft_conf_defaults;
	conf.periodic = periodic;
	conf.toeplitz = false;	// avoid infinite loop
	conf.lowmem = lowmem;

	//FIXME: do we need this? It was old behavior, but not sure if it is relevant for reproducibility.
	conf.precomp_linphase = vptr || use_compat_to_version("v0.8.00");
	conf.precomp_roll = vptr || use_compat_to_version("v0.8.00");
	conf.precomp_fftmod = vptr || use_compat_to_version("v0.8.00");

	return conf;
}


complex float* compute_psf(int N, const long img_dims[N], const long trj_dims[N], const complex float* traj,
				const long bas_dims[N], const complex float* basis,
				const long wgh_dims[N], const complex float* weights,
				bool periodic, bool lowmem)
{
	long ksp_dims[N];
	md_select_dims(N, ~MD_BIT(0), ksp_dims, trj_dims);

	if (NULL != weights)
		md_max_dims(N, ~0UL, ksp_dims, ksp_dims, wgh_dims);

	long sqr_bas_dims[N];

	complex float* sqr_basis = compute_square_basis(false, N, sqr_bas_dims, bas_dims, basis, ksp_dims);
	complex float* sqr_weights = compute_square_weights(N, wgh_dims, weights);

	long img_dims2[N];
	md_copy_dims(N, img_dims2, img_dims);

	if (NULL != sqr_basis) {

		img_dims2[6] *= img_dims2[6];
		img_dims2[5] = 1;
	}

	complex float* psf = md_alloc_sameplace(N, img_dims, CFL_SIZE, traj);

	complex float* ones = md_alloc_sameplace(N, ksp_dims, CFL_SIZE, traj);
	md_zfill(N, ksp_dims, ones, 1.);

	struct nufft_conf_s conf = compute_psf_nufft_conf(periodic, lowmem, is_vptr(traj));

	struct linop_s* lop_nufft = nufft_create2(N, ksp_dims, img_dims2, trj_dims, traj, wgh_dims, sqr_weights, sqr_bas_dims, sqr_basis, conf);
	lop_nufft = linop_reshape_in_F(lop_nufft, N, img_dims);

	md_free(sqr_weights);
	md_free(sqr_basis);

	linop_adjoint(lop_nufft, N, img_dims, psf, N, ksp_dims, ones);
	linop_free(lop_nufft);

	md_free(ones);


	return psf;
}


// This function computes decompose(fftuc(nufft^H(1; 2*traj)) on the factor 2 oversampled grid
// It computes the even and off frequencies independently and is hence more memory efficient
complex float* compute_psf2_decomposed(int N, const long psf_dims[N + 1], unsigned long flags, const long trj_dims[N + 1], const complex float* traj,
				const long bas_dims[N + 1], const complex float* basis, const long wgh_dims[N + 1], const complex float* weights,
				bool periodic, bool lowmem, bool upper_triag)
{
	long ksp_dims[N + 1];
	md_select_dims(N + 1, ~MD_BIT(0), ksp_dims, trj_dims);
	ksp_dims[N] = psf_dims[N];

	if (NULL != weights)
		md_max_dims(N + 1, ~0UL, ksp_dims, ksp_dims, wgh_dims);

	long sqr_bas_dims[N + 1];

	complex float* sqr_basis = compute_square_basis(upper_triag, N + 1, sqr_bas_dims, bas_dims, basis, ksp_dims);
	complex float* sqr_weights = compute_square_weights(N + 1, wgh_dims, weights);

	long psf_dims2[N + 1];
	md_copy_dims(N + 1, psf_dims2, psf_dims);

	if (upper_triag) {

		assert(1 == psf_dims2[5]);
	} else {

		if (NULL != sqr_basis) {

			psf_dims2[6] *= psf_dims2[6];
			psf_dims2[5] = 1;
		}
	}

	struct nufft_conf_s conf = compute_psf_nufft_conf(periodic, lowmem, is_vptr(traj));

	//Workaround, will be fixed with vptr_fun
	if (is_vptr(traj)) {

		conf.precomp_fftmod = true;
		conf.precomp_linphase = true;
		conf.precomp_roll = true;
	}

	struct linop_s* lop_nufft;

	if (lowmem) {

		long ksp_dims2[N + 1];
		long psf_dims3[N + 1];
		long loop_dims[N + 1];

		md_select_dims(N + 1, ~MD_BIT(N), ksp_dims2, ksp_dims);
		md_select_dims(N + 1, ~MD_BIT(N), psf_dims3, psf_dims2);
		md_select_dims(N + 1,  MD_BIT(N), loop_dims, psf_dims2);

		lop_nufft = nufft_create2(N + 1, ksp_dims2, psf_dims3, trj_dims, traj, wgh_dims, sqr_weights, sqr_bas_dims, sqr_basis, conf);
		lop_nufft = linop_loop_F(N + 1, loop_dims, lop_nufft);
	} else {

		lop_nufft = nufft_create2(N + 1, ksp_dims, psf_dims2, trj_dims, traj, wgh_dims, sqr_weights, sqr_bas_dims, sqr_basis, conf);
	}

	lop_nufft = linop_reshape_in_F(lop_nufft, N + 1, psf_dims);

	md_free(sqr_weights);
	md_free(sqr_basis);

	complex float* kern = md_alloc_sameplace(N + 1, ksp_dims, CFL_SIZE, traj);
	md_zfill(N + 1, ksp_dims, kern, 1.);

	long factors[N + 1];
	compute_factors(N + 1, flags, factors, psf_dims);

	long pos[N + 1];
	md_set_dims(N + 1, pos, 0);

	for (int i = 0, j = 0; i < N; i++) {

		if (!(MD_IS_SET(flags, i) && (1 < psf_dims[i])))
			continue;

		complex float* tkern = md_alloc_sameplace(N + 1, ksp_dims, CFL_SIZE, traj);
		md_copy2(N + 1, ksp_dims, MD_STRIDES(N + 1, ksp_dims, CFL_SIZE), tkern, MD_STRIDES(N + 1, trj_dims, CFL_SIZE), traj + j, CFL_SIZE);

		md_zsmul(N + 1, ksp_dims, tkern, tkern, M_PI);

		for (int k = 0; k < psf_dims[N]; k++) {

			float shift[3];
			compute_shift(3, shift, N, factors, k);
			(0. != shift[j] ? md_zsin : md_zcos)(N, ksp_dims, tkern + k * md_calc_size(N, ksp_dims), tkern + k * md_calc_size(N, ksp_dims));
		}

		md_zsmul(N + 1, ksp_dims, tkern, tkern, cexp(M_PI * 0.25I * psf_dims[j]));
		md_zmul(N + 1, ksp_dims, kern, tkern, kern);

		md_free(tkern);

		j++;
	}

	complex float* psf = md_alloc_sameplace(N + 1, psf_dims, CFL_SIZE, traj);

	linop_adjoint_unchecked(lop_nufft, psf, kern);

	md_free(kern);

	linphase_decomp(N + 1, factors, psf_dims, MD_STRIDES(N + 1, psf_dims, CFL_SIZE), psf, MD_STRIDES(N + 1, psf_dims, CFL_SIZE), psf);

	fft(N + 1, psf_dims, conf.flags, psf, psf);

	linop_free(lop_nufft);

	return psf;
}



complex float* compute_psf2(int N, const long psf_dims[N + 1], unsigned long flags, const long trj_dims[N + 1], const complex float* traj,
				const long bas_dims[N + 1], const complex float* basis, const long wgh_dims[N + 1], const complex float* weights,
				bool periodic, bool lowmem, bool upper_triag)
{

#ifdef USE_CUDA
	bool gpu = cuda_ondevice(traj);
#else
	bool gpu = false;
#endif
	bool odd = false;
	for (int i = 0; i < N; i++)
		if (MD_IS_SET(flags, i) && 1 < psf_dims[i] && 1 == psf_dims[i] % 2)
			odd = true;

	if (!odd && (upper_triag || lowmem || gpu))
		return compute_psf2_decomposed(N, psf_dims, flags,
					       trj_dims, traj, bas_dims, basis, wgh_dims, weights,
					       periodic, lowmem, upper_triag);

	int ND = N + 1;

	long img_dims[ND];
	long img_strs[ND];

	md_select_dims(ND, ~MD_BIT(N + 0), img_dims, psf_dims);
	md_calc_strides(ND, img_strs, img_dims, CFL_SIZE);

	// PSF 2x size

	long img2_dims[ND];
	long img2_strs[ND];

	md_copy_dims(ND, img2_dims, img_dims);

	for (int i = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			img2_dims[i] = (1 == img_dims[i]) ? 1 : (2 * img_dims[i]);

	md_calc_strides(ND, img2_strs, img2_dims, CFL_SIZE);

	complex float* traj2 = md_alloc_sameplace(ND, trj_dims, CFL_SIZE, traj);

	md_zsmul(ND, trj_dims, traj2, traj, 2.);

	complex float* psft = compute_psf(ND, img2_dims, trj_dims, traj2, bas_dims, basis, wgh_dims, weights, periodic, lowmem);

	md_free(traj2);

	fftuc(ND, img2_dims, flags, psft, psft);

	// reformat

	complex float* psf = md_alloc_sameplace(ND, psf_dims, CFL_SIZE, traj);

	long factors[N];

	for (int i = 0; i < N; i++)
		factors[i] = ((img_dims[i] > 1) && (MD_IS_SET(flags, i))) ? 2 : 1;

	md_decompose(N + 0, factors, psf_dims, psf, img2_dims, psft, CFL_SIZE);

	md_free(psft);

	return psf;
}


static struct nufft_data* nufft_create_data(int N,
			const long cim_dims[N], bool basis,
			struct nufft_conf_s conf)
{
	PTR_ALLOC(struct nufft_data, data);
	SET_TYPEID(nufft_data, data);

	conf.precomp_fftmod = conf.precomp_fftmod && !conf.zero_overhead && conf.precomp;
	conf.precomp_linphase = conf.precomp_linphase && !conf.zero_overhead && conf.precomp;
	conf.precomp_roll = conf.precomp_roll && !conf.zero_overhead && conf.precomp;
	conf.lowmem = conf.lowmem || conf.zero_overhead;

	data->N = N;

	// extend internal dimensions by one for linear phases
	int ND = N + 1;

	data->ksp_dims = *TYPE_ALLOC(long[ND]);
	data->cim_dims = *TYPE_ALLOC(long[ND]);
	data->cml_dims = *TYPE_ALLOC(long[ND]);
	data->img_dims = *TYPE_ALLOC(long[ND]);
	data->trj_dims = *TYPE_ALLOC(long[ND]);
	data->lph_dims = *TYPE_ALLOC(long[ND]);
	data->psf_dims = *TYPE_ALLOC(long[ND]);
	data->wgh_dims = *TYPE_ALLOC(long[ND]);
	data->bas_dims = *TYPE_ALLOC(long[ND]);
	data->out_dims = *TYPE_ALLOC(long[ND]);
	data->ciT_dims = *TYPE_ALLOC(long[ND]);
	data->cmT_dims = *TYPE_ALLOC(long[ND]);
	data->cm2_dims = *TYPE_ALLOC(long[ND]);
	data->com_dims = *TYPE_ALLOC(long[ND]);

	data->factors = *TYPE_ALLOC(long[ND]);

	md_singleton_dims(ND, data->factors);

	data->ksp_strs = *TYPE_ALLOC(long[ND]);
	data->cim_strs = *TYPE_ALLOC(long[ND]);
	data->cml_strs = *TYPE_ALLOC(long[ND]);
	data->img_strs = *TYPE_ALLOC(long[ND]);
	data->trj_strs = *TYPE_ALLOC(long[ND]);
	data->lph_strs = *TYPE_ALLOC(long[ND]);
	data->psf_strs = *TYPE_ALLOC(long[ND]);
	data->wgh_strs = *TYPE_ALLOC(long[ND]);
	data->bas_strs = *TYPE_ALLOC(long[ND]);
	data->out_strs = *TYPE_ALLOC(long[ND]);
	data->com_strs = *TYPE_ALLOC(long[ND]);

	data->traj = NULL;
	data->psf = NULL;
	data->weights = NULL;
	data->basis = NULL;
	data->compress = NULL;

	data->conf = conf;
	data->flags = conf.flags;

	data->width = conf.width;
	data->beta = calc_beta(conf.os, data->width);

	// For reproducibility (deep-deep-learning paper, v0.8.00, Figure_06):
	// kb_init needs to be called with double precision beta.
	// Initializing by calling "rolloff_correction" would initialize with float precision.
	// Hence, we need to call kb_init here. Later (v0.9.00) may init via "rolloff_correction",
	// hence we only init in v0.8.00.
	if (use_compat_to_version("v0.8.00"))
		kb_init(data->beta);

	// dim 0 must be transformed (we treat this special in the trajectory)
	assert(MD_IS_SET(data->flags, 0));

	assert(0 == (data->flags & conf.cfft));
	assert(!((!conf.decomp) && conf.toeplitz));

	struct grid_conf_s grid_conf = {

		.width = data->width,
		.os = conf.os,
		.periodic = data->conf.periodic,
		.beta = data->beta,

		.shift = { 0., 0., 0. },
	};

	data->grid_conf = grid_conf;



	md_copy_dims(N, data->cim_dims, cim_dims);
	data->cim_dims[N] = 1;

	md_copy_dims(ND, data->ciT_dims, data->cim_dims);

	md_select_dims(ND, data->flags, data->img_dims, data->cim_dims);

	md_calc_strides(ND, data->cim_strs, data->cim_dims, CFL_SIZE);
	md_calc_strides(ND, data->img_strs, data->img_dims, CFL_SIZE);


	if (conf.precomp_fftmod) {

		complex float* fftm = md_alloc(ND, data->img_dims, CFL_SIZE);

		md_zfill(ND, data->img_dims, fftm, 1.);

		fftmod(ND, data->img_dims, data->flags, fftm, fftm);

		data->fftmod = multiplace_move_F(ND, data->img_dims, CFL_SIZE, fftm);

	} else {

		data->fftmod = NULL;
	}


	if (conf.precomp_roll) {

		complex float* roll = md_alloc(ND, data->img_dims, CFL_SIZE);

		rolloff_correction(conf.decomp ? data->grid_conf.os : 1, data->width, data->beta, data->img_dims, roll);

		data->roll = multiplace_move_F(ND, data->img_dims, CFL_SIZE, roll);

	} else {

		data->roll = NULL;
	}


	if (conf.precomp_linphase) {

		complex float* linphase;

		if (conf.decomp) {

			linphase = compute_linphases(N, data->lph_dims, data->flags, data->img_dims);

			for (int i = 0; i < data->N; i++)
				if ((data->img_dims[i] > 1) && MD_IS_SET(data->flags, i))
					data->factors[i] = 2;

		} else {

			linphase = md_alloc(ND, data->img_dims, CFL_SIZE);

			// only fftscale of the not-oversampled dims is required
			float scale = powf(sqrtf(2.), bitcount((data->flags) & md_nontriv_dims(N, data->img_dims)));
			md_zfill(ND, data->img_dims, linphase, scale);

			md_copy_dims(ND, data->lph_dims, data->img_dims);
		}

		md_calc_strides(ND, data->lph_strs, data->lph_dims, CFL_SIZE);

		if (!conf.toeplitz) {

			assert(conf.precomp_roll);

			md_zmul2(ND, data->lph_dims, data->lph_strs, linphase, data->lph_strs, linphase, data->img_strs, multiplace_read(data->roll, linphase));

			multiplace_free(data->roll);

			data->roll = NULL;
		}


		fftmod(ND, data->lph_dims, data->flags, linphase, linphase);
		fftscale(ND, data->lph_dims, data->flags, linphase, linphase);

		data->linphase = multiplace_move_F(ND, data->lph_dims, CFL_SIZE, linphase);

	} else {

		for (int i = 0; i < data->N; i++)
			if ((data->img_dims[i] > 1) && MD_IS_SET(data->flags, i))
				data->factors[i] = conf.decomp ? 2 : 1;

		md_copy_dims(N, data->lph_dims, data->img_dims);
		data->lph_dims[N] = md_calc_size(N, data->factors);

		data->linphase = NULL;
	}

	md_copy_dims(ND, data->cml_dims, data->cim_dims);
	data->cml_dims[N + 0] = data->lph_dims[N + 0];

	md_copy_dims(ND, data->cmT_dims, data->cml_dims);

	if (basis) {

		assert(1 == data->cml_dims[5]);
		data->cmT_dims[5] = data->cml_dims[6];
		data->cmT_dims[6] = 1;

		assert(1 == data->cim_dims[5]);
		data->ciT_dims[5] = data->cim_dims[6];
		data->ciT_dims[6] = 1;
	}

	md_calc_strides(ND, data->cml_strs, data->cml_dims, CFL_SIZE);


	md_copy_dims(ND, data->cm2_dims, data->cim_dims);

	for (int i = 0; i < N; i++)
		if (conf.decomp && MD_IS_SET(data->flags, i))
			data->cm2_dims[i] = (1 == cim_dims[i]) ? 1 : (2 * cim_dims[i]);



	data->fft_op = linop_fft_create(ND, data->cml_dims, data->flags | data->conf.cfft);

	if (conf.pcycle || conf.lowmem) {

		debug_printf(DP_DEBUG1, "NUFFT: %s mode\n", conf.lowmem ? "low-mem" : "pcycle");
		data->cycle = 0;
		data->cfft_op = linop_fft_create(N, data->cim_dims, data->flags | data->conf.cfft);
	}

	return PTR_PASS(data);
}


static void nufft_set_traj(struct nufft_data* data, int N,
			   const long trj_dims[N], const complex float* traj,
			   const long wgh_dims[N], const complex float* weights,
			   const long bas_dims[N], const complex float* basis)
{
	int ND = N + 1;

	if (NULL != traj) {

		assert(md_check_equal_dims(N, trj_dims, data->trj_dims, ~0UL));

		multiplace_free(data->traj);

		data->traj = multiplace_move(N, trj_dims, CFL_SIZE, traj);
	}

	if (NULL != basis) {

		assert(1 == md_calc_size(5, bas_dims));
		data->out_dims[5] = bas_dims[5];	// TE
		data->out_dims[6] = 1;			// COEFF

		if (1 == data->ksp_dims[6])
			data->ksp_dims[6] = bas_dims[6];

		assert(data->ksp_dims[6] == bas_dims[6]);

		// recompute
		md_calc_strides(ND, data->out_strs, data->out_dims, CFL_SIZE);
		md_calc_strides(ND, data->ksp_strs, data->ksp_dims, CFL_SIZE);

		md_copy_dims(N, data->bas_dims, bas_dims);
		data->bas_dims[N] = 1;

		md_calc_strides(ND, data->bas_strs, data->bas_dims, CFL_SIZE);

		multiplace_free(data->basis);

		data->basis = multiplace_move(ND, data->bas_dims, CFL_SIZE, basis);
	}

	if (NULL != weights) {

		md_copy_dims(N, data->wgh_dims, wgh_dims);
		data->wgh_dims[N] = 1;

		md_calc_strides(ND, data->wgh_strs, data->wgh_dims, CFL_SIZE);

		multiplace_free(data->weights);

		data->weights = multiplace_move(ND, data->wgh_dims, CFL_SIZE, weights);
	}

	if (data->conf.toeplitz) {

		debug_printf(DP_DEBUG1, "NUFFT: Toeplitz mode\n");

		md_copy_dims(ND, data->psf_dims, data->lph_dims);

		for (int i = 0; i < N; i++)
			if (!MD_IS_SET(data->flags, i))
				data->psf_dims[i] = MAX(data->trj_dims[i], ((NULL != weights) ? data->wgh_dims[i] : 0));

		if (NULL != basis) {

			debug_printf(DP_DEBUG3, "psf_dims: ");
			debug_print_dims(DP_DEBUG3, N, data->psf_dims);

			if (data->conf.upper_triag) {

				data->psf_dims[6] = data->bas_dims[6] * (data->bas_dims[6] + 1) / 2;
				data->psf_dims[5] = 1;
			} else {

				data->psf_dims[6] = data->bas_dims[6];
				data->psf_dims[5] = data->bas_dims[6];
			}
		}

		md_calc_strides(ND, data->psf_strs, data->psf_dims, CFL_SIZE);

		if (!data->conf.nopsf && (NULL != data->traj)) {

			multiplace_free(data->psf);
			multiplace_free(data->compress);

			long max_idx = 0;

			if (data->conf.compress_psf) {

				md_select_dims(ND, FFT_FLAGS, data->com_dims, data->img_dims);

				complex float* grid = md_alloc_sameplace(ND, data->com_dims, CFL_SIZE, traj);
				md_clear(ND, data->com_dims, grid, CFL_SIZE);

				const complex float* pat = multiplace_read(data->weights, traj);
				if (NULL == pat)
					error("Compressed PSF only possible if pattern provided!\n.");

				struct grid_conf_s gconf = data->grid_conf;
				gconf.periodic = true;

				for (int i = 0; i < md_calc_size(data->N, data->factors); i++)
					grid2_decomp(&gconf, i, ND, data->factors, data->trj_dims, multiplace_read(data->traj, traj), data->com_dims, grid, data->wgh_dims, pat);

				md_zabs(ND, data->com_dims, grid, grid);

				complex float* grid_cpu = md_alloc(ND, data->com_dims, CFL_SIZE);
				md_copy(ND, data->com_dims, grid_cpu, grid, CFL_SIZE);
				md_free(grid);

				long* idx = md_alloc(ND, data->com_dims, sizeof(long));
				max_idx = md_compress_mask_to_index(ND, data->com_dims, idx, grid_cpu);
				md_free(grid_cpu);

				data->compress = multiplace_move_F(ND, data->com_dims, sizeof(long), idx);

				debug_printf(DP_DEBUG1, "Compressing PSF to %.0f%%\n", 100. * max_idx / md_calc_size(ND, data->com_dims));
			}

			const complex float* psf = compute_psf2(N, data->psf_dims, data->flags, data->trj_dims, traj,
						data->bas_dims, multiplace_read(data->basis, traj), data->wgh_dims, multiplace_read(data->weights, traj),
						true /*conf.periodic*/, data->conf.lowmem, data->conf.upper_triag);

			if (data->conf.real) {

				float* psf_real = md_alloc_sameplace(ND, data->psf_dims, FL_SIZE, traj);
				md_real(ND, data->psf_dims, psf_real, psf);
				md_free(psf);

				md_calc_strides(ND, data->psf_strs, data->psf_dims, FL_SIZE);

				data->psf = multiplace_move_F(ND, data->psf_dims, FL_SIZE, psf_real);
				md_calc_strides(ND, data->psf_strs, data->psf_dims, FL_SIZE);
			} else {

				data->psf = multiplace_move_F(ND, data->psf_dims, CFL_SIZE, psf);
			}

			if (NULL != data->compress) {

				long com_psf_dims[ND];
				md_compress_dims(ND, com_psf_dims, data->psf_dims, data->com_dims, max_idx);

				complex float* com_psf = md_alloc_sameplace(ND, com_psf_dims, data->conf.real ? FL_SIZE : CFL_SIZE, traj);
				md_compress(ND, com_psf_dims, com_psf, data->psf_dims, multiplace_read(data->psf, traj), data->com_dims, multiplace_read(data->compress, traj), data->conf.real ? FL_SIZE : CFL_SIZE);
				multiplace_free(data->psf);

				md_copy_dims(ND, data->psf_dims, com_psf_dims);
				md_calc_strides(ND, data->psf_strs, data->psf_dims, data->conf.real ? FL_SIZE : CFL_SIZE);
				data->psf = multiplace_move_F(ND, data->psf_dims, data->conf.real ? FL_SIZE : CFL_SIZE, com_psf);
			}
		}
	}
}


static void nufft_free_data(const linop_data_t* data);
static void nufft_apply_forward(const linop_data_t* _data, complex float* dst, const complex float* src);
static void nufft_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src);
static void nufft_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src);

static void nufft_apply_adjoint_lowmem(const linop_data_t* _data, complex float* dst, const complex float* src);
static void nufft_apply_forward_lowmem(const linop_data_t* _data, complex float* dst, const complex float* src);

static void nufft_apply_adjoint_zero_overhead(const linop_data_t* _data, complex float* dst, const complex float* src);
static void nufft_apply_forward_zero_overhead(const linop_data_t* _data, complex float* dst, const complex float* src);


static struct linop_s* nufft_create3(int N,
			     const long ksp_dims[N],
			     const long cim_dims[N],
			     const long traj_dims[N],
			     const complex float* traj,
			     const long wgh_dims[N],
			     const complex float* weights,
			     const long bas_dims[N],
			     const complex float* basis,
			     struct nufft_conf_s conf)
{
	if (2. != conf.os) {

		debug_printf(DP_DEBUG1, "Chained nuFFT!\n");

		struct grid_conf_s grid_conf = {

			.os = conf.os,
			.width = conf.width,
			.beta = calc_beta(grid_conf.os, grid_conf.width),
			.periodic = conf.periodic,
			.shift = { 0., 0., 0. },
		};

		return nufft_create_chain(N, ksp_dims, cim_dims, traj_dims, traj, wgh_dims, weights, bas_dims, basis, &grid_conf);
	}

	debug_printf(DP_DEBUG1, "ksp : ");
	debug_print_dims(DP_DEBUG1, N, ksp_dims);
	debug_printf(DP_DEBUG1, "cim : ");
	debug_print_dims(DP_DEBUG1, N, cim_dims);
	debug_printf(DP_DEBUG1, "traj: ");
	debug_print_dims(DP_DEBUG1, N, traj_dims);

	if (!md_check_compat(N, ~(MD_BIT(1) | MD_BIT(2)), ksp_dims, traj_dims))
		error("Incompatible dimensions of k-space and trajectory!\n");

	if (NULL != weights) {

		debug_printf(DP_DEBUG1, "wgh : ");
		debug_print_dims(DP_DEBUG1, N, wgh_dims);

		if (!md_check_compat(N, ~0UL, ksp_dims, wgh_dims))
			error("Incompatible dimensions of k-space and weights!\n");
	}

	if (NULL != basis) {

		debug_printf(DP_DEBUG1, "bas : ");
		debug_print_dims(DP_DEBUG1, N, bas_dims);
	}


//	assert(md_check_compat(N, ~data->flags, ksp_dims, cim_dims));
//	assert(md_check_bounds(N, ~data->flags, cim_dims, ksp_dims));
	assert((1 == md_calc_size(N, ksp_dims)) || md_check_bounds(N, ~(conf.flags | (NULL == basis ? 0 : (1 << 6))), cim_dims, ksp_dims));

	// extend internal dimensions by one for linear phases
	int ND = N + 1;

	assert((1 == md_calc_size(N, traj_dims)) || (bitcount(conf.flags) == traj_dims[0]));

	long chk_dims[N];
	md_select_dims(N, ~conf.flags, chk_dims, traj_dims);
	assert((1 == md_calc_size(N, ksp_dims)) || md_check_compat(N, ~0ul, chk_dims, ksp_dims));
//	assert(md_check_bounds(N, ~0ul, chk_dims, ksp_dims));


	auto data = nufft_create_data(N, cim_dims, NULL != basis, conf);


	md_copy_dims(N, data->ksp_dims, ksp_dims);
	data->ksp_dims[N] = 1;

	md_copy_dims(N, data->out_dims, ksp_dims);
	data->out_dims[N] = 1;

	md_copy_dims(N, data->trj_dims, traj_dims);
	data->trj_dims[N] = 1;

	md_calc_strides(ND, data->trj_strs, data->trj_dims, CFL_SIZE);
	md_calc_strides(ND, data->ksp_strs, data->ksp_dims, CFL_SIZE);
	md_calc_strides(ND, data->out_strs, data->out_dims, CFL_SIZE);

	nufft_set_traj(data, N, traj_dims, traj, wgh_dims, weights, bas_dims, basis);

	long out_dims[N];
	md_copy_dims(N, out_dims, data->out_dims);

	if (conf.zero_overhead)
		return linop_create(N, out_dims, N, cim_dims,
				CAST_UP(data), nufft_apply_forward_zero_overhead, nufft_apply_adjoint_zero_overhead, NULL, NULL, nufft_free_data);

	if (conf.lowmem) {

		return linop_create(N, out_dims, N, cim_dims,
				CAST_UP(data), nufft_apply_forward_lowmem, nufft_apply_adjoint_lowmem, conf.toeplitz ? nufft_apply_normal : NULL, NULL, nufft_free_data);
	} else {

		return linop_create(N, out_dims, N, cim_dims,
				CAST_UP(data), nufft_apply_forward, nufft_apply_adjoint, conf.toeplitz ? nufft_apply_normal : NULL, NULL, nufft_free_data);
	}
}


struct linop_s* nufft_create2(int N,
			     const long ksp_dims[N],
			     const long cim_dims[N],
			     const long traj_dims[N],
			     const complex float* traj,
			     const long wgh_dims[N],
			     const complex float* weights,
			     const long bas_dims[N],
			     const complex float* basis,
			     struct nufft_conf_s conf)
{
	if (0 <= conf.loopdim) {

		int d = conf.loopdim;
		const long L = ksp_dims[d];

		assert(d < N);
		assert((NULL == weights) || (1 == wgh_dims[d]));
		assert((NULL == basis) || (1 == bas_dims[d]));
		assert(1 == traj_dims[d]);
		assert(L == cim_dims[d]);

		if (1 < L) {

			debug_printf(DP_WARN, "NEW NUFFT LOOP CODE\n");

			long ksp1_dims[N];
			md_select_dims(N, ~MD_BIT(d), ksp1_dims, ksp_dims);

			long cim1_dims[N];
			md_select_dims(N, ~MD_BIT(d), cim1_dims, cim_dims);

			auto nu = nufft_create2(N, ksp1_dims, cim1_dims, traj_dims, traj, wgh_dims, weights, bas_dims, basis, conf);

			long loop_dims[N];
			md_select_dims(N, MD_BIT(d), loop_dims, cim_dims);

			auto nu2 = linop_loop(N, loop_dims, nu);

			linop_free(nu);

			return nu2;
		}
	}

	return nufft_create3(N, ksp_dims, cim_dims,
			traj_dims, traj, wgh_dims, weights,
			bas_dims, basis, conf);
}

static void nufft_normal_only(const linop_data_t* /*_data*/, complex float* /*dst*/, const complex float* /*src*/)
{
	error("NuFFT with normal operator only!\n");
}

struct linop_s* nufft_create_normal(int N, const long cim_dims[N],
				    int ND, const long psf_dims[ND], const complex float* psf,
				    bool basis, struct nufft_conf_s conf)
{
	debug_printf(DP_DEBUG1, "cim : ");
	debug_print_dims(DP_DEBUG1, N, cim_dims);

	debug_printf(DP_DEBUG1, "psf : ");
	debug_print_dims(DP_DEBUG1, ND, psf_dims);

	auto data = nufft_create_data(N, cim_dims, basis, conf);

	md_copy_dims(ND, data->psf_dims, psf_dims);
	md_calc_strides(ND, data->psf_strs, psf_dims, CFL_SIZE);

	assert(md_check_equal_dims(ND, data->psf_dims, data->lph_dims, data->flags));
	assert(conf.toeplitz);

	long out_dims[N];
	md_singleton_dims(N, out_dims);

	auto result = linop_create(N, out_dims, N, cim_dims,
			CAST_UP(data), nufft_normal_only, nufft_normal_only, nufft_apply_normal, NULL, nufft_free_data);

	if (NULL != psf)
		nufft_update_psf(result, ND, psf_dims, psf);

	return result;
}

struct linop_s* nufft_create(int N,				///< Number of dimension
			     const long ksp_dims[N],		///< kspace dimension
			     const long cim_dims[N],		///< Coil images dimension
			     const long traj_dims[N],		///< Trajectory dimension
			     const complex float* traj,		///< Trajectory
			     const complex float* weights,	///< Weights, ex, soft-gating or density compensation
			     struct nufft_conf_s conf)		///< NUFFT configuration options
{
	long wgh_dims[N];
	md_select_dims(N, ~MD_BIT(0), wgh_dims, traj_dims);

	return nufft_create2(N, ksp_dims, cim_dims, traj_dims, traj, wgh_dims, weights, NULL, NULL, conf);
}






static void nufft_free_data(const linop_data_t* _data)
{
	auto data = CAST_DOWN(nufft_data, _data);

	xfree(data->ksp_dims);
	xfree(data->cim_dims);
	xfree(data->cml_dims);
	xfree(data->img_dims);
	xfree(data->trj_dims);
	xfree(data->lph_dims);
	xfree(data->psf_dims);
	xfree(data->wgh_dims);
	xfree(data->bas_dims);
	xfree(data->out_dims);
	xfree(data->ciT_dims);
	xfree(data->cmT_dims);
	xfree(data->com_dims);

	xfree(data->ksp_strs);
	xfree(data->cim_strs);
	xfree(data->cml_strs);
	xfree(data->img_strs);
	xfree(data->trj_strs);
	xfree(data->lph_strs);
	xfree(data->psf_strs);
	xfree(data->wgh_strs);
	xfree(data->bas_strs);
	xfree(data->out_strs);
	xfree(data->com_strs);

	xfree(data->cm2_dims);
	xfree(data->factors);

	multiplace_free(data->linphase);
	multiplace_free(data->psf);
	multiplace_free(data->fftmod);
	multiplace_free(data->weights);
	multiplace_free(data->roll);
	multiplace_free(data->basis);
	multiplace_free(data->traj);
	multiplace_free(data->compress);

	linop_free(data->fft_op);

	if (data->conf.pcycle || data->conf.lowmem)
		linop_free(data->cfft_op);

	xfree(data);
}




// Forward: from image to kspace
static void nufft_apply_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nufft_data, _data);
	assert(!data->conf.lowmem);

	int ND = data->N + 1;

	if (data->conf.toeplitz || NULL == data->linphase) {

		complex float* tmp = md_alloc_sameplace(ND, data->cim_dims, CFL_SIZE, dst);

		if (NULL != data->roll)
			md_zmul2(ND, data->cim_dims, data->cim_strs, tmp, data->cim_strs, src, data->img_strs, multiplace_read(data->roll, src));
		else
			apply_rolloff_correction2(data->conf.os, data->conf.width, data->beta, data->N, data->cim_dims, data->cim_strs, tmp, data->cim_strs, src);

		src = tmp;
	}


	complex float* grid = md_alloc_sameplace(ND, data->cml_dims, CFL_SIZE, dst);

	if (NULL != data->linphase)
		md_zmul2(ND, data->cml_dims, data->cml_strs, grid, data->cim_strs, src, data->lph_strs, multiplace_read(data->linphase, dst));
	else
		linphase_decomp(ND, data->factors, data->cml_dims, data->cml_strs, grid, data->cim_strs, src);

	if (data->conf.toeplitz || NULL == data->linphase)
		md_free(src);

	linop_forward(data->fft_op, ND, data->cml_dims, grid, ND, data->cml_dims, grid);

	if (NULL != data->fftmod)
		md_zmul2(ND, data->cml_dims, data->cml_strs, grid, data->cml_strs, grid, data->img_strs, multiplace_read(data->fftmod, dst));
	else
		ifftmod2(ND, data->cml_dims, data->conf.flags, data->cml_strs, grid, data->cml_strs, grid);

	complex float* tmp = dst;

	if (NULL != data->basis)
		tmp = md_alloc_sameplace(ND, data->ksp_dims, CFL_SIZE, dst);

	md_clear(ND, data->ksp_dims, tmp, CFL_SIZE);

	if (!data->conf.decomp || use_compat_to_version("v0.9.00")) {

		complex float* gridX = md_alloc_sameplace(data->N, data->cm2_dims, CFL_SIZE, dst);

		md_recompose(data->N, data->factors, data->cm2_dims, gridX, data->cml_dims, grid, CFL_SIZE);
		md_free(grid);

		grid2H(&data->grid_conf, ND, data->trj_dims, multiplace_read(data->traj, dst), data->ksp_dims, tmp, data->cm2_dims, gridX);

		md_free(gridX);
	} else {

		for (int i = 0; i < md_calc_size(data->N, data->factors); i++)
			grid2H_decomp(&data->grid_conf, i, data->N + 1, data->factors, data->trj_dims, multiplace_read(data->traj, dst), data->ksp_dims, tmp, data->cim_dims, grid + i * md_calc_size(data->N, data->cml_dims));

		md_free(grid);
	}

	if (NULL != data->basis) {

		md_ztenmul(data->N, data->out_dims, dst, data->ksp_dims, tmp, data->bas_dims, multiplace_read(data->basis, dst));
		md_free(tmp);
	}

	if (NULL != data->weights)
		md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->out_strs, dst, data->wgh_strs, multiplace_read(data->weights, dst));
}


// Adjoint: from kspace to image
static void nufft_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nufft_data, _data);
	assert(!data->conf.lowmem);

#ifdef USE_CUDA
	assert(cuda_ondevice(src) == cuda_ondevice(dst));
#endif
	int ND = data->N + 1;

	complex float* wdat = NULL;

	if (NULL != data->weights) {

		wdat = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);
		md_zmulc2(data->N, data->out_dims, data->out_strs, wdat, data->out_strs, src, data->wgh_strs, multiplace_read(data->weights, dst));
		src = wdat;
	}

	complex float* bdat = NULL;

	if (NULL != data->basis) {

		bdat = md_alloc_sameplace(data->N, data->ksp_dims, CFL_SIZE, dst);
		md_ztenmulc(data->N, data->ksp_dims, bdat, data->out_dims, src, data->bas_dims, multiplace_read(data->basis, dst));
		src = bdat;

		md_free(wdat);
		wdat = NULL;
	}


	complex float* grid = md_alloc_sameplace(ND, data->cml_dims, CFL_SIZE, dst);

	if (data->conf.decomp && !use_compat_to_version("v0.8.00")) {

		md_clear(ND, data->cml_dims, grid, CFL_SIZE);

		for (int i = 0; i < md_calc_size(data->N, data->factors); i++)
			grid2_decomp(&data->grid_conf, i, data->N, data->factors, data->trj_dims, multiplace_read(data->traj, dst), data->cml_dims, grid + i * md_calc_size(data->N, data->cml_dims), data->ksp_dims, src);

	} else {

		complex float* gridX = md_alloc_sameplace(ND, data->cm2_dims, CFL_SIZE, dst);

		md_clear(data->N, data->cm2_dims, gridX, CFL_SIZE);

		grid2(&data->grid_conf, ND, data->trj_dims, multiplace_read(data->traj, dst), data->cm2_dims, gridX, data->ksp_dims, src);

		md_decompose(data->N, data->factors, data->cml_dims, grid, data->cm2_dims, gridX, CFL_SIZE);

		md_free(gridX);
	}

	if (NULL != data->fftmod)
		md_zmulc2(ND, data->cml_dims, data->cml_strs, grid, data->cml_strs, grid, data->img_strs, multiplace_read(data->fftmod, dst));
	else
		ifftmod2(ND, data->cml_dims, data->conf.flags, data->cml_strs, grid, data->cml_strs, grid);

	linop_adjoint(data->fft_op, ND, data->cml_dims, grid, ND, data->cml_dims, grid);

	if (NULL != data->linphase)
		md_ztenmulc2(ND, data->cml_dims, data->cim_strs, dst, data->cml_strs, grid, data->lph_strs, multiplace_read(data->linphase, dst));
	else
		linphaseH_decomp(ND, data->factors, data->cml_dims, data->cim_strs, dst, data->cml_strs, grid);

	md_free(grid);

	md_free(bdat);
	md_free(wdat);

	if (data->conf.toeplitz || NULL == data->linphase) {

		if (NULL != data->roll)
			md_zmul2(ND, data->cim_dims, data->cim_strs, dst, data->cim_strs, dst, data->img_strs, multiplace_read(data->roll, dst));
		else
			apply_rolloff_correction2(data->conf.os, data->conf.width, data->beta, data->N, data->cim_dims, data->cim_strs, dst, data->cim_strs, dst);
	}
}





static void toeplitz_mult(const struct nufft_data* data, complex float* dst, const complex float* src)
{
	int ND = data->N + 1;

	const complex float* linphase = multiplace_read(data->linphase, src);
	const void* psf = multiplace_read(data->psf, src);

	complex float* grid = md_alloc_sameplace(ND, data->cml_dims, CFL_SIZE, dst);

	if (NULL != linphase)
		md_zmul2(ND, data->cml_dims, data->cml_strs, grid, data->cim_strs, src, data->lph_strs, linphase);
	else
		linphase_decomp(ND, data->factors, data->cml_dims, data->cml_strs, grid, data->cim_strs, src);

	linop_forward(data->fft_op, ND, data->cml_dims, grid, ND, data->cml_dims, grid);

	long cml_dims[ND];
	long cmT_dims[ND];

	md_copy_dims(ND, cml_dims, data->cml_dims);
	md_copy_dims(ND, cmT_dims, data->cmT_dims);

	if (NULL != data->compress) {

		md_copy_dims(3, cml_dims, data->psf_dims);
		md_copy_dims(3, cmT_dims, data->psf_dims);

		const long* idx = multiplace_read(data->compress, grid);
		complex float* com_grid = md_alloc_sameplace(ND, cml_dims, CFL_SIZE, grid);
		md_compress(ND, cml_dims, com_grid, data->cml_dims, grid, data->com_dims, idx, CFL_SIZE);
		md_free(grid);
		grid = com_grid;
	}

	long max_dims[ND];
	md_max_dims(ND, ~0UL, max_dims, cmT_dims, cml_dims);

	complex float* gridT = md_alloc_sameplace(ND, cml_dims, CFL_SIZE, dst);

	if (data->conf.real) {

		long cmT_strs[ND];
		long cml_strs[ND];
		md_calc_strides(ND, cmT_strs, cmT_dims, CFL_SIZE);
		md_calc_strides(ND, cml_strs, cml_dims, CFL_SIZE);

		if (data->conf.upper_triag) // shifted indexing (6, 7) as real dim is first
			md_tenmul_upper_triag2(6, 7, ND + 1, MD_REAL_DIMS(ND, max_dims), MD_REAL_STRS(ND, cmT_strs, FL_SIZE), (float*)gridT, MD_REAL_STRS(ND, cml_strs, FL_SIZE), (float*)grid, MD_REAL_DIMS(ND, data->psf_dims), MD_REAL_STRS(ND, data->psf_strs, 0), (float*)psf);
		else
			md_tenmul2(ND + 1, MD_REAL_DIMS(ND, max_dims), MD_REAL_STRS(ND, cmT_strs, FL_SIZE), (float*)gridT, MD_REAL_STRS(ND, cml_strs, FL_SIZE), (float*)grid, MD_REAL_STRS(ND, data->psf_strs, 0), (float*)psf);
	} else {

		if (data->conf.upper_triag)
			md_ztenmul_upper_triag(5, 6, ND, cmT_dims, gridT, cml_dims, grid, data->psf_dims, psf);
		else
			md_ztenmul(ND, cmT_dims, gridT, cml_dims, grid, data->psf_dims, psf);
	}

	md_free(grid);
	grid = gridT;

	if (NULL != data->compress) {

		const long* idx = multiplace_read(data->compress, grid);
		complex float* dec_grid = md_alloc_sameplace(ND, data->cml_dims, CFL_SIZE, grid);
		md_clear(ND, data->cml_dims, dec_grid, CFL_SIZE);
		md_decompress(ND, data->cml_dims, dec_grid, cml_dims, grid, data->com_dims, idx, NULL, CFL_SIZE);
		md_free(grid);
		grid = dec_grid;
	}

	linop_adjoint(data->fft_op, ND, data->cml_dims, grid, ND, data->cml_dims, grid);

	if (NULL != linphase)
		md_ztenmulc2(ND, data->cml_dims, data->cim_strs, dst, data->cml_strs, grid, data->lph_strs, linphase);
	else
		linphaseH_decomp(ND, data->factors, data->cml_dims, data->cim_strs, dst, data->cml_strs, grid);

	md_free(grid);
}



static void toeplitz_mult_lowmem(const struct nufft_data* data, int i, complex float* dst, const complex float* src)
{
	const complex float* linphase = multiplace_read(data->linphase, src);
	const complex float* clinphase = linphase ? linphase + i * md_calc_size(data->N, data->lph_dims) : NULL;

	const void* psf = multiplace_read(data->psf, src);
	const void* cpsf = psf + i * md_calc_size(data->N, data->psf_dims) * (long)(data->conf.real ? FL_SIZE : CFL_SIZE);

	float shift[3];
	for (int j = 0; j < 3; j++)
		shift[j] = MD_IS_SET((unsigned long)i, j) ? -0.5 : 0;

	complex float* grid = md_alloc_sameplace(data->N, data->cim_dims, CFL_SIZE, dst);

	if (NULL != clinphase) {

		md_zmul2(data->N, data->cim_dims, data->cim_strs, grid, data->cim_strs, src, data->img_strs, clinphase);

	} else {

		float scale = 1. / sqrtf(md_calc_size(3, data->lph_dims));
		apply_linphases_3D(data->N, data->cim_dims, shift, grid, src, false, false, true, scale);
	}


	linop_forward(data->cfft_op, data->N, data->cim_dims, grid, data->N, data->cim_dims, grid);

	long mdims[data->N];
	md_max_dims(data->N, ~0UL, mdims, data->ciT_dims, data->cim_dims);

	if (!md_check_equal_dims(data->N, data->cim_dims, data->ciT_dims, ~0UL)) {

		complex float* gridT = md_alloc_sameplace(data->N, data->ciT_dims, CFL_SIZE, dst);

		if (data->conf.real) {

			long ciT_strs[data->N];
			md_calc_strides(data->N, ciT_strs, data->ciT_dims, CFL_SIZE);

			if (data->conf.upper_triag)
				md_tenmul_upper_triag2(5, 6, data->N + 1, MD_REAL_DIMS(data->N, mdims), MD_REAL_STRS(data->N, ciT_strs, FL_SIZE), (float*)gridT, MD_REAL_STRS(data->N, data->cim_strs, FL_SIZE), (float*)grid, MD_REAL_DIMS(data->N, data->psf_dims), MD_REAL_STRS(data->N, data->psf_strs, 0), (float*)cpsf);
			else
				md_tenmul2(data->N + 1, MD_REAL_DIMS(data->N, mdims), MD_REAL_STRS(data->N, ciT_strs, FL_SIZE), (float*)gridT, MD_REAL_STRS(data->N, data->cim_strs, FL_SIZE), (float*)grid, MD_REAL_STRS(data->N, data->psf_strs, 0), (float*)cpsf);
		} else {

			if (data->conf.upper_triag)
				md_ztenmul_upper_triag(5, 6, data->N, data->ciT_dims, gridT, data->cim_dims, grid, data->psf_dims, cpsf);
			else
				md_ztenmul(data->N, data->ciT_dims, gridT, data->cim_dims, grid, data->psf_dims, cpsf);
		}


		md_free(grid);

		grid = gridT;

	} else {

		if (data->conf.real)
			md_mul2(data->N, MD_REAL_DIMS(data->N, data->cim_dims), MD_REAL_STRS(data->N, data->cim_strs, FL_SIZE), (float*)grid, MD_REAL_STRS(data->N, data->cml_strs, FL_SIZE), (float*)grid, MD_REAL_STRS(data->N, data->psf_strs, 0), (float*)cpsf);
		else
			md_zmul2(data->N, data->cim_dims, data->cim_strs, grid, data->cim_strs, grid, data->psf_strs, cpsf);
	}

	linop_adjoint(data->cfft_op, data->N, data->cim_dims, grid, data->N, data->cim_dims, grid);

	if (NULL != clinphase) {

		md_zfmacc2(data->N, data->cim_dims, data->cim_strs, dst, data->cim_strs, grid, data->img_strs, clinphase);
	} else {

		float scale = 1. / sqrtf(md_calc_size(3, data->lph_dims));
		apply_linphases_3D(data->N, data->cim_dims, shift, dst, grid, true, true, true, scale);
	}

	md_free(grid);
}



static void nufft_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nufft_data, _data);

	assert(data->conf.toeplitz);

	if (!(data->conf.pcycle || data->conf.lowmem)) {

		toeplitz_mult(data, dst, src);

		return;
	}

	// low mem versions

	assert(dst != src);

	int ncycles = data->lph_dims[data->N];

	if (data->conf.pcycle)
		data->cycle = (data->cycle + 1) % ncycles;	// FIXME:

	md_clear(data->N, data->cim_dims, dst, CFL_SIZE);

	for (int i = 0; i < ncycles; i++) {

		if (data->conf.pcycle && (i != data->cycle))
			continue;

		toeplitz_mult_lowmem(data, i, dst, src);
	}
}


// Adjoint: from kspace to image
static void nufft_apply_adjoint_lowmem(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nufft_data, _data);

	int ND = data->N + 1;

	complex float* wdat = NULL;

	if (NULL != data->weights) {

		wdat = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);
		md_zmulc2(data->N, data->out_dims, data->out_strs, wdat, data->out_strs, src, data->wgh_strs, multiplace_read(data->weights, dst));
		src = wdat;
	}

	complex float* bdat = NULL;

	if (NULL != data->basis) {

		bdat = md_alloc_sameplace(data->N, data->ksp_dims, CFL_SIZE, dst);
		md_ztenmulc(data->N, data->ksp_dims, bdat, data->out_dims, src, data->bas_dims, multiplace_read(data->basis, dst));
		src = bdat;

		md_free(wdat);
		wdat = NULL;
	}

	md_clear(data->N, data->cim_dims, dst, CFL_SIZE);

	complex float* grid = md_alloc_sameplace(data->N, data->cim_dims, CFL_SIZE, dst);

	long pos_cml[ND];
	md_singleton_strides(ND, pos_cml);

	for (; pos_cml[data->N] < md_calc_size(data->N, data->factors); pos_cml[data->N]++) {

		md_clear(data->N, data->cim_dims, grid, CFL_SIZE);
		grid2_decomp(&(data->grid_conf), pos_cml[data->N], data->N, data->factors, data->trj_dims, multiplace_read(data->traj, dst), data->cim_dims, grid, data->ksp_dims, src);


		if (NULL != data->fftmod)
			md_zmulc2(data->N, data->cim_dims, data->cim_strs, grid, data->cim_strs, grid, data->img_strs, multiplace_read(data->fftmod, dst));
		else
			ifftmod(data->N, data->cim_dims, data->flags, grid, grid);

		linop_adjoint(data->cfft_op, data->N, data->cim_dims, grid, data->N, data->cim_dims, grid);

		if (NULL != data->linphase){

			md_zfmacc2(data->N, data->cim_dims, data->cim_strs, dst, data->cim_strs, grid, data->lph_strs, &MD_ACCESS(ND, data->lph_strs, pos_cml, (complex float*)multiplace_read(data->linphase, dst)));
		} else {

			float scale = 1. / sqrtf(md_calc_size(3, data->lph_dims));

			float shift[3];
			compute_shift(3, shift, data->N, data->factors, pos_cml[data->N]);

			apply_linphases_3D(data->N, data->cim_dims, shift, dst, grid, true, true, true, scale);
		}
	}

	md_free(grid);
	md_free(bdat);
	md_free(wdat);


	if ((NULL == data->linphase) || (data->conf.toeplitz)){

		if (NULL == data->roll)
			apply_rolloff_correction(2., data->grid_conf.width, data->grid_conf.beta, data->N, data->cim_dims, dst, dst);
		else
			md_zmul2(ND, data->cim_dims, data->cim_strs, dst, data->cim_strs, dst, data->img_strs, multiplace_read(data->roll, dst));;
	}
}


// Forward: from image to kspace
static void nufft_apply_forward_lowmem(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nufft_data, _data);

	int ND = data->N + 1;

	complex float* grid = md_alloc_sameplace(data->N, data->cim_dims, CFL_SIZE, dst);

	long pos_cml[ND];
	md_singleton_strides(ND, pos_cml);

	complex float* tmp = dst;
	if (NULL != data->basis)
		tmp = md_alloc_sameplace(ND, data->ksp_dims, CFL_SIZE, dst);

	md_clear(ND, data->ksp_dims, tmp, CFL_SIZE);

	for (; pos_cml[data->N] < md_calc_size(data->N, data->factors); pos_cml[data->N]++) {

		if (NULL != data->linphase){

			md_zmul2(data->N, data->cim_dims, data->cim_strs, grid, data->cim_strs, src, data->lph_strs, &MD_ACCESS(ND, data->lph_strs, pos_cml, (complex float*)multiplace_read(data->linphase, dst)));
		} else {

			float scale = 1. / sqrtf(md_calc_size(3, data->lph_dims));

			float shift[3];
			compute_shift(3, shift, data->N, data->factors, pos_cml[data->N]);

			apply_linphases_3D(data->N, data->cim_dims, shift, grid, src, false, false, true, scale);
		}

		if ((NULL == data->linphase) || (data->conf.toeplitz)) {

			if (NULL == data->roll)
				apply_rolloff_correction(2., data->grid_conf.width, data->grid_conf.beta, data->N, data->cim_dims, grid, grid);
			else
				md_zmul2(ND, data->cim_dims, data->cim_strs, grid, data->cim_strs, grid, data->img_strs, multiplace_read(data->roll, dst));;
		}

		linop_forward(data->cfft_op, data->N, data->cim_dims, grid, data->N, data->cim_dims, grid);

		if (NULL != data->fftmod)
			md_zmul2(data->N, data->cim_dims, data->cim_strs, grid, data->cim_strs, grid, data->img_strs, multiplace_read(data->fftmod, dst));
		else
			fftmod(data->N, data->cim_dims, data->flags, grid, grid);


		grid2H_decomp(&(data->grid_conf), pos_cml[data->N], data->N, data->factors, data->trj_dims, multiplace_read(data->traj, src), data->ksp_dims, tmp, data->cim_dims, grid);
	};

	md_free(grid);

	if (NULL != data->basis) {

		md_ztenmul(data->N, data->out_dims, dst, data->ksp_dims, tmp, data->bas_dims, multiplace_read(data->basis, src));
		md_free(tmp);
	}

	if (NULL != data->weights)
		md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->out_strs, dst, data->wgh_strs, multiplace_read(data->weights, src));
}



// Adjoint: from kspace to image
static void nufft_apply_adjoint_zero_overhead(const linop_data_t* _data, complex float* dst, const complex float* _src)
{
	auto data = CAST_DOWN(nufft_data, _data);

	int ND = data->N + 1;

	complex float* src = (complex float*) _src;

	if (NULL != data->weights)
		md_zmulc2(data->N, data->out_dims, data->out_strs, src, data->out_strs, src, data->wgh_strs, src);


	md_clear(data->N, data->cim_dims, dst, CFL_SIZE);

	long pos_fac[ND];
	long pos_cml[ND];

	md_singleton_strides(ND, pos_fac);
	md_singleton_strides(ND, pos_cml);

	for (; pos_cml[data->N] < md_calc_size(data->N, data->factors); pos_cml[data->N]++) {


		float scale = 1. / sqrtf(md_calc_size(3, data->lph_dims));

		float shift[3];
		compute_shift(3, shift, data->N, data->factors, pos_cml[data->N]);

		apply_linphases_3D(data->N, data->cim_dims, shift, dst, dst, false, false, true, scale);
		linop_forward(data->cfft_op, data->N, data->cim_dims, dst, data->N, data->cim_dims, dst);
		fftmod(data->N, data->cim_dims, data->flags, dst, dst);

		grid2_decomp(&(data->grid_conf), pos_cml[data->N], data->N, data->factors, data->trj_dims, multiplace_read(data->traj, dst), data->cim_dims, dst, data->ksp_dims, src);

		//recover src
		ifftmod(data->N, data->cim_dims, data->flags, dst, dst);
		linop_adjoint(data->cfft_op, data->N, data->cim_dims, dst, data->N, data->cim_dims, dst);
		apply_linphases_3D(data->N, data->cim_dims, shift, dst, dst, true, false, true, scale);

	}

	apply_rolloff_correction(2., data->grid_conf.width, data->grid_conf.beta, data->N, data->cim_dims, dst, dst);
}


// Forward: from image to kspace
static void nufft_apply_forward_zero_overhead(const linop_data_t* _data, complex float* dst, const complex float* _src)
{
	auto data = CAST_DOWN(nufft_data, _data);

	complex float* src = (complex float*) _src;

	int ND = data->N + 1;

	long pos_fac[ND];
	long pos_cml[ND];

	md_singleton_strides(ND, pos_fac);
	md_singleton_strides(ND, pos_cml);

	md_clear(ND, data->ksp_dims, dst, CFL_SIZE);

	apply_rolloff_correction(2., data->grid_conf.width, data->grid_conf.beta, data->N, data->cim_dims, src, src);

	do {
		float scale = 1. / sqrtf(md_calc_size(3, data->lph_dims));

		float shift[3];
		compute_shift(3, shift, data->N, data->factors, pos_cml[data->N]);

		apply_linphases_3D(data->N, data->cim_dims, shift, src, src, false, false, true, scale);
		linop_forward(data->cfft_op, data->N, data->cim_dims, src, data->N, data->cim_dims, src);
		fftmod(data->N, data->cim_dims, data->flags, src, src);

		grid2H_decomp(&(data->grid_conf), pos_cml[data->N], data->N, data->factors, data->trj_dims, multiplace_read(data->traj, src), data->ksp_dims, dst, data->cim_dims, src);

		// Recover src
		ifftmod(data->N, data->cim_dims, data->flags, src, src);
		linop_adjoint(data->cfft_op, data->N, data->cim_dims, src, data->N, data->cim_dims, src);
		apply_linphases_3D(data->N, data->cim_dims, shift, src, src, true, false, true, scale);

	} while (md_next(data->N, data->factors, data->conf.flags, pos_fac));

	if (NULL != data->weights)
		md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->out_strs, dst, data->wgh_strs, multiplace_read(data->weights, src));
}




int nufft_get_psf_dims(const struct linop_s* nufft, int N, long psf_dims[N])
{
	auto lop_data = linop_get_data(nufft);
	assert(NULL != lop_data);

	auto data = CAST_DOWN(nufft_data, lop_data);

	if (N > 0)
		md_copy_dims(N, psf_dims, data->psf_dims);

	return data->N + 1;
}


void nufft_get_psf2(const struct linop_s* nufft, int N, const long psf_dims[N], const long psf_strs[N], complex float* psf)
{
	auto lop_data = linop_get_data(nufft);
	assert(NULL != lop_data);

	auto data = CAST_DOWN(nufft_data, lop_data);

	assert(N == data->N + 1);
	md_check_equal_dims(N, psf_dims, data->psf_dims, ~0UL);

	md_copy2(N, psf_dims, psf_strs, psf, data->psf_strs, multiplace_read(data->psf, psf), CFL_SIZE);
}


void nufft_get_psf(const struct linop_s* nufft, int N, const long psf_dims[N], complex float* psf)
{
	nufft_get_psf2(nufft, N, psf_dims, MD_STRIDES(N, psf_dims, CFL_SIZE), psf);
}

void nufft_update_traj(	const struct linop_s* nufft, int N,
			const long trj_dims[N], const complex float* traj,
			const long wgh_dims[N], const complex float* weights,
			const long bas_dims[N], const complex float* basis)
{
	auto _data = linop_get_data_nested(nufft);
	assert (NULL != _data);

	auto data = CAST_DOWN(nufft_data, _data);

	assert(data->N == N);

	nufft_set_traj(data, N, trj_dims, traj, wgh_dims, weights, bas_dims, basis);
}

void nufft_update_psf2(const struct linop_s* nufft, int ND, const long psf_dims[ND], const long psf_strs[ND], const complex float* psf)
{
	auto _data = linop_get_data_nested(nufft);
	assert (NULL != _data);

	auto data = CAST_DOWN(nufft_data, _data);

	assert(md_check_equal_dims(ND, data->psf_dims, psf_dims, ~0UL));

	multiplace_free(data->psf);

	data->psf = multiplace_move2(ND, psf_dims, psf_strs, CFL_SIZE, psf);
}

void nufft_update_psf(const struct linop_s* nufft, int ND, const long psf_dims[ND], const complex float* psf)
{
	nufft_update_psf2(nufft, ND, psf_dims, MD_STRIDES(ND, psf_dims, CFL_SIZE), psf);
}


