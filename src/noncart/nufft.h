/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016-2020. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2022. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

struct operator_s;
struct linop_s;

struct nufft_conf_s {

	_Bool toeplitz; ///< Toeplitz embedding boolean for A^T A
	_Bool pcycle; /// < Phase cycling
	_Bool periodic;
	_Bool lowmem;
	int loopdim;
	unsigned long flags;
	unsigned long cfft;
	_Bool decomp;
	_Bool nopsf;
	_Bool cache_psf_grdding;

	_Bool precomp_linphase;
	_Bool precomp_fftmod;
	_Bool precomp_roll;
};

extern struct nufft_conf_s nufft_conf_defaults;


extern struct linop_s* nufft_create(int N,				///< Number of dimensions
				    const long ksp_dims[__VLA(N)],	///< Kspace dimension
				    const long coilim_dims[__VLA(N)],	///< Coil image dimension
				    const long traj_dims[__VLA(N)],	///< Trajectory dimension
				    const _Complex float* traj,		///< Trajectory
				    const _Complex float* weights,	///< Weights, ex, density-compensation
				    struct nufft_conf_s conf);		///< NUFFT configuration

extern struct linop_s* nufft_create2(int N,
			     const long ksp_dims[N],
			     const long cim_dims[N],
			     const long traj_dims[N],
			     const complex float* traj,
			     const long wgh_dims[N],
			     const complex float* weights,
			     const long bas_dims[N],
			     const complex float* basis,
			     struct nufft_conf_s conf);

extern _Complex float* compute_psf(int N,
				   const long img2_dims[__VLA(N)],
				   const long trj_dims[__VLA(N)],
				   const complex float* traj,
				   const long bas_dims[__VLA(N)],
				   const complex float* basis,
				   const long wgh_dims[__VLA(N)],
				   const complex float* weights,
				   _Bool periodic,
				   _Bool lowmem);


extern const struct operator_s* nufft_precond_create(const struct linop_s* nufft_op);

extern struct linop_s* nufft_create_normal(int N, const long cim_dims[__VLA(N)],
					   int ND, const long psf_dims[__VLA(ND)], const _Complex float* psf,
					   _Bool basis, struct nufft_conf_s conf);

extern void nufft_update_traj(const struct linop_s* nufft, int N,
			const long trj_dims[__VLA(N)], const _Complex float* traj,
			const long wgh_dims[__VLA(N)], const _Complex float* weights,
			const long bas_dims[__VLA(N)], const _Complex float* basis);
extern void nufft_update_psf(const struct linop_s* nufft, int ND, const long psf_dims[__VLA(ND)], const _Complex float* psf);
extern void nufft_update_psf2(const struct linop_s* nufft, int ND, const long psf_dims[__VLA(ND)], const long psf_strs[__VLA(ND)], const _Complex float* psf);

extern int nufft_get_psf_dims(const struct linop_s* nufft, int N, long psf_dims[N]);
extern void nufft_get_psf2(const struct linop_s* nufft, int N, const long psf_dims[N], const long psf_strs[N], _Complex float* psf);
extern void nufft_get_psf(const struct linop_s* nufft, int N, const long psf_dims[N], _Complex float* psf);

#include "misc/cppwrap.h"

