/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
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
	unsigned long flags;
	unsigned long cfft;
};

extern struct nufft_conf_s nufft_conf_defaults;


extern struct linop_s* nufft_create(unsigned int N,			///< Number of dimensions
				    const long ksp_dims[__VLA(N)],	///< Kspace dimension
				    const long coilim_dims[__VLA(N)],	///< Coil image dimension
				    const long traj_dims[__VLA(N)],	///< Trajectory dimension
				    const _Complex float* traj,		///< Trajectory
				    const _Complex float* weights,	///< Weights, ex, density-compensation
				    struct nufft_conf_s conf);		///< NUFFT configuration

extern struct linop_s* nufft_create2(unsigned int N,
			     const long ksp_dims[N],
			     const long cim_dims[N],
			     const long traj_dims[N],
			     const complex float* traj,
			     const long wgh_dims[N],
			     const complex float* weights,
			     const long bas_dims[N],
			     const complex float* basis,
			     struct nufft_conf_s conf);

extern _Complex float* compute_psf(unsigned int N,
				   const long img2_dims[__VLA(N)],
				   const long trj_dims[__VLA(N)],
				   const complex float* traj,
				   const long bas_dims[__VLA(N)],
				   const complex float* basis,
				   const long wgh_dims[__VLA(N)],
				   const complex float* weights,
				   _Bool periodic);

extern void estimate_im_dims(int N, unsigned long flags, long dims[__VLA(N)], const long tdims[__VLA(N)], const complex float* traj);

extern const struct operator_s* nufft_precond_create(const struct linop_s* nufft_op);

#include "misc/cppwrap.h"

