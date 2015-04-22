/* Copyright 2014-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */


struct linop_s;

struct nufft_conf_s {

	_Bool toeplitz;
};

extern struct nufft_conf_s nufft_conf_defaults;


extern struct linop_s* nufft_create(unsigned int N, const long ksp_dims[N], const long coilim_dims[N], const long traj_dims[N], const _Complex float* traj, const _Complex float* weights, struct nufft_conf_s conf, _Bool use_gpu);

extern void estimate_im_dims(unsigned int N, long dims[3], const long tdims[N], const _Complex float* traj);
extern _Complex float* compute_psf(unsigned int N, const long img2_dims[N], const long trj_dims[N], const complex float* traj, const complex float* weights);

