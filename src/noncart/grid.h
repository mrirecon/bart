/* Copyright 2014-2015 The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#include <complex.h>

#include "misc/cppwrap.h"


struct grid_conf_s {

	float os;
	float width;
	_Bool periodic;
	double beta;
};

extern void grid(const struct grid_conf_s* conf, const complex float* traj, const long grid_dims[4], complex float* grid, const long ksp_dims[4], const complex float* src);

extern void gridH(const struct grid_conf_s* conf, const complex float* traj, const long ksp_dims[4], complex float* dst, const long grid_dims[4], const complex float* grid);


extern void grid2(const struct grid_conf_s* conf, unsigned int D, const long trj_dims[__VLA(D)], const complex float* traj, const long grid_dims[__VLA(D)], complex float* grid, const long ksp_dims[__VLA(D)], const complex float* src);

extern void grid2H(const struct grid_conf_s* conf, unsigned int D, const long trj_dims[__VLA(D)], const complex float* traj, const long ksp_dims[__VLA(D)], complex float* dst, const long grid_dims[__VLA(D)], const complex float* grid);


extern void grid_pointH(unsigned int ch, int N, const long dims[__VLA(N)], const float pos[__VLA(N)], complex float val[__VLA(ch)], const complex float* src, _Bool periodic, float width, int kb_size, const float kb_table[__VLA(kb_size + 1)]);
extern void grid_point(unsigned int ch, int N, const long dims[__VLA(N)], const float pos[__VLA(N)], complex float* dst, const complex float val[__VLA(ch)], _Bool periodic, float width, int kb_size, const float kb_table[__VLA(kb_size + 1)]);

extern double calc_beta(float os, float width);

extern void rolloff_correction(float os, float width, float beta, const long dim[3], complex float* dst);


#include "misc/cppwrap.h"

