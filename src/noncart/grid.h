/* Copyright 2014 The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#include <complex.h>
#include "misc/mri.h"

float kb_table128[129];

extern void grid(float os, float width, double beta, float scale, const float shifts[3], const complex float* traj, const complex float* pat, const long grid_dims[DIMS], complex float* grid, const long ksp_dims[DIMS], const complex float* src);

extern void gridH(float os, float width, double beta, float scale, const float shifts[3], const complex float* traj, const complex float* pat, const long ksp_dims[DIMS], complex float* dst, const long grid_dims[DIMS], const complex float* grid);

extern void grid_pointH(unsigned int ch, const long dims[3], const float pos[3], complex float val[ch], const complex float* src, float width, int kb_size, float kb_table[kb_size+1]);
extern void grid_point(unsigned int ch, const long dims[3], const float pos[3], complex float* dst, const complex float val[ch], float width, int kb_size, float kb_table[kb_size+1]);


extern void rolloff_correction(float os, float width, const long dim[3], complex float* dst);


extern void grid_line3d(unsigned int n, unsigned int ch, const long dims[3], const float start[3], const float end[3], complex float* dst, const complex float* src, float width, int kb_size, float kb_table[kb_size+1]);
extern void grid_radial(const long dimensions[3], unsigned int samples, unsigned int channels, unsigned int spokes, unsigned int sp, complex float* dst, const complex float* src);
extern void grid_line(const long dimensions[3], unsigned int samples, unsigned int channels, unsigned int phencs, unsigned int pe, complex float* dst, const complex float* src);
extern void density_compensation(unsigned int samples, unsigned int channels, unsigned int spokes, complex float* dst, const complex float* src);
extern void density_comp3d(const long dim[3], complex float* kspace);

