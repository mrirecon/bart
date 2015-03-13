/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>


#include "misc/mri.h"

#ifndef MAX_LEV
#define MAX_LEV 100
#endif

struct operator_p_s;


// Low rank thresholding for arbitrary block sizes
extern const struct operator_p_s* lrthresh_create(const long dims_lev[DIMS], _Bool randshift, unsigned long mflags, const long blkdims[MAX_LEV][DIMS], float lambda, _Bool noise, int remove_mean, _Bool use_gpu);

// Returns nuclear norm using lrthresh operator
extern float lrnucnorm(const struct operator_p_s* op, const complex float* src);

// Generates multiscale block sizes
extern long multilr_blkdims(long blkdims[MAX_LEV][DIMS], unsigned long flags, const long dims[DIMS], int blkskip, long initblk);

// Generates locally low rank block size
extern long llr_blkdims(long blkdims[MAX_LEV][DIMS], unsigned long flags, const long dims[DIMS], long llrblk);

// Generates low rank plus sparse block size
extern long ls_blkdims(long blkdims[MAX_LEV][DIMS], const long dims[DIMS]);


extern void add_lrnoiseblk(long* level, long blkdims[MAX_LEV][DIMS], const long dims[DIMS]);

// Return the regularization parameter
extern float get_lrthresh_lambda(const struct operator_p_s* o);
