/* Copyright 2014 The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>


#include "misc/mri.h"

struct linop_s;
struct operator_p_s;

// Nufft operator
extern struct linop_s* nufft_create( const long ksp_dims[DIMS], const long coilim_dims[DIMS], const complex float* traj, const complex float* pat, _Bool toeplitz, _Bool precond, _Bool stoch, void* cgconf, _Bool use_gpu);



// Method to estimate image
extern void estimate_im_dims( long dims[DIMS], long tdims[2], complex float* traj );
