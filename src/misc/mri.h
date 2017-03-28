/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __MRI_H
#define __MRI_H

#include <stdbool.h>


#include "misc/cppwrap.h"


enum mri_dims {
	READ_DIM,
	PHS1_DIM,
	PHS2_DIM,
	COIL_DIM,
	MAPS_DIM,
	TE_DIM,
	COEFF_DIM,
	COEFF2_DIM,
	ITER_DIM,
	CSHIFT_DIM,
	TIME_DIM,
	TIME2_DIM,
	LEVEL_DIM,
	SLICE_DIM,
	AVG_DIM,
};

#ifdef BERKELEY_SVN
#define KSPACE_DIMS	16u
#endif

#ifndef DIMS
#define DIMS		16u
#endif

#define READ_FLAG (1u << READ_DIM)
#define PHS1_FLAG (1u << PHS1_DIM)
#define PHS2_FLAG (1u << PHS2_DIM)
#define COIL_FLAG (1u << COIL_DIM)
#define MAPS_FLAG (1u << MAPS_DIM)
#define TE_FLAG (1u << TE_DIM)
#define COEFF_FLAG (1u << COEFF_DIM)
#define COEFF2_FLAG (1u << COEFF2_DIM)
#define ITER_FLAG (1u << ITER_DIM)
#define CSHIFT_FLAG (1u << CSHIFT_DIM)
#define TIME_FLAG (1u << TIME_DIM)
#define TIME2_FLAG (1u << TIME2_DIM)
#define LEVEL_FLAG (1u << LEVEL_DIM)

#define FFT_FLAGS (READ_FLAG|PHS1_FLAG|PHS2_FLAG)
#define SENS_FLAGS (COIL_FLAG|MAPS_FLAG)
#define SLICE_FLAG (1u << SLICE_DIM)



extern void estimate_pattern(unsigned int D, const long dims[__VLA(D)], unsigned int flags, _Complex float* pattern, const _Complex float* kspace_data);
extern _Complex float* extract_calib(long caldims[DIMS], const long calsize[3], const long in_dims[DIMS], const _Complex float* in_data, _Bool fixed);
extern _Complex float* extract_calib2(long caldims[DIMS], const long calsize[3], const long in_dims[DIMS], const long in_strs[DIMS], const _Complex float* in_data, _Bool fixed);
extern void data_consistency(const long dims[DIMS], _Complex float* dst, const _Complex float* pattern, const _Complex float* kspace1, const _Complex float* kspace2);
extern void calib_geom(long caldims[DIMS], long calpos[DIMS], const long calsize[3], const long in_dims[DIMS], const _Complex float* in_data);

#include "misc/cppwrap.h"

#endif	// __MRI_H

