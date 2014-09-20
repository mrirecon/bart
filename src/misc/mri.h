/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __MRI_H
#define __MRI_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#ifndef __VLA
#define __VLA(x) 
#endif
#else
#ifndef __VLA
#define __VLA(x) static x
#endif
#endif

#define READ_DIM	0u
#define PHS1_DIM	1u
#define PHS2_DIM	2u
#define COIL_DIM	3u
#define MAPS_DIM	4u
#define TE_DIM		5u
#define COEFF_DIM	6u
#define ITER_DIM	7u
#define CSHIFT_DIM	8u
#define TIME_DIM	9u
#define TIME2_DIM	10u
#define LEVEL_DIM	11u
#define COEFF2_DIM	12u
#define SLICE_DIM	13u
#define KSPACE_DIMS	16u

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



struct transfer_data_s {
	
	long dims[DIMS];
	long strs[DIMS];
	long strs_tf[DIMS];
};

typedef void (*transfer_fun_t)(void* data, const _Complex float* pattern, _Complex float* dst, const _Complex float* src);

extern void transfer_function(void* _data, const _Complex float* pattern, _Complex float* dst, const _Complex float* src);
extern void estimate_pattern(unsigned int D, const long dims[__VLA(D)], unsigned int dim, _Complex float* pattern, const _Complex float* kspace_data);
extern _Complex float* extract_calib(long caldims[DIMS], const long calsize[3], const long in_dims[DIMS], const _Complex float* in_data, _Bool fixed);
extern _Complex float* extract_calib2(long caldims[DIMS], const long calsize[3], const long in_dims[DIMS], const long in_strs[DIMS], const _Complex float* in_data, _Bool fixed);
extern void data_consistency(const long dims[DIMS], _Complex float* dst, const _Complex float* pattern, const _Complex float* kspace1, const _Complex float* kspace2);
extern void calib_geom(long caldims[DIMS], long calpos[DIMS], const long calsize[3], const long in_dims[DIMS], const _Complex float* in_data);


#ifdef __cplusplus
}
#endif



#endif	// __MRI_H

