/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */ 

#ifdef __cplusplus
extern "C" {
#endif

#include "misc/mri.h"



typedef void (*grecon_fun_t)(void* param, const long dims1[DIMS], _Complex float* out1, 
	const long sens1_dims[DIMS], _Complex float* cov1, const long w1_dims[DIMS], const _Complex float* weights,
	_Complex float* kspace1, _Bool gpu);


extern void parslices(grecon_fun_t grecon, void* param, const long dims[DIMS],
	_Complex float* image, const long sens_dims[DIMS], const _Complex float* sens_maps,
	const long pat_dims[DIMS], const _Complex float* pattern,
	const _Complex float* kspace_data, _Bool output_ksp, _Bool gpu);

// strided version
extern void parslices2(grecon_fun_t grecon, void* param, const long dims[DIMS],
	const long ostr[DIMS], _Complex float* image,
	const long sens_dims[DIMS], const _Complex float* sens_maps,
	const long pat_dims[DIMS], const _Complex float* pattern,
	const long istr[DIMS], const _Complex float* kspace_data,
	_Bool output_ksp, _Bool gpu);


#ifdef __cplusplus
}
#endif



