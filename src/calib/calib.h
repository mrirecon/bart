/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __CALIB_H
#define __CALIB_H
 

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

#include "misc/mri.h"

struct ecalib_conf {

	long kdims[3];
	float threshold;
	int numsv;
	float percentsv;
	_Bool weighting;
	_Bool softcrop;
	float crop;
	_Bool orthiter;
	_Bool usegpu;
	float perturb;
};

extern const struct ecalib_conf ecalib_defaults;

extern void calib(const struct ecalib_conf* conf, const long out_dims[KSPACE_DIMS], _Complex float* out_data, _Complex float* eptr, 
			unsigned int SN, float svals[SN], const long calreg_dims[KSPACE_DIMS], const _Complex float* calreg_data);

extern void calib2(const struct ecalib_conf* conf, const long out_dims[KSPACE_DIMS], _Complex float* out_data, _Complex float* eptr, unsigned int SN, float svals[SN], const long calreg_dims[KSPACE_DIMS], const _Complex float* data, const long msk_dims[3], const _Bool* msk);

extern void eigenmaps(const long out_dims[KSPACE_DIMS], _Complex float* out_data, _Complex float* eptr, const _Complex float* imgcov, const long msk_dims[3], const _Bool* msk, _Bool orthiter, _Bool usegpu);
extern void calmat_aha(const long kdims[3], _Complex float* cov, const long calreg_dims[4], const _Complex float* calreg_data);

extern void fixphase(unsigned int D, const long dims[__VLA(D)], unsigned int dim, _Complex float* out, const _Complex float* in);
extern void crop_sens(const long dims[KSPACE_DIMS], _Complex float* ptr, float crth, const _Complex float* map);

extern void calone_dims(const struct ecalib_conf* conf, long cov_dims[4], long channels);
extern void calone(const struct ecalib_conf* conf, const long cov_dims[4], _Complex float* cov, unsigned int SN, float svals[SN], const long calreg_dims[KSPACE_DIMS], const _Complex float* cal_data);
extern void caltwo(const struct ecalib_conf* conf, const long out_dims[KSPACE_DIMS], _Complex float* out_data, _Complex float* emaps, const long in_dims[4], _Complex float* in_data, const long msk_dims[3], const _Bool* msk);
extern _Complex float* calibration_matrix(long calmat_dims[2], const long kdims[3], const long calreg_dims[4], const _Complex float* data);
extern void compute_imgcov(const long cov_dims[4], _Complex float* imgcov, const long nskerns_dims[5], const _Complex float* nskerns);
extern void compute_nskerns(const struct ecalib_conf* conf, long nskerns_dims[5], _Complex float** nskerns_ptr, unsigned int SN, float svals[SN], const long caldims[KSPACE_DIMS], const _Complex float* caldata);

#ifdef __cplusplus
}
#endif

#endif	// __CALIB_H
