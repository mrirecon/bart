/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __CALIB_H
#define __CALIB_H
 
#include "misc/cppwrap.h"
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
	_Bool intensity;
	_Bool rotphase;
	float var;
	_Bool automate;
};

extern const struct ecalib_conf ecalib_defaults;

extern void calib(const struct ecalib_conf* conf, const long out_dims[DIMS], _Complex float* out_data, _Complex float* eptr, 
			unsigned int SN, float svals[__VLA2(SN)], const long calreg_dims[DIMS], const _Complex float* calreg_data);

extern void calib2(const struct ecalib_conf* conf, const long out_dims[DIMS], _Complex float* out_data, _Complex float* eptr, unsigned int SN, float svals[__VLA2(SN)], const long calreg_dims[DIMS], const _Complex float* data, const long msk_dims[3], const _Bool* msk);

extern void eigenmaps(const long out_dims[DIMS], _Complex float* out_data, _Complex float* eptr, const _Complex float* imgcov, const long msk_dims[3], const _Bool* msk, _Bool orthiter, _Bool usegpu);


extern void crop_sens(const long dims[DIMS], _Complex float* ptr, bool soft, float crth, const _Complex float* map);

extern void calone_dims(const struct ecalib_conf* conf, long cov_dims[4], long channels);
extern void calone(const struct ecalib_conf* conf, const long cov_dims[4], _Complex float* cov, unsigned int SN, float svals[__VLA2(SN)], const long calreg_dims[DIMS], const _Complex float* cal_data);
extern void caltwo(const struct ecalib_conf* conf, const long out_dims[DIMS], _Complex float* out_data, _Complex float* emaps, const long in_dims[4], _Complex float* in_data, const long msk_dims[3], const _Bool* msk);
extern void compute_imgcov(const long cov_dims[4], _Complex float* imgcov, const long nskerns_dims[5], const _Complex float* nskerns);
extern void compute_kernels(const struct ecalib_conf* conf, long nskerns_dims[5], _Complex float** nskerns_ptr, unsigned int SN, float svals[__VLA2(SN)], const long caldims[DIMS], const _Complex float* caldata);

#include "misc/cppwrap.h"
#endif	// __CALIB_H
