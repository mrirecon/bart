
#ifndef _MRI2_H
#define _MRI2_H

#include "misc/cppwrap.h"

#ifndef DIMS
#define DIMS		16
#endif

extern void estimate_pattern(int D, const long dims[__VLA(D)], unsigned long flags, _Complex float* pattern, const _Complex float* kspace_data);
extern _Complex float* extract_calib(long caldims[DIMS], const long calsize[3], const long in_dims[DIMS], const _Complex float* in_data, _Bool fixed);
extern _Complex float* extract_calib2(long caldims[DIMS], const long calsize[3], const long in_dims[DIMS], const long in_strs[DIMS], const _Complex float* in_data, _Bool fixed);
extern void data_consistency(const long dims[DIMS], _Complex float* dst, const _Complex float* pattern, const _Complex float* kspace1, const _Complex float* kspace2);
extern void calib_geom(long caldims[DIMS], long calpos[DIMS], const long calsize[3], const long in_dims[DIMS], const _Complex float* in_data);

extern void estimate_im_dims(int N, unsigned long flags, long dims[__VLA(N)], const long tdims[__VLA(N)], const _Complex float* traj);
extern void estimate_fast_sq_im_dims(int N, long dims[3], const long tdims[__VLA(N)], const _Complex float* traj);

#include "misc/cppwrap.h"

#endif	// _MRI2_H

