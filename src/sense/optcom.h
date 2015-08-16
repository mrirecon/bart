/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#include "misc/mri.h"

extern void rss_combine(const long dims[DIMS], _Complex float* image, const _Complex float* data);
extern void optimal_combine(const long dims[DIMS], float alpha, _Complex float* image, const _Complex float* sens, const _Complex float* data);
extern float estimate_scaling_norm(float rescale, long imsize, _Complex float* tmpnorm, bool compat);
extern float estimate_scaling(const long dims[DIMS], const _Complex float* sens, const _Complex float* data);
extern float estimate_scaling2(const long dims[DIMS], const _Complex float* sens, const long strs[DIMS], const _Complex float* data);
extern float estimate_scaling_old2(const long dims[DIMS], const _Complex float* sens, const long strs[DIMS], const _Complex float* data);
extern void fake_kspace(const long dims[DIMS], _Complex float* kspace, const _Complex float* sens, const _Complex float* image);
extern void replace_kspace(const long dims[DIMS], _Complex float* out, const _Complex float* kspace, const _Complex float* sens, const _Complex float* image);
extern void replace_kspace2(const long dims[DIMS], _Complex float* out, const _Complex float* kspace, const _Complex float* sens, const _Complex float* image);

extern float estimate_scaling_cal(const long dims[DIMS], const _Complex float* sens, const long cal_dims[DIMS], const _Complex float* cal_data, _Bool compat);

#ifdef __cplusplus
}
#endif

