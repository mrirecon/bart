/*
 * Copyright 2013-2015 The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef __DFWAVELET_H
#define __DFWAVELET_H

#include "misc/cppwrap.h"

struct dfwavelet_plan_s;

extern struct dfwavelet_plan_s* prepare_dfwavelet_plan(int numdims, long* imSize, long* minSize_tr, _Complex float* res, int use_gpu);

extern void dfwavelet_forward(struct dfwavelet_plan_s* plan, _Complex float* out_wcdf1, _Complex float* out_wcdf2, _Complex float* out_wcn, _Complex float* in_vx, _Complex float* in_vy, _Complex float* in_vz);

extern void dfwavelet_inverse(struct dfwavelet_plan_s* plan, _Complex float* out_vx, _Complex float* out_vy, _Complex float* out_vz, _Complex float* in_wcdf1, _Complex float* in_wcdf2, _Complex float* in_wcn);

extern void dfsoft_thresh(struct dfwavelet_plan_s* plan, float dfthresh, float nthresh, _Complex float* wcdf1, _Complex float* wcdf2, _Complex float* wcn);

extern void dfwavelet_thresh(struct dfwavelet_plan_s* plan, float dfthresh, float nthresh,_Complex float* out_vx, _Complex float* out_vy, _Complex float* out_vz, _Complex float* in_vx,_Complex float* in_vy, _Complex float* in_vz);

extern void dfwavelet_new_randshift(struct dfwavelet_plan_s* plan);
extern void dfwavelet_clear_randshift(struct dfwavelet_plan_s* plan);
extern void dfwavelet_free(struct dfwavelet_plan_s* plan);
extern void print_plan(struct dfwavelet_plan_s* plan);

#include "misc/cppwrap.h"
#endif // __WAVELET_H

