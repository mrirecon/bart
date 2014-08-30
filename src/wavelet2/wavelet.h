/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 
 
#ifndef __WAVELET_H
#define __WAVELET_H

#ifdef __cplusplus
extern "C" {
#define __VLA(x) 
#define __VLA2(x) x
#else
#define __VLA(x) static x
#define __VLA2(x) x
#endif



struct wavelet_plan_s;

// Few things to note about the functions:
// 1) use_gpu accepts 2 values: 
//    0=cpu, 1=gpu_to_gpu
// 2) prepare_wavelet_plan uses Daubechies 2 by default
// 3) prepare_wavelet_plan_filters currently has 3 filters available:
//    [name, param1, param2]
//    a) ["Haar",0,0] b) ["Daubechies",2,0] c) ["CDF", 4,4] (also known as 9,7)
// 4) soft_thresh applies soft thresholding on wavelet coefficeints
//    while wavelet_thresh does fwt, soft_thresh and iwt and outputs an image
// 5) wavelet_thresh_randshift does wavelet_thresh + randshift and unshift
// 6) The input image for the forward transform is random shifted and then unshifted
//    at the end of the function to return the input image to the original state
//    (not strictly const).

extern const float wavelet2_haar[4][2];
extern const float wavelet2_dau2[4][4];
extern const float wavelet2_cdf44[4][10];

struct operator_p_s;

// operator interface

	extern const struct linop_s* wavelet_create(int numdims, const long imSize[__VLA(numdims)], unsigned int wave_flags, const long minSize_tr[__VLA(numdims)], _Bool randshift, _Bool use_gpu);

	extern const struct operator_p_s* prox_wavethresh_create(int numdims, const long imSize[__VLA(numdims)], unsigned int wave_flags, const long minSize_tr[__VLA(numdims)], float lambda, _Bool randshift, _Bool use_gpu);

extern struct wavelet_plan_s* prepare_wavelet_plan(int numdims, const long imSize[__VLA(numdims)], unsigned int flags, const long minSize_tr[__VLA(numdims)], int use_gpu);
  extern struct wavelet_plan_s* prepare_wavelet_plan_filters(int numdims, const long imSize[__VLA(numdims)], unsigned int flags, const long minSize_tr[__VLA(numdims)], int use_gpu, int filter_length, const float filter[4][__VLA2(filter_length)]);
	extern void wavelet_forward(const void* _data, _Complex float* outCoeff, const _Complex float* inImage);
extern void wavelet_inverse(const void* _data, _Complex float* outImage, const _Complex float* inCoeff);
extern void soft_thresh(struct wavelet_plan_s* plan, _Complex float* coeff, float thresh);
extern void wavelet_thresh(const void* _data, float thresh, _Complex float* outImage, const _Complex float* inImage);
extern void wavelet_new_randshift(struct wavelet_plan_s* plan);
extern void wavelet_clear_randshift(struct wavelet_plan_s* plan);
extern void wavelet_free(const struct wavelet_plan_s* plan);

extern long get_numCoeff_tr(struct wavelet_plan_s* plan);

#ifdef __cplusplus
}
#endif
#undef __VLA

#endif // __WAVELET_H

