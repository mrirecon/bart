/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */ 

#ifdef __cplusplus
extern "C" {
#define __VLA(x) 
#else
#define __VLA(x) static x
#endif



  struct wavelet_plan_s;

  extern void fwt2_gpuHost(struct wavelet_plan_s* plan, _Complex float* out, const _Complex float* in);
  extern void iwt2_gpuHost(struct wavelet_plan_s* plan, _Complex float* out, const _Complex float* in);
  extern void wavthresh2_gpuHost(struct wavelet_plan_s* plan, _Complex float* out, const _Complex float* in, float thresh);
  extern void fwt3_gpuHost(struct wavelet_plan_s* plan, _Complex float* out, const _Complex float* in);
  extern void iwt3_gpuHost(struct wavelet_plan_s* plan, _Complex float* out, const _Complex float* in);
  extern void wavthresh3_gpuHost(struct wavelet_plan_s* plan, _Complex float* out, const _Complex float* in, float thresh);
  extern void softthresh_gpuHost(struct wavelet_plan_s* plan, _Complex float* in, float thresh);

  extern void fwt2_gpu(struct wavelet_plan_s* plan, _Complex float* out, const _Complex float* in);
  extern void iwt2_gpu(struct wavelet_plan_s* plan, _Complex float* out, const _Complex float* in);
  extern void wavthresh2_gpu(struct wavelet_plan_s* plan, _Complex float* out, const _Complex float* in, float thresh);
  extern void fwt3_gpu(struct wavelet_plan_s* plan, _Complex float* out, const _Complex float* in);
  extern void iwt3_gpu(struct wavelet_plan_s* plan, _Complex float* out, const _Complex float* in);
  extern void wavthresh3_gpu(struct wavelet_plan_s* plan, _Complex float* out, const _Complex float* in, float thresh);
  extern void softthresh_gpu(struct wavelet_plan_s* plan, _Complex float* in, float thresh);
  extern void circshift_gpu(struct wavelet_plan_s* plan, _Complex float* data);
  extern void circunshift_gpu(struct wavelet_plan_s* plan, _Complex float* data);


  extern void prepare_wavelet_filters_gpu(struct wavelet_plan_s* plan,int filterLen,const float* filter);
  extern void prepare_wavelet_temp_gpu(struct wavelet_plan_s* plan);
  extern void wavelet_free_gpu(const struct wavelet_plan_s* plan);

#ifdef __cplusplus
}
#endif
