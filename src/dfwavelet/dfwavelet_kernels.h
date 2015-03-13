/*
 * Copyright 2013-2015 The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
 

#ifdef __cplusplus
extern "C" {
#endif




  struct dfwavelet_plan_s;

  /* GPU Host Funcstion */
  extern void dffwt3_gpuHost(struct dfwavelet_plan_s* plan, _Complex float* out_wcdf1,_Complex float* out_wcdf2,_Complex float* out_wcn, _Complex float* in_vx,_Complex float* in_vy,_Complex float* in_vz);
  extern void dfiwt3_gpuHost(struct dfwavelet_plan_s* plan, _Complex float* out_vx,_Complex float* out_vy,_Complex float* out_vz, _Complex float* in_wcdf1,_Complex float* in_wcdf2,_Complex float* in_wcn);
  extern void dfsoftthresh_gpuHost(struct dfwavelet_plan_s* plan,float dfthresh, float nthresh, _Complex float* out_wcdf1,_Complex float* out_wcdf2,_Complex float* out_wcn);

  extern void dfwavthresh3_gpuHost(struct dfwavelet_plan_s* plan,float dfthresh, float nthresh,_Complex float* out_vx,_Complex float* out_vy,_Complex float* out_vz,_Complex float* in_vx,_Complex float* in_vy,_Complex float* in_vz);

  /* GPU Funcstion */
  extern void dffwt3_gpu(struct dfwavelet_plan_s* plan, _Complex float* out_wcdf1,_Complex float* out_wcdf2,_Complex float* out_wcn, _Complex float* in_vx,_Complex float* in_vy,_Complex float* in_vz);
  extern void dfiwt3_gpu(struct dfwavelet_plan_s* plan, _Complex float* out_vx,_Complex float* out_vy,_Complex float* out_vz, _Complex float* in_wcdf1,_Complex float* in_wcdf2,_Complex float* in_wcn);
  extern void dfsoftthresh_gpu(struct dfwavelet_plan_s* plan,float dfthresh, float nthresh, _Complex float* out_wcdf1,_Complex float* out_wcdf2,_Complex float* out_wcn);

  extern void dfwavthresh3_gpu(struct dfwavelet_plan_s* plan,float dfthresh, float nthresh,_Complex float* out_vx,_Complex float* out_vy,_Complex float* out_vz,_Complex float* in_vx,_Complex float* in_vy,_Complex float* in_vz);

  extern void circshift_gpu(struct dfwavelet_plan_s* plan, _Complex float* data);
  extern void circunshift_gpu(struct dfwavelet_plan_s* plan, _Complex float* data);

#ifdef __cplusplus
}
#endif
