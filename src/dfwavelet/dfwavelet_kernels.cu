/* 
 * Copyright 2013-2015 The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Frank Ong, Martin Uecker, Pat Virtue, and Mark Murphy
 * frankong@berkeley.edu
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <cuda.h>

#include "num/multind.h"

#include "dfwavelet_kernels.h"
#include "dfwavelet_impl.h"

#  define _hdev_ __host__ __device__

// _data_t is the interal representation of data_t in CUDA
// Must be float2/double2 for data_t=Complex float/double or float/double for data_t=float/double
typedef float2 _data_t;

// Float2 Operators
inline _hdev_ float2 operator+ (float2 z1, float2 z2) {
  return make_float2 (z1.x + z2.x, z1.y + z2.y);		
}
inline _hdev_ float2 operator- (float2 z1, float2 z2) {
  return make_float2 (z1.x - z2.x, z1.y - z2.y);		
}
inline _hdev_ float2 operator* (float2 z1, float2 z2) {
  return make_float2 (z1.x*z2.x - z1.y*z2.y, z1.x*z2.y + z1.y*z2.x);		
}
inline _hdev_ float2 operator* (float2 z1, float alpha) {
  return make_float2 (z1.x*alpha, z1.y*alpha);		
}
inline _hdev_ float2 operator* (float alpha,float2 z1) {
  return make_float2 (z1.x*alpha, z1.y*alpha);		
}
inline _hdev_ float2 operator/ (float alpha,float2 z1) {
  return make_float2 (1.f/z1.x, 1.f/z1.y);		
}
inline _hdev_ void operator+= (float2 &z1, float2 z2) {
  z1.x += z2.x;
  z1.y += z2.y;		
}
inline _hdev_ float abs(float2 z1) {
  return sqrt(z1.x*z1.x + z1.y*z1.y);		
}

// Double2 Operators
inline _hdev_ double2 operator+ (double2 z1, double2 z2) {
  return make_double2 (z1.x + z2.x, z1.y + z2.y);		
}
inline _hdev_ double2 operator- (double2 z1, double2 z2) {
  return make_double2 (z1.x - z2.x, z1.y - z2.y);		
}
inline _hdev_ double2 operator* (double2 z1, double2 z2) {
  return make_double2 (z1.x*z2.x - z1.y*z2.y, z1.x*z2.y + z1.y*z2.x);		
}
inline _hdev_ double2 operator* (double2 z1, double alpha) {
  return make_double2 (z1.x*alpha, z1.y*alpha);		
}
inline _hdev_ double2 operator* (double alpha,double2 z1) {
  return make_double2 (z1.x*alpha, z1.y*alpha);		
}
inline _hdev_ double2 operator/ (double alpha,double2 z1) {
  return make_double2 (1.f/z1.x, 1.f/z1.y);		
}
inline _hdev_ void operator+= (double2 &z1, double2 z2) {
  z1.x += z2.x;
  z1.y += z2.y;		
}
inline _hdev_ double abs(double2 z1) {
  return sqrt(z1.x*z1.x + z1.y*z1.y);		
}

/********** Macros ************/
#define cuda(Call) do {					\
    cudaError_t err = cuda ## Call ;			\
    if (err != cudaSuccess){				\
      fprintf(stderr, "%s\n", cudaGetErrorString(err));	\
      throw;						\
    }							\
  } while(0)

#define cuda_sync() do{				\
    cuda (ThreadSynchronize());			\
    cuda (GetLastError());			\
  } while(0)


/********** Macros ************/
#define cuda(Call) do {					\
    cudaError_t err = cuda ## Call ;			\
    if (err != cudaSuccess){				\
      fprintf(stderr, "%s\n", cudaGetErrorString(err));	\
      throw;						\
    }							\
  } while(0)

#define cuda_sync() do{				\
    cuda (ThreadSynchronize());			\
    cuda (GetLastError());			\
  } while(0)

// ############################################################################
// Headers
// ############################################################################
static __global__ void cu_fwt3df_col(_data_t *Lx,_data_t *Hx,_data_t *in,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,scalar_t *lod,scalar_t *hid,int filterLen);
static __global__ void cu_fwt3df_row(_data_t *Ly,_data_t *Hy,_data_t *in,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,scalar_t *lod,scalar_t *hid,int filterLen);
static __global__ void cu_fwt3df_dep(_data_t *Lz,_data_t *Hz,_data_t *in,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,scalar_t *lod,scalar_t *hid,int filterLen);
static __global__ void cu_iwt3df_dep(_data_t *out,_data_t *Lz,_data_t *Hz,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,int xOffset,int yOffset,int zOffset,scalar_t *lod,scalar_t *hid,int filterLen);
static __global__ void cu_iwt3df_row(_data_t *out,_data_t *Ly,_data_t *Hy,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,int xOffset,int yOffset,int zOffset,scalar_t *lod,scalar_t *hid,int filterLen);
static __global__ void cu_iwt3df_col(_data_t *out,_data_t *Lx,_data_t *Hx,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,int xOffset,int yOffset,int zOffset,scalar_t *lod,scalar_t *hid,int filterLen);
static __global__ void cu_fwt3df_LC1(_data_t *HxLyLz_df1,_data_t *HxLyLz_df2,_data_t *HxLyLz_n,_data_t *LxHyLz_df1,_data_t *LxHyLz_df2,_data_t *LxHyLz_n,_data_t *LxLyHz_df1,_data_t *LxLyHz_df2,_data_t *LxLyHz_n,int dxNext, int dyNext, int dzNext);
static __global__ void cu_fwt3df_LC2(_data_t* HxHyLz_df1,_data_t* HxHyLz_df2,_data_t* HxHyLz_n,_data_t* HxLyHz_df1,_data_t* HxLyHz_df2,_data_t* HxLyHz_n,_data_t* LxHyHz_df1,_data_t* LxHyHz_df2,_data_t* LxHyHz_n,int dxNext, int dyNext, int dzNext);
static __global__ void cu_fwt3df_LC1_diff(_data_t *HxLyLz_df1,_data_t *HxLyLz_df2,_data_t *HxLyLz_n,_data_t *LxHyLz_df1,_data_t *LxHyLz_df2,_data_t *LxHyLz_n,_data_t *LxLyHz_df1,_data_t *LxLyHz_df2,_data_t *LxLyHz_n,int dxNext, int dyNext, int dzNext);
static __global__ void cu_fwt3df_LC2_diff(_data_t* HxHyLz_df1,_data_t* HxHyLz_df2,_data_t* HxHyLz_n,_data_t* HxLyHz_df1,_data_t* HxLyHz_df2,_data_t* HxLyHz_n,_data_t* LxHyHz_df1,_data_t* LxHyHz_df2,_data_t* LxHyHz_n,int dxNext, int dyNext, int dzNext);
static __global__ void cu_fwt3df_LC3(_data_t* HxHyHz_df1,_data_t* HxHyHz_df2,_data_t* HxHyHz_n,int dxNext, int dyNext, int dzNext);
static __global__ void cu_iwt3df_LC1(_data_t *HxLyLz_df1,_data_t *HxLyLz_df2,_data_t *HxLyLz_n,_data_t *LxHyLz_df1,_data_t *LxHyLz_df2,_data_t *LxHyLz_n,_data_t *LxLyHz_df1,_data_t *LxLyHz_df2,_data_t *LxLyHz_n,int dx, int dy, int dz);
static __global__ void cu_iwt3df_LC2(_data_t* HxHyLz_df1,_data_t* HxHyLz_df2,_data_t* HxHyLz_n,_data_t* HxLyHz_df1,_data_t* HxLyHz_df2,_data_t* HxLyHz_n,_data_t* LxHyHz_df1,_data_t* LxHyHz_df2,_data_t* LxHyHz_n,int dx, int dy, int dz);
static __global__ void cu_iwt3df_LC1_diff(_data_t *HxLyLz_df1,_data_t *HxLyLz_df2,_data_t *HxLyLz_n,_data_t *LxHyLz_df1,_data_t *LxHyLz_df2,_data_t *LxHyLz_n,_data_t *LxLyHz_df1,_data_t *LxLyHz_df2,_data_t *LxLyHz_n,int dx, int dy, int dz);
static __global__ void cu_iwt3df_LC2_diff(_data_t* HxHyLz_df1,_data_t* HxHyLz_df2,_data_t* HxHyLz_n,_data_t* HxLyHz_df1,_data_t* HxLyHz_df2,_data_t* HxLyHz_n,_data_t* LxHyHz_df1,_data_t* LxHyHz_df2,_data_t* LxHyHz_n,int dx, int dy, int dz);
static __global__ void cu_iwt3df_LC3(_data_t* HxHyHz_df1,_data_t* HxHyHz_df2,_data_t* HxHyHz_n,int dx, int dy, int dz);

static __global__ void cu_mult(_data_t* in, _data_t mult, int maxInd);
static __global__ void cu_soft_thresh (_data_t* in, scalar_t thresh, int numMax);
static __global__ void cu_circshift(_data_t* data, _data_t* dataCopy, int dx, int dy, int dz, int shift1, int shift2, int shift3);
static __global__ void cu_circunshift(_data_t* data, _data_t* dataCopy, int dx, int dy, int dz, int shift1, int shift2, int shift3);


extern "C" void dffwt3_gpuHost(struct dfwavelet_plan_s* plan, data_t* out_wcdf1,data_t* out_wcdf2,data_t* out_wcn, data_t* in_vx,data_t* in_vy,data_t* in_vz)
{
  assert(plan->use_gpu==2);
  data_t* dev_wcdf1,*dev_wcdf2,*dev_wcn,*dev_vx,*dev_vy,*dev_vz;
  cuda(Malloc( (void**)&dev_vx, plan->numPixel*sizeof(data_t) ));
  cuda(Malloc( (void**)&dev_vy, plan->numPixel*sizeof(data_t) ));
  cuda(Malloc( (void**)&dev_vz, plan->numPixel*sizeof(data_t) ));

  cuda(Memcpy( dev_vx, in_vx, plan->numPixel*sizeof(data_t), cudaMemcpyHostToDevice ));
  cuda(Memcpy( dev_vy, in_vy, plan->numPixel*sizeof(data_t), cudaMemcpyHostToDevice ));
  cuda(Memcpy( dev_vz, in_vz, plan->numPixel*sizeof(data_t), cudaMemcpyHostToDevice ));

  cuda(Malloc( (void**)&dev_wcdf1, plan->numCoeff*sizeof(data_t) ));
  cuda(Malloc( (void**)&dev_wcdf2, plan->numCoeff*sizeof(data_t) ));
  cuda(Malloc( (void**)&dev_wcn, plan->numCoeff*sizeof(data_t) ));

  dffwt3_gpu(plan,dev_wcdf1,dev_wcdf2,dev_wcn,dev_vx,dev_vy,dev_vz);

  cuda(Memcpy( out_wcdf1, dev_wcdf1, plan->numCoeff*sizeof(data_t), cudaMemcpyDeviceToHost ));
  cuda(Memcpy( out_wcdf2, dev_wcdf2, plan->numCoeff*sizeof(data_t), cudaMemcpyDeviceToHost ));
  cuda(Memcpy( out_wcn, dev_wcn, plan->numCoeff*sizeof(data_t), cudaMemcpyDeviceToHost ));

  cuda(Free( dev_wcdf1 ));
  cuda(Free( dev_wcdf2 ));
  cuda(Free( dev_wcn ));
  cuda(Free( dev_vx ));
  cuda(Free( dev_vy ));
  cuda(Free( dev_vz ));
}

extern "C" void dfiwt3_gpuHost(struct dfwavelet_plan_s* plan, data_t* out_vx,data_t* out_vy,data_t* out_vz, data_t* in_wcdf1,data_t* in_wcdf2,data_t* in_wcn)
{
  assert(plan->use_gpu==2);
  data_t* dev_wcdf1,*dev_wcdf2,*dev_wcn,*dev_vx,*dev_vy,*dev_vz;
  cuda(Malloc( (void**)&dev_wcdf1, plan->numCoeff*sizeof(data_t) ));
  cuda(Malloc( (void**)&dev_wcdf2, plan->numCoeff*sizeof(data_t) ));
  cuda(Malloc( (void**)&dev_wcn, plan->numCoeff*sizeof(data_t) ));

  cuda(Memcpy( dev_wcdf1, in_wcdf1, plan->numCoeff*sizeof(data_t), cudaMemcpyHostToDevice ));
  cuda(Memcpy( dev_wcdf2, in_wcdf2, plan->numCoeff*sizeof(data_t), cudaMemcpyHostToDevice ));
  cuda(Memcpy( dev_wcn, in_wcn, plan->numCoeff*sizeof(data_t), cudaMemcpyHostToDevice ));

  cuda(Malloc( (void**)&dev_vx, plan->numPixel*sizeof(data_t) ));
  cuda(Malloc( (void**)&dev_vy, plan->numPixel*sizeof(data_t) ));
  cuda(Malloc( (void**)&dev_vz, plan->numPixel*sizeof(data_t) ));

  dfiwt3_gpu(plan,dev_vx,dev_vy,dev_vz,dev_wcdf1,dev_wcdf2,dev_wcn);
  cuda(Memcpy( out_vx, dev_vx, plan->numPixel*sizeof(data_t), cudaMemcpyDeviceToHost ));
  cuda(Memcpy( out_vy, dev_vy, plan->numPixel*sizeof(data_t), cudaMemcpyDeviceToHost ));
  cuda(Memcpy( out_vz, dev_vz, plan->numPixel*sizeof(data_t), cudaMemcpyDeviceToHost ));

  cuda(Free( dev_wcdf1 ));
  cuda(Free( dev_wcdf2 ));
  cuda(Free( dev_wcn ));
  cuda(Free( dev_vx ));
  cuda(Free( dev_vy ));
  cuda(Free( dev_vz ));
}

extern "C" void dfsoftthresh_gpuHost(struct dfwavelet_plan_s* plan,scalar_t dfthresh, scalar_t nthresh, data_t* out_wcdf1,data_t* out_wcdf2,data_t* out_wcn)
{
  assert(plan->use_gpu==2);
  data_t* dev_wcdf1,*dev_wcdf2,*dev_wcn;
  cuda(Malloc( (void**)&dev_wcdf1, plan->numCoeff*sizeof(data_t) ));
  cuda(Malloc( (void**)&dev_wcdf2, plan->numCoeff*sizeof(data_t) ));
  cuda(Malloc( (void**)&dev_wcn, plan->numCoeff*sizeof(data_t) ));

  cuda(Memcpy( dev_wcdf1, out_wcdf1, plan->numCoeff*sizeof(data_t), cudaMemcpyHostToDevice ));
  cuda(Memcpy( dev_wcdf2, out_wcdf2, plan->numCoeff*sizeof(data_t), cudaMemcpyHostToDevice ));
  cuda(Memcpy( dev_wcn, out_wcn, plan->numCoeff*sizeof(data_t), cudaMemcpyHostToDevice ));

  dfsoftthresh_gpu(plan,dfthresh,nthresh,dev_wcdf1,dev_wcdf2,dev_wcn);

  cuda(Memcpy( out_wcdf1, dev_wcdf1, plan->numCoeff*sizeof(data_t), cudaMemcpyDeviceToHost ));
  cuda(Memcpy( out_wcdf2, dev_wcdf2, plan->numCoeff*sizeof(data_t), cudaMemcpyDeviceToHost ));
  cuda(Memcpy( out_wcn, dev_wcn, plan->numCoeff*sizeof(data_t), cudaMemcpyDeviceToHost ));

  cuda(Free( dev_wcdf1 ));
  cuda(Free( dev_wcdf2 ));
  cuda(Free( dev_wcn ));
}

extern "C" void dfwavthresh3_gpuHost(struct dfwavelet_plan_s* plan, scalar_t dfthresh,scalar_t nthresh,data_t* out_vx,data_t* out_vy,data_t* out_vz, data_t* in_vx,data_t* in_vy,data_t* in_vz)
{
  assert(plan->use_gpu==2);
  data_t*dev_vx,*dev_vy,*dev_vz;
  cuda(Malloc( (void**)&dev_vx, plan->numPixel*sizeof(data_t) ));
  cuda(Malloc( (void**)&dev_vy, plan->numPixel*sizeof(data_t) ));
  cuda(Malloc( (void**)&dev_vz, plan->numPixel*sizeof(data_t) ));

  cuda(Memcpy( dev_vx, in_vx, plan->numPixel*sizeof(data_t), cudaMemcpyHostToDevice ));
  cuda(Memcpy( dev_vy, in_vy, plan->numPixel*sizeof(data_t), cudaMemcpyHostToDevice ));
  cuda(Memcpy( dev_vz, in_vz, plan->numPixel*sizeof(data_t), cudaMemcpyHostToDevice ));

  dfwavthresh3_gpu(plan,dfthresh,nthresh,dev_vx,dev_vy,dev_vz,dev_vx,dev_vy,dev_vz);

  cuda(Memcpy( out_vx, dev_vx, plan->numPixel*sizeof(data_t), cudaMemcpyDeviceToHost ));
  cuda(Memcpy( out_vy, dev_vy, plan->numPixel*sizeof(data_t), cudaMemcpyDeviceToHost ));
  cuda(Memcpy( out_vz, dev_vz, plan->numPixel*sizeof(data_t), cudaMemcpyDeviceToHost ));

  cuda(Free( dev_vx ));
  cuda(Free( dev_vy ));
  cuda(Free( dev_vz ));
}

extern "C" void dffwt3_gpu(struct dfwavelet_plan_s* plan, data_t* out_wcdf1,data_t* out_wcdf2,data_t* out_wcn, data_t* in_vx,data_t* in_vy,data_t* in_vz)
{
  circshift_gpu(plan,in_vx);
  circshift_gpu(plan,in_vy);
  circshift_gpu(plan,in_vz);
  
  long numCoeff, filterLen,*waveSizes;
  numCoeff = plan->numCoeff;
  waveSizes = plan->waveSizes;
  filterLen = plan->filterLen;
  int numLevels = plan->numLevels;
  // Cast from generic data_t to device compatible _data_t
  _data_t* dev_wcdf1 = (_data_t*) out_wcdf1;
  _data_t* dev_wcdf2 = (_data_t*) out_wcdf2;
  _data_t* dev_wcn = (_data_t*) out_wcn;
  _data_t* dev_in_vx = (_data_t*) in_vx;
  _data_t* dev_in_vy = (_data_t*) in_vy;
  _data_t* dev_in_vz = (_data_t*) in_vz;
  _data_t* res = (_data_t*) plan->res;
  _data_t* dev_temp1,*dev_temp2;
  cuda(Malloc( (void**)&dev_temp1, numCoeff*sizeof(_data_t) ));
  cuda(Malloc( (void**)&dev_temp2, numCoeff*sizeof(_data_t) ));

  // Get dimensions
  int dx = plan->imSize[0];
  int dy = plan->imSize[1];
  int dz = plan->imSize[2];
  int dxNext = waveSizes[0 + 3*numLevels];
  int dyNext = waveSizes[1 + 3*numLevels];
  int dzNext = waveSizes[2 + 3*numLevels];
  int blockSize = dxNext*dyNext*dzNext;

  // allocate device memory and  copy filters to device
  scalar_t *dev_filters;
  cuda(Malloc( (void**)&dev_filters, 4*plan->filterLen*sizeof(scalar_t) ));
  scalar_t *dev_lod0 = dev_filters + 0*plan->filterLen;
  scalar_t *dev_hid0 = dev_filters + 1*plan->filterLen;
  scalar_t *dev_lod1 = dev_filters + 2*plan->filterLen;
  scalar_t *dev_hid1 = dev_filters + 3*plan->filterLen;
  cuda(Memcpy( dev_lod0, plan->lod0, 2*plan->filterLen*sizeof(scalar_t), cudaMemcpyHostToDevice ));
  cuda(Memcpy( dev_lod1, plan->lod1, 2*plan->filterLen*sizeof(scalar_t), cudaMemcpyHostToDevice ));

  // Initialize variables and Pointers for FWT
  int const SHMEM_SIZE = 16384;
  int const T = 512;
  int mem, K;
  dim3 numBlocks, numThreads;

  // Temp Pointers
  _data_t *dev_tempLx,*dev_tempHx;
  dev_tempLx = dev_temp1;
  dev_tempHx = dev_tempLx + numCoeff/2;
  _data_t *dev_tempLxLy,*dev_tempHxLy,*dev_tempLxHy,*dev_tempHxHy;
  dev_tempLxLy = dev_temp2;
  dev_tempHxLy = dev_tempLxLy + numCoeff/4;
  dev_tempLxHy = dev_tempHxLy + numCoeff/4;
  dev_tempHxHy = dev_tempLxHy + numCoeff/4;

  // wcdf1 Pointers
  _data_t *dev_LxLyLz_df1,*dev_HxLyLz_df1,*dev_LxHyLz_df1,*dev_HxHyLz_df1,*dev_LxLyHz_df1,*dev_HxLyHz_df1,*dev_LxHyHz_df1,*dev_HxHyHz_df1,*dev_current_vx;
  dev_LxLyLz_df1 = dev_wcdf1;
  dev_HxLyLz_df1 = dev_LxLyLz_df1 + waveSizes[0]*waveSizes[1]*waveSizes[2];
  for (int l = 1; l <= numLevels; ++l){
    dev_HxLyLz_df1 += 7*waveSizes[0 + 3*l]*waveSizes[1 + 3*l]*waveSizes[2 + 3*l];
  }
  dev_current_vx = dev_in_vx;

  // wcdf2 Pointers
  _data_t *dev_LxLyLz_df2,*dev_HxLyLz_df2,*dev_LxHyLz_df2,*dev_HxHyLz_df2,*dev_LxLyHz_df2,*dev_HxLyHz_df2,*dev_LxHyHz_df2,*dev_HxHyHz_df2,*dev_current_vy;
  dev_LxLyLz_df2 = dev_wcdf2;
  dev_HxLyLz_df2 = dev_LxLyLz_df2 + waveSizes[0]*waveSizes[1]*waveSizes[2];
  for (int l = 1; l <= numLevels; ++l){
    dev_HxLyLz_df2 += 7*waveSizes[0 + 3*l]*waveSizes[1 + 3*l]*waveSizes[2 + 3*l];
  }
  dev_current_vy = dev_in_vy;

  // wcn Pointers
  _data_t *dev_LxLyLz_n,*dev_HxLyLz_n,*dev_LxHyLz_n,*dev_HxHyLz_n,*dev_LxLyHz_n,*dev_HxLyHz_n,*dev_LxHyHz_n,*dev_HxHyHz_n,*dev_current_vz;
  dev_LxLyLz_n = dev_wcn;
  dev_HxLyLz_n = dev_LxLyLz_n + waveSizes[0]*waveSizes[1]*waveSizes[2];
  for (int l = 1; l <= numLevels; ++l){
    dev_HxLyLz_n += 7*waveSizes[0 + 3*l]*waveSizes[1 + 3*l]*waveSizes[2 + 3*l];
  }
  dev_current_vz = dev_in_vz;

  //*****************Loop through levels****************
  for (int l = numLevels; l >= 1; --l)
    {
      dxNext = waveSizes[0 + 3*l];
      dyNext = waveSizes[1 + 3*l];
      dzNext = waveSizes[2 + 3*l];
      blockSize = dxNext*dyNext*dzNext;

      // Update Pointers
      // df1
      dev_HxLyLz_df1 = dev_HxLyLz_df1 - 7*blockSize;
      dev_LxHyLz_df1 = dev_HxLyLz_df1 + blockSize;
      dev_HxHyLz_df1 = dev_LxHyLz_df1 + blockSize;
      dev_LxLyHz_df1 = dev_HxHyLz_df1 + blockSize;
      dev_HxLyHz_df1 = dev_LxLyHz_df1 + blockSize;
      dev_LxHyHz_df1 = dev_HxLyHz_df1 + blockSize;
      dev_HxHyHz_df1 = dev_LxHyHz_df1 + blockSize;
      // df2
      dev_HxLyLz_df2 = dev_HxLyLz_df2 - 7*blockSize;
      dev_LxHyLz_df2 = dev_HxLyLz_df2 + blockSize;
      dev_HxHyLz_df2 = dev_LxHyLz_df2 + blockSize;
      dev_LxLyHz_df2 = dev_HxHyLz_df2 + blockSize;
      dev_HxLyHz_df2 = dev_LxLyHz_df2 + blockSize;
      dev_LxHyHz_df2 = dev_HxLyHz_df2 + blockSize;
      dev_HxHyHz_df2 = dev_LxHyHz_df2 + blockSize;
      // n
      dev_HxLyLz_n = dev_HxLyLz_n - 7*blockSize;
      dev_LxHyLz_n = dev_HxLyLz_n + blockSize;
      dev_HxHyLz_n = dev_LxHyLz_n + blockSize;
      dev_LxLyHz_n = dev_HxHyLz_n + blockSize;
      dev_HxLyHz_n = dev_LxLyHz_n + blockSize;
      dev_LxHyHz_n = dev_HxLyHz_n + blockSize;
      dev_HxHyHz_n = dev_LxHyHz_n + blockSize;

      //************WCVX***********
      // FWT Columns
      K = (SHMEM_SIZE-16)/(dx*sizeof(_data_t));
      numBlocks = dim3(1,(dy+K-1)/K,dz);
      numThreads = dim3(T/K,K,1);
      mem = K*dx*sizeof(_data_t);

      cu_fwt3df_col <<< numBlocks,numThreads,mem >>>(dev_tempLx,dev_tempHx,dev_current_vx,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod1,dev_hid1,filterLen);
      cuda_sync();
      // FWT Rows
      K = (SHMEM_SIZE-16)/(dy*sizeof(_data_t));
      numBlocks = dim3(((dxNext)+K-1)/K,1,dz);
      numThreads = dim3(K,T/K,1);
      mem = K*dy*sizeof(_data_t);
      
      cu_fwt3df_row <<< numBlocks,numThreads,mem >>>(dev_tempLxLy,dev_tempLxHy,dev_tempLx,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod0,dev_hid0,filterLen);
      cu_fwt3df_row <<< numBlocks,numThreads,mem >>>(dev_tempHxLy,dev_tempHxHy,dev_tempHx,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod0,dev_hid0,filterLen);
      cuda_sync();
      // FWT Depths
      K = (SHMEM_SIZE-16)/(dz*sizeof(_data_t));
      numBlocks = dim3(((dxNext)+K-1)/K,dyNext,1);
      numThreads = dim3(K,1,T/K);
      mem = K*dz*sizeof(_data_t);
      
      cu_fwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_LxLyLz_df1,dev_LxLyHz_df1,dev_tempLxLy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod0,dev_hid0,filterLen);
      cu_fwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_LxHyLz_df1,dev_LxHyHz_df1,dev_tempLxHy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod0,dev_hid0,filterLen);
      cu_fwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_HxLyLz_df1,dev_HxLyHz_df1,dev_tempHxLy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod0,dev_hid0,filterLen);
      cu_fwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_HxHyLz_df1,dev_HxHyHz_df1,dev_tempHxHy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod0,dev_hid0,filterLen);
      cuda_sync();

      //************WCVY***********
      // FWT Columns
      K = (SHMEM_SIZE-16)/(dx*sizeof(_data_t));
      numBlocks = dim3(1,(dy+K-1)/K,dz);
      numThreads = dim3(T/K,K,1);
      mem = K*dx*sizeof(_data_t);
      
      cu_fwt3df_col <<< numBlocks,numThreads,mem >>>(dev_tempLx,dev_tempHx,dev_current_vy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod0,dev_hid0,filterLen);
      cuda_sync();
      // FWT Rows
      K = (SHMEM_SIZE-16)/(dy*sizeof(_data_t));
      numBlocks = dim3(((dxNext)+K-1)/K,1,dz);
      numThreads = dim3(K,T/K,1);
      mem = K*dy*sizeof(_data_t);
      
      cu_fwt3df_row <<< numBlocks,numThreads,mem >>>(dev_tempLxLy,dev_tempLxHy,dev_tempLx,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod1,dev_hid1,filterLen);
      cu_fwt3df_row <<< numBlocks,numThreads,mem >>>(dev_tempHxLy,dev_tempHxHy,dev_tempHx,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod1,dev_hid1,filterLen);
      cuda_sync();
      // FWT Depths
      K = (SHMEM_SIZE-16)/(dz*sizeof(_data_t));
      numBlocks = dim3(((dxNext)+K-1)/K,dyNext,1);
      numThreads = dim3(K,1,T/K);
      mem = K*dz*sizeof(_data_t);
      
      cu_fwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_LxLyLz_df2,dev_LxLyHz_df2,dev_tempLxLy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod0,dev_hid0,filterLen);
      cu_fwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_LxHyLz_df2,dev_LxHyHz_df2,dev_tempLxHy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod0,dev_hid0,filterLen);
      cu_fwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_HxLyLz_df2,dev_HxLyHz_df2,dev_tempHxLy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod0,dev_hid0,filterLen);
      cu_fwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_HxHyLz_df2,dev_HxHyHz_df2,dev_tempHxHy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod0,dev_hid0,filterLen);
      cuda_sync();

      //************WCVZ***********
      // FWT Columns
      K = (SHMEM_SIZE-16)/(dx*sizeof(_data_t));
      numBlocks = dim3(1,(dy+K-1)/K,dz);
      numThreads = dim3(T/K,K,1);
      mem = K*dx*sizeof(_data_t);

      cu_fwt3df_col <<< numBlocks,numThreads,mem >>>(dev_tempLx,dev_tempHx,dev_current_vz,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod0,dev_hid0,filterLen);
      cuda_sync();
      // FWT Rows
      K = (SHMEM_SIZE-16)/(dy*sizeof(_data_t));
      numBlocks = dim3(((dxNext)+K-1)/K,1,dz);
      numThreads = dim3(K,T/K,1);
      mem = K*dy*sizeof(_data_t);
      
      cu_fwt3df_row <<< numBlocks,numThreads,mem >>>(dev_tempLxLy,dev_tempLxHy,dev_tempLx,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod0,dev_hid0,filterLen);
      cu_fwt3df_row <<< numBlocks,numThreads,mem >>>(dev_tempHxLy,dev_tempHxHy,dev_tempHx,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod0,dev_hid0,filterLen);
      cuda_sync();
      // FWT Depths
      K = (SHMEM_SIZE-16)/(dz*sizeof(_data_t));
      numBlocks = dim3(((dxNext)+K-1)/K,dyNext,1);
      numThreads = dim3(K,1,T/K);
      mem = K*dz*sizeof(_data_t);
      
      cu_fwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_LxLyLz_n,dev_LxLyHz_n,dev_tempLxLy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod1,dev_hid1,filterLen);
      cu_fwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_LxHyLz_n,dev_LxHyHz_n,dev_tempLxHy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod1,dev_hid1,filterLen);
      cu_fwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_HxLyLz_n,dev_HxLyHz_n,dev_tempHxLy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod1,dev_hid1,filterLen);
      cu_fwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_HxHyLz_n,dev_HxHyHz_n,dev_tempHxHy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod1,dev_hid1,filterLen);
      cuda_sync();

      //******* Multi ******
      int maxInd = 7*blockSize;
      numThreads = T;
      numBlocks = (maxInd+numThreads.x-1)/numThreads.x;
      cu_mult <<< numBlocks, numThreads >>> (dev_HxLyLz_df1,1.f/res[0],maxInd);
      cu_mult <<< numBlocks, numThreads >>> (dev_HxLyLz_df2,1.f/res[1],maxInd);
      cu_mult <<< numBlocks, numThreads >>> (dev_HxLyLz_n,1.f/res[2],maxInd);
      cuda_sync();

      //*******Linear Combination******
      int t1 = min(dxNext,T);
      int t2 = T/t1;
      numBlocks = dim3( (dxNext+t1-1)/t1, (dyNext+t2-1)/t2, dzNext);
      numThreads = dim3(t1,t2,1);
	  
      cu_fwt3df_LC1 <<< numBlocks,numThreads >>> (dev_HxLyLz_df1,dev_HxLyLz_df2,dev_HxLyLz_n,dev_LxHyLz_df1,dev_LxHyLz_df2,dev_LxHyLz_n,dev_LxLyHz_df1,dev_LxLyHz_df2,dev_LxLyHz_n,dxNext,dyNext,dzNext);
      cu_fwt3df_LC2 <<< numBlocks,numThreads >>> (dev_HxHyLz_df1,dev_HxHyLz_df2,dev_HxHyLz_n,dev_HxLyHz_df1,dev_HxLyHz_df2,dev_HxLyHz_n,dev_LxHyHz_df1,dev_LxHyHz_df2,dev_LxHyHz_n,dxNext,dyNext,dzNext);
      cu_fwt3df_LC3 <<< numBlocks,numThreads >>> (dev_HxHyHz_df1,dev_HxHyHz_df2,dev_HxHyHz_n,dxNext,dyNext,dzNext);
      cuda_sync();
      cu_fwt3df_LC1_diff <<< numBlocks,numThreads >>> (dev_HxLyLz_df1,dev_HxLyLz_df2,dev_HxLyLz_n,dev_LxHyLz_df1,dev_LxHyLz_df2,dev_LxHyLz_n,dev_LxLyHz_df1,dev_LxLyHz_df2,dev_LxLyHz_n,dxNext,dyNext,dzNext);
      cu_fwt3df_LC2_diff <<< numBlocks,numThreads >>> (dev_HxHyLz_df1,dev_HxHyLz_df2,dev_HxHyLz_n,dev_HxLyHz_df1,dev_HxLyHz_df2,dev_HxLyHz_n,dev_LxHyHz_df1,dev_LxHyHz_df2,dev_LxHyHz_n,dxNext,dyNext,dzNext);
      cuda_sync();

      dev_current_vx = dev_wcdf1;
      dev_current_vy = dev_wcdf2;
      dev_current_vz = dev_wcn;

      dx = dxNext;
      dy = dyNext;
      dz = dzNext;
    }
  cuda(Free( dev_filters ));
  cuda(Free( dev_temp1 ));
  cuda(Free( dev_temp2 ));
  
  circunshift_gpu(plan,in_vx);
  circunshift_gpu(plan,in_vy);
  circunshift_gpu(plan,in_vz);
}

extern "C" void dfiwt3_gpu(struct dfwavelet_plan_s* plan, data_t* out_vx,data_t* out_vy,data_t* out_vz, data_t* in_wcdf1,data_t* in_wcdf2,data_t* in_wcn)
{
  long numCoeff, filterLen,*waveSizes;
  numCoeff = plan->numCoeff;
  waveSizes = plan->waveSizes;
  filterLen = plan->filterLen;
  int numLevels = plan->numLevels;
  // Cast from generic data_t to device compatible _data_t
  _data_t* dev_out_vx = (_data_t*)out_vx;
  _data_t* dev_out_vy = (_data_t*)out_vy;
  _data_t* dev_out_vz = (_data_t*)out_vz;
  _data_t* dev_wcdf1 = (_data_t*)in_wcdf1;
  _data_t* dev_wcdf2 = (_data_t*)in_wcdf2;
  _data_t* dev_wcn = (_data_t*)in_wcn;
  _data_t* res = (_data_t*) plan->res;
  _data_t* dev_temp1, *dev_temp2;
  cuda(Malloc( (void**)&dev_temp1, numCoeff*sizeof(_data_t) ));
  cuda(Malloc( (void**)&dev_temp2, numCoeff*sizeof(_data_t)) );
  // allocate device memory
  scalar_t *dev_filters;
  cuda(Malloc( (void**)&dev_filters, 4*(plan->filterLen)*sizeof(scalar_t) ));
  scalar_t *dev_lor0 = dev_filters + 0*plan->filterLen;
  scalar_t *dev_hir0 = dev_filters + 1*plan->filterLen;
  scalar_t *dev_lor1 = dev_filters + 2*plan->filterLen;
  scalar_t *dev_hir1 = dev_filters + 3*plan->filterLen;
  cuda(Memcpy( dev_lor0, plan->lor0, 2*plan->filterLen*sizeof(scalar_t), cudaMemcpyHostToDevice ));
  cuda(Memcpy( dev_lor1, plan->lor1, 2*plan->filterLen*sizeof(scalar_t), cudaMemcpyHostToDevice ));
      
  // Workspace dimensions
  int dxWork = waveSizes[0 + 3*numLevels]*2-1 + filterLen-1;
  int dyWork = waveSizes[1 + 3*numLevels]*2-1 + filterLen-1;
  int dzWork = waveSizes[2 + 3*numLevels]*2-1 + filterLen-1;

  // Initialize variables and pointers for IWT
  int const SHMEM_SIZE = 16384;
  int const T = 512;
  int mem,K;
  dim3 numBlocks, numThreads;
  int dx = waveSizes[0];
  int dy = waveSizes[1];
  int dz = waveSizes[2];

  // Temp Pointers
  _data_t *dev_tempLxLy,*dev_tempHxLy,*dev_tempLxHy,*dev_tempHxHy;
  dev_tempLxLy = dev_temp1;
  dev_tempHxLy = dev_tempLxLy + numCoeff/4;
  dev_tempLxHy = dev_tempHxLy + numCoeff/4;
  dev_tempHxHy = dev_tempLxHy + numCoeff/4;
  _data_t *dev_tempLx,*dev_tempHx;
  dev_tempLx = dev_temp2;
  dev_tempHx = dev_tempLx + numCoeff/2;
  // wcdf1 Pointers
  _data_t *dev_LxLyLz_df1,*dev_HxLyLz_df1,*dev_LxHyLz_df1,*dev_HxHyLz_df1,*dev_LxLyHz_df1,*dev_HxLyHz_df1,*dev_LxHyHz_df1,*dev_HxHyHz_df1,*dev_current_vx;
  dev_LxLyLz_df1 = dev_wcdf1;
  dev_HxLyLz_df1 = dev_LxLyLz_df1 + dx*dy*dz;
  dev_current_vx = dev_LxLyLz_df1;
  // wcdf2 Pointers
  _data_t *dev_LxLyLz_df2,*dev_HxLyLz_df2,*dev_LxHyLz_df2,*dev_HxHyLz_df2,*dev_LxLyHz_df2,*dev_HxLyHz_df2,*dev_LxHyHz_df2,*dev_HxHyHz_df2,*dev_current_vy;
  dev_LxLyLz_df2 = dev_wcdf2;
  dev_HxLyLz_df2 = dev_LxLyLz_df2 + dx*dy*dz;
  dev_current_vy = dev_LxLyLz_df2;
  // wcn Pointers
  _data_t *dev_LxLyLz_n,*dev_HxLyLz_n,*dev_LxHyLz_n,*dev_HxHyLz_n,*dev_LxLyHz_n,*dev_HxLyHz_n,*dev_LxHyHz_n,*dev_HxHyHz_n,*dev_current_vz;
  dev_LxLyLz_n = dev_wcn;
  dev_HxLyLz_n = dev_LxLyLz_n + dx*dy*dz;
  dev_current_vz = dev_LxLyLz_n;

  for (int level = 1; level < numLevels+1; ++level)
    {
      dx = waveSizes[0 + 3*level];
      dy = waveSizes[1 + 3*level];
      dz = waveSizes[2 + 3*level];
      int blockSize = dx*dy*dz;
      int dxNext = waveSizes[0+3*(level+1)];
      int dyNext = waveSizes[1+3*(level+1)];
      int dzNext = waveSizes[2+3*(level+1)];
	  
      // Calclate Offset
      dxWork = (2*dx-1 + filterLen-1);
      dyWork = (2*dy-1 + filterLen-1);
      dzWork = (2*dz-1 + filterLen-1);
      int xOffset = (int) floor((dxWork - dxNext) / 2.0);
      int yOffset = (int) floor((dyWork - dyNext) / 2.0);
      int zOffset = (int) floor((dzWork - dzNext) / 2.0);

      // Update Pointers
      // df1
      dev_LxHyLz_df1 = dev_HxLyLz_df1 + blockSize;
      dev_HxHyLz_df1 = dev_LxHyLz_df1 + blockSize;
      dev_LxLyHz_df1 = dev_HxHyLz_df1 + blockSize;
      dev_HxLyHz_df1 = dev_LxLyHz_df1 + blockSize;
      dev_LxHyHz_df1 = dev_HxLyHz_df1 + blockSize;
      dev_HxHyHz_df1 = dev_LxHyHz_df1 + blockSize;
      // df2
      dev_LxHyLz_df2 = dev_HxLyLz_df2 + blockSize;
      dev_HxHyLz_df2 = dev_LxHyLz_df2 + blockSize;
      dev_LxLyHz_df2 = dev_HxHyLz_df2 + blockSize;
      dev_HxLyHz_df2 = dev_LxLyHz_df2 + blockSize;
      dev_LxHyHz_df2 = dev_HxLyHz_df2 + blockSize;
      dev_HxHyHz_df2 = dev_LxHyHz_df2 + blockSize;
      // n
      dev_LxHyLz_n = dev_HxLyLz_n + blockSize;
      dev_HxHyLz_n = dev_LxHyLz_n + blockSize;
      dev_LxLyHz_n = dev_HxHyLz_n + blockSize;
      dev_HxLyHz_n = dev_LxLyHz_n + blockSize;
      dev_LxHyHz_n = dev_HxLyHz_n + blockSize;
      dev_HxHyHz_n = dev_LxHyHz_n + blockSize;

      //*******Linear Combination******

      int t1 = min(dxNext,T);
      int t2 = T/t1;
      numBlocks = dim3( (dx+t1-1)/t1, (dy+t2-1)/t2, dz);
      numThreads = dim3(t1,t2,1);

      cu_iwt3df_LC1 <<< numBlocks,numThreads >>> (dev_HxLyLz_df1,dev_HxLyLz_df2,dev_HxLyLz_n,dev_LxHyLz_df1,dev_LxHyLz_df2,dev_LxHyLz_n,dev_LxLyHz_df1,dev_LxLyHz_df2,dev_LxLyHz_n,dx,dy,dz);
      cu_iwt3df_LC2 <<< numBlocks,numThreads >>> (dev_HxHyLz_df1,dev_HxHyLz_df2,dev_HxHyLz_n,dev_HxLyHz_df1,dev_HxLyHz_df2,dev_HxLyHz_n,dev_LxHyHz_df1,dev_LxHyHz_df2,dev_LxHyHz_n,dx,dy,dz);
      cu_iwt3df_LC3 <<< numBlocks,numThreads >>> (dev_HxHyHz_df1,dev_HxHyHz_df2,dev_HxHyHz_n,dx,dy,dz);
      cuda_sync();
      cu_iwt3df_LC1_diff <<< numBlocks,numThreads >>> (dev_HxLyLz_df1,dev_HxLyLz_df2,dev_HxLyLz_n,dev_LxHyLz_df1,dev_LxHyLz_df2,dev_LxHyLz_n,dev_LxLyHz_df1,dev_LxLyHz_df2,dev_LxLyHz_n,dx,dy,dz);
      cu_iwt3df_LC2_diff <<< numBlocks,numThreads >>> (dev_HxHyLz_df1,dev_HxHyLz_df2,dev_HxHyLz_n,dev_HxLyHz_df1,dev_HxLyHz_df2,dev_HxLyHz_n,dev_LxHyHz_df1,dev_LxHyHz_df2,dev_LxHyHz_n,dx,dy,dz);
      cuda_sync();
      
      //******* Multi ******
      int maxInd = 7*blockSize;
      numThreads = T;
      numBlocks = (maxInd+numThreads.x-1)/numThreads.x;
      cu_mult <<< numBlocks, numThreads >>> (dev_HxLyLz_df1,res[0],maxInd);
      cu_mult <<< numBlocks, numThreads >>> (dev_HxLyLz_df2,res[1],maxInd);
      cu_mult <<< numBlocks, numThreads >>> (dev_HxLyLz_n,res[2],maxInd);
      cuda_sync();

      //************WCX************
      // Update Pointers
      if (level==numLevels)
	dev_current_vx = dev_out_vx;
      // IWT Depths
      K = (SHMEM_SIZE-16)/(2*dz*sizeof(_data_t));
      numBlocks = dim3((dx+K-1)/K,dy,1);
      numThreads = dim3(K,1,(T/K));
      mem = K*2*dz*sizeof(_data_t);

      cu_iwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_tempLxLy,dev_LxLyLz_df1,dev_LxLyHz_df1,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor0,dev_hir0,filterLen);
      cu_iwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_tempHxLy,dev_HxLyLz_df1,dev_HxLyHz_df1,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor0,dev_hir0,filterLen);
      cu_iwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_tempLxHy,dev_LxHyLz_df1,dev_LxHyHz_df1,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor0,dev_hir0,filterLen);
      cu_iwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_tempHxHy,dev_HxHyLz_df1,dev_HxHyHz_df1,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor0,dev_hir0,filterLen);
      cuda_sync();
      // IWT Rows
      K = (SHMEM_SIZE-16)/(2*dy*sizeof(_data_t));
      numBlocks = dim3((dx+K-1)/K,1,dzNext);
      numThreads = dim3(K,(T/K),1);
      mem = K*2*dy*sizeof(_data_t);

      cu_iwt3df_row <<< numBlocks,numThreads,mem >>>(dev_tempLx,dev_tempLxLy,dev_tempLxHy,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor0,dev_hir0,plan->filterLen);
      cu_iwt3df_row <<< numBlocks,numThreads,mem >>>(dev_tempHx,dev_tempHxLy,dev_tempHxHy,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor0,dev_hir0,plan->filterLen);
      cuda_sync();
      // IWT Columns
      K = (SHMEM_SIZE-16)/(2*dx*sizeof(_data_t));
      numBlocks = dim3(1,(dyNext+K-1)/K,dzNext);
      numThreads = dim3((T/K),K,1);
      mem = K*2*dx*sizeof(_data_t);

      cu_iwt3df_col <<< numBlocks,numThreads,mem >>>(dev_current_vx,dev_tempLx,dev_tempHx,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor1,dev_hir1,plan->filterLen);
      cuda_sync();

      //************WCY************
      // Update Pointers
      if (level==numLevels)
	dev_current_vy = dev_out_vy;
      // IWT Depths
      K = (SHMEM_SIZE-16)/(2*dz*sizeof(_data_t));
      numBlocks = dim3((dx+K-1)/K,dy,1);
      numThreads = dim3(K,1,(T/K));
      mem = K*2*dz*sizeof(_data_t);

      cu_iwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_tempLxLy,dev_LxLyLz_df2,dev_LxLyHz_df2,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor0,dev_hir0,filterLen);
      cu_iwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_tempHxLy,dev_HxLyLz_df2,dev_HxLyHz_df2,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor0,dev_hir0,filterLen);
      cu_iwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_tempLxHy,dev_LxHyLz_df2,dev_LxHyHz_df2,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor0,dev_hir0,filterLen);
      cu_iwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_tempHxHy,dev_HxHyLz_df2,dev_HxHyHz_df2,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor0,dev_hir0,filterLen);
      cuda_sync();
      // IWT Rows
      K = (SHMEM_SIZE-16)/(2*dy*sizeof(_data_t));
      numBlocks = dim3((dx+K-1)/K,1,dzNext);
      numThreads = dim3(K,(T/K),1);
      mem = K*2*dy*sizeof(_data_t);

      cu_iwt3df_row <<< numBlocks,numThreads,mem >>>(dev_tempLx,dev_tempLxLy,dev_tempLxHy,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor1,dev_hir1,plan->filterLen);
      cu_iwt3df_row <<< numBlocks,numThreads,mem >>>(dev_tempHx,dev_tempHxLy,dev_tempHxHy,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor1,dev_hir1,plan->filterLen);
      cuda_sync();
      // IWT Columns
      K = (SHMEM_SIZE-16)/(2*dx*sizeof(_data_t));
      numBlocks = dim3(1,(dyNext+K-1)/K,dzNext);
      numThreads = dim3((T/K),K,1);
      mem = K*2*dx*sizeof(_data_t);

      cu_iwt3df_col <<< numBlocks,numThreads,mem >>>(dev_current_vy,dev_tempLx,dev_tempHx,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor0,dev_hir0,plan->filterLen);
      cuda_sync();

      //************WCZ************
      // Update Pointers
      if (level==numLevels)
	dev_current_vz = dev_out_vz;
      // IWT Depths
      K = (SHMEM_SIZE-16)/(2*dz*sizeof(_data_t));
      numBlocks = dim3((dx+K-1)/K,dy,1);
      numThreads = dim3(K,1,(T/K));
      mem = K*2*dz*sizeof(_data_t);

      cu_iwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_tempLxLy,dev_LxLyLz_n,dev_LxLyHz_n,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor1,dev_hir1,filterLen);
      cu_iwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_tempHxLy,dev_HxLyLz_n,dev_HxLyHz_n,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor1,dev_hir1,filterLen);
      cu_iwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_tempLxHy,dev_LxHyLz_n,dev_LxHyHz_n,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor1,dev_hir1,filterLen);
      cu_iwt3df_dep <<< numBlocks,numThreads,mem >>>(dev_tempHxHy,dev_HxHyLz_n,dev_HxHyHz_n,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor1,dev_hir1,filterLen);
      cuda_sync();
      // IWT Rows
      K = (SHMEM_SIZE-16)/(2*dy*sizeof(_data_t));
      numBlocks = dim3((dx+K-1)/K,1,dzNext);
      numThreads = dim3(K,(T/K),1);
      mem = K*2*dy*sizeof(_data_t);

      cu_iwt3df_row <<< numBlocks,numThreads,mem >>>(dev_tempLx,dev_tempLxLy,dev_tempLxHy,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor0,dev_hir0,plan->filterLen);
      cu_iwt3df_row <<< numBlocks,numThreads,mem >>>(dev_tempHx,dev_tempHxLy,dev_tempHxHy,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor0,dev_hir0,plan->filterLen);
      cuda_sync();
      // IWT Columns
      K = (SHMEM_SIZE-16)/(2*dx*sizeof(_data_t));
      numBlocks = dim3(1,(dyNext+K-1)/K,dzNext);
      numThreads = dim3((T/K),K,1);
      mem = K*2*dx*sizeof(_data_t);

      cu_iwt3df_col <<< numBlocks,numThreads,mem >>>(dev_current_vz,dev_tempLx,dev_tempHx,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor0,dev_hir0,plan->filterLen);
      cuda_sync();
      dev_HxLyLz_df1 += 7*blockSize;
      dev_HxLyLz_df2 += 7*blockSize;
      dev_HxLyLz_n += 7*blockSize;

    }
  cuda(Free( dev_filters ));
  cuda(Free( dev_temp1 ));
  cuda(Free( dev_temp2 ));
  
  circunshift_gpu(plan,out_vx);
  circunshift_gpu(plan,out_vy);
  circunshift_gpu(plan,out_vz);
}

int rand_lim(int limit) {

  int divisor = RAND_MAX/(limit+1);
  int retval;

  do { 
    retval = rand() / divisor;
  } while (retval > limit);

  return retval;
}

void dfwavelet_new_randshift_gpu (struct dfwavelet_plan_s* plan) {
  int i;
  i = rand();
  for(i = 0; i < plan->numdims; i++) {
    // Determine maximum shift value for this dimension
    int log2dim = 1;
    while( (1<<log2dim) < plan->imSize[i]) {
      log2dim++;
    }
    int maxShift = 1 << (log2dim-plan->numLevels);
    if (maxShift > 8) {
      maxShift = 8;
    }
    // Generate random shift value between 0 and maxShift
    plan->randShift[i] = rand_lim(maxShift);
  }
}

extern "C" void dfwavthresh3_gpu(struct dfwavelet_plan_s* plan,scalar_t dfthresh, scalar_t nthresh,data_t* out_vx,data_t* out_vy,data_t* out_vz,data_t* in_vx,data_t* in_vy,data_t* in_vz)
{
  data_t* dev_wcdf1,*dev_wcdf2,*dev_wcn;
  cuda(Malloc( (void**)&dev_wcdf1, plan->numCoeff*sizeof(_data_t) ));
  cuda(Malloc( (void**)&dev_wcdf2, plan->numCoeff*sizeof(_data_t) ));
  cuda(Malloc( (void**)&dev_wcn, plan->numCoeff*sizeof(_data_t) ));

  dffwt3_gpu(plan,dev_wcdf1,dev_wcdf2,dev_wcn,in_vx,in_vy,in_vz);
  dfsoftthresh_gpu(plan,dfthresh,nthresh,dev_wcdf1,dev_wcdf2,dev_wcn);
  dfiwt3_gpu(plan,out_vx,out_vy,out_vz,dev_wcdf1,dev_wcdf2,dev_wcn);

  cuda(Free( dev_wcdf1 ));
  cuda(Free( dev_wcdf2 ));
  cuda(Free( dev_wcn ));
}


extern "C" void dfsoftthresh_gpu(struct dfwavelet_plan_s* plan,scalar_t dfthresh, scalar_t nthresh, data_t* out_wcdf1,data_t* out_wcdf2,data_t* out_wcn)
{

  assert(plan->use_gpu==1||plan->use_gpu==2);

  _data_t* dev_wcdf1,*dev_wcdf2,*dev_wcn;
  dev_wcdf1 = (_data_t*) out_wcdf1;
  dev_wcdf2 = (_data_t*) out_wcdf2;
  dev_wcn = (_data_t*) out_wcn;
  int numMax;
  int const T = 512;
  dim3 numBlocks, numThreads;
  numMax = plan->numCoeff-plan->numCoarse;
  numBlocks = dim3((numMax+T-1)/T,1,1);
  numThreads = dim3(T,1,1);
  cu_soft_thresh <<< numBlocks,numThreads>>> (dev_wcdf1+plan->numCoarse,dfthresh,numMax);

  cu_soft_thresh <<< numBlocks,numThreads>>> (dev_wcdf2+plan->numCoarse,dfthresh,numMax);

  cu_soft_thresh <<< numBlocks,numThreads>>> (dev_wcn+plan->numCoarse,nthresh,numMax);

}



/********** Aux functions **********/
extern "C" void circshift_gpu(struct dfwavelet_plan_s* plan, data_t* data_c) {
  // Return if no shifts
  int zeroShift = 1;
  int i;
  for (i = 0; i< plan->numdims; i++)
    {
      zeroShift &= (plan->randShift[i]==0);
    }
  if(zeroShift) {
    return;
  }
  _data_t* data = (_data_t*) data_c;
  // Copy data
  _data_t* dataCopy;
  cuda(Malloc((void**)&dataCopy, plan->numPixel*sizeof(_data_t)));
  cuda(Memcpy(dataCopy, data, plan->numPixel*sizeof(_data_t), cudaMemcpyDeviceToDevice));
  int T = 512;
  if (plan->numdims==2)
    {
      int dx,dy,r0,r1;
      dx = plan->imSize[0];
      dy = plan->imSize[1];
      r0 = plan->randShift[0];
      r1 = plan->randShift[1];
      cu_circshift <<< (plan->numPixel+T-1)/T, T>>>(data,dataCopy,dx,dy,1,r0,r1,0);
    } else if (plan->numdims==3)
    {
      int dx,dy,dz,r0,r1,r2;
      dx = plan->imSize[0];
      dy = plan->imSize[1];
      dz = plan->imSize[2];
      r0 = plan->randShift[0];
      r1 = plan->randShift[1];
      r2 = plan->randShift[2];
      cu_circshift <<< (plan->numPixel+T-1)/T, T>>>(data,dataCopy,dx,dy,dz,r0,r1,r2);
    }
  cuda(Free(dataCopy));
}

extern "C" void circunshift_gpu(struct dfwavelet_plan_s* plan, data_t* data_c) {
  // Return if no shifts
  int zeroShift = 1;
  int i;
  for (i = 0; i< plan->numdims; i++)
    {
      zeroShift &= (plan->randShift[i]==0);
    }
  if(zeroShift) {
    return;
  }
  _data_t* data = (_data_t*) data_c;
  // Copy data
  _data_t* dataCopy;
  cuda(Malloc((void**)&dataCopy, plan->numPixel*sizeof(_data_t)));
  cuda(Memcpy(dataCopy, data, plan->numPixel*sizeof(_data_t), cudaMemcpyDeviceToDevice));
  int T = 512;
  if (plan->numdims==2)
    {
      int dx,dy,r0,r1;
      dx = plan->imSize[0];
      dy = plan->imSize[1];
      r0 = plan->randShift[0];
      r1 = plan->randShift[1];
      cu_circunshift <<< (plan->numPixel+T-1)/T, T>>>(data,dataCopy,dx,dy,1,r0,r1,0);
    } else if (plan->numdims==3)
    {
      int dx,dy,dz,r0,r1,r2;
      dx = plan->imSize[0];
      dy = plan->imSize[1];
      dz = plan->imSize[2];
      r0 = plan->randShift[0];
      r1 = plan->randShift[1];
      r2 = plan->randShift[2];
      cu_circunshift <<< (plan->numPixel+T-1)/T, T>>>(data,dataCopy,dx,dy,dz,r0,r1,r2);
    }
  cuda(Free(dataCopy));
}

// ############################################################################
// CUDA function of fwt column convolution
// Loads data to scratchpad (shared memory) and convolve w/ low pass and high pass
// Output: Lx, Hx
// Input:  in, dx, dy, dz, dxNext, lod, hid, filterLen
// ############################################################################
extern "C" __global__ void cu_fwt3df_col(_data_t *Lx,_data_t *Hx,_data_t *in,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,scalar_t *lod,scalar_t *hid,int filterLen)
{
  extern __shared__ _data_t cols [];
  int ti = threadIdx.x;
  int tj = threadIdx.y;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  int k = blockIdx.z*blockDim.z+threadIdx.z;

  if (j>=dy) {
    return;
  }

  // Load Input to Temp Array
  for (int i = ti; i < dx; i += blockDim.x){
    cols[i + tj*dx] = in[i + j*dx + k*dx*dy];
  }
  __syncthreads();
  // Low-Pass and High-Pass Downsample
  int ind, lessThan, greaThan;
  for (int i = ti; i < dxNext; i += blockDim.x){
    _data_t y = cols[0]-cols[0];
    _data_t z = cols[0]-cols[0];
#pragma unroll
    for (int f = 0; f < filterLen; f++){
      ind = 2*i+1 - (filterLen-1)+f;

      lessThan = (int) (ind<0);
      greaThan = (int) (ind>=dx);
      ind = -1*lessThan+ind*(-2*lessThan+1);
      ind = (2*dx-1)*greaThan+ind*(-2*greaThan+1);

      y += cols[ind + tj*dx] * lod[filterLen-1-f];
      z += cols[ind + tj*dx] * hid[filterLen-1-f];
    }
    Lx[i + j*dxNext + k*dxNext*dy] = y;
    Hx[i + j*dxNext + k*dxNext*dy] = z;
  }
}

// ############################################################################
// CUDA function of fwt row convolution. Assumes fwt_col() has already been called
// Loads data to scratchpad (shared memory) and convolve w/ low pass and high pass
// Output: LxLy, LxHy / HxLy, HxHy
// Input:  Lx/Hx, dx, dy, dxNext, dyNext, lod, hid, filterLen
// ############################################################################
extern "C" __global__ void cu_fwt3df_row(_data_t *Ly,_data_t *Hy,_data_t *in,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,scalar_t *lod,scalar_t *hid,int filterLen)
{
  extern __shared__ _data_t rows [];
  int const K = blockDim.x;
  int ti = threadIdx.x;
  int tj = threadIdx.y;
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int k = blockIdx.z*blockDim.z+threadIdx.z;

  if (i>=dxNext)
    {
      return;
    }

  for (int j = tj; j < dy; j += blockDim.y){
    rows[ti + j*K] = in[i + j*dxNext + k*dxNext*dy];
  }
  __syncthreads();

  // Low-Pass and High Pass Downsample
  int ind, lessThan, greaThan;
  for (int j = tj; j < dyNext; j += blockDim.y){
    _data_t y = rows[0]-rows[0];
    _data_t z = rows[0]-rows[0];
#pragma unroll
    for (int f = 0; f < filterLen; f++){
      ind = 2*j+1 - (filterLen-1)+f;
      lessThan = (int) (ind<0);
      greaThan = (int) (ind>=dy);
      ind = -1*lessThan+ind*(-2*lessThan+1);
      ind = (2*dy-1)*greaThan+ind*(-2*greaThan+1);
      y += rows[ti + ind*K] * lod[filterLen-1-f];
      z += rows[ti + ind*K] * hid[filterLen-1-f];
    }
    Ly[i + j*dxNext + k*dxNext*dyNext] = y;
    Hy[i + j*dxNext + k*dxNext*dyNext] = z;
  }
}

// ############################################################################
// CUDA function of fwt depth convolution. Assumes fwt_row() has already been called
// Loads data to scratchpad (shared memory) and convolve w/ low pass and high pass
// Output: LxLy, LxHy / HxLy, HxHy
// Input:  Lx/Hx, dx, dy, dxNext, dyNext, lod, hid, filterLen
// ############################################################################
extern "C" __global__ void cu_fwt3df_dep(_data_t *Lz,_data_t *Hz,_data_t *in,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,scalar_t *lod,scalar_t *hid,int filterLen)
{
  extern __shared__ _data_t deps [];
  int const K = blockDim.x;
  int ti = threadIdx.x;
  int tk = threadIdx.z;
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;

  if (i>=dxNext)
    {
      return;
    }

  for (int k = tk; k < dz; k += blockDim.z){
    deps[ti + k*K] = in[i + j*dxNext + k*dxNext*dyNext];
  }
  __syncthreads();

  // Low-Pass and High Pass Downsample
  int ind, lessThan, greaThan;
  for (int k = tk; k < dzNext; k += blockDim.z){
    _data_t y = deps[0]-deps[0];
    _data_t z = deps[0]-deps[0];
#pragma unroll
    for (int f = 0; f < filterLen; f++){
      ind = 2*k+1 - (filterLen-1)+f;
      lessThan = (int) (ind<0);
      greaThan = (int) (ind>=dz);
      ind = -1*lessThan+ind*(-2*lessThan+1);
      ind = (2*dz-1)*greaThan+ind*(-2*greaThan+1);
      y += deps[ti + ind*K] * lod[filterLen-1-f];
      z += deps[ti + ind*K] * hid[filterLen-1-f];
    }
    Lz[i + j*dxNext + k*dxNext*dyNext] = y;
    Hz[i + j*dxNext + k*dxNext*dyNext] = z;
  }
}

extern "C" __global__ void cu_fwt3df_LC1(_data_t *HxLyLz_df1,_data_t *HxLyLz_df2,_data_t *HxLyLz_n,_data_t *LxHyLz_df1,_data_t *LxHyLz_df2,_data_t *LxHyLz_n,_data_t *LxLyHz_df1,_data_t *LxLyHz_df2,_data_t *LxLyHz_n,int dxNext, int dyNext, int dzNext)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  int k = blockIdx.z*blockDim.z+threadIdx.z;
  _data_t x,y,z;
  scalar_t xGreatZero,yGreatZero,zGreatZero;
  if ((i>=dxNext)||(j>=dyNext)||(k>=dzNext))
    {
      return;
    }

  //HLL
  x = HxLyLz_df1[i+j*dxNext+k*dxNext*dyNext];
  y = HxLyLz_df2[i+j*dxNext+k*dxNext*dyNext];
  z = HxLyLz_n[i+j*dxNext+k*dxNext*dyNext];
  HxLyLz_df1[i+j*dxNext+k*dxNext*dyNext] = y;
  HxLyLz_df2[i+j*dxNext+k*dxNext*dyNext] = z;
  yGreatZero = j>0;
  zGreatZero = k>0;
  HxLyLz_n[i+j*dxNext+k*dxNext*dyNext] = x + yGreatZero*0.25f*y + zGreatZero*0.25f*z;

  //LHL
  x = LxHyLz_df1[i+j*dxNext+k*dxNext*dyNext];
  y = LxHyLz_df2[i+j*dxNext+k*dxNext*dyNext];
  z = LxHyLz_n[i+j*dxNext+k*dxNext*dyNext];
  LxHyLz_df2[i+j*dxNext+k*dxNext*dyNext] = z;
  xGreatZero = i>0;
  zGreatZero = k>0;
  LxHyLz_n[i+j*dxNext+k*dxNext*dyNext] = y + xGreatZero*0.25f*x + zGreatZero*0.25f*z;
      
  //LLH
  x = LxLyHz_df1[i+j*dxNext+k*dxNext*dyNext];
  y = LxLyHz_df2[i+j*dxNext+k*dxNext*dyNext];
  z = LxLyHz_n[i+j*dxNext+k*dxNext*dyNext];
  LxLyHz_df1[i+j*dxNext+k*dxNext*dyNext] = y;
  LxLyHz_df2[i+j*dxNext+k*dxNext*dyNext] = x;
  yGreatZero = j>0;
  xGreatZero = i>0;
  LxLyHz_n[i+j*dxNext+k*dxNext*dyNext] = z + yGreatZero*0.25*y + xGreatZero*0.25*x;
}
extern "C" __global__ void cu_fwt3df_LC1_diff(_data_t *HxLyLz_df1,_data_t *HxLyLz_df2,_data_t *HxLyLz_n,_data_t *LxHyLz_df1,_data_t *LxHyLz_df2,_data_t *LxHyLz_n,_data_t *LxLyHz_df1,_data_t *LxLyHz_df2,_data_t *LxLyHz_n,int dxNext, int dyNext, int dzNext)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  int k = blockIdx.z*blockDim.z+threadIdx.z;
  _data_t x,y,z;
  _data_t zero = make_float2(0.f,0.f);
  if ((i>=dxNext)||(j>=dyNext)||(k>=dzNext))
    {
      return;
    }

  //HLL
  if (j>0)
    y = HxLyLz_df1[i+(j-1)*dxNext+k*dxNext*dyNext];
  else
    y = zero;
  if (k>0)
    z = HxLyLz_df2[i+j*dxNext+(k-1)*dxNext*dyNext];
  else
    z = zero;
  HxLyLz_n[i+j*dxNext+k*dxNext*dyNext] += -0.25*y - 0.25*z;

  //LHL
  if (i>0)
    x = LxHyLz_df1[(i-1)+j*dxNext+k*dxNext*dyNext];
  else
    x = zero;
  if (k>0)
    z = LxHyLz_df2[i+j*dxNext+(k-1)*dxNext*dyNext];
  else
    z = zero;
  LxHyLz_n[i+j*dxNext+k*dxNext*dyNext] += -0.25*x - 0.25*z;

  //LLH
  if (j>0)
    y = LxLyHz_df1[i+(j-1)*dxNext+k*dxNext*dyNext];
  else
    y = zero;
  if (i>0)
    x = LxLyHz_df2[(i-1)+j*dxNext+k*dxNext*dyNext];
  else
    x = zero;
  LxLyHz_n[i+j*dxNext+k*dxNext*dyNext] += -0.25*y - 0.25*x;
}
extern "C" __global__ void cu_fwt3df_LC2(_data_t* HxHyLz_df1,_data_t* HxHyLz_df2,_data_t* HxHyLz_n,_data_t* HxLyHz_df1,_data_t* HxLyHz_df2,_data_t* HxLyHz_n,_data_t* LxHyHz_df1,_data_t* LxHyHz_df2,_data_t* LxHyHz_n,int dxNext, int dyNext, int dzNext)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  int k = blockIdx.z*blockDim.z+threadIdx.z;
  _data_t x,y,z;
scalar_t xGreatZero,yGreatZero,zGreatZero;
  if ((i>=dxNext)||(j>=dyNext)||(k>=dzNext))
    {
      return;
    }

  //HHL
  x = HxHyLz_df1[i+j*dxNext+k*dxNext*dyNext];
  y = HxHyLz_df2[i+j*dxNext+k*dxNext*dyNext];
  z = HxHyLz_n[i+j*dxNext+k*dxNext*dyNext];
  HxHyLz_df1[i+j*dxNext+k*dxNext*dyNext] = 0.5*(x-y);
  HxHyLz_df2[i+j*dxNext+k*dxNext*dyNext] = z;
  zGreatZero = k>0;
  HxHyLz_n[i+j*dxNext+k*dxNext*dyNext] = 0.5*(x+y) + zGreatZero*0.125*z;

  //HLH
  x = HxLyHz_df1[i+j*dxNext+k*dxNext*dyNext];
  y = HxLyHz_df2[i+j*dxNext+k*dxNext*dyNext];
  z = HxLyHz_n[i+j*dxNext+k*dxNext*dyNext];
  HxLyHz_df1[i+j*dxNext+k*dxNext*dyNext] = 0.5*(z-x);
  HxLyHz_df2[i+j*dxNext+k*dxNext*dyNext] = y;
  yGreatZero = j>0;
  HxLyHz_n[i+j*dxNext+k*dxNext*dyNext] = 0.5*(z+x) + yGreatZero*0.125*y;
      
  //LHH
  x = LxHyHz_df1[i+j*dxNext+k*dxNext*dyNext];
  y = LxHyHz_df2[i+j*dxNext+k*dxNext*dyNext];
  z = LxHyHz_n[i+j*dxNext+k*dxNext*dyNext];
  LxHyHz_df1[i+j*dxNext+k*dxNext*dyNext] = 0.5*(y-z);
  LxHyHz_df2[i+j*dxNext+k*dxNext*dyNext] = x;
  xGreatZero = i>0;
  LxHyHz_n[i+j*dxNext+k*dxNext*dyNext] = 0.5*(y+z) + xGreatZero*0.125*x;
}

extern "C" __global__ void cu_fwt3df_LC2_diff(_data_t* HxHyLz_df1,_data_t* HxHyLz_df2,_data_t* HxHyLz_n,_data_t* HxLyHz_df1,_data_t* HxLyHz_df2,_data_t* HxLyHz_n,_data_t* LxHyHz_df1,_data_t* LxHyHz_df2,_data_t* LxHyHz_n,int dxNext, int dyNext, int dzNext)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  int k = blockIdx.z*blockDim.z+threadIdx.z;
  _data_t x,y,z;
  _data_t zero = make_float2(0.f,0.f);
  if ((i>=dxNext)||(j>=dyNext)||(k>=dzNext))
    {
      return;
    }

  //HHL
  if (k>0)
    z = HxHyLz_df2[i+j*dxNext+(k-1)*dxNext*dyNext];
  else 
    z = zero;
  HxHyLz_n[i+j*dxNext+k*dxNext*dyNext] += -0.125*z;

  //HLH
  if (j>0)
    y = HxLyHz_df2[i+(j-1)*dxNext+k*dxNext*dyNext];
  else 
    y = zero;
  HxLyHz_n[i+j*dxNext+k*dxNext*dyNext] += -0.125*y;
      
  //LHH
  if (i>0)
    x = LxHyHz_df2[(i-1)+j*dxNext+k*dxNext*dyNext];
  else 
    x = zero;
  LxHyHz_n[i+j*dxNext+k*dxNext*dyNext] += -0.125*x;
}

extern "C" __global__ void cu_fwt3df_LC3(_data_t* HxHyHz_df1,_data_t* HxHyHz_df2,_data_t* HxHyHz_n,int dxNext, int dyNext, int dzNext)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  int k = blockIdx.z*blockDim.z+threadIdx.z;
  _data_t x,y,z;
  if ((i>=dxNext)||(j>=dyNext)||(k>=dzNext))
    {
      return;
    }

  //HHH
  x = HxHyHz_df1[i+j*dxNext+k*dxNext*dyNext];
  y = HxHyHz_df2[i+j*dxNext+k*dxNext*dyNext];
  z = HxHyHz_n[i+j*dxNext+k*dxNext*dyNext];
  HxHyHz_df1[i+j*dxNext+k*dxNext*dyNext] = 1.0/3.0*(-2.0*x+y+z);
  HxHyHz_df2[i+j*dxNext+k*dxNext*dyNext] = 1.0/3.0*(2*y-x-z);
  HxHyHz_n[i+j*dxNext+k*dxNext*dyNext] = 1.0/3.0*(x+y+z);
}

// ############################################################################
// CUDA function of iwt depth convolution.
// Loads data to scratchpad (shared memory) and convolve w/ low pass and high pass
// Scratchpad size: K x 2*dy
// Output: Lz/Hz
// Input:  LxLy,LxHy / HxLy, HxHy, dx, dy, dxNext, dyNext,xOffset, yOffset,lod, hid, filterLen
// ############################################################################
extern "C" __global__ void cu_iwt3df_dep(_data_t *out, _data_t *Lz, _data_t *Hz, int dx, int dy,int dz,int dxNext, int dyNext, int dzNext,int xOffset, int yOffset,int zOffset,scalar_t *lod, scalar_t *hid, int filterLen)
{
  extern __shared__ _data_t deps [];
  int const K = blockDim.x;

  int ti = threadIdx.x;
  int tk = threadIdx.z;
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  if (i>=dx){
    return;
  }
  for (int k = tk; k < dz; k += blockDim.z){
    deps[ti + k*K] = Lz[i + j*dx + k*dx*dy];
    deps[ti + (k+dz)*K] = Hz[i + j*dx + k*dx*dy];
  }
  __syncthreads();

  // Low-Pass and High Pass Downsample
  int ind;
  for (int k = tk+zOffset; k < dzNext+zOffset; k += blockDim.z){
	
    _data_t y = deps[0]-deps[0];
#pragma unroll
    for (int f = (k-(filterLen-1)) % 2; f < filterLen; f+=2){
      ind = (k-(filterLen-1)+f)>>1;
      if ((ind >= 0) && (ind < dz)) {
	y += deps[ti + ind*K] * lod[filterLen-1-f];
	y += deps[ti + (ind+dz)*K] * hid[filterLen-1-f];
      }
    }
	
    out[i + j*dx + (k-zOffset)*dx*dy] = y;
  }
}

// ############################################################################
// CUDA function of iwt row convolution. Assumes fwt_col() has already been called.
// Loads data to scratchpad (shared memory) and convolve w/ low pass and high pass
// Scratchpad size: K x 2*dy
// Output: Lx/Hx
// Input:  LxLy,LxHy / HxLy, HxHy, dx, dy, dxNext, dyNext,xOffset, yOffset,lod, hid, filterLen
// ############################################################################
extern "C" __global__ void cu_iwt3df_row(_data_t *out, _data_t *Ly, _data_t *Hy, int dx, int dy,int dz,int dxNext, int dyNext,int dzNext,int xOffset, int yOffset, int zOffset,scalar_t *lod, scalar_t *hid, int filterLen)
{
  extern __shared__ _data_t rows [];
  int const K = blockDim.x;

  int ti = threadIdx.x;
  int tj = threadIdx.y;
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int k = blockIdx.z*blockDim.z+threadIdx.z;
  if (i>=dx){
    return;
  }
  for (int j = tj; j < dy; j += blockDim.y){
    rows[ti + j*K] = Ly[i + j*dx + k*dx*dy];
    rows[ti + (j+dy)*K] = Hy[i + j*dx + k*dx*dy];
  }
  __syncthreads();

  // Low-Pass and High Pass Downsample
  int ind;
  for (int j = tj+yOffset; j < dyNext+yOffset; j += blockDim.y){
	
    _data_t y = rows[0]-rows[0];
#pragma unroll
    for (int f = (j-(filterLen-1)) % 2; f < filterLen; f+=2){
      ind = (j-(filterLen-1)+f)>>1;
      if ((ind >= 0) && (ind < dy)) {
	y += rows[ti + ind*K] * lod[filterLen-1-f];
	y += rows[ti + (ind+dy)*K] * hid[filterLen-1-f];
      }
    }
	
    out[i + (j-yOffset)*dx + k*dx*dyNext] = y;
  }
}

// ############################################################################
// CUDA function of iwt column convolution
// Loads data to scratchpad (shared memory) and convolve w/ low pass and high pass
// Scratchpad size: 2*dx x K
// Output: out
// Input:  Lx, Hx, dx, dy, dxNext, dyNext, lod, hid, filterLen
// ############################################################################
extern "C" __global__ void cu_iwt3df_col(_data_t *out, _data_t *Lx, _data_t *Hx, int dx, int dy,int dz,int dxNext, int dyNext, int dzNext,int xOffset, int yOffset, int zOffset,scalar_t *lod, scalar_t *hid, int filterLen)
{
  extern __shared__ _data_t cols [];

  int ti = threadIdx.x;
  int tj = threadIdx.y;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  int k = blockIdx.z*blockDim.z+threadIdx.z;
  if (j>=dyNext){
    return;
  }
  int dx2 = 2*dx;
  // Load Input to Temp Array
  for (int i = ti; i < dx; i += blockDim.x){
    cols[i + tj*dx2] = Lx[i + j*dx + k*dx*dyNext];
    cols[dx+i + tj*dx2] = Hx[i + j*dx + k*dx*dyNext];
  }
  __syncthreads();

  // Low-Pass and High Pass Downsample
  int ind;
  for (int i = ti+xOffset; i < dxNext+xOffset; i += blockDim.x){
    _data_t y = cols[0]-cols[0];
#pragma unroll
    for (int f = (i-(filterLen-1)) % 2; f < filterLen; f+=2){
      ind = (i-(filterLen-1)+f)>>1;
      if (ind >= 0 && ind < dx) {
	y += cols[ind + tj*dx2] * lod[filterLen-1-f];
	y += cols[dx+ind + tj*dx2] * hid[filterLen-1-f];
      }
    }
    out[(i-xOffset) + j*dxNext + k*dxNext*dyNext] = y;
  }
}

extern "C" __global__ void cu_iwt3df_LC1 (_data_t *HxLyLz_df1,_data_t *HxLyLz_df2,_data_t *HxLyLz_n,_data_t *LxHyLz_df1,_data_t *LxHyLz_df2,_data_t *LxHyLz_n,_data_t *LxLyHz_df1,_data_t *LxLyHz_df2,_data_t *LxLyHz_n,int dx, int dy, int dz)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  int k = blockIdx.z*blockDim.z+threadIdx.z;
  _data_t df1,df2,n;
  scalar_t xGreatZero,yGreatZero,zGreatZero;
  if ((i>=dx)||(j>=dy)||(k>=dz))
    {
      return;
    }

  //HLL
  df1 = HxLyLz_df1[i+j*dx+k*dx*dy];
  df2 = HxLyLz_df2[i+j*dx+k*dx*dy];
  n = HxLyLz_n[i+j*dx+k*dx*dy];
  HxLyLz_df2[i+j*dx+k*dx*dy] = df1;
  HxLyLz_n[i+j*dx+k*dx*dy] = df2;
  yGreatZero = j>0;
  zGreatZero = k>0;
  HxLyLz_df1[i+j*dx+k*dx*dy] = n - yGreatZero*0.25*df1 - zGreatZero*0.25*df2;

  //LHL
  df1 = LxHyLz_df1[i+j*dx+k*dx*dy];
  df2 = LxHyLz_df2[i+j*dx+k*dx*dy];
  n = LxHyLz_n[i+j*dx+k*dx*dy];
  LxHyLz_n[i+j*dx+k*dx*dy] = df2;
  xGreatZero = i>0;
  zGreatZero = k>0;
  LxHyLz_df2[i+j*dx+k*dx*dy] = n - xGreatZero*0.25*df1 - zGreatZero*0.25*df2;
      
  //LLH
  df1 = LxLyHz_df1[i+j*dx+k*dx*dy];
  df2 = LxLyHz_df2[i+j*dx+k*dx*dy];
  n = LxLyHz_n[i+j*dx+k*dx*dy];
  LxLyHz_df1[i+j*dx+k*dx*dy] = df2;
  LxLyHz_df2[i+j*dx+k*dx*dy] = df1;
  yGreatZero = j>0;
  xGreatZero = i>0;
  LxLyHz_n[i+j*dx+k*dx*dy] = n - yGreatZero*0.25*df1 - xGreatZero*0.25*df2;
}

extern "C" __global__ void cu_iwt3df_LC1_diff (_data_t *HxLyLz_df1,_data_t *HxLyLz_df2,_data_t *HxLyLz_n,_data_t *LxHyLz_df1,_data_t *LxHyLz_df2,_data_t *LxHyLz_n,_data_t *LxLyHz_df1,_data_t *LxLyHz_df2,_data_t *LxLyHz_n,int dx, int dy, int dz)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  int k = blockIdx.z*blockDim.z+threadIdx.z;
  _data_t x,y,z;
  _data_t zero = make_float2(0.f,0.f);
  if ((i>=dx)||(j>=dy)||(k>=dz))
    {
      return;
    }

  //HLL
  if (j>0)
    y = HxLyLz_df2[i+(j-1)*dx+k*dx*dy];
  else
    y = zero;
  if (k>0)
    z = HxLyLz_n[i+j*dx+(k-1)*dx*dy];
  else
    z = zero;
  HxLyLz_df1[i+j*dx+k*dx*dy] += 0.25*y + 0.25*z;

  //LHL
  if (i>0)
    x = LxHyLz_df1[(i-1)+j*dx+k*dx*dy];
  else
    x = zero;
  if (k>0)
    z = LxHyLz_n[i+j*dx+(k-1)*dx*dy];
  else
    z = zero;
  LxHyLz_df2[i+j*dx+k*dx*dy] += 0.25*x + 0.25*z;

  //LLH
  if (j>0)
    y = LxLyHz_df2[i+(j-1)*dx+k*dx*dy];
  else
    y = zero;
  if (i>0)
    x = LxLyHz_df1[(i-1)+j*dx+k*dx*dy];
  else
    x = zero;
  LxLyHz_n[i+j*dx+k*dx*dy] += 0.25*y + 0.25*x;
}

extern "C" __global__ void cu_iwt3df_LC2 (_data_t* HxHyLz_df1,_data_t* HxHyLz_df2,_data_t* HxHyLz_n,_data_t* HxLyHz_df1,_data_t* HxLyHz_df2,_data_t* HxLyHz_n,_data_t* LxHyHz_df1,_data_t* LxHyHz_df2,_data_t* LxHyHz_n,int dx, int dy, int dz)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  int k = blockIdx.z*blockDim.z+threadIdx.z;
  _data_t df1,df2,n;
  scalar_t xGreatZero,yGreatZero,zGreatZero;
  if ((i>=dx)||(j>=dy)||(k>=dz))
    {
      return;
    }

  //HHL
  df1 = HxHyLz_df1[i+j*dx+k*dx*dy];
  df2 = HxHyLz_df2[i+j*dx+k*dx*dy];
  n = HxHyLz_n[i+j*dx+k*dx*dy];
  HxHyLz_n[i+j*dx+k*dx*dy] = df2;
  zGreatZero = k>0;
  HxHyLz_df1[i+j*dx+k*dx*dy] = df1+n-zGreatZero*0.125*df2;
  HxHyLz_df2[i+j*dx+k*dx*dy] = n-df1-zGreatZero*0.125*df2;

  //HLH
  df1 = HxLyHz_df1[i+j*dx+k*dx*dy];
  df2 = HxLyHz_df2[i+j*dx+k*dx*dy];
  n = HxLyHz_n[i+j*dx+k*dx*dy];
  HxLyHz_df2[i+j*dx+k*dx*dy] = df2;
  yGreatZero = j>0;
  HxLyHz_n[i+j*dx+k*dx*dy] = df1+n-yGreatZero*0.125*df2;
  HxLyHz_df1[i+j*dx+k*dx*dy] = n-df1-yGreatZero*0.125*df2;
      
  //LHH
  df1 = LxHyHz_df1[i+j*dx+k*dx*dy];
  df2 = LxHyHz_df2[i+j*dx+k*dx*dy];
  n = LxHyHz_n[i+j*dx+k*dx*dy];
  LxHyHz_df1[i+j*dx+k*dx*dy] = df2;
  xGreatZero = i>0;
  LxHyHz_df2[i+j*dx+k*dx*dy] = df1+n-xGreatZero*0.125*df2;
  LxHyHz_n[i+j*dx+k*dx*dy] = n-df1-xGreatZero*0.125*df2;
}

extern "C" __global__ void cu_iwt3df_LC2_diff (_data_t* HxHyLz_df1,_data_t* HxHyLz_df2,_data_t* HxHyLz_n,_data_t* HxLyHz_df1,_data_t* HxLyHz_df2,_data_t* HxLyHz_n,_data_t* LxHyHz_df1,_data_t* LxHyHz_df2,_data_t* LxHyHz_n,int dx, int dy, int dz)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  int k = blockIdx.z*blockDim.z+threadIdx.z;
  _data_t x,y,z;
  _data_t zero = make_float2(0.f,0.f);
  if ((i>=dx)||(j>=dy)||(k>=dz))
    {
      return;
    }

  //HHL
  if (k>0)
    z = HxHyLz_n[i+j*dx+(k-1)*dx*dy];
  else 
    z = zero;
  HxHyLz_df1[i+j*dx+k*dx*dy] += 0.125*z;
  HxHyLz_df2[i+j*dx+k*dx*dy] += 0.125*z;

  //HLH
  if (j>0)
    y = HxLyHz_df2[i+(j-1)*dx+k*dx*dy];
  else 
    y = zero;
  HxLyHz_df1[i+j*dx+k*dx*dy] += 0.125*y;
  HxLyHz_n[i+j*dx+k*dx*dy] += 0.125*y;
      
  //LHH
  if (i>0)
    x = LxHyHz_df1[(i-1)+j*dx+k*dx*dy];
  else 
    x = zero;
  LxHyHz_df2[i+j*dx+k*dx*dy] += 0.125*x;
  LxHyHz_n[i+j*dx+k*dx*dy] += 0.125*x;
}

extern "C" __global__ void cu_iwt3df_LC3 (_data_t* HxHyHz_df1,_data_t* HxHyHz_df2,_data_t* HxHyHz_n,int dx, int dy, int dz)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  int k = blockIdx.z*blockDim.z+threadIdx.z;
  _data_t df1,df2,n;
  if ((i>=dx)||(j>=dy)||(k>=dz))
    {
      return;
    }

  //HHH
  df1 = HxHyHz_df1[i+j*dx+k*dx*dy];
  df2 = HxHyHz_df2[i+j*dx+k*dx*dy];
  n = HxHyHz_n[i+j*dx+k*dx*dy];
  HxHyHz_df1[i+j*dx+k*dx*dy] = n-df1;
  HxHyHz_df2[i+j*dx+k*dx*dy] = df2+n;
  HxHyHz_n[i+j*dx+k*dx*dy] = df1-df2+n;
}
extern "C" __global__ void cu_mult(_data_t* in, _data_t mult, int maxInd)
{
  int ind = blockIdx.x*blockDim.x+threadIdx.x;
  if (ind > maxInd)
    {
      return;
    }
  in[ind] = in[ind]*mult;
}

extern "C" __global__ void cu_add_mult(_data_t* out, _data_t* in, _data_t mult, int maxInd)
{
  int ind = blockIdx.x*blockDim.x+threadIdx.x;
  if (ind > maxInd)
    {
      return;
    }
  _data_t i = out[ind];
  out[ind] = i+(out[ind]-i)*mult;
}

__global__ void cu_soft_thresh (_data_t* in, scalar_t thresh, int numMax)
{
  int const i = threadIdx.x + blockDim.x*blockIdx.x;
  if (i>numMax)
    return;
  scalar_t norm = abs(in[i]);
  scalar_t red = norm - thresh;
  in[i] = (red > 0.f) ? ((red / norm) * (in[i])) : in[i]-in[i];
}

__global__ void cu_circshift(_data_t* data, _data_t* dataCopy, int dx, int dy, int dz,int shift1, int shift2,int shift3) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;

  if (index >= dx*dy*dz) {
    return;
  }
  int indexShifted = (index+shift1+shift2*dx+shift3*dx*dy)%(dx*dy*dz);
  data[indexShifted] = dataCopy[index];
}

__global__ void cu_circunshift(_data_t* data, _data_t* dataCopy, int dx, int dy, int dz,int shift1, int shift2,int shift3) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;

  if (index >= dx*dy*dz) {
    return;
  }
  int indexShifted = (index+shift1+shift2*dx+shift3*dx*dy)%(dx*dy*dz);
  data[index] = dataCopy[indexShifted];
}

