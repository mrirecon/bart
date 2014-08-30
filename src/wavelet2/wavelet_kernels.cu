/* Copyright 2013. The Regents of the University of California.
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

#include "wavelet_kernels.h"
#include "wavelet_impl.h"

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
inline _hdev_ double2 operator* (double2 z1, float alpha) {
  return make_double2 (z1.x*alpha, z1.y*alpha);		
}
inline _hdev_ double2 operator* (float alpha,double2 z1) {
  return make_double2 (z1.x*alpha, z1.y*alpha);		
}
inline _hdev_ void operator+= (double2 &z1, double2 z2) {
  z1.x += z2.x;
  z1.y += z2.y;		
}
inline _hdev_ float abs(double2 z1) {
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

/********** Headers ************/
__global__ void cu_fwt2_col(_data_t *Lx, _data_t *Hx, _data_t *in, int dx, int dy, int dxNext, const scalar_t *lod, const scalar_t *hid, int filterLen);
__global__ void cu_fwt2_row(_data_t *Ly,_data_t *Hy,_data_t *in,int dx,int dy,int dxNext,int dyNext,const scalar_t *lod,const scalar_t *hid,int filterLen);
__global__ void cu_iwt2_row(_data_t *out, _data_t *Ly, _data_t *Hy, int dx, int dy,int dxNext, int dyNext, int xOffset, int yOffset, const scalar_t *lod, const scalar_t *hid, int filterLen);
__global__ void cu_iwt2_col(_data_t *out,_data_t *Lx,_data_t *Hx,int dx,int dy,int dxNext,int dyNext,int xOffset,int yOffset,const scalar_t *lod,const scalar_t *hid,int filterLen);
__global__ void cu_fwt3_col(_data_t *Lx,_data_t *Hx,_data_t *in,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,const scalar_t *lod,const scalar_t *hid,int filterLen);
__global__ void cu_fwt3_row(_data_t *Ly,_data_t *Hy,_data_t *in,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,const scalar_t *lod,const scalar_t *hid,int filterLen);
__global__ void cu_fwt3_dep(_data_t *Lz,_data_t *Hz,_data_t *in,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,const scalar_t *lod,const scalar_t *hid,int filterLen);
__global__ void cu_iwt3_dep(_data_t *out,_data_t *Lz,_data_t *Hz,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,int xOffset,int yOffset,int zOffset,const scalar_t *lod,const scalar_t *hid,int filterLen);
__global__ void cu_iwt3_row(_data_t *out,_data_t *Ly,_data_t *Hy,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,int xOffset,int yOffset,int zOffset,const scalar_t *lod,const scalar_t *hid,int filterLen);
__global__ void cu_iwt3_col(_data_t *out,_data_t *Lx,_data_t *Hx,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,int xOffset,int yOffset,int zOffset,const scalar_t *lod,const scalar_t *hid,int filterLen);
__global__ void cu_soft_thresh (_data_t* in, scalar_t thresh, int numMax);


/********** 2D GPU Functions ************/
extern "C" void fwt2_gpu(struct wavelet_plan_s* plan, data_t *dev_coeff_c, const data_t *dev_inImage_c)
{
  circshift(plan, (data_t*)dev_inImage_c); // FIXME
  _data_t* dev_coeff = (_data_t*) dev_coeff_c;
  _data_t* dev_inImage = (_data_t*) dev_inImage_c;
  _data_t* dev_temp = (_data_t*) plan->tmp_mem_tr;

  // Get dimensions
  int dx = plan->imSize_tr[0];
  int dy = plan->imSize_tr[1];

  int dxNext = plan->waveSizes_tr[0 + 2*plan->numLevels_tr];
  int dyNext = plan->waveSizes_tr[1 + 2*plan->numLevels_tr];
  int blockSize = dxNext*dyNext;

  // Initialize filters
  const scalar_t *dev_lod = plan->lod;
  const scalar_t *dev_hid = plan->hid;

  // Initialize variables and Pointers for FWT
  int const SHMEM_SIZE = 16384;
  int const T = 512;
  int mem, K;
  dim3 numBlocks, numThreads;

  _data_t* dev_tempLx,*dev_tempHx,*dev_LxLy,*dev_HxLy,*dev_LxHy,*dev_HxHy,*dev_currentInImage;
  dev_tempLx = dev_temp;
  dev_tempHx = dev_tempLx+plan->numCoeff_tr/2;
  dev_LxLy = dev_coeff;
  dev_HxLy = dev_LxLy + plan->waveSizes_tr[0]*plan->waveSizes_tr[1];
  for (int l = 1; l <= plan->numLevels_tr; ++l){
    dev_HxLy += 3*plan->waveSizes_tr[0 + 2*l]*plan->waveSizes_tr[1 + 2*l];
  }
  dev_currentInImage = dev_inImage;
  
  // Loop through levels
  int l;
  for (l = plan->numLevels_tr; l >= 1; --l)
    {
      dxNext = plan->waveSizes_tr[0 + 2*l];
      dyNext = plan->waveSizes_tr[1 + 2*l];
      blockSize = dxNext*dyNext;
      
      dev_HxLy = dev_HxLy - 3*blockSize;
      dev_LxHy = dev_HxLy + blockSize;
      dev_HxHy = dev_HxLy + 2*blockSize;
      
      // FWT Columns
      K = (SHMEM_SIZE-16)/(dx*sizeof(_data_t));
      numBlocks = dim3(1,(dy+K-1)/K);
      numThreads = dim3(T/K,K);
      mem = K*dx*sizeof(_data_t);

      cu_fwt2_col <<< numBlocks,numThreads,mem >>>(dev_tempLx,dev_tempHx,dev_currentInImage,dx,dy,dxNext,dev_lod,dev_hid,plan->filterLen);
      cuda_sync();	
  
      // FWT Rows
      K = (SHMEM_SIZE-16)/(dy*sizeof(_data_t));
      numBlocks = dim3(((dxNext)+K-1)/K,1);
      numThreads = dim3(K,T/K);
      mem = K*dy*sizeof(_data_t);
      
      cu_fwt2_row <<< numBlocks,numThreads,mem >>>(dev_LxLy,dev_LxHy,dev_tempLx,dx,dy,dxNext,dyNext,dev_lod,dev_hid,plan->filterLen);
      cu_fwt2_row <<< numBlocks,numThreads,mem >>>(dev_HxLy,dev_HxHy,dev_tempHx,dx,dy,dxNext,dyNext,dev_lod,dev_hid,plan->filterLen);
      cuda_sync();      
	  
      // Update variables for next iteration
      dev_currentInImage = dev_LxLy;
      dx = dxNext;
      dy = dyNext;
    }
  circunshift(plan, (data_t*)dev_inImage_c); // FIXME
}

extern "C" void iwt2_gpu(struct wavelet_plan_s* plan, data_t *dev_outImage_c, const data_t *dev_coeff_c)
{
  _data_t* dev_coeff = (_data_t*) dev_coeff_c;
  _data_t* dev_outImage = (_data_t*) dev_outImage_c;
  _data_t* dev_temp = (_data_t*) plan->tmp_mem_tr;

  // Workspace dimensions
  int dxWork = plan->waveSizes_tr[0 + 2*plan->numLevels_tr]*2-1 + plan->filterLen-1;
  int dyWork = plan->waveSizes_tr[1 + 2*plan->numLevels_tr]*2-1 + plan->filterLen-1;

  // Initialize filters
  const scalar_t *dev_lor = plan->lor;
  const scalar_t *dev_hir = plan->hir;

  // Initialize variables and pointers for IWT
  int const SHMEM_SIZE = 16384;
  int const T = 512;
  int mem,K;
  dim3 numBlocks, numThreads;
  int dx = plan->waveSizes_tr[0];
  int dy = plan->waveSizes_tr[1];

  _data_t* dev_tempLx, *dev_tempHx, *dev_LxLy,*dev_HxLy,*dev_LxHy,*dev_HxHy,*dev_currentOutImage;
  dev_tempLx = dev_temp;
  dev_tempHx = dev_tempLx + plan->numCoeff_tr/2;
  dev_LxLy = dev_coeff;
  dev_HxLy = dev_LxLy + dx*dy;
  dev_currentOutImage = dev_LxLy;
  int level;
  for (level = 1; level < plan->numLevels_tr+1; ++level)
    {
      dx = plan->waveSizes_tr[0 + 2*level];
      dy = plan->waveSizes_tr[1 + 2*level];
      int blockSize = dx*dy;
      int dxNext = plan->waveSizes_tr[0+2*(level+1)];
      int dyNext = plan->waveSizes_tr[1+2*(level+1)];

      dev_LxHy = dev_HxLy + blockSize;
      dev_HxHy = dev_HxLy + 2*blockSize;

      // Calclate Offset
      dxWork = (2*dx-1 + plan->filterLen-1);
      dyWork = (2*dy-1 + plan->filterLen-1);
      int xOffset = (int) floor((dxWork - dxNext) / 2.0);
      int yOffset = (int) floor((dyWork - dyNext) / 2.0);

      // Set Current OutImage
      if (level == plan->numLevels_tr)
	dev_currentOutImage = dev_outImage;

      // IWT Rows
      K = (SHMEM_SIZE-16)/(2*dy*sizeof(_data_t));
      numBlocks = dim3((dx+K-1)/K,1);
      numThreads = dim3(K,(T/K));
      mem = K*2*dy*sizeof(_data_t);

      cu_iwt2_row <<< numBlocks,numThreads,mem >>>(dev_tempLx,dev_LxLy,dev_LxHy,dx,dy,dxNext,dyNext,xOffset,yOffset,dev_lor,dev_hir,plan->filterLen);
      cu_iwt2_row <<< numBlocks,numThreads,mem >>>(dev_tempHx,dev_HxLy,dev_HxHy,dx,dy,dxNext,dyNext,xOffset,yOffset,dev_lor,dev_hir,plan->filterLen);
      cuda_sync();
  
      // IWT Columns
      K = (SHMEM_SIZE-16)/(2*dx*sizeof(_data_t));
      numBlocks = dim3(1,(dyNext+K-1)/K);
      numThreads = dim3((T/K),K);
      mem = K*2*dx*sizeof(_data_t);

      cu_iwt2_col <<< numBlocks,numThreads,mem >>>(dev_currentOutImage,dev_tempLx,dev_tempHx,dx,dy,dxNext,dyNext,xOffset,yOffset,dev_lor,dev_hir,plan->filterLen);
      cuda_sync();

      dev_HxLy += 3*blockSize;
    }
  circunshift(plan,dev_outImage_c);
}

extern "C" void wavthresh2_gpu(struct wavelet_plan_s* plan, data_t *dev_outImage_c, const data_t *dev_inImage_c, scalar_t thresh)
{
  const data_t* dev_inImage = dev_inImage_c;
  data_t* dev_outImage = dev_outImage_c;
  data_t* dev_coeff;
  cuda(Malloc( (void**)&dev_coeff, plan->numCoeff_tr*sizeof(_data_t) ));

  fwt2_gpu (plan, dev_coeff, dev_inImage);
  softthresh_gpu (plan, dev_coeff, thresh);
  iwt2_gpu (plan, dev_outImage, dev_coeff);

  cuda(Free( dev_coeff ));
}

/********** 3D GPU Functions ************/
extern "C" void fwt3_gpu(struct wavelet_plan_s* plan, data_t *dev_coeff_c, const data_t *dev_inImage_c)
{
  circshift(plan, (data_t*)dev_inImage_c);
  _data_t* dev_coeff = (_data_t*) dev_coeff_c;
  _data_t* dev_inImage = (_data_t*) dev_inImage_c;
  // Cast from generic _data_type to device compatible _data_t
  _data_t* dev_temp1,*dev_temp2;
  dev_temp1 = (_data_t*) plan->tmp_mem_tr;
  dev_temp2 = (_data_t*) plan->tmp_mem_tr + plan->numCoeff_tr;

  // Get dimensions
  int dx = plan->imSize_tr[0];
  int dy = plan->imSize_tr[1];
  int dz = plan->imSize_tr[2];

  int dxNext = plan->waveSizes_tr[0 + 3*plan->numLevels_tr];
  int dyNext = plan->waveSizes_tr[1 + 3*plan->numLevels_tr];
  int dzNext = plan->waveSizes_tr[2 + 3*plan->numLevels_tr];
  int blockSize = dxNext*dyNext*dzNext;

  // Initialize filters
  const scalar_t *dev_lod = plan->lod;
  const scalar_t *dev_hid = plan->hid;

  // Initialize variables and Pointers for FWT
  int const SHMEM_SIZE = 16384;
  int const T = 512;
  int mem, K;
  dim3 numBlocks, numThreads;

  _data_t *dev_tempLx,*dev_tempHx;
  dev_tempLx = dev_temp1;
  dev_tempHx = dev_tempLx + plan->numCoeff_tr/2;
  _data_t *dev_tempLxLy,*dev_tempHxLy,*dev_tempLxHy,*dev_tempHxHy;
  dev_tempLxLy = dev_temp2;
  dev_tempHxLy = dev_tempLxLy + plan->numCoeff_tr/4;
  dev_tempLxHy = dev_tempHxLy + plan->numCoeff_tr/4;
  dev_tempHxHy = dev_tempLxHy + plan->numCoeff_tr/4;
  _data_t *dev_LxLyLz,*dev_HxLyLz,*dev_LxHyLz,*dev_HxHyLz,*dev_LxLyHz,*dev_HxLyHz,*dev_LxHyHz,*dev_HxHyHz,*dev_currentInImage;
  dev_LxLyLz = dev_coeff;
  dev_HxLyLz = dev_LxLyLz + plan->waveSizes_tr[0]*plan->waveSizes_tr[1]*plan->waveSizes_tr[2];
  for (int l = 1; l <= plan->numLevels_tr; ++l){
    dev_HxLyLz += 7*plan->waveSizes_tr[0 + 3*l]*plan->waveSizes_tr[1 + 3*l]*plan->waveSizes_tr[2 + 3*l];
  }
  dev_currentInImage = dev_inImage;

  // Loop through levels
  for (int l = plan->numLevels_tr; l >= 1; --l)
    {
      dxNext = plan->waveSizes_tr[0 + 3*l];
      dyNext = plan->waveSizes_tr[1 + 3*l];
      dzNext = plan->waveSizes_tr[2 + 3*l];
      blockSize = dxNext*dyNext*dzNext;

      dev_HxLyLz = dev_HxLyLz - 7*blockSize;
      dev_LxHyLz = dev_HxLyLz + blockSize;
      dev_HxHyLz = dev_LxHyLz + blockSize;
      dev_LxLyHz = dev_HxHyLz + blockSize;
      dev_HxLyHz = dev_LxLyHz + blockSize;
      dev_LxHyHz = dev_HxLyHz + blockSize;
      dev_HxHyHz = dev_LxHyHz + blockSize;

      // FWT Columns
      K = (SHMEM_SIZE-16)/(dx*sizeof(_data_t));
      numBlocks = dim3(1,(dy+K-1)/K,dz);
      numThreads = dim3(T/K,K,1);
      mem = K*dx*sizeof(_data_t);
      
      cu_fwt3_col <<< numBlocks,numThreads,mem >>>(dev_tempLx,dev_tempHx,dev_currentInImage,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod,dev_hid,plan->filterLen);
      cuda_sync();
      
      // FWT Rows
      K = (SHMEM_SIZE-16)/(dy*sizeof(_data_t));
      numBlocks = dim3(((dxNext)+K-1)/K,1,dz);
      numThreads = dim3(K,T/K,1);
      mem = K*dy*sizeof(_data_t);
      
      cu_fwt3_row <<< numBlocks,numThreads,mem >>>(dev_tempLxLy,dev_tempLxHy,dev_tempLx,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod,dev_hid,plan->filterLen);
      cu_fwt3_row <<< numBlocks,numThreads,mem >>>(dev_tempHxLy,dev_tempHxHy,dev_tempHx,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod,dev_hid,plan->filterLen);
      cuda_sync();

      // FWT Depths
      K = (SHMEM_SIZE-16)/(dz*sizeof(_data_t));
      numBlocks = dim3(((dxNext)+K-1)/K,dyNext,1);
      numThreads = dim3(K,1,T/K);
      mem = K*dz*sizeof(_data_t);
      
      cu_fwt3_dep <<< numBlocks,numThreads,mem >>>(dev_LxLyLz,dev_LxLyHz,dev_tempLxLy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod,dev_hid,plan->filterLen);
      cu_fwt3_dep <<< numBlocks,numThreads,mem >>>(dev_LxHyLz,dev_LxHyHz,dev_tempLxHy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod,dev_hid,plan->filterLen);
      cu_fwt3_dep <<< numBlocks,numThreads,mem >>>(dev_HxLyLz,dev_HxLyHz,dev_tempHxLy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod,dev_hid,plan->filterLen);
      cu_fwt3_dep <<< numBlocks,numThreads,mem >>>(dev_HxHyLz,dev_HxHyHz,dev_tempHxHy,dx,dy,dz,dxNext,dyNext,dzNext,dev_lod,dev_hid,plan->filterLen);
      cuda_sync();
	  
      dev_currentInImage = dev_LxLyLz;
      dx = dxNext;
      dy = dyNext;
      dz = dzNext;
    }
  circunshift(plan, (data_t*)dev_inImage_c);
}

extern "C" void iwt3_gpu(struct wavelet_plan_s* plan, data_t *dev_outImage_c, const data_t *dev_coeff_c)
{
  _data_t* dev_coeff = (_data_t*) dev_coeff_c;
  _data_t* dev_outImage = (_data_t*) dev_outImage_c;
  // Allocate Temp Memory
  _data_t* dev_temp1, *dev_temp2;
  dev_temp1 = (_data_t*) plan->tmp_mem_tr;
  dev_temp2 = (_data_t*) plan->tmp_mem_tr + plan->numCoeff_tr;

  // Workspace dimensions
  int dxWork = plan->waveSizes_tr[0 + 3*plan->numLevels_tr]*2-1 + plan->filterLen-1;
  int dyWork = plan->waveSizes_tr[1 + 3*plan->numLevels_tr]*2-1 + plan->filterLen-1;
  int dzWork = plan->waveSizes_tr[2 + 3*plan->numLevels_tr]*2-1 + plan->filterLen-1;

  // Initialize filters
  const scalar_t *dev_lor = plan->lor;
  const scalar_t *dev_hir = plan->hir;

  // Initialize variables and pointers for IWT
  int const SHMEM_SIZE = 16384;
  int const T = 512;
  int mem,K;
  dim3 numBlocks, numThreads;
  int dx = plan->waveSizes_tr[0];
  int dy = plan->waveSizes_tr[1];
  int dz = plan->waveSizes_tr[2];

  _data_t *dev_LxLyLz,*dev_HxLyLz,*dev_LxHyLz,*dev_HxHyLz,*dev_LxLyHz,*dev_HxLyHz,*dev_LxHyHz,*dev_HxHyHz;
  dev_LxLyLz = dev_coeff;
  dev_HxLyLz = dev_LxLyLz + dx*dy*dz;
  _data_t *dev_tempLxLy,*dev_tempHxLy,*dev_tempLxHy,*dev_tempHxHy;
  dev_tempLxLy = dev_temp1;
  dev_tempHxLy = dev_tempLxLy + plan->numCoeff_tr/4;
  dev_tempLxHy = dev_tempHxLy + plan->numCoeff_tr/4;
  dev_tempHxHy = dev_tempLxHy + plan->numCoeff_tr/4;
  _data_t *dev_tempLx,*dev_tempHx;
  dev_tempLx = dev_temp2;
  dev_tempHx = dev_tempLx + plan->numCoeff_tr/2;
  _data_t *dev_currentOutImage;
  dev_currentOutImage = dev_LxLyLz;

  for (int level = 1; level < plan->numLevels_tr+1; ++level)
    {
      dx = plan->waveSizes_tr[0 + 3*level];
      dy = plan->waveSizes_tr[1 + 3*level];
      dz = plan->waveSizes_tr[2 + 3*level];
      int blockSize = dx*dy*dz;
      int dxNext = plan->waveSizes_tr[0+3*(level+1)];
      int dyNext = plan->waveSizes_tr[1+3*(level+1)];
      int dzNext = plan->waveSizes_tr[2+3*(level+1)];

      dev_LxHyLz = dev_HxLyLz + blockSize;
      dev_HxHyLz = dev_LxHyLz + blockSize;
      dev_LxLyHz = dev_HxHyLz + blockSize;
      dev_HxLyHz = dev_LxLyHz + blockSize;
      dev_LxHyHz = dev_HxLyHz + blockSize;
      dev_HxHyHz = dev_LxHyHz + blockSize;
	  
      // Calclate Offset
      dxWork = (2*dx-1 + plan->filterLen-1);
      dyWork = (2*dy-1 + plan->filterLen-1);
      dzWork = (2*dz-1 + plan->filterLen-1);
      int xOffset = (int) floor((dxWork - dxNext) / 2.0);
      int yOffset = (int) floor((dyWork - dyNext) / 2.0);
      int zOffset = (int) floor((dzWork - dzNext) / 2.0);

      // Set Current OutImage
      if (level==plan->numLevels_tr)
	dev_currentOutImage = dev_outImage;

      // IWT Depths
      K = (SHMEM_SIZE-16)/(2*dz*sizeof(_data_t));
      numBlocks = dim3((dx+K-1)/K,dy,1);
      numThreads = dim3(K,1,(T/K));
      mem = K*2*dz*sizeof(_data_t);

      cu_iwt3_dep <<< numBlocks,numThreads,mem >>>(dev_tempLxLy,dev_LxLyLz,dev_LxLyHz,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor,dev_hir,plan->filterLen);
      cu_iwt3_dep <<< numBlocks,numThreads,mem >>>(dev_tempHxLy,dev_HxLyLz,dev_HxLyHz,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor,dev_hir,plan->filterLen);
      cu_iwt3_dep <<< numBlocks,numThreads,mem >>>(dev_tempLxHy,dev_LxHyLz,dev_LxHyHz,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor,dev_hir,plan->filterLen);
      cu_iwt3_dep <<< numBlocks,numThreads,mem >>>(dev_tempHxHy,dev_HxHyLz,dev_HxHyHz,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor,dev_hir,plan->filterLen);
      cuda_sync();

      // IWT Rows
      K = (SHMEM_SIZE-16)/(2*dy*sizeof(_data_t));
      numBlocks = dim3((dx+K-1)/K,1,dzNext);
      numThreads = dim3(K,(T/K),1);
      mem = K*2*dy*sizeof(_data_t);

      cu_iwt3_row <<< numBlocks,numThreads,mem >>>(dev_tempLx,dev_tempLxLy,dev_tempLxHy,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor,dev_hir,plan->filterLen);
      cu_iwt3_row <<< numBlocks,numThreads,mem >>>(dev_tempHx,dev_tempHxLy,dev_tempHxHy,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor,dev_hir,plan->filterLen);
      cuda_sync();

      // IWT Columns
      K = (SHMEM_SIZE-16)/(2*dx*sizeof(_data_t));
      numBlocks = dim3(1,(dyNext+K-1)/K,dzNext);
      numThreads = dim3((T/K),K,1);
      mem = K*2*dx*sizeof(_data_t);

      cu_iwt3_col <<< numBlocks,numThreads,mem >>>(dev_currentOutImage,dev_tempLx,dev_tempHx,dx,dy,dz,dxNext,dyNext,dzNext,xOffset,yOffset,zOffset,dev_lor,dev_hir,plan->filterLen);
      cuda_sync();
      dev_HxLyLz += 7*blockSize;
    }
  circunshift(plan,dev_outImage_c);
}

extern "C" void wavthresh3_gpu(struct wavelet_plan_s* plan, data_t *dev_outImage_c, const data_t *dev_inImage_c, scalar_t thresh)
{
  const data_t* dev_inImage = dev_inImage_c;
  data_t* dev_outImage = dev_outImage_c;
  data_t* dev_coeff;
  cuda(Malloc( (void**)&dev_coeff, plan->numCoeff_tr*sizeof(_data_t) ));

  fwt3_gpu (plan, dev_coeff, dev_inImage);
  softthresh_gpu (plan, dev_coeff, thresh);
  iwt3_gpu (plan, dev_outImage, dev_coeff);

  cuda(Free( dev_coeff ));
}

/********** Aux functions **********/
extern "C" void softthresh_gpu (struct wavelet_plan_s* plan, data_t* coeff_c,scalar_t thresh)
{
  assert(plan->use_gpu==1||plan->use_gpu==2);

  _data_t* dev_coeff;
  dev_coeff = (_data_t*) coeff_c;
  int numMax;
  int const T = 512;
  dim3 numBlocks, numThreads;
  numMax = plan->numCoeff_tr;//-plan->numCoarse_tr;
  numBlocks = dim3((numMax+T-1)/T,1,1);
  numThreads = dim3(T,1,1);
  cu_soft_thresh <<< numBlocks,numThreads>>> (dev_coeff,thresh,numMax);
}




extern "C" void prepare_wavelet_filters_gpu(struct wavelet_plan_s* plan,int filterLen,const float* filter)
{
  // copy filters to device
  cuda(Malloc( (void**)&plan->lod, 4*plan->filterLen*sizeof(scalar_t) ));
  cuda(Memcpy( (void*) plan->lod, filter, 4*plan->filterLen*sizeof(scalar_t), cudaMemcpyHostToDevice ));
  plan->hid = plan->lod+plan->filterLen;
  plan->lor = plan->hid+plan->filterLen;
  plan->hir = plan->lor+plan->filterLen;
}

extern "C" void prepare_wavelet_temp_gpu(struct wavelet_plan_s* plan)
{
  cuda(Malloc( (void**)&plan->tmp_mem_tr, (sizeof(data_t)*plan->numCoeff_tr*2)));
}

extern "C" void wavelet_free_gpu(const struct wavelet_plan_s* plan)
{
  cuda(Free(plan->tmp_mem_tr));
  cuda(Free((void*)plan->lod));
}



/********** 2D Kernels ************/
__global__ void cu_fwt2_col(_data_t *Lx, _data_t *Hx, _data_t *in, int dx, int dy, int dxNext, const scalar_t *lod, const scalar_t *hid, int filterLen)
{
  extern __shared__ _data_t cols [];
  int ti = threadIdx.x;
  int tj = threadIdx.y;
  int j = blockIdx.y*blockDim.y+threadIdx.y;

  if (j>=dy) {
    return;
  }
  int i;
  // Load Input to Temp Array
  for (i = ti; i < dx; i += blockDim.x){
    cols[i + tj*dx] = in[i + j*dx];
  }
  __syncthreads();
  // Low-Pass and High-Pass Downsample
  int ind, lessThan, greaThan;
  for (i = ti; i < dxNext; i += blockDim.x){
    _data_t y = cols[0]-cols[0];
    _data_t z = cols[0]-cols[0];
    int f;
#pragma unroll
    for (f = 0; f < filterLen; f++){
      ind = 2*i+1 - (filterLen-1)+f;
      lessThan = (int) (ind<0);
      greaThan = (int) (ind>=dx);
      ind = -1*lessThan+ind*(-2*lessThan+1);
      ind = (2*dx-1)*greaThan+ind*(-2*greaThan+1);
	  
      y += cols[ind + tj*dx] * lod[filterLen-1-f];
      z += cols[ind + tj*dx] * hid[filterLen-1-f];
    }
    Lx[i + j*dxNext] = y;
    Hx[i + j*dxNext] = z;
  }
}

__global__ void cu_fwt2_row(_data_t *Ly,_data_t *Hy,_data_t *in,int dx,int dy,int dxNext,int dyNext,const scalar_t *lod,const scalar_t *hid,int filterLen)
{
  extern __shared__ _data_t rows [];
  int const K = blockDim.x;
  int ti = threadIdx.x;
  int tj = threadIdx.y;
  int i = blockIdx.x*blockDim.x+threadIdx.x;

  if (i>=dxNext)
    {
      return;
    }
  // Load Chunks of data into scratchpad
  int j;
  for (j = tj; j < dy; j += blockDim.y){
    rows[ti + j*K] = in[i + j*dxNext];
  }
  __syncthreads();

  // Low-Pass and High Pass Convolution
  int ind, lessThan, greaThan;
  for (j = tj; j < dyNext; j += blockDim.y){
    _data_t y = rows[0]-rows[0];
    _data_t z = rows[0]-rows[0];
    int f;
#pragma unroll
    for (f = 0; f < filterLen; f++){
      ind = 2*j+1 - (filterLen-1)+f;
      lessThan = (int) (ind<0);
      greaThan = (int) (ind>=dy);
      ind = -1*lessThan+ind*(-2*lessThan+1);
      ind = (2*dy-1)*greaThan+ind*(-2*greaThan+1);
	  
      y += rows[ti + ind*K] * lod[filterLen-1-f];
      z += rows[ti + ind*K] * hid[filterLen-1-f];
    }
    Ly[i + j*dxNext] = y;
    Hy[i + j*dxNext] = z;
  }
}

__global__ void cu_iwt2_row(_data_t *out, _data_t *Ly, _data_t *Hy, int dx, int dy,int dxNext, int dyNext, int xOffset, int yOffset, const scalar_t *lod, const scalar_t *hid, int filterLen)
{
  extern __shared__ _data_t rows [];
  int const K = blockDim.x;

  int ti = threadIdx.x;
  int tj = threadIdx.y;
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=dx){
    return;
  }
  int j;
  for (j = tj; j < dy; j += blockDim.y){
    rows[ti + j*K] = Ly[i + j*dx];
    rows[ti + (j+dy)*K] = Hy[i + j*dx];
  }
  __syncthreads();

  // Low-Pass and High Pass Downsample
  int ind;
  for (j = tj+yOffset; j < dyNext+yOffset; j += blockDim.y){
	
    _data_t y = rows[0]-rows[0];
    int f;
#pragma unroll
    for (f = (j-(filterLen-1)) & 1; f < filterLen; f+=2){
      ind = (j-(filterLen-1)+f)>>1;
      if ((ind >= 0) and (ind < dy)) {
	y += rows[ti + ind*K] * lod[filterLen-1-f];
	y += rows[ti + (ind+dy)*K] * hid[filterLen-1-f];
      }
    }
	
    out[i + (j-yOffset)*dx] = y;
  }
}

__global__ void cu_iwt2_col(_data_t *out, _data_t *Lx, _data_t *Hx, int dx, int dy,int dxNext, int dyNext, int xOffset, int yOffset, const scalar_t *lod, const scalar_t *hid, int filterLen)
{
  extern __shared__ _data_t cols [];

  int ti = threadIdx.x;
  int tj = threadIdx.y;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  if (j>=dyNext){
    return;
  }
  int dx2 = 2*dx;
  int i;
  // Load Input to Temp Array
  for (i = ti; i < dx; i += blockDim.x){
    cols[i + tj*dx2] = Lx[i + j*dx];
    cols[dx+i + tj*dx2] = Hx[i + j*dx];
  }
  __syncthreads();

  // Low-Pass and High Pass Downsample
  int ind;
  for (i = ti+xOffset; i < dxNext+xOffset; i += blockDim.x){
    _data_t y = cols[0]-cols[0];
    int f;
#pragma unroll
    for (f = (i-(filterLen-1)) & 1; f < filterLen; f+=2){
      ind = (i-(filterLen-1)+f)>>1;
      if (ind >= 0 and ind < dx) {
	y += cols[ind + tj*dx2] * lod[filterLen-1-f];
	y += cols[dx+ind + tj*dx2] * hid[filterLen-1-f];
      }
    }
    out[(i-xOffset) + j*dxNext] = y;
  }
}

/********** 3D Kernels ************/
__global__ void cu_fwt3_col(_data_t *Lx,_data_t *Hx,_data_t *in,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,const scalar_t *lod,const scalar_t *hid,int filterLen)
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

__global__ void cu_fwt3_row(_data_t *Ly,_data_t *Hy,_data_t *in,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,const scalar_t *lod,const scalar_t *hid,int filterLen)
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

__global__ void cu_fwt3_dep(_data_t *Lz,_data_t *Hz,_data_t *in,int dx,int dy,int dz,int dxNext,int dyNext,int dzNext,const scalar_t *lod,const scalar_t *hid,int filterLen)
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
      // y += deps[ti + ind*K] * lod[filterLen-1-f];
      y += deps[ti + ind*K] * lod[filterLen-1-f];
      z += deps[ti + ind*K] * hid[filterLen-1-f];
    }
    Lz[i + j*dxNext + k*dxNext*dyNext] = y;
    Hz[i + j*dxNext + k*dxNext*dyNext] = z;
  }
}

__global__ void cu_iwt3_dep(_data_t *out, _data_t *Lz, _data_t *Hz, int dx, int dy,int dz,int dxNext, int dyNext, int dzNext,int xOffset, int yOffset,int zOffset,const scalar_t *lod, const scalar_t *hid, int filterLen)
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
    for (int f = (k-(filterLen-1)) & 1; f < filterLen; f+=2){
      ind = (k-(filterLen-1)+f)>>1;
      if ((ind >= 0) and (ind < dz)) {
	y += deps[ti + ind*K] * lod[filterLen-1-f];
	y += deps[ti + (ind+dz)*K] * hid[filterLen-1-f];
      }
    }
	
    out[i + j*dx + (k-zOffset)*dx*dy] = y;
  }
}

__global__ void cu_iwt3_row(_data_t *out, _data_t *Ly, _data_t *Hy, int dx, int dy,int dz,int dxNext, int dyNext,int dzNext,int xOffset, int yOffset, int zOffset,const scalar_t *lod, const scalar_t *hid, int filterLen)
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
    for (int f = (j-(filterLen-1)) & 1; f < filterLen; f+=2){
      ind = (j-(filterLen-1)+f)>>1;
      if ((ind >= 0) and (ind < dy)) {
	y += rows[ti + ind*K] * lod[filterLen-1-f];
	y += rows[ti + (ind+dy)*K] * hid[filterLen-1-f];
      }
    }
	
    out[i + (j-yOffset)*dx + k*dx*dyNext] = y;
  }
}

__global__ void cu_iwt3_col(_data_t *out, _data_t *Lx, _data_t *Hx, int dx, int dy,int dz,int dxNext, int dyNext, int dzNext,int xOffset, int yOffset, int zOffset,const scalar_t *lod, const scalar_t *hid, int filterLen)
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
    for (int f = (i-(filterLen-1)) & 1; f < filterLen; f+=2){
      ind = (i-(filterLen-1)+f)>>1;
      if (ind >= 0 and ind < dx) {
	y += cols[ind + tj*dx2] * lod[filterLen-1-f];
	y += cols[dx+ind + tj*dx2] * hid[filterLen-1-f];
      }
    }
    out[(i-xOffset) + j*dxNext + k*dxNext*dyNext] = y;
  }
}

__global__ void cu_soft_thresh (_data_t* in, scalar_t thresh, int numMax)
{
  int const i = threadIdx.x + blockDim.x*blockIdx.x;
  if (i>numMax)
    return;
  _data_t orig = in[i];
  scalar_t norm = abs(orig);
  scalar_t red = norm - thresh;
  in[i] = (red > 0.) ? ((red / norm) * (orig)) : make_float2(0., 0.);
}


