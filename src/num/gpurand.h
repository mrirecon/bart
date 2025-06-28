
#ifndef _GPURAND_H
#define _GPURAND_H

#include "misc/cppwrap.h"


extern void cuda_gaussian_rand(long N, _Complex float* dst, uint64_t state, uint64_t ctr1, uint64_t offset);
extern void cuda_uniform_rand(long N, _Complex float* dst, uint64_t state, uint64_t ctr1, uint64_t offset);
extern void cuda_rand_one(long N, _Complex float* dst, double p, uint64_t state, uint64_t ctr1, uint64_t offset);


#include "misc/cppwrap.h"

#endif // _GPURAND_H
  
