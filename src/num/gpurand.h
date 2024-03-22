#ifndef GPURAND_H
#define GPURAND_H



#include "misc/cppwrap.h"


extern void cuda_gaussian_rand(long N, _Complex float* dst, uint64_t state, uint64_t ctr1);
extern void cuda_uniform_rand(long N, _Complex float* dst, uint64_t state, uint64_t ctr1);
extern void cuda_rand_one(long N, _Complex float* dst, double p, uint64_t state, uint64_t ctr1);


#include "misc/cppwrap.h"

#endif // GPURAND_H
