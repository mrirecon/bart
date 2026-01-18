#ifndef _RAND_H
#define _RAND_H

#include "misc/dllspec.h"
#include "misc/cppwrap.h"


extern double uniform_rand(void);
extern _Complex double gaussian_rand(void);
extern void md_gaussian_rand(int D, const long dims[__VLA(D)], _Complex float* dst);
extern void md_zgaussian_rand(int D, const long dims[__VLA(D)], _Complex float* dst);
extern void md_uniform_rand(int D, const long dims[__VLA(D)], _Complex float* dst);
extern void md_rand_one(int D, const long dims[__VLA(D)], _Complex float* dst, double p);

extern void gaussian_rand_vec(long N, float* dst);
extern void uniform_rand_vec(long N, float* dst);

BARTLIB_API extern void BARTLIB_CALL num_rand_init(unsigned long long seed);

struct bart_rand_state;

extern struct bart_rand_state* rand_state_create(unsigned long long seed);
extern void rand_state_update(struct bart_rand_state* state, unsigned long long seed);

extern unsigned int rand_range(unsigned int range);
extern unsigned int rand_range_state(struct bart_rand_state* state, unsigned int range);

extern unsigned long long rand_ull_state(struct bart_rand_state* state);

#include "misc/cppwrap.h"
#endif // _RAND_H
 
