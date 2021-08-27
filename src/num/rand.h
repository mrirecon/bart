
#include "misc/cppwrap.h"

extern double uniform_rand(void);
extern _Complex double gaussian_rand(void);
extern void md_gaussian_rand(unsigned int D, const long dims[__VLA(D)], _Complex float* dst);
extern void md_uniform_rand(unsigned int D, const long dims[__VLA(D)], _Complex float* dst);
extern void md_rand_one(unsigned int D, const long dims[__VLA(D)], _Complex float* dst, double p);

extern void num_rand_init(unsigned int seed);

#include "misc/cppwrap.h"
