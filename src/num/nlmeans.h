
#include <complex.h>

extern void md_znlmeans2(int D, const long dims[D], unsigned long flags,
		const long ostrs[D], complex float* optr,
		const long istrs[D], const complex float* iptr,
		long patch_size, long patch_dist, float h, float a);

extern void md_znlmeans(int D, const long dims[D], unsigned long flags,
		complex float* optr, const complex float* iptr,
		long patch_size, long patch_dist, float h, float a);

extern void md_znlmeans_distance2(int D, const long idims[D], int xD,
		const long odims[xD], unsigned long flags,
		const long ostrs[xD], complex float* optr,
		const long istrs[D], const complex float* iptr);

extern void md_znlmeans_distance(int D, const long idims[D], int xD,
		const long odims[xD], unsigned long flags,
		complex float* optr, const complex float* iptr);

