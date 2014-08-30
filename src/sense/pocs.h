/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 

#include "misc/mri.h"
#include "iter/iter2.h"


#ifdef __cplusplus
extern "C" {
#endif

struct operator_p_s;
struct linop_s;

extern void pocs_recon(const long dims[DIMS], const struct operator_p_s* thresh_data, int maxiter, float alpha, float lambda, _Complex float* result, const _Complex float* maps, const _Complex float* pattern, const _Complex float* kspace);
extern void pocs_recon2(italgo_fun2_t italgo, void* iconf, const struct linop_s** ops, const long dims[DIMS], const struct operator_p_s* thresh_data, float alpha, float lambda, _Complex float* result, const _Complex float* maps, const _Complex float* pattern, const _Complex float* kspace);

#ifdef USE_CUDA
extern void pocs_recon_gpu(const long dims[DIMS], const struct operator_p_s* thresh, int maxiter, float alpha, float lambda, _Complex float* result, const _Complex float* maps, const _Complex float* pattern, const _Complex float* kspace);
extern void pocs_recon_gpu2(italgo_fun2_t italgo, void* iconf, const struct linop_s** ops, const long dims[DIMS], const struct operator_p_s* thresh, float alpha, float lambda, _Complex float* result, const _Complex float* maps, const _Complex float* pattern, const _Complex float* kspace);
#endif


#ifdef __cplusplus
}
#endif

