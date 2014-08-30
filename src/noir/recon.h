/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "misc/mri.h"

extern void noir_recon(const long dims[DIMS], unsigned int iter, float l1, complex float* image, complex float* sens, const complex float* pattern, const complex float* mask, const complex float* kspace_data, bool usegpu);

#ifdef __cplusplus
}
#endif


