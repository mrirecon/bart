/* Copyright 2013-2018. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_add_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2);
extern void cuda_zadd_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2);

extern void cuda_mul_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2);
extern void cuda_zmul_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2);
extern void cuda_zmulc_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2);

extern void cuda_fmac_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2);
extern void cuda_zfmac_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2);
extern void cuda_zfmacc_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2);


#ifdef __cplusplus
}
#endif
