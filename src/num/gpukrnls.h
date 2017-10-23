/* Copyright 2013-2017. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_float2double(long size, double* dst, const float* src);
extern void cuda_double2float(long size, float* dst, const double* src);
extern void cuda_sxpay(long size, float* y, float alpha, const float* src);
extern void cuda_xpay(long N, float beta, float* dst, const float* src);
extern void cuda_axpbz(long N, float* dst, const float a, const float* x, const float b, const float* z);
extern void cuda_smul(long N, float alpha, float* dst, const float* src);
extern void cuda_mul(long N, float* dst, const float* src1, const float* src2);
extern void cuda_div(long N, float* dst, const float* src1, const float* src2);
extern void cuda_add(long N, float* dst, const float* src1, const float* src2);
extern void cuda_sub(long N, float* dst, const float* src1, const float* src2);
extern void cuda_fmac(long N, float* dst, const float* src1, const float* src2);
extern void cuda_fmac2(long N, double* dst, const float* src1, const float* src2);
extern void cuda_zmul(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zdiv(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zfmac(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zfmac2(long N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zmulc(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zfmacc(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zfmacc2(long N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_pow(long N, float* dst, const float* src1, const float* src2);
extern void cuda_zpow(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_sqrt(long N, float* dst, const float* src);
extern void cuda_zconj(long N, _Complex float* dst, const _Complex float* src);
extern void cuda_zphsr(long N, _Complex float* dst, const _Complex float* src);
extern void cuda_zexpj(long N, _Complex float* dst, const _Complex float* src);
extern void cuda_zarg(long N, _Complex float* dst, const _Complex float* src);
extern void cuda_zsoftthresh_half(long N, float lambda, _Complex float* d, const _Complex float* x);
extern void cuda_zsoftthresh(long N, float lambda, _Complex float* d, const _Complex float* x);
extern void cuda_softthresh_half(long N, float lambda, float* d, const float* x);
extern void cuda_softthresh(long N, float lambda, float* d, const float* x);
extern void cuda_zreal(long N, _Complex float* dst, const _Complex float* src);
extern void cuda_zcmp(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zdiv_reg(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2, _Complex float lambda);
extern void cuda_le(long N, float* dst, const float* src1, const float* src2);
extern void cuda_ge(long N, float* dst, const float* src1, const float* src2);
extern void cuda_zfftmod(long N, _Complex float* dst, const _Complex float* src, unsigned int n, _Bool inv, double phase);
extern void cuda_max(long N, float* dst, const float* src1, const float* src2);
extern void cuda_min(long N, float* dst, const float* src1, const float* src2);

#ifdef __cplusplus
}
#endif
