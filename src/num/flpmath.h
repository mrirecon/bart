/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2015-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef __FLPMATH
#define __FLPMATH

#include "misc/cppwrap.h"


#define CFL_SIZE	sizeof(_Complex float)
#define  FL_SIZE	sizeof(float)
#define CDL_SIZE	sizeof(_Complex double)
#define  DL_SIZE	sizeof(double)


extern void md_mul2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_mul(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);

extern void md_zrmul2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zrmul(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_zmul2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zmul(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_zdiv2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zdiv(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_zdiv_reg2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2, _Complex float lambda);
extern void md_zdiv_reg(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2, _Complex float lambda);

extern void md_div2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_div(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);

extern void md_zmulc2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zmulc(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_zsmul2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr, _Complex float val);
extern void md_zsmul(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr, _Complex float val);

extern void md_smul2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr, float val);
extern void md_smul(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr, float val);

extern void md_zpow2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zpow(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_pow2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_pow(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);

extern void md_sqrt2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)],  float* optr, const long istr[__VLA(D)], const float* iptr);
extern void md_sqrt(unsigned int D, const long dim[__VLA(D)],  float* optr, const float* iptr);

extern void md_zsqrt2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1);
extern void md_zsqrt(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1);

extern void md_zspow2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr, _Complex float val);
extern void md_zspow(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr, _Complex float val);

extern void md_zaxpy2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, _Complex float val, const long istr1[__VLA(D)], const _Complex float* iptr1);
extern void md_zaxpy(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, _Complex float val, const _Complex float* iptr1);

extern void md_axpy2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, float val, const long istr1[__VLA(D)], const float* iptr1);
extern void md_axpy(unsigned int D, const long dim[__VLA(D)], float* optr, float val, const float* iptr);

extern void md_zfmac2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zfmac(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_tenmul_dims(unsigned int D, long max_dims[__VLA(D)], const long out_dims[__VLA(D)], const long in1_dims[__VLA(D)], const long in2_dims[__VLA(D)]);

extern void md_ztenmul2(unsigned int D, const long max_dims[__VLA(D)], const long out_strs[__VLA(D)], _Complex float* out, const long in1_strs[__VLA(D)], const _Complex float* in1, const long in2_strs[__VLA(D)], const _Complex float* in2);
extern void md_ztenmul(unsigned int D, const long out_dims[__VLA(D)], _Complex float* out, const long in1_dims[__VLA(D)], const _Complex float* in1, const long in2_dims[__VLA(D)], const _Complex float* int2);

extern void md_ztenmulc2(unsigned int D, const long max_dims[__VLA(D)], const long out_strs[__VLA(D)], _Complex float* out, const long in1_strs[__VLA(D)], const _Complex float* in1, const long in2_strs[__VLA(D)], const _Complex float* in2);
extern void md_ztenmulc(unsigned int D, const long out_dims[__VLA(D)], _Complex float* out, const long in1_dims[__VLA(D)], const _Complex float* in1, const long in2_dims[__VLA(D)], const _Complex float* int2);


extern void md_matmul_dims(unsigned int D, long max_dims[__VLA(D)], const long out_dims[__VLA(D)], const long mat_dims[__VLA(D)], const long in_dims[__VLA(D)]);

extern void md_zmatmul2(unsigned int D, const long out_dims[__VLA(D)], const long out_strs[__VLA(D)], _Complex float* dst, const long mat_dims[__VLA(D)], const long mat_strs[__VLA(D)], const _Complex float* mat, const long in_dims[__VLA(D)], const long in_strs[__VLA(D)], const _Complex float* src);
extern void md_zmatmul(unsigned int D, const long out_dims[__VLA(D)], _Complex float* dst, const long mat_dims[__VLA(D)], const _Complex float* mat, const long in_dims[__VLA(D)], const _Complex float* src);

extern void md_zmatmulc2(unsigned int D, const long out_dims[__VLA(D)], const long out_strs[__VLA(D)], _Complex float* dst, const long mat_dims[__VLA(D)], const long mat_strs[__VLA(D)], const _Complex float* mat, const long in_dims[__VLA(D)], const long in_strs[__VLA(D)], const _Complex float* src);
extern void md_zmatmulc(unsigned int D, const long out_dims[__VLA(D)], _Complex float* dst, const long mat_dims[__VLA(D)], const _Complex float* mat, const long in_dims[__VLA(D)], const _Complex float* src);

extern void md_fmac2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_fmac(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);

extern void md_zfmacc2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zfmacc(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);
extern void md_zfmaccD(unsigned int D, const long dim[__VLA(D)], _Complex double* optr, const _Complex float* iptr1, const _Complex float* iptr2);
extern void md_zfmacD2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex double* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zfmacD(unsigned int D, const long dim[__VLA(D)], _Complex double* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_fmacD2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], double* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_fmacD(unsigned int D, const long dim[__VLA(D)], double* optr, const float* iptr1, const float* iptr2);

extern void md_zfmaccD2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex double* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zfmaccD(unsigned int D, const long dim[__VLA(D)], _Complex double* optr, const _Complex float* iptr1, const _Complex float* iptr2);


extern void md_zadd2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zadd(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_zsadd2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr, _Complex float val);
extern void md_zsadd(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr, _Complex float val);

extern void md_zsub2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zsub(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_add2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_add(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);

extern void md_sadd2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr, float val);
extern void md_sadd(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr, float val);

extern void md_sub2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_sub(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);

extern void md_zphsr(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zphsr2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_abs(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr);
extern void md_abs2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr);


extern void md_zabs(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zabs2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);


extern void md_max(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);
extern void md_max2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);

extern void md_min(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);
extern void md_min2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr);

extern void md_zsoftthresh_half2(unsigned int D, const long dim[__VLA(D)], float lambda, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);
extern void md_zsoftthresh_half(unsigned int D, const long dim[__VLA(D)], float lambda, _Complex float* optr, const _Complex float* iptr);

extern void md_softthresh_half2(unsigned int D, const long dim[__VLA(D)], float lambda, const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr);
extern void md_softthresh_half(unsigned int D, const long dim[__VLA(D)], float lambda, float* optr, const float* iptr);

extern void md_softthresh2(unsigned int D, const long dim[__VLA(D)], float lambda, unsigned int flags, const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr);
extern void md_softthresh(unsigned int D, const long dim[__VLA(D)], float lambda, unsigned int flags, float* optr, const float* iptr);


extern void md_softthresh_core2(unsigned int D, const long dims[__VLA(D)], float lambda, unsigned int flags, float* tmp_norm, const long ostrs[__VLA(D)], float* optr, const long istrs[__VLA(D)], const float* iptr);

extern void md_zsoftthresh_core2(unsigned int D, const long dims[__VLA(D)], float lambda, unsigned int flags, _Complex float* tmp_norm, const long ostrs[__VLA(D)], _Complex float* optr, const long istrs[__VLA(D)], const _Complex float* iptr);

extern void md_zsoftthresh2(unsigned int D, const long dim[__VLA(D)], float lambda, unsigned int flags, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);
extern void md_zsoftthresh(unsigned int D, const long dim[__VLA(D)], float lambda, unsigned int flags, _Complex float* optr, const _Complex float* iptr);




extern void md_zconj(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zconj2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zreal(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zreal2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zimag(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zimag2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zcmp(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);
extern void md_zcmp2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);

extern void md_zexpj(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zexpj2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zarg(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zarg2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);


extern void md_lessequal(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);
extern void md_lessequal2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_slessequal(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr, float val);
extern void md_slessequal2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr, float val);

extern void md_greatequal(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);
extern void md_greatequal2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_sgreatequal(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr, float val);
extern void md_sgreatequal2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr, float val);

extern float md_znorm2(unsigned int D, const long dim[__VLA(D)], const long str[__VLA(D)], const _Complex float* ptr);
extern float md_znorm(unsigned int D, const long dim[__VLA(D)], const _Complex float* ptr);
extern _Complex float md_zscalar2(unsigned int D, const long dim[__VLA(D)], const long str1[__VLA(D)], const _Complex float* ptr1, const long str2[__VLA(D)], const _Complex float* ptr2);
extern _Complex float md_zscalar(unsigned int D, const long dim[__VLA(D)], const _Complex float* ptr1, const _Complex float* ptr2);
extern float md_zscalar_real2(unsigned int D, const long dim[__VLA(D)], const long str1[__VLA(D)], const _Complex float* ptr1, const long str2[__VLA(D)], const _Complex float* ptr2);
extern float md_zscalar_real(unsigned int D, const long dim[__VLA(D)], const _Complex float* ptr1, const _Complex float* ptr2);


extern float md_asum2(unsigned int D, const long dim[__VLA(D)], const long str[__VLA(D)], const float* ptr);
extern float md_asum(unsigned int D, const long dim[__VLA(D)], const float* ptr);

extern float md_zasum2(unsigned int D, const long dim[__VLA(D)], const long str[__VLA(D)], const _Complex float* ptr);
extern float md_zasum(unsigned int D, const long dim[__VLA(D)], const _Complex float* ptr);

extern float md_z1norm2(unsigned int D, const long dim[__VLA(D)], const long str[__VLA(D)], const _Complex float* ptr);
extern float md_z1norm(unsigned int D, const long dim[__VLA(D)], const _Complex float* ptr);

extern float md_asum2(unsigned int D, const long dim[__VLA(D)], const long str[__VLA(D)], const float* ptr);
extern float md_asum(unsigned int D, const long dim[__VLA(D)], const float* ptr);

extern float md_zasum2(unsigned int D, const long dim[__VLA(D)], const long str[__VLA(D)], const _Complex float* ptr);
extern float md_zasum(unsigned int D, const long dim[__VLA(D)], const _Complex float* ptr);

extern float md_z1norm2(unsigned int D, const long dim[__VLA(D)], const long str[__VLA(D)], const _Complex float* ptr);
extern float md_z1norm(unsigned int D, const long dim[__VLA(D)], const _Complex float* ptr);

extern float md_norm2(unsigned int D, const long dim[__VLA(D)], const long str[__VLA(D)], const float* ptr);
extern float md_norm(unsigned int D, const long dim[__VLA(D)], const float* ptr);
extern float md_scalar2(unsigned int D, const long dim[__VLA(D)], const long str1[__VLA(D)], const float* ptr1, const long str2[__VLA(D)], const float* ptr2);
extern float md_scalar(unsigned int D, const long dim[__VLA(D)], const float* ptr1, const float* ptr2);

extern void md_rss(unsigned int D, const long dims[__VLA(D)], unsigned int flags, float* dst, const float* src);
extern void md_zrss(unsigned int D, const long dims[__VLA(D)], unsigned int flags, _Complex float* dst, const _Complex float* src);

extern void md_zavg(unsigned int D, const long dims[__VLA(D)], unsigned int flags, _Complex float* optr, const _Complex float* iptr);
extern void md_zavg2(unsigned int D, const long dims[__VLA(D)], unsigned int flags, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zwavg(unsigned int D, const long dims[__VLA(D)], unsigned int flags, _Complex float* optr, const _Complex float* iptr);
extern void md_zwavg2(unsigned int D, const long dims[__VLA(D)], unsigned int flags, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);
extern void md_zwavg2_core1(unsigned int D, const long dims[__VLA(D)], unsigned int flags, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* weights);
extern void md_zwavg2_core2(unsigned int D, const long dims[__VLA(D)], unsigned int flags, const long ostr[__VLA(D)], _Complex float* optr, const _Complex float* weights, const long istr[__VLA(D)], const _Complex float* iptr);

extern float md_zrms(unsigned int D, const long dim[__VLA(D)], const _Complex float* in);
extern float md_zrmse(unsigned int D, const long dim[__VLA(D)], const _Complex float* in1, const _Complex float* in2);
extern float md_znrmse(unsigned int D, const long dim[__VLA(D)], const _Complex float* ref, const _Complex float* in);
extern float md_znorme(unsigned int D, const long dim[__VLA(D)], const _Complex float* in1, const _Complex float* in2);
extern float md_zrnorme(unsigned int D, const long dim[__VLA(D)], const _Complex float* ref, const _Complex float* in);

extern void md_zdouble2float(unsigned int D, const long dims[__VLA(D)], _Complex float* dst, const _Complex double* src);
extern void md_zfloat2double(unsigned int D, const long dims[__VLA(D)], _Complex double* dst, const _Complex float* src);
extern void md_float2double(unsigned int D, const long dims[__VLA(D)], double* dst, const float* src);
extern void md_double2float(unsigned int D, const long dims[__VLA(D)], float* dst, const double* src);
extern void md_zdouble2float2(unsigned int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], _Complex float* dst, const long istr[__VLA(D)], const _Complex double* src);
extern void md_zfloat2double2(unsigned int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], _Complex double* dst, const long istr[__VLA(D)], const _Complex float* src);
extern void md_float2double2(unsigned int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], double* dst, const long istr[__VLA(D)], const float* src);
extern void md_double2float2(unsigned int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], float* dst, const long istr[__VLA(D)], const double* src);

extern void md_zfill2(unsigned int D, const long dim[__VLA(D)], const long str[__VLA(D)], _Complex float* ptr, _Complex float val);
extern void md_zfill(unsigned int D, const long dim[__VLA(D)], _Complex float* ptr, _Complex float val);


extern void md_smin2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr, float val);
extern void md_smin(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr, float val);
extern void md_smax2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr, float val);
extern void md_smax(unsigned int D, const long dim[__VLA(D)], float* optr, const float* iptr, float val);

extern void md_fdiff2(unsigned int D, const long dims[__VLA(D)], unsigned int d, const long ostr[__VLA(D)], float* out, const long istr[__VLA(D)], const float* in);
extern void md_fdiff(unsigned int D, const long dims[__VLA(D)], unsigned int d, float* out, const float* in);
extern void md_fdiff_backwards2(unsigned int D, const long dims[__VLA(D)], unsigned int d, const long ostr[__VLA(D)], float* out, const long istr[__VLA(D)], const float* in);
extern void md_fdiff_backwards(unsigned int D, const long dims[__VLA(D)], unsigned int d, float* out, const float* in);

extern void md_zfdiff2(unsigned int D, const long dims[__VLA(D)], unsigned int d, const long ostr[__VLA(D)], _Complex float* out, const long istr[__VLA(D)], const _Complex float* in);
extern void md_zfdiff(unsigned int D, const long dims[__VLA(D)], unsigned int d, _Complex float* out, const _Complex float* in);
extern void md_zfdiff_backwards2(unsigned int D, const long dims[__VLA(D)], unsigned int d, const long ostr[__VLA(D)], _Complex float* out, const long istr[__VLA(D)], const _Complex float* in);
extern void md_zfdiff_backwards(unsigned int D, const long dims[__VLA(D)], unsigned int d, _Complex float* out, const _Complex float* in);


extern void md_zfftmod(unsigned int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr, _Bool inv, double phase);
extern void md_zfftmod2(unsigned int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr, _Bool inv, double phase);


#include "misc/cppwrap.h"

#endif


