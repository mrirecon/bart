/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2015-2020. Martin Uecker.
 * Copyright 2017. University of Oxford.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __FLPMATH
#define __FLPMATH

#include "misc/cppwrap.h"

#include <stdbool.h>

#define CFL_SIZE	sizeof(_Complex float)
#define  FL_SIZE	sizeof(float)
#define CDL_SIZE	sizeof(_Complex double)
#define  DL_SIZE	sizeof(double)

#define MD_REAL_DIMS(N, dims)				\
({							\
	int _N = (N);					\
	long* _dims = alloca((_N + 1) * sizeof(long));	\
	md_copy_dims(_N, _dims + 1, dims);		\
	_dims[0] = 2;					\
	_dims;						\
})

#define MD_REAL_STRS(N, strs, size)			\
({							\
	int _N = (N);					\
	long* _strs = alloca((_N + 1) * sizeof(long));	\
	md_copy_dims(_N, _strs + 1, strs);		\
	_strs[0] = (size);				\
	_strs;						\
})


extern void md_mul2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_mul(int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);

extern void md_zrmul2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zrmul(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_zmul2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zmul(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_zdiv2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zdiv(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_zdiv_reg2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2, _Complex float lambda);
extern void md_zdiv_reg(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2, _Complex float lambda);

extern void md_div2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_div(int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);

extern void md_zmulc2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zmulc(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_zsmul2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr, _Complex float val);
extern void md_zsmul(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr, _Complex float val);

extern void md_smul2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr, float val);
extern void md_smul(int D, const long dim[__VLA(D)], float* optr, const float* iptr, float val);

extern void md_zpow2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zpow(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_pow2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_pow(int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);

extern void md_sqrt2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)],  float* optr, const long istr[__VLA(D)], const float* iptr);
extern void md_sqrt(int D, const long dim[__VLA(D)],  float* optr, const float* iptr);

extern void md_zsqrt2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1);
extern void md_zsqrt(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1);

extern void md_zspow2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr, _Complex float val);
extern void md_zspow(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr, _Complex float val);

extern void md_zaxpy2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, _Complex float val, const long istr1[__VLA(D)], const _Complex float* iptr1);
extern void md_zaxpy(int D, const long dim[__VLA(D)], _Complex float* optr, _Complex float val, const _Complex float* iptr1);

extern void md_axpy2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, float val, const long istr1[__VLA(D)], const float* iptr1);
extern void md_axpy(int D, const long dim[__VLA(D)], float* optr, float val, const float* iptr);

extern void md_zfmac2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zfmac(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_tenmul_dims(int D, long max_dims[__VLA(D)], const long out_dims[__VLA(D)], const long in1_dims[__VLA(D)], const long in2_dims[__VLA(D)]);

extern void md_ztenmul2(int D, const long max_dims[__VLA(D)], const long out_strs[__VLA(D)], _Complex float* out, const long in1_strs[__VLA(D)], const _Complex float* in1, const long in2_strs[__VLA(D)], const _Complex float* in2);
extern void md_ztenmul(int D, const long out_dims[__VLA(D)], _Complex float* out, const long in1_dims[__VLA(D)], const _Complex float* in1, const long in2_dims[__VLA(D)], const _Complex float* int2);

extern void md_ztenmulc2(int D, const long max_dims[__VLA(D)], const long out_strs[__VLA(D)], _Complex float* out, const long in1_strs[__VLA(D)], const _Complex float* in1, const long in2_strs[__VLA(D)], const _Complex float* in2);
extern void md_ztenmulc(int D, const long out_dims[__VLA(D)], _Complex float* out, const long in1_dims[__VLA(D)], const _Complex float* in1, const long in2_dims[__VLA(D)], const _Complex float* int2);

extern void md_tenmul2(int D, const long max_dims[__VLA(D)], const long out_strs[__VLA(D)], float* out, const long in1_strs[__VLA(D)], const float* in1, const long in2_strs[__VLA(D)], const float* in2);
extern void md_tenmul(int D, const long out_dims[__VLA(D)], float* out, const long in1_dims[__VLA(D)], const float* in1, const long in2_dims[__VLA(D)], const float* in2);

extern void md_zcorr2(	int N, unsigned long flags,
			const long odims[__VLA(N)], const long ostrs[__VLA(N)], _Complex float* out,
			const long kdims[__VLA(N)], const long kstrs[__VLA(N)], const _Complex float* krn,
			const long idims[__VLA(N)], const long istrs[__VLA(N)], const _Complex float* in);
extern void md_zcorr(	int N, unsigned long flags,
			const long odims[__VLA(N)], _Complex float* out,
			const long kdims[__VLA(N)], const _Complex float* krn,
			const long idims[__VLA(N)], const _Complex float* in);

extern void md_zconv2(	int N, unsigned long flags,
			const long odims[__VLA(N)], const long ostrs[__VLA(N)], _Complex float* out,
			const long kdims[__VLA(N)], const long kstrs[__VLA(N)], const _Complex float* krn,
			const long idims[__VLA(N)], const long istrs[__VLA(N)], const _Complex float* in);
extern void md_zconv(	int N, unsigned long flags,
			const long odims[__VLA(N)], _Complex float* out,
			const long kdims[__VLA(N)], const _Complex float* krn,
			const long idims[__VLA(N)], const _Complex float* in);

extern int calc_convcorr_geom(int N, unsigned long flags,
			long mdims[__VLA(2 * N)], long ostrs2[__VLA(2 * N)], long kstrs2[__VLA(2 * N)], long istrs2[__VLA(2 * N)],
			const long odims[__VLA(N)], const long ostrs[__VLA(N)],
			const long kdims[__VLA(N)], const long kstrs[__VLA(N)],
			const long idims[__VLA(N)], const long istrs[__VLA(N)], bool conv);
extern int calc_convcorr_geom_strs_dil(int N, unsigned long flags,
				       long mdims[__VLA(2 * N)], long ostrs2[__VLA(2 * N)], long kstrs2[__VLA(2 * N)], long istrs2[__VLA(2 * N)],
				       const long odims[__VLA(N)], const long ostrs[__VLA(N)], const long kdims[__VLA(N)], const long kstrs[__VLA(N)], const long idims[__VLA(N)], const long istrs[__VLA(N)],
				       const long dilation[__VLA(N)], const long strides[__VLA(N)], bool conv, bool test_mode);

extern void md_matmul_dims(int D, long max_dims[__VLA(D)], const long out_dims[__VLA(D)], const long mat_dims[__VLA(D)], const long in_dims[__VLA(D)]);

extern void md_zmatmul2(int D, const long out_dims[__VLA(D)], const long out_strs[__VLA(D)], _Complex float* dst, const long mat_dims[__VLA(D)], const long mat_strs[__VLA(D)], const _Complex float* mat, const long in_dims[__VLA(D)], const long in_strs[__VLA(D)], const _Complex float* src);
extern void md_zmatmul(int D, const long out_dims[__VLA(D)], _Complex float* dst, const long mat_dims[__VLA(D)], const _Complex float* mat, const long in_dims[__VLA(D)], const _Complex float* src);

extern void md_zmatmulc2(int D, const long out_dims[__VLA(D)], const long out_strs[__VLA(D)], _Complex float* dst, const long mat_dims[__VLA(D)], const long mat_strs[__VLA(D)], const _Complex float* mat, const long in_dims[__VLA(D)], const long in_strs[__VLA(D)], const _Complex float* src);
extern void md_zmatmulc(int D, const long out_dims[__VLA(D)], _Complex float* dst, const long mat_dims[__VLA(D)], const _Complex float* mat, const long in_dims[__VLA(D)], const _Complex float* src);

extern void md_fmac2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_fmac(int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);

extern void md_zfmacc2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zfmacc(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);
extern void md_zfmaccD(int D, const long dim[__VLA(D)], _Complex double* optr, const _Complex float* iptr1, const _Complex float* iptr2);
extern void md_zfmacD2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex double* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zfmacD(int D, const long dim[__VLA(D)], _Complex double* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_fmacD2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], double* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_fmacD(int D, const long dim[__VLA(D)], double* optr, const float* iptr1, const float* iptr2);

extern void md_zfmaccD2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex double* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zfmaccD(int D, const long dim[__VLA(D)], _Complex double* optr, const _Complex float* iptr1, const _Complex float* iptr2);


extern void md_zadd2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zadd(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_zsadd2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr, _Complex float val);
extern void md_zsadd(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr, _Complex float val);

extern void md_zsub2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zsub(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);

extern void md_add2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_add(int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);

extern void md_sadd2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr, float val);
extern void md_sadd(int D, const long dim[__VLA(D)], float* optr, const float* iptr, float val);

extern void md_sub2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_sub(int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);

extern void md_zphsr(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zphsr2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_abs(int D, const long dim[__VLA(D)], float* optr, const float* iptr);
extern void md_abs2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr);


extern void md_zabs(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zabs2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zatanr(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zatanr2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zmax(int D, const long dims[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);
extern void md_zmax2(int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);

extern void md_max(int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);
extern void md_max2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);

extern void md_min(int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);
extern void md_min2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr);

extern void md_zsoftthresh_half2(int D, const long dim[__VLA(D)], float lambda, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);
extern void md_zsoftthresh_half(int D, const long dim[__VLA(D)], float lambda, _Complex float* optr, const _Complex float* iptr);

extern void md_softthresh_half2(int D, const long dim[__VLA(D)], float lambda, const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr);
extern void md_softthresh_half(int D, const long dim[__VLA(D)], float lambda, float* optr, const float* iptr);

extern void md_softthresh2(int D, const long dim[__VLA(D)], float lambda, unsigned long flags, const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr);
extern void md_softthresh(int D, const long dim[__VLA(D)], float lambda, unsigned long flags, float* optr, const float* iptr);


extern void md_softthresh_core2(int D, const long dims[__VLA(D)], float lambda, unsigned long flags, float* tmp_norm, const long ostrs[__VLA(D)], float* optr, const long istrs[__VLA(D)], const float* iptr);

extern void md_zsoftthresh_core2(int D, const long dims[__VLA(D)], float lambda, unsigned long flags, _Complex float* tmp_norm, const long ostrs[__VLA(D)], _Complex float* optr, const long istrs[__VLA(D)], const _Complex float* iptr);

extern void md_zsoftthresh2(int D, const long dim[__VLA(D)], float lambda, unsigned long flags, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);
extern void md_zsoftthresh(int D, const long dim[__VLA(D)], float lambda, unsigned long flags, _Complex float* optr, const _Complex float* iptr);

void md_zhardthresh_mask2(int D, const long dim[__VLA(D)], int k, unsigned long flags, _Complex float* tmp_norm, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zhardthresh_mask(int D, const long dim[__VLA(D)], int k, unsigned long flags, _Complex float* optr, const _Complex float* iptr);

extern void md_zhardthresh_joint2(int D, const long dims[__VLA(D)], int k, unsigned long flags, _Complex float* tmp_norm, const long ostrs[__VLA(D)], _Complex float* optr, const long istrs[__VLA(D)], const _Complex float* iptr);

extern void md_zhardthresh2(int D, const long dims[__VLA(D)], int k, unsigned long flags, const long ostrs[__VLA(D)], _Complex float* optr, const long istrs[__VLA(D)], const _Complex float* iptr);
extern void md_zhardthresh(int D, const long dims[__VLA(D)], int k, unsigned long flags, _Complex float* optr, const _Complex float* iptr);

extern void md_zconj(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zconj2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zreal(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zreal2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zimag(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zimag2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zcmp(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);
extern void md_zcmp2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);

extern void md_zexpj(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zexpj2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zexp(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zexp2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_exp(int D, const long dim[__VLA(D)], float* optr, const float* iptr);
extern void md_exp2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr);

extern void md_zlog(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zlog2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_log(int D, const long dim[__VLA(D)], float* optr, const float* iptr);
extern void md_log2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr);

extern void md_zarg(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zarg2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zsin(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zsin2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zcos(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zcos2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zacos(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zacos2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zsinh(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zsinh2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zcosh(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr);
extern void md_zcosh2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zlessequal(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);
extern void md_zlessequal2(int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zslessequal(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, float val);
extern void md_zslessequal2(int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, float val);


extern void md_lessequal(int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);
extern void md_lessequal2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_slessequal(int D, const long dim[__VLA(D)], float* optr, const float* iptr, float val);
extern void md_slessequal2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr, float val);

extern void md_zgreatequal(int D, const long dims[__VLA(D)], _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);
extern void md_zgreatequal2(int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);
extern void md_zsgreatequal(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr, float val);
extern void md_zsgreatequal2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr, float val);

extern void md_greatequal(int D, const long dim[__VLA(D)], float* optr, const float* iptr1, const float* iptr2);
extern void md_greatequal2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr1[__VLA(D)], const float* iptr1, const long istr2[__VLA(D)], const float* iptr2);
extern void md_sgreatequal(int D, const long dim[__VLA(D)], float* optr, const float* iptr, float val);
extern void md_sgreatequal2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr, float val);

extern float md_znorm2(int D, const long dim[__VLA(D)], const long str[__VLA(D)], const _Complex float* ptr);
extern float md_znorm(int D, const long dim[__VLA(D)], const _Complex float* ptr);
extern _Complex float md_zscalar2(int D, const long dim[__VLA(D)], const long str1[__VLA(D)], const _Complex float* ptr1, const long str2[__VLA(D)], const _Complex float* ptr2);
extern _Complex float md_zscalar(int D, const long dim[__VLA(D)], const _Complex float* ptr1, const _Complex float* ptr2);
extern float md_zscalar_real2(int D, const long dim[__VLA(D)], const long str1[__VLA(D)], const _Complex float* ptr1, const long str2[__VLA(D)], const _Complex float* ptr2);
extern float md_zscalar_real(int D, const long dim[__VLA(D)], const _Complex float* ptr1, const _Complex float* ptr2);


extern float md_asum2(int D, const long dim[__VLA(D)], const long str[__VLA(D)], const float* ptr);
extern float md_asum(int D, const long dim[__VLA(D)], const float* ptr);

extern float md_zasum2(int D, const long dim[__VLA(D)], const long str[__VLA(D)], const _Complex float* ptr);
extern float md_zasum(int D, const long dim[__VLA(D)], const _Complex float* ptr);

extern float md_z1norm2(int D, const long dim[__VLA(D)], const long str[__VLA(D)], const _Complex float* ptr);
extern float md_z1norm(int D, const long dim[__VLA(D)], const _Complex float* ptr);

extern float md_asum2(int D, const long dim[__VLA(D)], const long str[__VLA(D)], const float* ptr);
extern float md_asum(int D, const long dim[__VLA(D)], const float* ptr);

extern float md_zasum2(int D, const long dim[__VLA(D)], const long str[__VLA(D)], const _Complex float* ptr);
extern float md_zasum(int D, const long dim[__VLA(D)], const _Complex float* ptr);

extern float md_z1norm2(int D, const long dim[__VLA(D)], const long str[__VLA(D)], const _Complex float* ptr);
extern float md_z1norm(int D, const long dim[__VLA(D)], const _Complex float* ptr);

extern float md_norm2(int D, const long dim[__VLA(D)], const long str[__VLA(D)], const float* ptr);
extern float md_norm(int D, const long dim[__VLA(D)], const float* ptr);
extern float md_scalar2(int D, const long dim[__VLA(D)], const long str1[__VLA(D)], const float* ptr1, const long str2[__VLA(D)], const float* ptr2);
extern float md_scalar(int D, const long dim[__VLA(D)], const float* ptr1, const float* ptr2);

extern void md_rss(int D, const long dims[__VLA(D)], unsigned long flags, float* dst, const float* src);
extern void md_zrss(int D, const long dims[__VLA(D)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void md_zss(int D, const long dims[__VLA(D)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void md_zss2(int D, const long dims[__VLA(D)], unsigned long flags, const long str2[__VLA(D)], _Complex float* dst, const long str1[__VLA(D)], const _Complex float* src);

extern void md_zstd(int D, const long dims[__VLA(D)], unsigned long flags, _Complex float* optr, const _Complex float* iptr);
extern void md_zstd2(int D, const long dims[__VLA(D)], unsigned long flags, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zvar(int D, const long dims[__VLA(D)], unsigned long flags, _Complex float* optr, const _Complex float* iptr);
extern void md_zvar2(int D, const long dims[__VLA(D)], unsigned long flags, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zcovar(int D, const long dims[__VLA(D)], unsigned long flags, _Complex float* optr, const _Complex float* iptr1, const _Complex float* iptr2);
extern void md_zcovar2(int D, const long dims[__VLA(D)], unsigned long flags, const long ostr[__VLA(D)], _Complex float* optr, const long istr1[__VLA(D)], const _Complex float* iptr1, const long istr2[__VLA(D)], const _Complex float* iptr2);

extern void md_zavg(int D, const long dims[__VLA(D)], unsigned long flags, _Complex float* optr, const _Complex float* iptr);
extern void md_zavg2(int D, const long dims[__VLA(D)], unsigned long flags, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zwavg(int D, const long dims[__VLA(D)], unsigned long flags, _Complex float* optr, const _Complex float* iptr);
extern void md_zwavg2(int D, const long dims[__VLA(D)], unsigned long flags, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);
extern void md_zwavg2_core1(int D, const long dims[__VLA(D)], unsigned long flags, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* weights);
extern void md_zwavg2_core2(int D, const long dims[__VLA(D)], unsigned long flags, const long ostr[__VLA(D)], _Complex float* optr, const _Complex float* weights, const long istr[__VLA(D)], const _Complex float* iptr);

extern float md_zrms(int D, const long dim[__VLA(D)], const _Complex float* in);
extern float md_zrmse(int D, const long dim[__VLA(D)], const _Complex float* in1, const _Complex float* in2);
extern float md_znrmse(int D, const long dim[__VLA(D)], const _Complex float* ref, const _Complex float* in);
extern float md_znorme(int D, const long dim[__VLA(D)], const _Complex float* in1, const _Complex float* in2);
extern float md_zrnorme(int D, const long dim[__VLA(D)], const _Complex float* ref, const _Complex float* in);

extern float md_rms(int D, const long dim[__VLA(D)], const float* in);
extern float md_rmse(int D, const long dim[__VLA(D)], const float* in1, const float* in2);
extern float md_nrmse(int D, const long dim[__VLA(D)], const float* ref, const float* in);

extern void md_zdouble2float(int D, const long dims[__VLA(D)], _Complex float* dst, const _Complex double* src);
extern void md_zfloat2double(int D, const long dims[__VLA(D)], _Complex double* dst, const _Complex float* src);
extern void md_float2double(int D, const long dims[__VLA(D)], double* dst, const float* src);
extern void md_double2float(int D, const long dims[__VLA(D)], float* dst, const double* src);
extern void md_zdouble2float2(int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], _Complex float* dst, const long istr[__VLA(D)], const _Complex double* src);
extern void md_zfloat2double2(int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], _Complex double* dst, const long istr[__VLA(D)], const _Complex float* src);
extern void md_float2double2(int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], double* dst, const long istr[__VLA(D)], const float* src);
extern void md_double2float2(int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], float* dst, const long istr[__VLA(D)], const double* src);

extern void md_zfill2(int D, const long dim[__VLA(D)], const long str[__VLA(D)], _Complex float* ptr, _Complex float val);
extern void md_zfill(int D, const long dim[__VLA(D)], _Complex float* ptr, _Complex float val);

extern void md_zsmax2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr, float val);
extern void md_zsmax(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr, float val);
extern void md_zsmin2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr, float val);
extern void md_zsmin(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr, float val);

extern void md_smin2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr, float val);
extern void md_smin(int D, const long dim[__VLA(D)], float* optr, const float* iptr, float val);
extern void md_smax2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr, float val);
extern void md_smax(int D, const long dim[__VLA(D)], float* optr, const float* iptr, float val);

extern void md_fdiff2(int D, const long dims[__VLA(D)], int d, const long ostr[__VLA(D)], float* out, const long istr[__VLA(D)], const float* in);
extern void md_fdiff(int D, const long dims[__VLA(D)], int d, float* out, const float* in);
extern void md_fdiff_backwards2(int D, const long dims[__VLA(D)], int d, const long ostr[__VLA(D)], float* out, const long istr[__VLA(D)], const float* in);
extern void md_fdiff_backwards(int D, const long dims[__VLA(D)], int d, float* out, const float* in);

extern void md_zfdiff2(int D, const long dims[__VLA(D)], int d, const long ostr[__VLA(D)], _Complex float* out, const long istr[__VLA(D)], const _Complex float* in);
extern void md_zfdiff(int D, const long dims[__VLA(D)], int d, _Complex float* out, const _Complex float* in);
extern void md_zfdiff_backwards2(int D, const long dims[__VLA(D)], int d, const long ostr[__VLA(D)], _Complex float* out, const long istr[__VLA(D)], const _Complex float* in);
extern void md_zfdiff_backwards(int D, const long dims[__VLA(D)], int d, _Complex float* out, const _Complex float* in);


extern void md_zfftmod(int D, const long dim[__VLA(D)], _Complex float* optr, const _Complex float* iptr, bool inv, double phase);
extern void md_zfftmod2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr, bool inv, double phase);

extern void md_zsum(int D, const long dims[__VLA(D)], unsigned long flags, _Complex float* dst, const _Complex float* src);

extern void md_imag2(int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], float* dst, const long istr[__VLA(D)], const _Complex float* src);
extern void md_imag(int D, const long dims[__VLA(D)], float* dst, const _Complex float* src);
extern void md_real2(int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], float* dst, const long istr[__VLA(D)], const _Complex float* src);
extern void md_real(int D, const long dims[__VLA(D)], float* dst, const _Complex float* src);
extern void md_zcmpl_real2(int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], _Complex float* dst, const long istr[__VLA(D)], const float* src);
extern void md_zcmpl_real(int D, const long dims[__VLA(D)], _Complex float* dst, const float* src);
extern void md_zcmpl_imag2(int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], _Complex float* dst, const long istr[__VLA(D)], const float* src);
extern void md_zcmpl_imag(int D, const long dims[__VLA(D)], _Complex float* dst, const float* src);
extern void md_zcmpl2(int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], _Complex float* dst, const long istr1[__VLA(D)], const float* src_real, const long istr2[__VLA(D)], const float* src_imag);
extern void md_zcmpl(int D, const long dims[__VLA(D)], _Complex float* dst, const float* src_real, const float* src_imag);

extern void md_pdf_gauss2(int D, const long dims[__VLA(D)], const long ostr[__VLA(D)], float* optr, const long istr[__VLA(D)], const float* iptr, float mu, float sigma);
extern void md_pdf_gauss(int D, const long dims[__VLA(D)], float* optr, const float* iptr, float mu, float sigma);

extern float md_zmaxnorm2(int D, const long dims[__VLA(D)], const long strs[__VLA(D)], const _Complex float* ptr);
extern float md_zmaxnorm(int D, const long dims[__VLA(D)], const _Complex float* ptr);

#include "misc/cppwrap.h"

#endif

