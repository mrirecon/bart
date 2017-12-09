/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012 Martin Uecker uecker@eecs.berkeley.edu
 */

#include <assert.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/conv.h"
#include "num/vecops.h"

#include "convoaa.h"



void overlapandadd(int N, const long dims[N], const long blk[N], complex float* dst, complex float* src1, const long dim2[N], complex float* src2)
{
	long ndims[2 * N];
	long L[2 * N];
	long ndim2[2 * N];
	long ndim3[2 * N];

	for (int i = 0; i < N; i++) {

		assert(0 == dims[i] % blk[i]);
		assert(dim2[i] <= blk[i]);

		ndims[i * 2 + 1] = dims[i] / blk[i];
		ndims[i * 2 + 0] = blk[i];

		L[i * 2 + 1] = dims[i] / blk[i];
		L[i * 2 + 0] = blk[i] + dim2[i] - 1;

		ndim2[i * 2 + 1] = 1;
		ndim2[i * 2 + 0] = dim2[i];

		ndim3[i * 2 + 1] = dims[i] / blk[i] + 1;
		ndim3[i * 2 + 0] = blk[i];
	}

	complex float* tmp = md_alloc(2 * N, L, CFL_SIZE);

//	conv_causal_extend(2 * N, L, tmp, ndims, src1, ndim2, src2);
	conv(2 * N, ~0, CONV_EXTENDED, CONV_CAUSAL, L, tmp, ndims, src1, ndim2, src2);
	// [------++++||||||||

	//long str1[2 * N];
	long str2[2 * N];
	long str3[2 * N];

	//md_calc_strides(2 * N, str1, ndims, 8);
	md_calc_strides(2 * N, str2, L, 8);
	md_calc_strides(2 * N, str3, ndim3, 8);

	md_clear(2 * N, ndim3, dst, CFL_SIZE);
	md_zadd2(2 * N, L, str3, dst, str3, dst, str2, tmp);
	
	md_free(tmp);
}



void overlapandsave(int N, const long dims[N], const long blk[N], complex float* dst, complex float* src1, const long dim2[N], complex float* src2)
{
	// [------++++
	// [------

	long ndims[2 * N];
	long L[2 * N];
	long ndim2[2 * N];
	long ndim3[2 * N];

	for (int i = 0; i < N; i++) {

		assert(0 == dims[i] % blk[i]);
		assert(dim2[i] <= blk[i]);

		ndims[i * 2 + 1] = dims[i] / blk[i];
		ndims[i * 2 + 0] = blk[i];

		L[i * 2 + 1] = dims[i] / blk[i];
		L[i * 2 + 0] = blk[i] + dim2[i] - 1;

		ndim2[i * 2 + 1] = 1;
		ndim2[i * 2 + 0] = dim2[i];

		ndim3[i * 2 + 1] = dims[i] / blk[i] - 0;
		ndim3[i * 2 + 0] = blk[i];
	}

	complex float* tmp = md_alloc(2 * N, L, CFL_SIZE);

	long str1[2 * N];
	long str2[2 * N];
	long str3[2 * N];

	md_calc_strides(2 * N, str1, ndims, 8);
	md_calc_strides(2 * N, str2, L, 8);
	md_calc_strides(2 * N, str3, ndim3, 8);

	md_clear(2 * N, L, tmp, 8);
	md_copy2(2 * N, ndim3, str2, tmp, str1, src1, 8);
	conv(2 * N, ~0, CONV_VALID, CONV_CAUSAL, ndims, dst, L, tmp, ndim2, src2);

	md_free(tmp);
}


#if 0
struct conv_plan* overlapandsave_plan(int N, const long dims[N], const long blk[N], const long dim2[N], complex float* src2)
{
	return conv_plan(2 * N, ~0, CONV_VALID, CONV_CAUSAL, ndims, L, ndim2, src2);
}


void overlapandsave_exec(struct conv_plan* plan, int N, const long dims[N], const long blk[N], complex float* dst, complex float* src1, const long dim2[N])
{
	md_clear(2 * N, L, tmp, 8);
	md_copy2(2 * N, ndim3, str2, tmp, str1, src1, 8);
	conv_exec(plan, dst, tmp);
	
	xfree(tmp);
}
#endif


void overlapandsave2(int N, unsigned int flags, const long blk[N], const long odims[N], complex float* dst, const long dims1[N], const complex float* src1, const long dims2[N], const complex float* src2)
{
	long dims1B[N];

	long tdims[2 * N];
	long nodims[2 * N];
	long ndims1[2 * N];
	long ndims2[2 * N];

	long shift[2 * N];

	unsigned int nflags = 0;

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(flags, i)) {

			nflags = MD_SET(nflags, 2 * i);

			assert(1 == dims2[i] % 2);
			assert(0 == blk[i] % 2);
			assert(0 == dims1[i] % 2);
			assert(0 == odims[i] % blk[i]);
			assert(0 == dims1[i] % blk[i]);
			assert(dims1[i] == odims[i]);
			assert(dims2[i] <= blk[i]);
			assert(dims1[i] >= dims2[i]);

			// blocked output

			nodims[i * 2 + 1] = odims[i] / blk[i];
			nodims[i * 2 + 0] = blk[i];

			// expanded temporary storage

			tdims[i * 2 + 1] = dims1[i] / blk[i];
			tdims[i * 2 + 0] = blk[i] + dims2[i] - 1;

			// blocked input

			// ---|---,---,---|---
			//   + +++ +
			//       + +++ +

			// resized input

			dims1B[i] = dims1[i] + 2 * blk[i];

			ndims1[i * 2 + 1] = dims1[i] / blk[i] + 2; // do we need two full blocks?
			ndims1[i * 2 + 0] = blk[i];

			shift[i * 2 + 1] = 0;
			shift[i * 2 + 0] = blk[i] - (dims2[i] - 1) / 2;

			// kernel

			ndims2[i * 2 + 1] = 1;
			ndims2[i * 2 + 0] = dims2[i];

		} else {

			nodims[i * 2 + 1] = 1;
			nodims[i * 2 + 0] = odims[i];

			tdims[i * 2 + 1] = 1;
			tdims[i * 2 + 0] = dims1[i];

			ndims1[i * 2 + 1] = 1;
			ndims1[i * 2 + 0] = dims1[i];

			shift[i * 2 + 1] = 0;
			shift[i * 2 + 0] = 0;


			dims1B[i] = dims1[i];

			ndims2[i * 2 + 1] = 1;
			ndims2[i * 2 + 0] = dims2[i];
		}
	}

	complex float* src1B = md_alloc(N, dims1B, CFL_SIZE);

	md_resize_center(N, dims1B, src1B, dims1, src1, CFL_SIZE);


	complex float* tmp = md_alloc(2 * N, tdims, CFL_SIZE);

	long str1[2 * N];
	long str2[2 * N];

	md_calc_strides(2 * N, str1, ndims1, CFL_SIZE);
	md_calc_strides(2 * N, str2, tdims, CFL_SIZE);

	long off = md_calc_offset(2 * N, str1, shift);
	md_copy2(2 * N, tdims, str2, tmp, str1, ((void*)src1B) + off, CFL_SIZE);

	md_free(src1B);

	conv(2 * N, nflags, CONV_VALID, CONV_SYMMETRIC, nodims, dst, tdims, tmp, ndims2, src2);

	md_free(tmp);
}


void overlapandsave2H(int N, unsigned int flags, const long blk[N], const long dims1[N], complex float* dst, const long odims[N], const complex float* src1, const long dims2[N], const complex float* src2)
{
	long dims1B[N];

	long tdims[2 * N];
	long nodims[2 * N];
	long ndims1[2 * N];
	long ndims2[2 * N];

	long shift[2 * N];
	
	unsigned int nflags = 0;

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(flags, i)) {

			nflags = MD_SET(nflags, 2 * i);

			assert(1 == dims2[i] % 2);
			assert(0 == blk[i] % 2);
			assert(0 == dims1[i] % 2);
			assert(0 == odims[i] % blk[i]);
			assert(0 == dims1[i] % blk[i]);
			assert(dims1[i] == odims[i]);
			assert(dims2[i] <= blk[i]);
			assert(dims1[i] >= dims2[i]);

			// blocked output

			nodims[i * 2 + 1] = odims[i] / blk[i];
			nodims[i * 2 + 0] = blk[i];

			// expanded temporary storage

			tdims[i * 2 + 1] = dims1[i] / blk[i];
			tdims[i * 2 + 0] = blk[i] + dims2[i] - 1;

			// blocked input

			// ---|---,---,---|---
			//   + +++ +
			//       + +++ +

			// resized input

			dims1B[i] = dims1[i] + 2 * blk[i];

			ndims1[i * 2 + 1] = dims1[i] / blk[i] + 2; // do we need two full blocks?
			ndims1[i * 2 + 0] = blk[i];

			shift[i * 2 + 1] = 0;
			shift[i * 2 + 0] = blk[i] - (dims2[i] - 1) / 2;

			// kernel

			ndims2[i * 2 + 1] = 1;
			ndims2[i * 2 + 0] = dims2[i];

		} else {

			nodims[i * 2 + 1] = 1;
			nodims[i * 2 + 0] = odims[i];

			tdims[i * 2 + 1] = 1;
			tdims[i * 2 + 0] = dims1[i];

			ndims1[i * 2 + 1] = 1;
			ndims1[i * 2 + 0] = dims1[i];

			shift[i * 2 + 1] = 0;
			shift[i * 2 + 0] = 0;

			dims1B[i] = dims1[i];

			ndims2[i * 2 + 1] = 1;
			ndims2[i * 2 + 0] = dims2[i];
		}
	}


	complex float* tmp = md_alloc(2 * N, tdims, CFL_SIZE);


	//conv(2 * N, flags, CONV_VALID, CONV_SYMMETRIC, nodims, dst, tdims, tmp, ndims2, src2);
	convH(2 * N, nflags, CONV_VALID, CONV_SYMMETRIC, tdims, tmp, nodims, src1, ndims2, src2);

 

	complex float* src1B = md_alloc(N, dims1B, CFL_SIZE);


	long str1[2 * N];
	long str2[2 * N];

	md_calc_strides(2 * N, str1, ndims1, CFL_SIZE);
	md_calc_strides(2 * N, str2, tdims, CFL_SIZE);



	long off = md_calc_offset(2 * N, str1, shift);
	md_clear(N, dims1B, src1B, CFL_SIZE);

	//md_copy2(2 * N, tdims, str1, ((void*)src1B) + off, str2, tmp, sizeof(complex float));// FIXME:
	md_zadd2(2 * N, tdims, str1, ((void*)src1B) + off, str1, ((void*)src1B) + off, str2, tmp);


	md_resize_center(N, dims1, dst, dims1B, src1B, CFL_SIZE);

	md_free(src1B);
	md_free(tmp);
}







void overlapandsave2NE(int N, unsigned int flags, const long blk[N], const long odims[N], complex float* dst, const long dims1[N], complex float* src1, const long dims2[N], complex float* src2, const long mdims[N], complex float* msk)
{
	long dims1B[N];

	long tdims[2 * N];
	long nodims[2 * N];
	long ndims1[2 * N];
	long ndims2[2 * N];

	long shift[2 * N];

	unsigned int nflags = 0;

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(flags, i)) {

			nflags = MD_SET(nflags, 2 * i);

			assert(1 == dims2[i] % 2);
			assert(0 == blk[i] % 2);
			assert(0 == dims1[i] % 2);
			assert(0 == odims[i] % blk[i]);
			assert(0 == dims1[i] % blk[i]);
			assert(dims1[i] == odims[i]);
			assert(dims2[i] <= blk[i]);
			assert(dims1[i] >= dims2[i]);

			// blocked output

			nodims[i * 2 + 1] = odims[i] / blk[i];
			nodims[i * 2 + 0] = blk[i];

			// expanded temporary storage

			tdims[i * 2 + 1] = dims1[i] / blk[i];
			tdims[i * 2 + 0] = blk[i] + dims2[i] - 1;

			// blocked input

			// ---|---,---,---|---
			//   + +++ +
			//       + +++ +

			// resized input

			dims1B[i] = dims1[i] + 2 * blk[i];

			ndims1[i * 2 + 1] = dims1[i] / blk[i] + 2; // do we need two full blocks?
			ndims1[i * 2 + 0] = blk[i];

			shift[i * 2 + 1] = 0;
			shift[i * 2 + 0] = blk[i] - (dims2[i] - 1) / 2;

			// kernel

			ndims2[i * 2 + 1] = 1;
			ndims2[i * 2 + 0] = dims2[i];

		} else {

			nodims[i * 2 + 1] = 1;
			nodims[i * 2 + 0] = odims[i];

			tdims[i * 2 + 1] = 1;
			tdims[i * 2 + 0] = dims1[i];

			ndims1[i * 2 + 1] = 1;
			ndims1[i * 2 + 0] = dims1[i];

			shift[i * 2 + 1] = 0;
			shift[i * 2 + 0] = 0;


			dims1B[i] = dims1[i];

			ndims2[i * 2 + 1] = 1;
			ndims2[i * 2 + 0] = dims2[i];
		}
	}

	complex float* src1B = md_alloc(N, dims1B, CFL_SIZE);
	complex float* tmp = md_alloc(2 * N, tdims, CFL_SIZE);
	complex float* tmpX = md_alloc(N, odims, CFL_SIZE);

	long str1[2 * N];
	long str2[2 * N];

	md_calc_strides(2 * N, str1, ndims1, sizeof(complex float));
	md_calc_strides(2 * N, str2, tdims, sizeof(complex float));

	long off = md_calc_offset(2 * N, str1, shift);

	md_resize_center(N, dims1B, src1B, dims1, src1, sizeof(complex float));

	// we can loop here

	md_copy2(2 * N, tdims, str2, tmp, str1, ((void*)src1B) + off, sizeof(complex float));

	conv(2 * N, nflags, CONV_VALID, CONV_SYMMETRIC, nodims, tmpX, tdims, tmp, ndims2, src2);

	long ostr[N];
	long mstr[N];

	md_calc_strides(N, ostr, odims, sizeof(complex float));
	md_calc_strides(N, mstr, mdims, sizeof(complex float));

	md_zmul2(N, odims, ostr, tmpX, ostr, tmpX, mstr, msk);

	convH(2 * N, nflags, CONV_VALID, CONV_SYMMETRIC, tdims, tmp, nodims, tmpX, ndims2, src2);

	md_clear(N, dims1B, src1B, sizeof(complex float));
	md_zadd2(2 * N, tdims, str1, ((void*)src1B) + off, str1, ((void*)src1B) + off, str2, tmp);

	//

	md_resize_center(N, dims1, dst, dims1B, src1B, sizeof(complex float));

	md_free(src1B);
	md_free(tmpX);
	md_free(tmp);
}





void overlapandsave2NEB(int N, unsigned int flags, const long blk[N], const long odims[N], complex float* dst, const long dims1[N], const complex float* src1, const long dims2[N], const complex float* src2, const long mdims[N], const complex float* msk)
{
	long dims1B[N];

	long tdims[2 * N];
	long nodims[2 * N];
	long ndims2[2 * N];
	long nmdims[2 * N];


	int e = N;

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(flags, i)) {

			assert(1 == dims2[i] % 2);
			assert(0 == blk[i] % 2);
			assert(0 == dims1[i] % 2);
			assert(0 == odims[i] % blk[i]);
			assert(0 == dims1[i] % blk[i]);
			assert(dims1[i] == odims[i]);
			assert(dims2[i] <= blk[i]);
			assert(dims1[i] >= dims2[i]);
			assert((1 == mdims[i]) || (mdims[i] == dims1[i]));

			// blocked output

			nodims[e] = odims[i] / blk[i];
			nodims[i] = blk[i];

			// expanded temporary storage

			tdims[e] = dims1[i] / blk[i];
			tdims[i] = blk[i] + dims2[i] - 1;

			// blocked input

			// ---|---,---,---|---
			//   + +++ +
			//       + +++ +

			if (1 == mdims[i]) {

				nmdims[2 * i + 1] = 1;
				nmdims[2 * i + 1] = 1;

			} else {

				nmdims[2 * i + 1] = mdims[i] / blk[i];
				nmdims[2 * i + 0] = blk[i];
			}

			// resized input
			// minimal padding
			dims1B[i] = dims1[i] + (dims2[i] - 1);

			// kernel

			ndims2[e] = 1;
			ndims2[i] = dims2[i];

			e++;

		} else {

			nodims[i] = odims[i];
			tdims[i] = dims1[i];
			nmdims[2 * i + 1] = 1;
			nmdims[2 * i + 0] = mdims[i];

			dims1B[i] = dims1[i];
			ndims2[i] = dims2[i];
		}
	}

	int NE = e;

	//long S = md_calc_size(N, dims1B, 1);

	long str1[NE];

	long str1B[N];
	md_calc_strides(N, str1B, dims1B, sizeof(complex float));

	e = N;
	for (int i = 0; i < N; i++) {

		str1[i] = str1B[i];

		if (MD_IS_SET(flags, i))
			str1[e++] = str1B[i] * blk[i];
	}
	assert(NE == e);


	long str2[NE];
	md_calc_strides(NE, str2, tdims, sizeof(complex float));


	long ostr[NE];
	long mstr[NE];
	long mstrB[2 * N];

	md_calc_strides(NE, ostr, nodims, sizeof(complex float));
	md_calc_strides(2 * N, mstrB, nmdims, sizeof(complex float));

	e = N;
	for (int i = 0; i < N; i++) {

		mstr[i] = mstrB[2 * i + 0];

		if (MD_IS_SET(flags, i))
			mstr[e++] = mstrB[2 * i + 1];
	}
	assert(NE == e);


	const complex float* src1B = src1;//!
	//complex float* src1B = xmalloc(S * sizeof(complex float));
	//md_resizec(N, dims1B, src1B, dims1, src1, sizeof(complex float));

	// we can loop here
	assert(NE == N + 3);
	assert(1 == ndims2[N + 0]);
	assert(1 == ndims2[N + 1]);
	assert(1 == ndims2[N + 2]);
	assert(tdims[N + 0] == nodims[N + 0]);
	assert(tdims[N + 1] == nodims[N + 1]);
	assert(tdims[N + 2] == nodims[N + 2]);

	//complex float* src1C = xmalloc(S * sizeof(complex float));
	complex float* src1C = dst;

	md_clear(N, dims1B, src1C, sizeof(complex float));	// must be done here

	#pragma omp parallel for collapse(3)
	for (int k = 0; k < nodims[N + 2]; k++) {
	for (int j = 0; j < nodims[N + 1]; j++) {
	for (int i = 0; i < nodims[N + 0]; i++) {

		complex float* tmp = md_alloc_sameplace(N, tdims, CFL_SIZE, dst);
		complex float* tmpX = md_alloc_sameplace(N, nodims, CFL_SIZE, dst);

		long off1 = str1[N + 0] * i + str1[N + 1] * j + str1[N + 2] * k;
		long off2 = mstr[N + 0] * i + mstr[N + 1] * j + mstr[N + 2] * k;

		md_copy2(N, tdims, str2, tmp, str1, ((const void*)src1B) + off1, sizeof(complex float));
		conv(N, flags, CONV_VALID, CONV_SYMMETRIC, nodims, tmpX, tdims, tmp, ndims2, src2);
		md_zmul2(N, nodims, ostr, tmpX, ostr, tmpX, mstr, ((const void*)msk) + off2);
		convH(N, flags, CONV_VALID, CONV_SYMMETRIC, tdims, tmp, nodims, tmpX, ndims2, src2);

		#pragma omp critical
		md_zadd2(N, tdims, str1, ((void*)src1C) + off1, str1, ((void*)src1C) + off1, str2,  tmp);

		md_free(tmpX);
		md_free(tmp);
	}}}

	//md_resizec(N, dims1, dst, dims1B, src1C, sizeof(complex float));
	//xfree(src1C);
	//xfree(src1B);
}





void overlapandsave2HB(int N, unsigned int flags, const long blk[N], const long dims1[N], complex float* dst, const long odims[N], const complex float* src1, const long dims2[N], const complex float* src2, const long mdims[N], const complex float* msk)
{
	long dims1B[N];

	long tdims[2 * N];
	long nodims[2 * N];
	long ndims2[2 * N];
	long nmdims[2 * N];


	int e = N;

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(flags, i)) {

			assert(1 == dims2[i] % 2);
			assert(0 == blk[i] % 2);
			assert(0 == dims1[i] % 2);
			assert(0 == odims[i] % blk[i]);
			assert(0 == dims1[i] % blk[i]);
			assert(dims1[i] == odims[i]);
			assert(dims2[i] <= blk[i]);
			assert(dims1[i] >= dims2[i]);
			assert((1 == mdims[i]) || (mdims[i] == dims1[i]));

			// blocked output

			nodims[e] = odims[i] / blk[i];
			nodims[i] = blk[i];

			// expanded temporary storage

			tdims[e] = dims1[i] / blk[i];
			tdims[i] = blk[i] + dims2[i] - 1;

			// blocked input

			// ---|---,---,---|---
			//   + +++ +
			//       + +++ +

			if (1 == mdims[i]) {

				nmdims[2 * i + 1] = 1;
				nmdims[2 * i + 1] = 1;

			} else {

				nmdims[2 * i + 1] = mdims[i] / blk[i];
				nmdims[2 * i + 0] = blk[i];
			}

			// resized input
			// minimal padding
			dims1B[i] = dims1[i] + (dims2[i] - 1);

			// kernel

			ndims2[e] = 1;
			ndims2[i] = dims2[i];

			e++;

		} else {

			nodims[i] = odims[i];
			tdims[i] = dims1[i];
			nmdims[2 * i + 1] = 1;
			nmdims[2 * i + 0] = mdims[i];

			dims1B[i] = dims1[i];
			ndims2[i] = dims2[i];
		}
	}

	int NE = e;

	// long S = md_calc_size(N, dims1B, 1);

	long str1[NE];

	long str1B[N];
	md_calc_strides(N, str1B, dims1B, sizeof(complex float));

	e = N;
	for (int i = 0; i < N; i++) {

		str1[i] = str1B[i];

		if (MD_IS_SET(flags, i))
			str1[e++] = str1B[i] * blk[i];
	}
	assert(NE == e);



	long str2[NE];
	md_calc_strides(NE, str2, tdims, sizeof(complex float));


	long ostr[NE];
	long mstr[NE];
	long mstrB[2 * N];

	md_calc_strides(NE, ostr, nodims, sizeof(complex float));
	md_calc_strides(2 * N, mstrB, nmdims, sizeof(complex float));

	e = N;
	for (int i = 0; i < N; i++) {

		mstr[i] = mstrB[2 * i + 0];

		if (MD_IS_SET(flags, i))
			mstr[e++] = mstrB[2 * i + 1];
	}
	assert(NE == e);
	
	// we can loop here
	assert(NE == N + 3);
	assert(1 == ndims2[N + 0]);
	assert(1 == ndims2[N + 1]);
	assert(1 == ndims2[N + 2]);
	assert(tdims[N + 0] == nodims[N + 0]);
	assert(tdims[N + 1] == nodims[N + 1]);
	assert(tdims[N + 2] == nodims[N + 2]);


	//complex float* src1C = xmalloc(S * sizeof(complex float));
	complex float* src1C = dst;

	md_clear(N, dims1B, src1C, CFL_SIZE);	// must be done here

	#pragma omp parallel for collapse(3)
	for (int k = 0; k < nodims[N + 2]; k++) {
	for (int j = 0; j < nodims[N + 1]; j++) {
	for (int i = 0; i < nodims[N + 0]; i++) {

		    complex float* tmp = md_alloc_sameplace(N, tdims, CFL_SIZE, dst);
		    complex float* tmpX = md_alloc_sameplace(N, nodims, CFL_SIZE, dst);

		    long off1 = str1[N + 0] * i + str1[N + 1] * j + str1[N + 2] * k;
		    long off2 = mstr[N + 0] * i + mstr[N + 1] * j + mstr[N + 2] * k;
		    long off3 = ostr[N + 0] * i + ostr[N + 1] * j + ostr[N + 2] * k;

		    md_zmul2(N, nodims, ostr, tmpX, ostr, ((const void*)src1) + off3, mstr, ((const void*)msk) + off2);
		    convH(N, flags, CONV_VALID, CONV_SYMMETRIC, tdims, tmp, nodims, tmpX, ndims2, src2);

		    #pragma omp critical
		    md_zadd2(N, tdims, str1, ((void*)src1C) + off1, str1, ((void*)src1C) + off1, str2,  tmp);

		    md_free(tmpX);
		    md_free(tmp);
	}}}
}


