/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>

#include "misc/misc.h"

#include "num/blas.h"
#include "num/multind.h"

#include "blas_md_wrapper.h"

// In this file we use units of elementsize for strides as in BLAS conventions
// x > 0 is an positive integer

static bool check_blas_strides(int N, const long str[N], long size)
{
	for (int i = 0; i < N; i++) {

		if ((0 != str[i] % size) || (0 > str[i]))
			return false;
	}

	return true;
}

/****************************************************************************************************
 *
 * Wrappers for zfmac
 *
 ****************************************************************************************************/

/**
 * cgemm for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {1, 0, dim[0]}
 * @param optr
 * @param istr1 must be of the form {1, dim[0] + x, 0} or {dim[1] + x, 1, 0}
 * @param iptr1
 * @param istr1 must be of the form {0, 1, dim[1] + x} or {0, dim[2] + x, 1}
 * @param iptr1
 **/
void blas_zfmac_cgemm(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert(3 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr1, size));
	assert(check_blas_strides(N, istr2, size));

	assert((size == ostr[0]) && (0 == ostr[1]) && (dims[0] * size <= ostr[2]));

	assert(    ((size == istr1[0]) && (0 == istr1[2]) && (dims[0] * size <= istr1[1]))
		|| ((size == istr1[1]) && (0 == istr1[2]) && (dims[1] * size <= istr1[0])));

	assert(    ((size == istr2[2]) && (0 == istr2[0]) && (dims[2] * size <= istr2[1]))
		|| ((size == istr2[1]) && (0 == istr2[0]) && (dims[1] * size <= istr2[2])));


	char transa = (size == istr1[0]) ? 'N' : 'T';
	char transb = (size == istr2[1]) ? 'N' : 'T';

	long lda = (size == istr1[0]) ? istr1[1] / size : istr1[0] / size;
	long ldb = (size == istr2[1]) ? istr2[2] / size : istr2[1] / size;
	long ldc = ostr[2] / size;

	lda = MAX(1, lda);
	ldb = MAX(1, ldb);
	ldc = MAX(1, ldc);

	blas_cgemm(transa, transb, dims[0], dims[2], dims[1], 1., lda, iptr1, ldb, iptr2, 1., ldc, optr);
}


/**
 * cgemv for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {1+x, 0}
 * @param optr
 * @param istr1 must be of the form {1, dim[0]+x} or {dim[1]+x, 1}
 * @param iptr1
 * @param istr1 must be of the form {0, 1+x}
 * @param iptr1
 **/
void blas_zfmac_cgemv(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert(2 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr1, size));
	assert(check_blas_strides(N, istr2, size));

	assert((0 < ostr[0]) && (0 == ostr[1]));
	assert(    ((size == istr1[0]) && (dims[0] * size <= istr1[1]))
		|| ((size == istr1[1]) && (dims[1] * size <= istr1[0])));
	assert((0 == istr2[0]) && (0 < istr2[1]));

	char trans = (size == istr1[0]) ? 'N' : 'T';

	long lda = (size == istr1[0]) ? istr1[1] / size : istr1[0] / size;
	long incx = istr2[1] / size;
	long incy = ostr[0] / size;

	long m = (size == istr1[0]) ? dims[0] : dims[1];
	long n = (size == istr1[0]) ? dims[1] : dims[0];

	lda = MAX(1, lda);

	blas_cgemv(trans, m, n, 1., lda, iptr1, incx, iptr2, 1., incy, optr);
}


/**
 * cgeru for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {1, dims[0]+x}
 * @param optr
 * @param istr1 must be of the form {1+x, 0}
 * @param iptr1
 * @param istr1 must be of the form {0, 1+x}
 * @param iptr1
 **/
void blas_zfmac_cgeru(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert(2 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr1, size));
	assert(check_blas_strides(N, istr2, size));

	assert((size == ostr[0]) && (dims[0] * size <= ostr[1]));
	assert((0 == istr1[1]) && (0 < istr1[0]));
	assert((0 == istr2[0]) && (0 < istr2[1]));

	long lda = ostr[1] / size;
	long incx = istr1[0] / size;
	long incy = istr2[1] / size;

	lda = MAX(1, lda);

	blas_cgeru(dims[0], dims[1], 1., incx, iptr1, incy, iptr2, lda, optr);
}


/**
 * caxpy for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {1+x}
 * @param optr
 * @param istr1 must be of the form {1+x}
 * @param iptr1
 * @param istr2 must be of the form {0}
 * @param iptr2
 **/
void blas_zfmac_caxpy(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert(1 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr1, size));
	assert(check_blas_strides(N, istr2, size));

	assert(0 < ostr[0]);
	assert(0 < istr1[0]);
	assert(0 == istr2[0]);

	long incx = istr1[0] / size;
	long incy = ostr[0] / size;

	blas2_caxpy(dims[0], iptr2, incx, iptr1, incy, optr);
}


/**
 * cdotu for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {0}
 * @param optr
 * @param istr1 must be of the form {1+x}
 * @param iptr1
 * @param istr2 must be of the form {1+x}
 * @param iptr2
 **/
void blas_zfmac_cdotu(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert(1 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr1, size));
	assert(check_blas_strides(N, istr2, size));

	assert(0 == ostr[0]);
	assert(0 < istr1[0]);
	assert(0 < istr2[0]);

	long incx = istr1[0] / size;
	long incy = istr2[0] / size;


	complex float* tmp = md_alloc_sameplace(1, MAKE_ARRAY(1l), size, optr);

	blas2_cdotu(tmp, dims[0], incx, iptr1, incy, iptr2);
	blas_caxpy(1, 1., size, tmp, size, optr);

	md_free(tmp);
}



/****************************************************************************************************
 *
 * Wrappers for fmac
 *
 ****************************************************************************************************/

/**
 * sgemm for inner fmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {1, 0, dim[0]}
 * @param optr
 * @param istr1 must be of the form {1, dim[0], 0} or {dim[1], 1, 0}
 * @param iptr1
 * @param istr1 must be of the form {0, 1, dim[1]} or {0, dim[2], 1}
 * @param iptr1
 **/
void blas_fmac_sgemm(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	long size = 4;

	assert(3 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr1, size));
	assert(check_blas_strides(N, istr2, size));

	assert((size == ostr[0]) && (0 == ostr[1]) && (dims[0] * size <= ostr[2]));

	assert(    ((size == istr1[0]) && (0 == istr1[2]) && (dims[0] * size <= istr1[1]))
		|| ((size == istr1[1]) && (0 == istr1[2]) && (dims[1] * size <= istr1[0])));

	assert(    ((size == istr2[2]) && (0 == istr2[0]) && (dims[2] * size <= istr2[1]))
		|| ((size == istr2[1]) && (0 == istr2[0]) && (dims[1] * size <= istr2[2])));

	char transa = (size == istr1[0]) ? 'N' : 'T';
	char transb = (size == istr2[1]) ? 'N' : 'T';

	long lda = (size == istr1[0]) ? istr1[1] / size : istr1[0] / size;
	long ldb = (size == istr2[1]) ? istr2[2] / size : istr2[1] / size;
	long ldc = ostr[2] / size;

	lda = MAX(1, lda);
	ldb = MAX(1, ldb);
	ldc = MAX(1, ldc);

	blas_sgemm(transa, transb, dims[0], dims[2], dims[1], 1., lda, iptr1, ldb, iptr2, 1., ldc, optr);
}


/**
 * sgemv for inner fmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {1+x, 0}
 * @param optr
 * @param istr1 must be of the form {1, dim[0]+x} or {dim[1]+x, 1}
 * @param iptr1
 * @param istr1 must be of the form {0, 1+x}
 * @param iptr1
 **/
void blas_fmac_sgemv(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	long size = 4;

	assert(2 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr1, size));
	assert(check_blas_strides(N, istr2, size));

	assert((0 < ostr[0]) && (0 == ostr[1]));
	assert(    ((size == istr1[0]) && (dims[0] * size <= istr1[1]))
		|| ((size == istr1[1]) && (dims[1] * size <= istr1[0])));
	assert((0 == istr2[0]) && (0 < istr2[1]));

	char trans = (size == istr1[0]) ? 'N' : 'T';

	long lda = (size == istr1[0]) ? istr1[1] / size : istr1[0] / size;
	long incx = istr2[1] / size;
	long incy = ostr[0] / size;

	lda = MAX(1, lda);

	long m = (size == istr1[0]) ? dims[0] : dims[1];
	long n = (size == istr1[0]) ? dims[1] : dims[0];

	blas_sgemv(trans, m, n, 1., lda, iptr1, incx, iptr2, 1., incy, optr);
}


/**
 * sger for inner fmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {1, dims[0] + x}
 * @param optr
 * @param istr1 must be of the form {1+x, 0}
 * @param iptr1
 * @param istr1 must be of the form {0, 1+x}
 * @param iptr1
 **/
void blas_fmac_sger(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	long size = 4;

	assert(2 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr1, size));
	assert(check_blas_strides(N, istr2, size));

	assert((size == ostr[0]) && (dims[0] * size <= ostr[1]));
	assert((0 == istr1[1]) && (0 < istr1[0]));
	assert((0 == istr2[0]) && (0 < istr2[1]));


	long lda = ostr[1] / size;
	long incx = istr1[0] / size;
	long incy = istr2[1] / size;

	lda = MAX(1, lda);

	blas_sger(dims[0], dims[1], 1., incx, iptr1, incy, iptr2, lda, optr);
}


/**
 * saxpy for inner fmac kernel
 *
 * @param dims dimension
 * @param ostr  must be of the form {1+x}
 * @param optr
 * @param istr1 must be of the form {1+x}
 * @param iptr1
 * @param istr2 must be of the form {0}
 * @param iptr2
 **/
void blas_fmac_saxpy(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	long size = 4;

	assert(1 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr1, size));
	assert(check_blas_strides(N, istr2, size));

	assert(0 < ostr[0]);
	assert(0 < istr1[0]);
	assert(0 == istr2[0]);

	long incx = istr1[0] / size;
	long incy = ostr[0] / size;

	blas2_saxpy(dims[0], iptr2, incx, iptr1, incy, optr);
}


/**
 * sdot for inner fmac kernel
 *
 * @param dims dimension
 * @param ostr  must be of the form {0}
 * @param optr
 * @param istr1 must be of the form {1+x}
 * @param iptr1
 * @param istr2 must be of the form {1+x}
 * @param iptr2
 **/
void blas_fmac_sdot(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	long size = 4;

	assert(1 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr1, size));
	assert(check_blas_strides(N, istr2, size));

	assert(0 == ostr[0]);
	assert(0 < istr1[0]);
	assert(0 < istr2[0]);

	long incx = istr1[0] / size;
	long incy = istr2[0] / size;


	float* tmp = md_alloc_sameplace(1, MAKE_ARRAY(1l), size, optr);

	blas2_sdot(tmp, dims[0], incx, iptr1, incy, iptr2);
	blas_saxpy(1, 1., size, tmp, size, optr);

	md_free(tmp);
}



/****************************************************************************************************
 *
 * Wrappers for zmul / zsmul
 *
 ****************************************************************************************************/

/**
 *
 * @param dims dimension
 * @param ostr  must be of the form {1, dim[0]}
 * @param optr
 * @param istr1 must be of the form {1, dim[0]} or {dim[1], 1}
 * @param iptr1
 * @param istr1 must be of the form {0, 0}
 * @param iptr1
 **/
void blas_zmul_cmatcopy(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert(2 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr1, size));
	assert(check_blas_strides(N, istr2, size));

	assert((size == ostr[0]) && (size * dims[0] == ostr[1]));
	assert(    ((size == istr1[0]) && (size * dims[0] == istr1[1]))
		|| ((size == istr1[1]) && (size * dims[1] == istr1[0])));
	assert((0 == istr2[0]) && (0 == istr2[1]));

	char trans = (size == istr1[0]) ? 'N' : 'T';

	long lda = (size == istr1[0]) ? istr1[1] / size : istr1[0] / size;
	long ldb = ostr[1] / size;

	lda = MAX(1, lda);
	ldb = MAX(1, ldb);

	blas2_cmatcopy(trans, dims[0], dims[1], iptr2, iptr1, lda, optr, ldb);
}


/**
 *
 * @param dims dimension
 * @param ostr must be of the form {1, dim[0] + x}
 * @param optr
 * @param istr must be of the form {1, dim[0]+x} or {dim[1]+x, 1}
 * @param iptr
 * @param val
 **/
void blas_zsmul_cmatcopy(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr[N], const complex float* iptr, complex float val)
{
	long size = 8;

	assert(2 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr, size));

	assert((size == ostr[0]) && (size * dims[0] == ostr[1]));
	assert(    ((size == istr[0]) && (size * dims[0] == istr[1]))
		|| ((size == istr[1]) && (size * dims[1] == istr[0])));

	char trans = (size == istr[0]) ? 'N' : 'T';

	long lda = (size == istr[0]) ? istr[1] / size : istr[0] / size;
	long ldb = ostr[1] / size;

	lda = MAX(1, lda);
	ldb = MAX(1, ldb);

	blas_cmatcopy(trans, dims[0], dims[1], val, iptr, lda, optr, ldb);
}


/**
 *
 * @param dims dimension
 * @param ostr must be of the form {1, dim[0] + x}
 * @param optr
 * @param istr1 must be of the form {1, dim[0] + x}
 * @param iptr1
 * @param istr1 must be of the form {1+x, 0} or {0, 1+x}
 * @param iptr1
 **/
void blas_zmul_cdgmm(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert(2 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr1, size));
	assert(check_blas_strides(N, istr2, size));

	assert((size == ostr[0]) && (0 == ostr[1] % size) && (dims[0] * ostr[0] <= ostr[1]));
	assert((size == istr1[0]) && (0 == istr1[1] % size) && (dims[0] * istr1[0] <= istr1[1]));
	assert((0 == istr2[0] * istr2[1]));
	assert((0 < istr2[0]) || (0 < istr2[1]));

	long lda = istr1[1] / size;
	long ldc = ostr[1] / size;
	long incx = (0 == istr2[1]) ? istr2[0] / size : istr2[1] / size;

	lda = MAX(1, lda);
	ldc = MAX(1, ldc);

	blas_cdgmm(dims[0], dims[1], 0 == istr2[1], iptr1, lda, iptr2, incx, optr, ldc);
}


/**
 * sger for inner mul kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {1, dims[0]+x}
 * @param optr
 * @param istr1 must be of the form {1+x, 0}
 * @param iptr1
 * @param istr1 must be of the form {0, 1+x}
 * @param iptr1
 **/
void blas_zmul_cgeru(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	md_clear2(N, dims, ostr, optr, size);

	blas_zfmac_cgeru(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}


/**
 *
 * @param dims dimension
 * @param ostr must be of the form {1+x}
 * @param optr
 * @param istr1 must be of the form {1+x}
 * @param iptr1
 * @param istr1 must be of the form {0}
 * @param iptr1
 **/
void blas_zmul_cscal(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert(((optr != iptr1) || (ostr[0] == istr1[0])) && (0 == ostr[0] % size) && (0 == istr2[0]));
	assert(1 == N);

	if (optr != iptr1)
		md_copy2(N, dims, ostr, optr, istr1, iptr1, size);

	blas2_cscal(dims[0], iptr2, ostr[0] / size, optr);
}




/****************************************************************************************************
 *
 * Wrappers for mul / smul
 *
 ****************************************************************************************************/

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {1, dim[0]+x}
 * @param optr
 * @param istr1 must be of the form {1, dim[0]+x} or {dim[1]+x, 1}
 * @param iptr1
 * @param istr1 must be of the form {0, 0}
 * @param iptr1
 **/
void blas_mul_smatcopy(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	long size = 4;

	assert(2 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr1, size));
	assert(check_blas_strides(N, istr2, size));

	assert((size == ostr[0]) && (size * dims[0] == ostr[1]));
	assert(    ((size == istr1[0]) && (size * dims[0] == istr1[1]))
		|| ((size == istr1[1]) && (size * dims[1] == istr1[0])));
	assert((0 == istr2[0]) && (0 == istr2[1]));

	char trans = (size == istr1[0]) ? 'N' : 'T';

	long lda = (size == istr1[0]) ? istr1[1] / size : istr1[0] / size;
	long ldb = ostr[1] / size;

	lda = MAX(1, lda);
	ldb = MAX(1, ldb);

	blas2_smatcopy(trans, dims[0], dims[1], iptr2, iptr1, lda, optr, ldb);
}


/**
 *
 * @param dims dimension
 * @param ostr must be of the form {1, dim[0]}
 * @param optr
 * @param istr must be of the form {1, dim[0]} or {dim[1]+x, 1}
 * @param iptr
 * @param val
 **/
void blas_smul_smatcopy(int N, const long dims[N], const long ostr[N], float* optr, const long istr[N], const float* iptr, float val)
{
	long size = 4;

	assert(2 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr, size));

	assert((size == ostr[0]) && (size * dims[0] == ostr[1]));
	assert(    ((size == istr[0]) && (size * dims[0] == istr[1]))
		|| ((size == istr[1]) && (size * dims[1] == istr[0])));

	char trans = (size == istr[0]) ? 'N' : 'T';

	long lda = (size == istr[0]) ? istr[1] / size : istr[0] / size;
	long ldb = ostr[1] / size;

	lda = MAX(1, lda);
	ldb = MAX(1, ldb);

	blas_smatcopy(trans, dims[0], dims[1], val, iptr, lda, optr, ldb);
}


/**
 *
 * @param dims dimension
 * @param ostr must be of the form {1, dim[0] + x}
 * @param optr
 * @param istr1 must be of the form {1, dim[0] + x}
 * @param iptr1
 * @param istr1 must be of the form {1+x, 0} or {0, 1+x}
 * @param iptr1
 **/
void blas_mul_sdgmm(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	long size = 4;

	assert(2 == N);
	assert(check_blas_strides(N, ostr, size));
	assert(check_blas_strides(N, istr1, size));
	assert(check_blas_strides(N, istr2, size));

	assert((size == ostr[0]) && (0 == ostr[1] % size) && (dims[0] * ostr[0] <= ostr[1]));
	assert((size == istr1[0]) && (0 == istr1[1] % size) && (dims[0] * istr1[0] <= istr1[1]));
	assert((0 == istr2[0] * istr2[1]));
	assert((0 < istr2[0]) || (0 < istr2[1]));

	long lda = istr1[1] / size;
	long ldc = ostr[1] / size;
	long incx = (0 == istr2[1]) ? istr2[0] / size : istr2[1] / size;
	lda = MAX(1, lda);
	ldc = MAX(1, ldc);

	blas_sdgmm(dims[0], dims[1], 0 == istr2[1], iptr1, lda, iptr2, incx, optr, ldc);
}


/**
 * sger for inner mul kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {1, dims[0]+x}
 * @param optr
 * @param istr1 must be of the form {1+x, 0}
 * @param iptr1
 * @param istr1 must be of the form {0, 1+x}
 * @param iptr1
 **/
void blas_mul_sger(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	long size = 4;

	md_clear2(N, dims, ostr, optr, size);

	blas_fmac_sger(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}


/**
 *
 * @param dims dimension
 * @param ostr must be of the form {1}
 * @param optr
 * @param istr1 must be of the form {1}
 * @param iptr1
 * @param istr1 must be of the form {0}
 * @param iptr1
 **/
void blas_mul_sscal(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	long size = 4;

	assert((ostr[0] == istr1[0]) && (0 == ostr[0] % size) && (0 == istr2[0]));
	assert(1 == N);

	if (optr != iptr1)
		md_copy2(N, dims, ostr, optr, istr1, iptr1, size);

	blas2_sscal(dims[0], iptr2, istr1[0] / size, optr);
}
