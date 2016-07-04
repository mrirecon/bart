

#include <assert.h>

#include "num/fft.h"
#include "num/flpmath.h"

#include "na/na.h"

#include "math.h"


static void fft_dims_check(na dst, na src)
{
	assert(na_rank(dst) == na_rank(src));
//	assert(dst->iov.size == src->iov.size);
//	assert(dst->iov.size == sizeof(complex float));
}

void na_fft(unsigned int flags, na dst, na src)
{
	fft_dims_check(dst, src);
	fft2(na_rank(dst), *NA_DIMS(dst), flags, *NA_STRS(dst), na_ptr(dst), *NA_STRS(src), na_ptr(src));
}

void na_ifft(unsigned int flags, na dst, na src)
{
	fft_dims_check(dst, src);
	ifft2(na_rank(dst), *NA_DIMS(dst), flags, *NA_STRS(dst), na_ptr(dst), *NA_STRS(src), na_ptr(src));
}

void na_fftc(unsigned int flags, na dst, na src)
{
	fft_dims_check(dst, src);
	fftc2(na_rank(dst), *NA_DIMS(dst), flags, *NA_STRS(dst), na_ptr(dst), *NA_STRS(src), na_ptr(src));
}

void na_ifftc(unsigned int flags, na dst, na src)
{
	fft_dims_check(dst, src);
	ifftc2(na_rank(dst), *NA_DIMS(dst), flags, *NA_STRS(dst), na_ptr(dst), *NA_STRS(src), na_ptr(src));
}

void na_fftu(unsigned int flags, na dst, na src)
{
	fft_dims_check(dst, src);
	fftu2(na_rank(dst), *NA_DIMS(dst), flags, *NA_STRS(dst), na_ptr(dst), *NA_STRS(src), na_ptr(src));
}

void na_ifftu(unsigned int flags, na dst, na src)
{
	fft_dims_check(dst, src);
	ifftu2(na_rank(dst), *NA_DIMS(dst), flags, *NA_STRS(dst), na_ptr(dst), *NA_STRS(src), na_ptr(src));
}

void na_fftuc(unsigned int flags, na dst, na src)
{
	fft_dims_check(dst, src);
	fftuc2(na_rank(dst), *NA_DIMS(dst), flags, *NA_STRS(dst), na_ptr(dst), *NA_STRS(src), na_ptr(src));
}

void na_ifftuc(unsigned int flags, na dst, na src)
{
	fft_dims_check(dst, src);
	ifftuc2(na_rank(dst), *NA_DIMS(dst), flags, *NA_STRS(dst), na_ptr(dst), *NA_STRS(src), na_ptr(src));
}


void na_fmac(na dst, na src1, na src2)
{
	unsigned int N = na_rank(dst);

	assert(N == na_rank(src1));
	assert(N == na_rank(src2));

	long dims[N];
	long (*dst_dims)[N] = NA_DIMS(dst);
	long (*src1_dims)[N] = NA_DIMS(src1);
	long (*src2_dims)[N] = NA_DIMS(src2);;
	
	for (unsigned int i = 0; i < N; i++) {

		if (1 == (*dst_dims)[i]) {

			if (1 == (*src1_dims)[i])
				dims[i] = (*src2_dims)[i];

			dims[i] = (*src1_dims)[i];

		} else {

			dims[i] = (*src2_dims)[i];
		}
	}

	md_fmac2(na_rank(dst), dims,
			*NA_STRS(dst), na_ptr(dst),
			*NA_STRS(src1), na_ptr(src1),
			*NA_STRS(src2), na_ptr(src2));
}


