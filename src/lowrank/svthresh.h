/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */ 


#define GWIDTH( M, N, B) ( (sqrtf( M ) + sqrtf( N )) + sqrtf( logf( B * ((M > N) ? N : M )) ))

//#define GWIDTH( M, N, B) ( sqrtf( M ) + sqrtf( N ) )

//#define GWIDTH( M, N, B) sqrtf( ((M > N) ? M : N) )


#include <complex.h>

// Singular value thresholding for matrix
extern float svthresh(long M, long N, float lambda, complex float* dst, const complex float* src);

extern float svthresh2(long M, long N, float lambda, complex float* dst, const complex float* src, complex float* U, float* S, complex float* VT);

extern float svthresh_nomeanu(long M, long N, float lambda, complex float* dst, const complex float* src);

extern float svthresh_nomeanv(long M, long N, float lambda, complex float* dst, const complex float* src);


// Singular value analysis (maybe useful to help determining regularization parameter for min nuclear norm)
extern float nuclearnorm(long M, long N, const complex float* d);

extern float maxsingular(long M, long N, const complex float* d);



extern struct svthresh_blockproc_data* svthresh_blockproc_create(unsigned long mflags, float lambda, int remove_mean);
extern float svthresh_blockproc(const void* _data, const long blkdims[DIMS], complex float* dst, const complex float* src);


extern float nucnorm_blockproc(const void* _data, const long blkdims[DIMS], complex float* dst, const complex float* src);
