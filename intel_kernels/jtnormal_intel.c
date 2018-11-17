#include <complex.h>
#include <fftw/fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include <omp.h>
#include <xmmintrin.h>
#include <mkl.h>

#define BLK 4
inline void TransposeBLKxBLK(complex float * __restrict__ A, complex float * __restrict__ B) {
int i, j;
for (i = 0; i < BLK; i++)
for (j = 0; j < BLK; j++)
B[i*BLK + j] = A[j*BLK + i];
}

inline void TransposePanel(complex float * __restrict__ cor_out, 
                           complex float * __restrict__ cor_out2, 
			   int _p, 
			   int tid,
			   int dim0, 
			   int dim1)
{
	int nblk0 = dim0 / BLK;
	int nblk1 = _p / BLK;
	for(int cc = 0 ; cc < nblk1 ; cc++)
	{
	for(int bb = 0 ; bb < nblk0 ; bb++)
	{
	  int mine = (bb+tid)%nblk0;
	  int b = mine * BLK;
	    int c = cc * BLK;
	    complex float buf1[BLK*BLK];
	    complex float buf2[BLK*BLK];
	    for(int i = 0 ; i < BLK ; i++)
	    {
	      #pragma simd
	      for(int j = 0 ; j < BLK ; j++)
	      {
	        buf1[j + i*BLK] = cor_out[b + j + (c+i)*dim0];
	      }
	    }
	    TransposeBLKxBLK(buf1, buf2);
	    for(int i = 0 ; i < BLK ; i++)
	    {
	      #pragma simd
	      for(int j = 0 ; j < BLK ; j++)
	      {
	        cor_out2[c + j + (b+i)*dim1] = buf2[j + i*BLK];
	      }
	    }
	  }
	}
	for(int cc = nblk1*BLK ; cc < _p ; cc++)
	{
	  #pragma simd
	  for(int i = 0 ; i < dim0 ; i++)
	  {
	    cor_out2[cc + i*dim1] = cor_out[i + cc*dim0];
	  }
	}
    // Do extra columns
    for(int bb = nblk0*BLK ; bb < dim0 ; bb++)
    {
      for(int cc = 0 ; cc < _p ; cc++)
      {
	    cor_out2[cc + bb*dim1] = cor_out[bb + cc*dim0];
      }
    }
}


void jtmodel_normal_benchmark_fast_parallel(
    const complex float * __restrict__ sens, const float * __restrict__ stkern_mat, 
    complex float * dst, const complex float * src,
    const unsigned long dim0,
    const unsigned long dim1,
    const unsigned long ncoils,
    const unsigned long nmaps,
    const unsigned long ncfimg,
    DFTI_DESCRIPTOR_HANDLE plan1d_0, DFTI_DESCRIPTOR_HANDLE plan1d_1,
    complex float * cfksp3,
    complex float * cfksp4) {

	struct timeval start, end;
	int nthr = omp_get_max_threads();
	int P = (dim1 + nthr-1) / nthr;
	int P0 = (dim0 + nthr-1) / nthr;
	float sc = 1.0 / sqrt((double)dim0 * dim1);

	assert(nmaps == 1 || nmaps == 2);

	for(int coil = 0 ; coil < ncoils ; coil++)
	{
#pragma omp parallel num_threads(nthr)
		{
			int tid = omp_get_thread_num();
			int row_start = tid * P;
			int row_end = (tid+1) * P;
			if(row_end > dim1) row_end = dim1;

			for(int cfimg = 0 ; cfimg < ncfimg ; cfimg++)
			{
				for(int row = row_start ; row < row_end; row++)
				{
					const complex float *map0 = sens + coil * dim1 * dim0 + dim0 * row;
					const complex float *map1 = NULL;
					if (nmaps == 2)
						map1 = sens + coil * dim1 * dim0 + ncoils * dim0 * dim1 + dim0 * row;
					const  complex float *cfimg0 = src + cfimg * dim0 * dim1 * nmaps + dim0 * row;
					const complex float *cfimg1 = NULL;
					if (nmaps == 2)
						cfimg1 = src + dim0 * dim1 + cfimg * dim0 * dim1 * nmaps + dim0 * row;
					complex float *cor_out =
						cfksp3 + cfimg * dim1 * dim0 + dim0 * row;

#pragma simd
					for (int i = 0; i < dim0; i++) {
						if (nmaps == 2)
							cor_out[i] = (map0[i] * cfimg0[i] + map1[i] * cfimg1[i]) * sc;
						else
							cor_out[i] = (map0[i] * cfimg0[i]) * sc;
					}
					DftiComputeForward(plan1d_0, cor_out, cor_out);
				}

				complex float *cor_out =
					cfksp3 + cfimg * dim1 * dim0 + dim0 * row_start;
				complex float *cor_out2 =
					cfksp4 + cfimg * dim1 * dim0 + row_start;

				TransposePanel(cor_out, cor_out2, row_end-row_start, tid, dim0, dim1);
			}
		}

#pragma omp parallel num_threads(nthr)
		{
			int tid = omp_get_thread_num();
			int row_start = tid * P0;
			int row_end = (tid+1) * P0;
			if(row_end > dim0) row_end = dim0;
			complex float * stkern_tmp = (complex float*) malloc(dim1 * ncfimg * sizeof(complex float));
			for (int row = row_start ; row < row_end ; row++) {
				for(int cfimg = 0 ; cfimg < ncfimg ; cfimg++)
				{
					complex float *cor_out =
						cfksp4 + cfimg * dim1 * dim0 + dim1 * row;
					DftiComputeForward(plan1d_1, cor_out, cor_out);
				}
				for(int cfimg_i = 0 ; cfimg_i < ncfimg ; cfimg_i++)
				{
					complex float *tmp = stkern_tmp + cfimg_i * dim1;
					for (int cfimg_j = 0; cfimg_j < ncfimg; cfimg_j++) {
						complex float *cfimg_in = cfksp4 + 
							cfimg_j * dim0 * dim1 + row * dim1;
						const float *mat = (cfimg_i > cfimg_j) ? stkern_mat + cfimg_i * dim1 * dim0 + cfimg_j * dim1 * dim0 * ncfimg + row * dim1 :
							stkern_mat + cfimg_j * dim1 * dim0 + cfimg_i * dim1 * dim0 * ncfimg + row * dim1;
						if(cfimg_j == 0)
						{
#pragma simd
							for (int pix = 0; pix < dim1; pix++) {
								tmp[pix] = (cfimg_in[pix] * mat[pix]);
							}
						}
						else
						{
#pragma simd
							for (int pix = 0; pix < dim1; pix++) {
								tmp[pix] += (cfimg_in[pix] * mat[pix]);
							}
						}
					}
					DftiComputeBackward(plan1d_1, tmp, tmp);
				}
				for(int cfimg_i = 0 ; cfimg_i < ncfimg ; cfimg_i++)
				{
					complex float *cfimg_in = cfksp4 + 
						cfimg_i * dim0 * dim1 + row * dim1;
#pragma simd
					for (int pix = 0; pix < dim1; pix++) {
						cfimg_in[pix] = stkern_tmp[pix + cfimg_i*dim1];
					}
				}
			}
			free(stkern_tmp);

			for(int cfimg_i = 0 ; cfimg_i < ncfimg ; cfimg_i++)
			{
				complex float *cfimg_in = cfksp4 + 
					cfimg_i * dim0 * dim1 + row_start * dim1;
				complex float *cfimg_in2 = cfksp3 + 
					cfimg_i * dim0 * dim1 + row_start;
				TransposePanel(cfimg_in, cfimg_in2, row_end-row_start, tid, dim1, dim0);
			}
		}

#pragma omp parallel num_threads(nthr)
		{
			int tid = omp_get_thread_num();
			int row_start = tid * P;
			int row_end = (tid+1) * P;
			if(row_end > dim1) row_end = dim1;
			for (int row = row_start ; row < row_end ; row++) {
				for (int cfimg = 0; cfimg < ncfimg; cfimg++) {
					const complex float *map0 = sens + coil*dim1*dim0 + row * dim0;
					const complex float *map1 = NULL;
					if (nmaps == 2)
						map1 = sens + coil*dim1*dim0 + ncoils *dim0 * dim1 + row * dim0;
					complex float *cor0 = dst + cfimg *dim1*dim0*nmaps + row * dim0;
					complex float* cor1 = NULL;
					if (nmaps == 2)
						cor1 = dst + dim1*dim0+cfimg*dim1*dim0*nmaps + row * dim0;
					complex float *cfimg_in = cfksp3 + cfimg*dim0*dim1 + row * dim0;
					DftiComputeBackward(plan1d_0, cfimg_in, cfimg_in);
					if(coil == 0)
					{
#pragma simd
						for (int i = 0; i < dim0; i++) {
							cor0[i] = 0;
							if (nmaps == 2)
								cor1[i] = 0;
						}
					}
#pragma simd
					for (int i = 0; i < dim0; i++) {
						float r0 = __real__ map0[i];
						float i0 = __imag__ map0[i];
						float r1 = 0;
						float i1 = 0;
						if (nmaps == 2) {

							r1 = __real__ map1[i];
							i1 = __imag__ map1[i];
						}
						float _r = __real__ cfimg_in[i];
						float _i = __imag__ cfimg_in[i];
						cor0[i] += ((r0 * _r + i0 * _i) + (r0 * _i - i0 * _r) * _Complex_I) * sc;
						if (nmaps == 2)
							cor1[i] += ((r1 * _r + i1 * _i) + (r1 * _i - i1 * _r) * _Complex_I) * sc;
					}
				}
			}
		}
	}
}

void jtmodel_adjoint_benchmark_fast_parallel(
    const complex float * __restrict__ sens, 
    complex float * dst, const complex float * src,
    const unsigned long dim0,
    const unsigned long dim1,
    const unsigned long ncoils,
    const unsigned long nmaps,
    const unsigned long ncfimg,
    DFTI_DESCRIPTOR_HANDLE plan2d,
    complex float * cfksp3)
{
	assert(nmaps == 1 || nmaps == 2);
	float sc = 1.0 / sqrt((double)dim0 * dim1);
	for(int coil = 0 ; coil < ncoils ; coil++)
	{
		const complex float * map0 = sens + coil * dim0 * dim1;
		const complex float * map1 = NULL;
		if (nmaps == 2)
			map1 = sens + coil * dim0 * dim1 + ncoils * dim0*dim1;

		for(int cfimg = 0 ; cfimg < ncfimg ; cfimg++)
		{
			complex float * ksp = (complex float*)src + coil*dim0*dim1 + cfimg*ncoils*dim0*dim1;

			DftiComputeBackward(plan2d, ksp, cfksp3);

			complex float * cor0 = dst + nmaps * cfimg * dim0 * dim1;
			complex float * cor1 = NULL;
			if (nmaps == 2)
				cor1 = dst + nmaps * cfimg * dim0 * dim1 + dim0*dim1;

			if(coil == 0)
			{
#pragma omp parallel for
#pragma simd
				for (int i = 0; i < dim0*dim1; i++) {
					cor0[i] = 0;
					if (nmaps == 2)
						cor1[i] = 0;
				}
			}
#pragma omp parallel for
#pragma simd
			for (int i = 0; i < dim0*dim1; i++) {
				float r0 = __real__ map0[i];
				float i0 = __imag__ map0[i];
				float r1 = 0.;
				float i1 = 0;
				if (nmaps == 2) {
					r1 = __real__ map1[i];
					i1 = __imag__ map1[i];
				}
				float _r = __real__ cfksp3[i];
				float _i = __imag__ cfksp3[i];
				cor0[i] += ((r0 * _r + i0 * _i) + (r0 * _i - i0 * _r) * _Complex_I) * sc;
				if (nmaps == 2)
					cor1[i] += ((r1 * _r + i1 * _i) + (r1 * _i - i1 * _r) * _Complex_I) * sc;
			}
		}
	}
}

