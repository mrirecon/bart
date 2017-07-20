/* Copyright 2015. The Regents of the University of California.
 * Copyright 2016-2017. University of Oxford.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2016-2017 Sofia Dimoudi <sofia.dimoudi@cardiov.ox.ac.uk>
 */

#include <assert.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <strings.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/loop.h"

#include "misc/misc.h"

#include "filter.h"



static void zabs(long N, float* dst, const complex float* src)
{
	for (long i = 0; i < N; i++)
		dst[i] = cabsf(src[i]);
}

static int cmp_float(const void* a, const void* b)
{
	return (*(float*)a - *(float*)b > 0.) ? 1. : -1.;
}

static int cmp_complex_float(const void* a, const void* b) // gives sign for 0. (not 0)
{
	return (cabsf(*(complex float*)a) - cabsf(*(complex float*)b) > 0.) ? 1. : -1.;
}

static void sort_floats(int N, float ar[N])
{
	qsort((void*)ar, N, sizeof(float), cmp_float);
}

static void sort_complex_floats(int N, complex float ar[N])
{
	qsort((void*)ar, N, sizeof(complex float), cmp_complex_float);
}

/* Using dynamically allocated array (pointer) */
static void sort_complex_floats_dyn(int N, complex float* ar)
{
	qsort((void*)ar, N, sizeof(complex float), cmp_complex_float);
}

float median_float(int N, float ar[N])
{
	float tmp[N];
	memcpy(tmp, ar, N * sizeof(float));
	sort_floats(N, tmp);
	return (1 == N % 2) ? tmp[(N - 1) / 2] : ((tmp[(N - 1) / 2 + 0] + tmp[(N - 1) / 2 + 1]) / 2.);
}

complex float median_complex_float(int N, complex float ar[N])
{
	complex float tmp[N];
	memcpy(tmp, ar, N * sizeof(complex float));
	sort_complex_floats(N, tmp);
	return (1 == N % 2) ? tmp[(N - 1) / 2] : ((tmp[(N - 1) / 2 + 0] + tmp[(N - 1) / 2 + 1]) / 2.);
}


/**
 * Quickselect adapted from §8.5 in Numerical Recipes in C, 
 * The Art of Scientific Computing
 * Second Edition, William H. Press, 1992.
 */
static float quickselect(float *arr, unsigned int n, unsigned int k) {
  unsigned long i,ir,j,l,mid;
  float a;
   
  l=0;
  ir=n-1;
  for(;;) {
    if (ir <= l+1) { 
      if (ir == l+1 && arr[ir] > arr[l]) {
	SWAP(arr[l],arr[ir], float);
      }
      return arr[k];
    }
    else {
      mid=(l+ir) >> 1; 
      SWAP(arr[mid],arr[l+1], float);
      if (arr[l] < arr[ir]) {
	SWAP(arr[l],arr[ir], float);
      }
      if (arr[l+1] < arr[ir]) {
	SWAP(arr[l+1],arr[ir], float);
      }
      if (arr[l] < arr[l+1]) {
	SWAP(arr[l],arr[l+1], float);
      }
      i=l+1; 
      j=ir;
      a=arr[l+1];

      for (;;) { 
	do i++; while (arr[i] > a); 
	do j--; while (arr[j] < a); 
	if (j < i) break; 
	SWAP(arr[i],arr[j], float);
      } 
      arr[l+1]=arr[j]; 
      arr[j]=a;
      
      if (j >= k) ir=j-1; 
      if (j <= k) l=i;
    }
  }
}

static float quickselect_complex(complex float *arr, unsigned int n, unsigned int k) {
  unsigned long i,ir,j,l,mid;
  float a;
  complex float ca;
   
  l=0;
  ir=n-1;
  for(;;) {
    if (ir <= l+1) { 
      if (ir == l+1 && cabsf(arr[ir]) > cabsf(arr[l])) {
	SWAP(arr[l],arr[ir], complex float);
      }
      return cabsf(arr[k]);
    }
    else {
      mid=(l+ir) >> 1; 
      SWAP(arr[mid],arr[l+1], complex float);
      if (cabsf(arr[l]) < cabsf(arr[ir])) {
	SWAP(arr[l],arr[ir], complex float);
      }
      if (cabsf(arr[l+1]) < cabsf(arr[ir])) {
	SWAP(arr[l+1],arr[ir], complex float);
      }
      if (cabsf(arr[l]) < cabsf(arr[l+1])) {
	SWAP(arr[l],arr[l+1], complex float);
      }
      i=l+1; 
      j=ir;
      a=cabsf(arr[l+1]);
      ca = arr[l+1];
      for (;;) { 
	do i++; while (cabsf(arr[i]) > a); 
	do j--; while (cabsf(arr[j]) < a); 
	if (j < i) break; 
	SWAP(arr[i],arr[j], complex float);
      } 
      arr[l+1]=arr[j]; 
      arr[j]=ca;
      
      if (j >= k) ir=j-1; 
      if (j <= k) l=i;
    }
  }
}

/**
 * Return the absolute value of the kth largest array element
 * To be used for hard thresholding
 * using full sort
 *
 * @param N number of elements
 * @param k the sorted element index to pick
 * @param r the input complex array
 *
 */

float klargest_complex_sort( unsigned int N,  unsigned int k, const complex float* ar)
{
  
        complex float* tmp = (complex float*)malloc(N * sizeof(complex float));
	memcpy(tmp, ar, N * sizeof(complex float));

	
	
	sort_complex_floats_dyn(N, tmp);
	
	float thr = (N >= k) ? cabsf(tmp[N-k]) : 0.;

	free(tmp);

	return thr;

}

/**
 * Return the absolute value of the kth largest array element
 * To be used for hard thresholding
 * using partial sort (quickselect) on the absolute values of the complex array
 *
 * @param N number of elements
 * @param k the sorted element index to pick
 * @param r the input complex array
 *
 */
float klargest_complex_sort_part_selfloat(unsigned int N, unsigned int k, const complex float* ar)
{
  assert(k <= N);
  float* tmp =  (float*)malloc(N * sizeof(float));
  zabs(N, tmp, ar); 

	
	float thr = quickselect(tmp, N, k);

	free(tmp);

	return thr;

}

/**
 * Return the absolute value of the kth largest array element
 * To be used for hard thresholding
 * using partial sort (quickselect) on the complex array (copy of)
 *
 * @param N number of elements
 * @param k the sorted element index to pick
 * @param r the input complex array
 *
 */
float klargest_complex_sort_part_selcpx( unsigned int N,  unsigned int k, const complex float* ar)
{
  assert(k <= N);
  complex float* tmp =  (complex float*)malloc(N * sizeof(complex float));
  memcpy(tmp, ar, N * sizeof(complex float));
	
	float thr = quickselect_complex(tmp, N, k);

	free(tmp);

	return thr;

}

struct median_s {

	long length;
	long stride;
};

static void nary_medianz(void* _data, void* ptr[])
{
	struct median_s* data = (struct median_s*)_data;

	complex float tmp[data->length];

	for (long i = 0; i < data->length; i++)
		tmp[i] = *((complex float*)(ptr[1] + i * data->stride));

	*(complex float*)ptr[0] = median_complex_float(data->length, tmp);
}

void md_medianz2(int D, int M, long dim[D], long ostr[D], complex float* optr, long istr[D], complex float* iptr)
{
	assert(M < D);
	const long* nstr[2] = { ostr, istr };
	void* nptr[2] = { optr, iptr };

	struct median_s data = { dim[M], istr[M] };

	long dim2[D];
	for (int i = 0; i < D; i++)
		dim2[i] = dim[i];

	dim2[M] = 1;

	md_nary(2, D, dim2, nstr, nptr, (void*)&data, &nary_medianz);
}

void md_medianz(int D, int M, long dim[D], complex float* optr, complex float* iptr)
{
	assert(M < D);

	long dim2[D];
	for (int i = 0; i < D; i++)
		dim2[i] = dim[i];

	dim2[M] = 1;

	long istr[D];
	long ostr[D];

	md_calc_strides(D, istr, dim, 8);
	md_calc_strides(D, ostr, dim2, 8);

	md_medianz2(D, M, dim, ostr, optr, istr, iptr);
}




void centered_gradient(unsigned int N, const long dims[N], const complex float grad[N], complex float* out)
{
	md_zgradient(N, dims, out, grad);

	long dims0[N];
	md_singleton_dims(N, dims0);

	long strs0[N];
	md_calc_strides(N, strs0, dims0, CFL_SIZE);

	complex float cn = 0.;

	for (unsigned int n = 0; n < N; n++)
		 cn -= grad[n] * (float)dims[n] / 2.;

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	md_zadd2(N, dims, strs, out, strs, out, strs0, &cn);
}

void linear_phase(unsigned int N, const long dims[N], const float pos[N], complex float* out)
{
	complex float grad[N];

	for (unsigned int n = 0; n < N; n++)
		grad[n] = 2.i * M_PI * (float)(pos[n]) / ((float)dims[n]);

	centered_gradient(N, dims, grad, out);
	md_zmap(N, dims, out, out, cexpf);
}


void klaplace_scaled(unsigned int N, const long dims[N], unsigned int flags, const float sc[N], complex float* out)
{
	unsigned int flags2 = flags;

	complex float* tmp = md_alloc(N, dims, CFL_SIZE);

	md_clear(N, dims, out, CFL_SIZE);

	for (unsigned int i = 0; i < bitcount(flags); i++) {

		unsigned int lsb = ffs(flags2) - 1;
		flags2 = MD_CLEAR(flags2, lsb);

		complex float grad[N];
		for (unsigned int j = 0; j < N; j++)
			grad[j] = 0.;

		grad[lsb] = sc[lsb];
		centered_gradient(N, dims, grad, tmp);
		md_zspow(N, dims, tmp, tmp, 2.);
		md_zadd(N, dims, out, out, tmp);
	}

	md_free(tmp);
}


void klaplace(unsigned int N, const long dims[N], unsigned int flags, complex float* out)
{
	float sc[N];
	for (unsigned int j = 0; j < N; j++)
		sc[j] = 1. / (float)dims[j];

	klaplace_scaled(N, dims, flags, sc, out);
}
