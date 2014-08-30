/* Copyright 2014 The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 *
 * Frank Ong, 2014
 * frankong@berkeley.edu
 *
 * Martin Uecker, 2014
 * uecker@eecs.berkeley.edu
 */

#include <math.h>
#include <complex.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "num/linop.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "nufft.h"

#include "grid.h"

#include "misc/mri.h"
#include "misc/mmio.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/vecops.h"

#include "iter/iter.h"


/**
 *
 * NUFFT internal data structure
 *
 */
struct nufft_data {
	void* cgconf;

	_Bool use_gpu;

	float scale;

	const complex float* traj;
	const complex float* pat;
	complex float* grid;
	complex float* roll;


	const struct operator_s* fft_op;
	const struct operator_s* ifft_op;

	complex float* fftmod;
	complex float* fftmodk;

	long ksp_dims[DIMS];
	long coilim_dims[DIMS];
	long coilim2_dims[DIMS];
	long img_dims[DIMS];
	long img2_dims[DIMS];


	long ksp_strs[DIMS];
	long coilim_strs[DIMS];
	long coilim2_strs[DIMS];
	long img_strs[DIMS];
	long img2_strs[DIMS];


	_Bool toeplitz;
	_Bool precond;

	complex float* psf;
	complex float* pre;
	complex float* tmp;

	unsigned int maxiter;

	float rho;
};


// Forward: from image to kspace
static struct nufft_data* nufft_create_data( const long ksp_dims[DIMS], const long coilim_dims[DIMS], const complex float* traj, const complex float* pat, _Bool toeplitz, _Bool precond, void* cgconf, _Bool use_gpu);
static void nufft_free_data( const void* data );
static void nufft_apply(const void* _data, complex float* dst, const complex float* src);
static void nufft_apply_adjoint(const void* _data, complex float* dst, const complex float* src);
static void nufft_apply_normal(const void* _data, complex float* dst, const complex float* src );
static void nufft_apply_pinverse(const void* _data, float lambda, complex float* dst, const complex float* src );


static void toeplitz_mult( const struct nufft_data* data, complex float* dst, const complex float* src );
static void precondition( const struct nufft_data* data, complex float* dst, const complex float* src );



/**
 *
 * NUFFT operator initialization
 *
 * @param ksp_dims      -     kspace dimension
 * @param coilim_dims   -     coil images dimension
 * @param traj          -     trajectory
 * @param pat           -     pattern / weights / density compensation factor
 * @param toeplitz      -     toeplitz boolean
 * @param precond       -     preconditioner boolean
 * @param cgconf        -     conjugate gradient configuration
 * @param use_gpu       -     use gpu boolean
 *
 */
struct linop_s* nufft_create( const long ksp_dims[DIMS], const long coilim_dims[DIMS], const complex float* traj, const complex float* pat, _Bool toeplitz, _Bool precond, void* cgconf, _Bool use_gpu)
{
	struct nufft_data* data = nufft_create_data( ksp_dims, coilim_dims, traj, pat, toeplitz, precond, cgconf, use_gpu);

	return linop_create(DIMS, data->ksp_dims, data->coilim_dims, data, nufft_apply, nufft_apply_adjoint, nufft_apply_normal, nufft_apply_pinverse, nufft_free_data);
}





static void triangular_window( const long dims[DIMS], complex float* triang )
{
	float scale = sqrtf(2);

	for( int i = 0; i < dims[0]; i++)
		for( int j = 0; j < dims[1]; j++)
			for( int k = 0; k < dims[2]; k++)
				triang[i + j*dims[0] + k*dims[0]*dims[1]] = 
					(1. - cabsf((float)i + 0.5 - (float)dims[0]/2.)/((float)dims[0] / 2.)) *
					(1. - cabsf((float)j + 0.5 - (float)dims[1]/2.)/((float)dims[1] / 2.)) *
					(1. - cabsf((float)k + 0.5 - (float)dims[2]/2.)/((float)dims[2] / 2.)) * scale;

}



static struct nufft_data* nufft_create_data( const long ksp_dims[DIMS], const long coilim_dims[DIMS], const complex float* traj, const complex float* pat, _Bool toeplitz, _Bool precond, void* cgconf, _Bool use_gpu)
{
	
	struct nufft_data* data = (struct nufft_data*) xmalloc( sizeof( struct nufft_data ) );

	data->cgconf = cgconf;
	data->traj = traj;
	data->pat = pat;
	md_copy_dims( DIMS, data->coilim_dims, coilim_dims );
	md_copy_dims( DIMS, data->ksp_dims, ksp_dims );
	md_select_dims( DIMS, ~COIL_FLAG, data->img_dims, coilim_dims );


	// Initialize oversampled data
	data->scale = 1.;
	md_copy_dims(DIMS, data->coilim2_dims, coilim_dims);

	for (int i = 0; i < 3; i++) {
		if (coilim_dims[i] > 1) {

			data->coilim2_dims[i] = 2 * coilim_dims[i];
			data->scale /= sqrtf( coilim_dims[i] ) * 2;
		}
	}
	data->grid = md_alloc(DIMS, data->coilim2_dims, sizeof(complex float));

	// get strides
	md_select_dims( DIMS, ~COIL_FLAG, data->img2_dims, data->coilim2_dims );
	md_calc_strides( DIMS, data->coilim_strs, data->coilim_dims, CFL_SIZE );
	md_calc_strides( DIMS, data->img_strs, data->img_dims, CFL_SIZE );
	md_calc_strides( DIMS, data->coilim2_strs, data->coilim2_dims, CFL_SIZE );
	md_calc_strides( DIMS, data->img2_strs, data->img2_dims, CFL_SIZE );

	// Initialize fft operators
	data->fft_op = fft_create(DIMS, data->coilim2_dims, FFT_FLAGS, data->grid, data->grid, false);
	data->ifft_op = fft_create(DIMS, data->coilim2_dims, FFT_FLAGS, data->grid, data->grid, true);

	// Create fftmod matrix
#ifdef USE_CUDA
	data->fftmod = (use_gpu ? md_alloc_gpu : md_alloc)(DIMS, data->img2_dims, CFL_SIZE);
#else
	data->fftmod = md_alloc(DIMS, data->img2_dims, CFL_SIZE);
#endif
	complex float one[1] = {1.};
	md_fill( DIMS, data->img2_dims, data->fftmod, one, CFL_SIZE );
	fftscale(DIMS, data->img2_dims, FFT_FLAGS, data->fftmod, data->fftmod);
	fftmod(DIMS, data->img2_dims, FFT_FLAGS, data->fftmod, data->fftmod);

	// Create fftmodk matrix
#ifdef USE_CUDA
	data->fftmodk = (use_gpu ? md_alloc_gpu : md_alloc)(DIMS, data->img2_dims, CFL_SIZE);
#else
	data->fftmodk = md_alloc(DIMS, data->img2_dims, CFL_SIZE);
#endif
	md_fill( DIMS, data->img2_dims, data->fftmodk, one, CFL_SIZE );
	fftmodk(DIMS, data->img2_dims, FFT_FLAGS, data->fftmodk, data->fftmodk);


	// initialize roll-off
	data->roll = md_alloc(DIMS, data->img2_dims, sizeof(complex float));
	rolloff_correction(data->img2_dims, data->roll);

	data->toeplitz = toeplitz;

	data->precond = precond;

	if (toeplitz)
	{
		// get traj * 2
		long traj_dims[DIMS];
		md_select_dims( DIMS, ~(COIL_FLAG|MAPS_FLAG), traj_dims, data->ksp_dims );
		traj_dims[0] = 3;

		complex float* traj2 = md_alloc_sameplace( DIMS, traj_dims, CFL_SIZE, data->traj );
		md_zsmul( DIMS, traj_dims, traj2, traj, 2 );

		// Get dims with 1 coil
		long ksp_dims1[DIMS];
		complex float scale2 = 1;
		md_select_dims( DIMS, ~(COIL_FLAG|MAPS_FLAG), ksp_dims1, data->ksp_dims );

		// get scale2
		for (int i = 0; i < 3; i++)
			if (data->img_dims[i] > 1)
				scale2  *= 2.  ;


		// Set tmp data for psf
		struct nufft_data* data2 = nufft_create_data( ksp_dims1, data->img2_dims, traj2, data->pat, false, false, NULL, use_gpu);

		// compute psf by gridding
		data->psf = md_alloc( DIMS, data->img2_dims , sizeof(complex float));
		complex float* delta = md_alloc( DIMS, ksp_dims1, sizeof(complex float));

		md_fill( DIMS, ksp_dims1, delta, &(scale2) , CFL_SIZE);

		// grid
		nufft_apply_adjoint( data2, data->psf, delta);

		md_free( delta );
		nufft_free_data( data2 );
		md_free( traj2 );


		if (precond)
		{
			complex float* triang = md_alloc_sameplace( DIMS, data->img2_dims, CFL_SIZE, data->traj );
			triangular_window( data->img2_dims, triang );
			md_zmul( DIMS, data->img2_dims, triang, triang, data->psf );

			// go to kspace
			md_zmul2( DIMS, data->img2_dims, data->img2_strs, triang, data->img2_strs, data->fftmod, data->img2_strs, triang );
			fft( DIMS, data->img2_dims, FFT_FLAGS, triang, triang );
			md_zmul2( DIMS, data->img2_dims, data->img2_strs, triang, data->img2_strs, data->fftmodk, data->img2_strs, triang );

			data->pre = md_alloc_sameplace( DIMS, data->img_dims, CFL_SIZE, data->traj );
			// subsample
			long sub2_strs[DIMS];
			md_copy_strides( DIMS, sub2_strs, data->img_strs );
			for( int i = 0; i < 3; i++)
				sub2_strs[i] <<= (i+1);
			
			md_copy2(DIMS, data->img_dims, data->img_strs, data->pre, sub2_strs, triang, CFL_SIZE);
			md_free( triang );

		}

		// go to kspace
		md_zmul2( DIMS, data->img2_dims, data->img2_strs, data->psf, data->img2_strs, data->fftmod, data->img2_strs, data->psf );
		fft( DIMS, data->img2_dims, FFT_FLAGS, data->psf, data->psf );
		md_zmul2( DIMS, data->img2_dims, data->img2_strs, data->psf, data->img2_strs, data->fftmodk, data->img2_strs, data->psf );


	} else
	{
		data->tmp = md_alloc_sameplace( DIMS, data->ksp_dims, CFL_SIZE, data->traj );
	}
	data->use_gpu = use_gpu;

	return data;
}


// Free nufft operator

static void nufft_free_data( const void* _data )
{
	struct nufft_data* data = (struct nufft_data*) _data;

	md_free(data->grid);
	md_free(data->roll);

	fft_free(data->fft_op);
	fft_free(data->ifft_op);

	md_free( data->fftmod );
	md_free( data->fftmodk );

	if (data->toeplitz)
		md_free(data->psf);
	else
		md_free(data->tmp);

	if (data->precond)
		md_free(data->pre);

	free(data);
}




// Forward: from image to kspace
static void nufft_apply(const void* _data, complex float* dst, const complex float* src)
{
	struct nufft_data* data = (struct nufft_data*) _data;

	// zero-pad
	md_clear(DIMS, data->coilim2_dims, data->grid, sizeof(complex float));
	md_resizec(DIMS, data->coilim2_dims, data->grid, data->coilim_dims, src, sizeof(complex float));

	// rolloff compensation
	md_zmulc2(DIMS, data->coilim2_dims, data->coilim2_strs, data->grid, data->coilim2_strs, data->grid, data->img2_strs, data->roll);

	// oversampled FFT
	md_zmul2( DIMS, data->coilim2_dims, data->coilim2_strs, data->grid, data->img2_strs, data->fftmod, data->coilim2_strs, data->grid );
	fft_exec( data->fft_op, data->grid, data->grid );
	md_zmul2( DIMS, data->coilim2_dims, data->coilim2_strs, data->grid, data->img2_strs, data->fftmodk, data->coilim2_strs, data->grid );

	// gridH
	float os = 2.;
	float width = 3.;
	gridH(  os, width, data->scale, data->traj, data->pat, data->ksp_dims, dst, data->coilim2_dims, data->grid);


}

// Adjoint: from kspace to image
static void nufft_apply_adjoint(const void* _data, complex float* dst, const complex float* src)
{
	const struct nufft_data* data = _data;

	// grid
	float os = 2.;
	float width = 3.;
	grid(  os, width, data->scale, data->traj, data->pat, data->coilim2_dims, data->grid, data->ksp_dims, src );

	// oversampled ifft
	md_zmul2( DIMS, data->coilim2_dims, data->coilim2_strs, data->grid, data->img2_strs, data->fftmodk, data->coilim2_strs, data->grid );
	fft_exec( data->ifft_op, data->grid, data->grid );
	md_zmul2( DIMS, data->coilim2_dims, data->coilim2_strs, data->grid, data->img2_strs, data->fftmod, data->coilim2_strs, data->grid );

	// rolloff compensation
	md_zmul2(DIMS, data->coilim2_dims, data->coilim2_strs, data->grid, data->coilim2_strs, data->grid, data->img2_strs, data->roll);

	// crop
	md_resizec(DIMS, data->coilim_dims, dst, data->coilim2_dims, data->grid, sizeof(complex float));
}



/** Normal: from image to image
* A^T A
*/
static void nufft_apply_normal(const void* _data, complex float* dst, const complex float* src )
{
	const struct nufft_data* data = _data;

	complex float* tmp = md_alloc_sameplace( DIMS, data->coilim_dims, CFL_SIZE, src );
	md_copy( DIMS, data->coilim_dims, tmp, src, CFL_SIZE );

	if (!data->toeplitz)
	{
		nufft_apply ( (const void*) data, data->tmp, src);
		nufft_apply_adjoint ( (const void*) data, dst, data->tmp);
	} else
		toeplitz_mult( data, dst, src );

	// l2 add
	md_zaxpy( DIMS, data->coilim_dims, dst, data->rho, tmp );
	md_free( tmp );

	// precond
	if (data->precond)
		precondition( data, dst, dst );
}

static void normaleq( const void* _data, complex float* dst, const complex float* src )
{
	const struct nufft_data* data = _data;

	nufft_apply_normal( data, dst, src );
}


/* 
 * x = (A^T A + r I)^-1  b 
 *
 *
 */

static void nufft_apply_pinverse(const void* _data, float rho, complex float* dst, const complex float* src )
{
	struct nufft_data* data = (struct nufft_data*) _data;

	// calculate parameters
	long size = md_calc_size( DIMS, data->coilim_dims ) * 2;

	complex float * rhs;

#ifdef USE_CUDA
	rhs = (data->use_gpu ? md_alloc_gpu : md_alloc)(DIMS, data->coilim_dims, CFL_SIZE);
#else
	rhs = md_alloc(DIMS, data->coilim_dims, CFL_SIZE);
#endif
	md_copy( DIMS, data->coilim_dims, rhs, src, CFL_SIZE );


	struct iter_conjgrad_conf* cgconf = data->cgconf;
	cgconf->l2lambda = 0;
	
	data->rho = rho;

	if (data->precond)
	{
		md_zsadd( DIMS, data->img_dims, data->pre, data->pre, rho);

		// precond
		precondition( data, rhs, src );
	}

	const struct operator_s* normaleq_op = operator_create(DIMS, data->coilim_dims, data->coilim_dims, (void*)data, normaleq, NULL);

	iter_conjgrad( cgconf, normaleq_op, NULL, size, (float*) dst, (float*) rhs, NULL, NULL, NULL );


	if (data->precond)
		md_zsadd( DIMS, data->img_dims, data->pre, data->pre, -rho);


	md_free( rhs );
}



/**
 * Estimate image dimensions from trajectory
 */
void estimate_im_dims( long dims[DIMS], long tdims[2], complex float* traj )
{
	float max_dims[3] = {0.,0.,0.};
	float min_dims[3] = {0.,0.,0.};

	for (long i = 0; i < tdims[1]; i++)
	{
		for( int j = 0; j < 3; j++)
		{
			max_dims[j] = MAX(  crealf(traj[ j + tdims[0] * i]) , max_dims[j]);
			min_dims[j] = MIN(  crealf(traj[ j + tdims[0] * i]) , min_dims[j]);
		}
	}

	for (int j = 0; j < 3; j++)
	{
		float d = max_dims[j] - min_dims[j] + 1;
		if ( d != 1 )
		{
			dims[j] = 2 * (long) ( (d + 0.5)/2);
		}
	}
}


static void precondition( const struct nufft_data* data, complex float* dst, const complex float* src )
{	
	md_zmul2( DIMS, data->coilim_dims, data->coilim_strs, dst, data->img2_strs, data->fftmod, data->coilim_strs, src );
	fft( DIMS, data->coilim_dims, FFT_FLAGS, dst, dst );

	md_zdiv2( DIMS, data->coilim_dims, data->coilim_strs, dst, data->coilim_strs, dst, data->img_strs, data->pre );

	ifft( DIMS, data->coilim_dims, FFT_FLAGS, dst, dst );
	md_zmul2( DIMS, data->coilim_dims, data->coilim_strs, dst, data->img2_strs, data->fftmodk, data->coilim_strs, dst );
}



static void toeplitz_mult( const struct nufft_data* data, complex float* dst, const complex float* src )
{
	// zeropad
	md_resizec( DIMS, data->coilim2_dims, data->grid, data->coilim_dims, src, sizeof(complex float));

	// oversampled FFT
	md_zmul2( DIMS, data->coilim2_dims, data->coilim2_strs, data->grid, data->img2_strs, data->fftmod, data->coilim2_strs, data->grid );
	fft_exec( data->fft_op, data->grid, data->grid );

	// multiply psf
	md_zmul2( DIMS, data->coilim2_dims, data->coilim2_strs, data->grid, data->coilim2_strs, data->grid, data->img2_strs, data->psf );

	// oversampled IFFT
	fft_exec( data->ifft_op, data->grid, data->grid );
	md_zmul2( DIMS, data->coilim2_dims, data->coilim2_strs, data->grid, data->img2_strs, data->fftmod, data->coilim2_strs, data->grid );

	// crop
	md_resizec( DIMS, data->coilim_dims, dst, data->coilim2_dims, data->grid, sizeof(complex float));
}

#if 0


void toeplitz_mult_prune( const struct nufft_data* data, complex float* dst, complex float* src )
{

	long img_size = md_calc_size( DIMS, data->img_dims );

	md_clear( DIMS, data->coilim_dims, dst, CFL_SIZE );

	for( int i = 0; i < data->fft_channels; i++)
	{
		md_zmul2( DIMS, data->coilim_dims, data->coilim_strs, data->tmp, data->coilim_strs, src, data->img_strs, data->linphase + i * img_size );

		fft_exec( data->fft_op, data->tmp , data->tmp );

		md_zmul2( DIMS, data->coilim_dims, data->coilim_strs, data->tmp, data->coilim_strs, data->tmp, data->img_strs, data->psf );

		fft_exec( data->ifft_op, data->tmp , data->tmp );

		md_zfmacc( DIMS, data->coilim_dims, data->coilim_strs, dst, data->coilim_strs, data->tmp, data->img_strs, data->linphase + i * img_size );
	}
}



static void linear_phase( const long dims[DIMS], float shifts[3], complex float* linphase )
{
	for( int i = 0; i < dims[0]; i++)
		for( int j = 0; j < dims[1]; j++)
			for( int k = 0; k < dims[2]; k++)
				linphase[i + j*dims[0] + k*dims[0]*dims[1]] = 
					cexpf( - I * 2. * M_PI / (float) (dims[0]*dims[1]*dims[2]) * ( shifts[0] * i + shifts[1] * j + shifts[2] *k ) );

}

int fft_channels = 1;
for( int i = 0; i < 3; i++)
	if (coilim_dims[i] > 1)
		fft_channels*=2;

long img_size = md_calc_size( DIMS, data->img_dims );
long linphase_dims[DIMS];
md_copy_dims( DIMS, linphase_dims, img_dims );
linphase_dims[3] = fft_channels;

linphase = md_alloc( DIMS, linphase_dims, CFL_SIZE );

for( int i = 0; i < 8; i++)
{
	float shifts[3];

	for( int j = 0; j < 3; j++)
		if ( i | (1 << j) )
			shifts[j] = 1.;

	linear_phase( data->img_dims, shifts, linphase + img_size );
}
#endif
