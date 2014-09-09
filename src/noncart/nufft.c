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
#include "num/ops.h"
#include "num/iovec.h"

#include "linops/linop.h"
#include "linops/someops.h"

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

	int nshifts;
	float* shifts;
	complex float* linphase;
	

	const complex float* traj;
	const complex float* pat;
	complex float* grid;
	complex float* roll;

	complex float* tmp_img;
	complex float* tmp_ksp;

	complex float* psf;
	complex float* pre;

	float os;
	float width;
	double beta;

	const struct linop_s* fft_op;

	long ksp_dims[DIMS];
	long coilim_dims[DIMS];
	long img_dims[DIMS];


	long ksp_strs[DIMS];
	long coilim_strs[DIMS];
	long img_strs[DIMS];

	long img_size;

	_Bool toeplitz;
	_Bool precond;


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
static void linear_phaseX( const long dims[DIMS], const float shifts[3], complex float* linphase );
static void triangular_window( const long dims[DIMS], complex float* triang );
static void fill_psf( struct nufft_data* data );
static void fill_linphases( struct nufft_data* data );


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

	return linop_create(DIMS, data->ksp_dims, DIMS, data->coilim_dims,
		data, nufft_apply, nufft_apply_adjoint, nufft_apply_normal, nufft_apply_pinverse, nufft_free_data);
}





static struct nufft_data* nufft_create_data( const long ksp_dims[DIMS], const long coilim_dims[DIMS], const complex float* traj, const complex float* pat, _Bool toeplitz, _Bool precond, void* cgconf, _Bool use_gpu)
{
	
	struct nufft_data* data = (struct nufft_data*) xmalloc( sizeof( struct nufft_data ) );

	data->use_gpu = use_gpu;
	data->cgconf = cgconf;
	data->traj = traj;
	data->pat = pat;

	data->os = 2;
	data->width = 3.;
	data->beta = M_PI * sqrt( pow( (data->width * 2. / data->os ) * (data->os - 0.5 ), 2. ) - 0.8 );

	// get dims

	md_copy_dims( DIMS, data->coilim_dims, coilim_dims );
	md_copy_dims( DIMS, data->ksp_dims, ksp_dims );
	md_select_dims( DIMS, ~COIL_FLAG, data->img_dims, coilim_dims );

	// get strides

	md_calc_strides( DIMS, data->coilim_strs, data->coilim_dims, CFL_SIZE );
	md_calc_strides( DIMS, data->img_strs, data->img_dims, CFL_SIZE );
	data->img_size = md_calc_size( DIMS, data->img_dims );



	// initialize tmp_memory

	data->tmp_img = md_alloc( DIMS, data->coilim_dims, CFL_SIZE );
	data->tmp_ksp = md_alloc( DIMS, data->ksp_dims, CFL_SIZE );

	// initialize grid

	data->scale = 1.;
	data->grid = md_alloc(DIMS, data->coilim_dims, sizeof(complex float));

	// get linear phases

	data->nshifts = 1;
	for( int i = 0; i < 3; i++)
		if (data->img_dims[i] > 1)
		{
			data->nshifts*=2;
			data->scale /= 2;
		}

	long linphase_dims[DIMS];
	md_select_dims( DIMS, FFT_FLAGS, linphase_dims, data->img_dims );
	linphase_dims[3] = data->nshifts;
	data->linphase = md_alloc( DIMS, linphase_dims, CFL_SIZE );
	data->shifts = md_alloc( 1, MD_DIMS( 8 * 3 ), FL_SIZE );

	fill_linphases( data );

	// initialize fft operators

	data->fft_op = linop_fftc_create( DIMS, data->coilim_dims, FFT_FLAGS, use_gpu );

	// initialize roll-off

	data->roll = md_alloc(DIMS, data->img_dims, sizeof(complex float));
	rolloff_correction(data->os, data->width, data->img_dims, data->roll);

	data->toeplitz = toeplitz;
	data->precond = precond && toeplitz;

	if (toeplitz)
	{
		data->psf = md_alloc( DIMS, linphase_dims, CFL_SIZE );
		fill_psf( data );
	}


	return data;
}



static void linear_phaseX(const long dims[DIMS], const float shifts[3], complex float* linphase)
{
#if 0
	// slower... does it matter?
	float pos[DIMS] = { shifts[0], shifts[1], shifts[2], [3 ... DIMS - 1] = 0. };
	linear_phase(DIMS, dims, pos, linphase);
#else
#pragma omp parallel for
	for( int k = 0; k < dims[2]; k++)
		for( int j = 0; j < dims[1]; j++)
			for( int i = 0; i < dims[0]; i++)
				linphase[i + j*dims[0] + k*dims[0]*dims[1]] = 
					cexpf( I *  2. * M_PI * ( shifts[0] *  ((float ) i - (float ) dims[0] / 2.) / (float) dims[0] 
								  + shifts[1] * ((float ) j - (float) dims[1] / 2.) / (float) dims[1]
								  + shifts[2] * ((float ) k - (float) dims[2] / 2.) / (float) dims[2] ) );
#endif
}

static void fill_linphases( struct nufft_data* data )
{

	int s = 0;
	for( int i = 0; i < 8; i++)
	{
		float ishifts[3] = {0., 0., 0.};
		bool skip = false;

		for( int j = 0; j < 3; j++)
		{
			if ( i & (1 << j) )
			{
				if ( data->img_dims[j] == 1)
					skip = true;
				else
					ishifts[j] = 0.5;
			}
		}

		if (!skip)
		{
			data->shifts[ 3 * s + 0 ] = ishifts[0];
			data->shifts[ 3 * s + 1 ] = ishifts[1];
			data->shifts[ 3 * s + 2 ] = ishifts[2];
			linear_phaseX( data->img_dims, ishifts, data->linphase + s * data->img_size );
			s++;
		}
	}
}


static void fill_psf(struct nufft_data* data)
{      
	// get traj * 2
	long traj_dims[DIMS];
	md_select_dims( DIMS, 2 , traj_dims, data->ksp_dims );
	traj_dims[0] = 3;

	complex float* traj2 = md_alloc_sameplace( DIMS, traj_dims, CFL_SIZE, data->traj );
	md_zsmul( DIMS, traj_dims, traj2, data->traj, 2. );


	// Get dims with 1 coil
	long ksp_dims1[DIMS];
	md_select_dims( DIMS, 1 | 2, ksp_dims1, data->ksp_dims );

	// get img2_dims, scale2
	long img2_dims[DIMS];
	long img2_strs[DIMS];
	md_copy_dims( DIMS, img2_dims, data->img_dims );
	for (int i = 0; i < 3; i++) {
		if (data->img_dims[i] > 1)
			img2_dims[i] = 2 * data->img_dims[i];
	}
	md_calc_strides( DIMS, img2_strs, img2_dims, CFL_SIZE );
	complex float* psf = md_alloc( DIMS, img2_dims, CFL_SIZE );


	// set tmp data for psf
	struct nufft_data* data2 = nufft_create_data( ksp_dims1, img2_dims, traj2, data->pat, false, false, NULL, data->use_gpu);


	// get ones
	complex float* ones = md_alloc( DIMS, ksp_dims1, sizeof(complex float));
	complex float one = 1;
	md_fill( DIMS, ksp_dims1, ones, &(one) , CFL_SIZE);

	// grid ones to get psf
	nufft_apply_adjoint( data2, psf, ones);

	md_free( ones );
	nufft_free_data( data2 );
	md_free( traj2 );


	// compensate for linear phase
	complex float* linphase = md_alloc( DIMS, img2_dims, CFL_SIZE );
	float shifts[3] = {0., 0., 0.};
	for( int i = 0; i < 3; i++)
		if (data->img_dims[i] > 1)
			shifts[i] = 1.;
	linear_phaseX( img2_dims, shifts, linphase );
	md_zmul( DIMS, img2_dims, psf, linphase, psf );
	md_free( linphase );

	// subsample strides
	long sub2_strs[DIMS];
	md_copy_strides( DIMS, sub2_strs, data->img_strs );
	for( int i = 0; i < 3; i++)
		sub2_strs[i] <<= (i+1);


	// get circular preconditioner
	if (data->precond)
	{

		complex float* triang = md_alloc_sameplace( DIMS, img2_dims, CFL_SIZE, data->traj );
		triangular_window( img2_dims, triang );
		md_zmul( DIMS, img2_dims, triang, triang, psf );


		// go to kspace
		fftuc( DIMS, img2_dims, FFT_FLAGS, triang, triang );

		data->pre = md_alloc_sameplace( DIMS, data->img_dims, CFL_SIZE, data->traj );
			
		// subsample
		md_copy2(DIMS, data->img_dims, data->img_strs, data->pre, sub2_strs, triang, CFL_SIZE);
		md_free( triang );

	}


	// get psf kspace
	fftuc( DIMS, img2_dims, FFT_FLAGS, psf, psf );

	float max = 0.;
	for( int i = 0; i < md_calc_size( DIMS, img2_dims ); i++)
		max = MAX( max, cabsf( psf[i] ) );

	md_zsmul( DIMS, img2_dims, psf, psf, 1. / max );
	data->scale = data->scale / sqrtf( max );

	if (data->precond)
		md_zsmul( DIMS, data->img_dims, data->pre, data->pre, 1. / max );


	for ( int s = 0; s < data->nshifts; s++ )
	{
		complex float* psf_s = data->psf + s * data->img_size;
		float* shifts_s = data->shifts + 3 * s;
		long off = ( (1. - 2. * shifts_s[0]) * img2_strs[0] + (1. - 2. * shifts_s[1]) * img2_strs[1] + (1. - 2. * shifts_s[2]) * img2_strs[2] ) / CFL_SIZE;
		

		md_copy2(DIMS, data->img_dims, data->img_strs, psf_s, sub2_strs, psf + off, CFL_SIZE);
	}

	md_free( psf );

}





// Free nufft operator

static void nufft_free_data( const void* _data )
{
	struct nufft_data* data = (struct nufft_data*) _data;

	md_free(data->grid);
	md_free(data->roll);
	md_free(data->linphase);
	md_free(data->tmp_img);
	md_free(data->tmp_ksp);

	linop_free( data->fft_op );

	if (data->toeplitz)
		md_free(data->psf);

	if (data->precond)
		md_free(data->pre);

	free(data);
}




// Forward: from image to kspace
static void nufft_apply(const void* _data, complex float* dst, const complex float* src)
{
	struct nufft_data* data = (struct nufft_data*) _data;

	// clear output
	md_clear( DIMS, data->ksp_dims, dst, CFL_SIZE );

	// rolloff compensation
	md_zmulc2(DIMS, data->coilim_dims, data->coilim_strs, data->tmp_img, data->coilim_strs, src, data->img_strs, data->roll);


	for (int s = 0; s < data->nshifts; s++)
	{
		complex float* grid_s = data->grid;
		complex float* linphase_s = data->linphase + s * data->img_size;
		float* shifts_s = data->shifts + 3 * s;
		
		// linphase shift
		md_zmul2(DIMS, data->coilim_dims, data->coilim_strs, grid_s, data->coilim_strs, data->tmp_img, data->img_strs, linphase_s);

		// fft
		linop_forward( data->fft_op, DIMS, data->coilim_dims, grid_s, DIMS, data->coilim_dims, grid_s );

		// gridH
		gridH(  1., data->width / 2., data->beta, data->scale, shifts_s, data->traj, data->pat, data->ksp_dims, dst, data->coilim_dims, grid_s);
	}


}

// Adjoint: from kspace to image
static void nufft_apply_adjoint(const void* _data, complex float* dst, const complex float* src)
{
	const struct nufft_data* data = _data;

	// clear output
	md_clear( DIMS, data->coilim_dims, dst, CFL_SIZE);

	for ( int s = 0; s < data->nshifts; s++)
	{
		complex float* grid_s = data->grid;
		complex float* linphase_s = data->linphase + s * data->img_size;
		float* shifts_s = data->shifts + 3 * s;
		md_clear( DIMS, data->coilim_dims, grid_s, CFL_SIZE );

		// grid
		grid(  1., data->width / 2., data->beta, data->scale, shifts_s, data->traj,  data->pat, data->coilim_dims, grid_s, data->ksp_dims, src );

		// ifft
		linop_adjoint( data->fft_op, DIMS, data->coilim_dims, grid_s, DIMS, data->coilim_dims, grid_s );

		// linphase unshift
		md_zfmacc2(DIMS, data->coilim_dims, data->coilim_strs, dst, data->coilim_strs, grid_s, data->img_strs, linphase_s);
	}

	// rolloff compensation
	md_zmul2(DIMS, data->coilim_dims, data->coilim_strs, dst, data->coilim_strs, dst, data->img_strs, data->roll);
}



/** Normal: from image to image
* A^T A
*/
static void nufft_apply_normal(const void* _data, complex float* dst, const complex float* src )
{
	const struct nufft_data* data = _data;

	md_copy( DIMS, data->coilim_dims, data->tmp_img, src, CFL_SIZE );

	if (!data->toeplitz)
	{
		nufft_apply ( (const void*) data, data->tmp_ksp, src);
		nufft_apply_adjoint ( (const void*) data, dst, data->tmp_ksp);
	} else
		toeplitz_mult( data, dst, src );

	// l2 add
	md_zaxpy( DIMS, data->coilim_dims, dst, data->rho, data->tmp_img );

	// precond
	if (data->precond)
		precondition( data, dst, dst );
}


static void toeplitz_mult( const struct nufft_data* data, complex float* dst, const complex float* src )
{

	md_copy( DIMS, data->coilim_dims, data->tmp_img, src, CFL_SIZE ); 
	md_clear( DIMS, data->coilim_dims, dst, CFL_SIZE ); 
	for (int s = 0; s < data->nshifts; s++)
	{
		complex float* grid_s = data->grid;
		complex float* linphase_s = data->linphase + s * data->img_size;
		complex float* psf_s = data->psf + s * data->img_size;
		md_clear( DIMS, data->coilim_dims, grid_s, CFL_SIZE );

		// linphase shift
		md_zmul2(DIMS, data->coilim_dims, data->coilim_strs, grid_s, data->coilim_strs, data->tmp_img, data->img_strs, linphase_s);

		// fft
		linop_forward( data->fft_op, DIMS, data->coilim_dims, grid_s, DIMS, data->coilim_dims, grid_s );

		// multiply psf
		md_zmul2( DIMS, data->coilim_dims, data->coilim_strs, grid_s, data->coilim_strs, grid_s, data->img_strs, psf_s );
		

		// ifft
		linop_adjoint( data->fft_op, DIMS, data->coilim_dims, grid_s, DIMS, data->coilim_dims, grid_s );

		// linphase unshift
		md_zfmacc2(DIMS, data->coilim_dims, data->coilim_strs, dst, data->coilim_strs, grid_s, data->img_strs, linphase_s);
	}
}


static void normaleq( const void* _data, unsigned int N, void* args[N] )
{
	const struct nufft_data* data = _data;
	assert(2 == N);
	nufft_apply_normal( data, (complex float*)args[0], (const complex float*)args[1] );
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

	const struct operator_s* normaleq_op = operator_create(DIMS, data->coilim_dims, DIMS, data->coilim_dims, (void*)data, normaleq, NULL);

	iter_conjgrad( cgconf, normaleq_op, NULL, size, (float*) dst, (float*) rhs, NULL, NULL, NULL );


	if (data->precond)
		md_zsadd( DIMS, data->img_dims, data->pre, data->pre, -rho);


	data->rho = 0.;
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
	linop_forward( data->fft_op, DIMS, data->coilim_dims, dst, DIMS, data->coilim_dims, src );

	md_zdiv2( DIMS, data->coilim_dims, data->coilim_strs, dst, data->coilim_strs, dst, data->img_strs, data->pre );

	linop_adjoint( data->fft_op, DIMS, data->coilim_dims, dst, DIMS, data->coilim_dims, dst );
}




static void triangular_window( const long dims[DIMS], complex float* triang )
{

#pragma omp parallel for
	for( int k = 0; k < dims[2]; k++)
		for( int j = 0; j < dims[1]; j++)
			for( int i = 0; i < dims[0]; i++)
				triang[i + j*dims[0] + k*dims[0]*dims[1]] = 
					(1. - abs((float)i - (float)dims[0]/2.)/((float)dims[0] / 2.)) *
					(1. - abs((float)j - (float)dims[1]/2.)/((float)dims[1] / 2.)) *
					(1. - abs((float)k - (float)dims[2]/2.)/((float)dims[2] / 2.));

}
