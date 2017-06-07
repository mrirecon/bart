/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014 Frank Ong <frankong@berkeley.edu>
 *
 */

#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "blockproc.h"


float lineproc2( unsigned int D,  const long dims[D], const long blkdims[D], const long line_dims[D], const void* data,
		 float (*op)(const void* data, const long blkdims[D], complex float* dst, const complex float* src), 
		 const long ostrs[D], complex float* dst, const long istrs[D], const complex float* src )
{
	// Get number of blocks per dimension
	long nblocks[D];
	long shifts[D];
	for (unsigned int i = 0; i < D; i++)
	{
		nblocks[i] = dims[i] - blkdims[i] + 1;
		shifts[i] = ( dims[i] - nblocks[i] * line_dims[i]) / 2;
	}
	long line_strs[D];
	md_calc_strides( D, line_strs, line_dims, CFL_SIZE );

	long numblocks = md_calc_size( D, nblocks );
	float info = 0;

	// Loop over blocks
	complex float* blk = md_alloc_sameplace(D, blkdims, sizeof( complex float ), src);
	complex float* line = md_alloc_sameplace(D, line_dims, sizeof( complex float ), src);
	for (long b = 0; b < numblocks; b++)
	{
		// Get block position and actual block size
		long blkpos[D];
		long linepos[D];

		long ind = b;
		for ( unsigned int i = 0; i < D; i++)
		{
			long blkind = ind % nblocks[i];
			blkpos[i] = blkind;
			linepos[i] = blkind + shifts[i];
			ind = (ind - blkind) / nblocks[i];
		}
		long blkstrs[D];
		md_calc_strides( D, blkstrs, blkdims, CFL_SIZE );

		// Extract block
		md_copy_block2( D, blkpos, blkdims, blkstrs, blk, dims, istrs, src, sizeof(complex float) );
		// Process block
		info += op( data, blkdims, line, blk );

		// Put back block
		md_copy_block2( D, linepos, dims, ostrs, dst, line_dims, line_strs, line, sizeof( complex float ));

	}
	md_free( blk );
	md_free( line );
	return info;
}


float lineproc( unsigned int D, const long dims[D], const long blkdims[D], const long line_dims[D], const void* data,
		 float (*op)(const void* data, const long blkdims[D], complex float* dst, const complex float* src), 
		 complex float* dst, const complex float* src )
{
	long strs[D];
	md_calc_strides( D, strs, dims, CFL_SIZE );

	return lineproc2( D, dims, blkdims, line_dims, data, op, strs, dst, strs, src );
}


float blockproc_shift_mult2( unsigned int D, const long dims[D], const long blkdims[D], const long shifts[D], const long mult[D], const void* data,
			float (*op)(const void* data, const long blkdims[D], complex float* dst, const complex float* src), 
			const long ostrs[D], complex float* dst, const long istrs[D], const complex float* src )
{
	float info = 0;
	long pos[D];

	for (unsigned int i = 0; i < D; i++) {	

		pos[i] = shifts[i];

		while (pos[i] < 0)
			pos[i] += dims[i];
	}

	unsigned int i = 0;
	while ((i < D) && (0 == pos[i]))
		i++;

	if (D == i) {

		info += blockproc2( D, dims, blkdims, data, op, ostrs, dst, istrs, src );
		return info;
	}

	long shift = pos[i];

	assert(shift != 0);

	long dim0[D];
	long dim1[D];
	long dim2[D];
	long dim3[D];

	md_copy_dims(D, dim0, dims);
	md_copy_dims(D, dim1, dims);
	md_copy_dims(D, dim2, dims);
	md_copy_dims(D, dim3, dims);

	dim0[i] = shift - (shift/mult[i]) * mult[i];
	dim1[i] = (shift/mult[i]) * mult[i];
	dim2[i] = ((dims[i] - shift) / mult[i]) * mult[i];
	dim3[i] = dims[i] - (((dims[i] - shift) / mult[i]) * mult[i]) - shift;

	long off0 = 0;
	long off1 = off0 + dim0[i] * ostrs[i] / CFL_SIZE;
	long off2 = off1 + dim1[i] * ostrs[i] / CFL_SIZE;
	long off3 = off2 + dim2[i] * ostrs[i] / CFL_SIZE;

	pos[i] = 0;

	info += blockproc_shift_mult2( D, dim0, blkdims, pos, mult, data, op, ostrs, dst + off0, istrs, src + off0 );

	info += blockproc_shift_mult2( D, dim1, blkdims, pos, mult, data, op, ostrs, dst + off1, istrs, src + off1 );

	info += blockproc_shift_mult2( D, dim2, blkdims, pos, mult, data, op, ostrs, dst + off2, istrs, src + off2 );

	info += blockproc_shift_mult2( D, dim3, blkdims, pos, mult, data, op, ostrs, dst + off3, istrs, src + off3 );


	return info;
}


float blockproc_shift_mult( unsigned int D,  const long dims[D], const long blkdims[D], const long shifts[D], const long mult[D], const void* data,
		 float (*op)(const void* data, const long blkdims[D], complex float* dst, const complex float* src), 
		 complex float* dst, const complex float* src )
{
	long strs[D];
	md_calc_strides( D, strs, dims, CFL_SIZE );

	return blockproc_shift_mult2( D, dims, blkdims, shifts, mult, data, op, strs, dst, strs, src );
}



float blockproc_shift2( unsigned int D, const long dims[D], const long blkdims[D], const long shifts[D], const void* data,
			float (*op)(const void* data, const long blkdims[D], complex float* dst, const complex float* src), 
			const long ostrs[D], complex float* dst, const long istrs[D], const complex float* src )
{
	float info = 0;
	long pos[D];

	for (unsigned int i = 0; i < D; i++) {	

		pos[i] = shifts[i];

		while (pos[i] < 0)
			pos[i] += dims[i];
	}

	unsigned int i = 0;
	while ((i < D) && (0 == pos[i]))
		i++;

	if (D == i) {

		info += blockproc2( D, dims, blkdims, data, op, ostrs, dst, istrs, src );
		return info;
	}

	long shift = pos[i];

	assert(shift != 0);

	long dim1[D];
	long dim2[D];

	md_copy_dims(D, dim1, dims);
	md_copy_dims(D, dim2, dims);

	dim1[i] = shift;
	dim2[i] = dims[i] - shift;

	pos[i] = 0;

	info += blockproc_shift2( D, dim1, blkdims, pos, data, op, ostrs, dst, istrs, src );
	info += blockproc_shift2( D, dim2, blkdims, pos, data, op, ostrs, dst + dim1[i]*ostrs[i]/CFL_SIZE, istrs, src + dim1[i]*istrs[i]/CFL_SIZE );

	return info;
}



float blockproc_shift( unsigned int D,  const long dims[D], const long blkdims[D], const long shifts[D], const void* data,
		 float (*op)(const void* data, const long blkdims[D], complex float* dst, const complex float* src), 
		 complex float* dst, const complex float* src )
{
	long strs[D];
	md_calc_strides( D, strs, dims, CFL_SIZE );

	return blockproc_shift2( D, dims, blkdims, shifts, data, op, strs, dst, strs, src );
}



float blockproc_circshift( unsigned int D,  const long dims[D], const long blkdims[D], const long shifts[D], const void* data,
		 float (*op)(const void* data, const long blkdims[D], complex float* dst, const complex float* src), 
		 complex float* dst, const complex float* src )
{
	complex float* tmp = md_alloc( D, dims, CFL_SIZE );
	
	long unshifts[D];
	for (unsigned int i = 0; i < D; i++)
		unshifts[i] = -shifts[i];

	md_circ_shift( D, dims, shifts, tmp, src, CFL_SIZE );

	return blockproc( D, dims, blkdims, data, op, tmp, tmp );

	md_circ_shift( D, dims, unshifts, dst, tmp, CFL_SIZE );

	md_free( tmp );
}





float blockproc2( unsigned int D,  const long dims[D], const long blkdims[D], const void* data,
		 float (*op)(const void* data, const long blkdims[D], complex float* dst, const complex float* src), 
		 const long ostrs[D], complex float* dst, const long istrs[D], const complex float* src )
{
	// Get number of blocks per dimension
	long nblocks[D];
	for (unsigned int i = 0; i < D; i++)
	{
		nblocks[i] = (float) ( dims[i] + blkdims[i] - 1 ) / (float) blkdims[i];
	}

	long numblocks = md_calc_size( D, nblocks );
	float info = 0;

	// Loop over blocks
	complex float* blk = md_alloc_sameplace(D, blkdims, sizeof( complex float ), src);
	for (long b = 0; b < numblocks; b++)
	{
		// Get block position and actual block size
		long blkpos[D];
		long blkdims_b[D]; // actual block size

		long ind = b;
		for ( unsigned int i = 0; i < D; i++)
		{
			long blkind = ind % nblocks[i];
			blkpos[i] = blkind * blkdims[i];
			ind = (ind - blkind) / nblocks[i];

			blkdims_b[i] = MIN( dims[i] - blkpos[i], blkdims[i] );
		}
		long blkstrs[D];
		md_calc_strides( D, blkstrs, blkdims_b, CFL_SIZE );

		// Extract block
		md_copy_block2( D, blkpos, blkdims_b, blkstrs, blk, dims, istrs, src, sizeof(complex float) );
		// Process block
		info += op( data, blkdims_b, blk, blk );

		// Put back block
		md_copy_block2( D, blkpos, dims, ostrs, dst, blkdims_b, blkstrs, blk, sizeof( complex float ));

	}
	md_free( blk );
	return info;
}


float blockproc( unsigned int D, const long dims[D], const long blkdims[D], const void* data,
		 float (*op)(const void* data, const long blkdims[D], complex float* dst, const complex float* src), 
		 complex float* dst, const complex float* src )
{
	long strs[D];
	md_calc_strides( D, strs, dims, CFL_SIZE );

	return blockproc2( D, dims, blkdims, data, op, strs, dst, strs, src );
}



float stackproc2( unsigned int D, const long dims[D], const long blkdims[D], unsigned int stkdim, const void* data,
		float (*op)(const void* data, const long stkdims[D], complex float* dst, const complex float* src), 
		 const long ostrs[D], complex float* dst, const long istrs[D], const complex float* src )
{
	// Get number of blocks per dimension
	long nblocks[D];
	for (unsigned int i = 0; i < D; i++)
	{
		nblocks[i] = (float) ( dims[i] + blkdims[i] - 1 ) / (float) blkdims[i];
	}

	long numblocks = md_calc_size( D, nblocks );
	float info = 0;

	// Initialize stack
	long stkdims[D];
	md_copy_dims( D, stkdims, blkdims );
	stkdims[stkdim] = numblocks;
	long stkstrs[D];
	md_calc_strides( D, stkstrs, stkdims, CFL_SIZE );
	long stkstr1[D];
	md_calc_strides( D, stkstr1, stkdims, 1 );

	complex float* stk = md_alloc(D, stkdims, sizeof( complex float ));
	md_clear( D, stkdims, stk, sizeof( complex float ) );

	// Loop over blocks and stack them up
	for (long b = 0; b < numblocks; b++)
	{
		// Get block position and actual block size
		long blkpos[D];
		long blkdims_b[D]; // actual block size
		long ind = b;
		for ( unsigned int i = 0; i < D; i++)
		{
			long blkind = ind % nblocks[i];
			blkpos[i] = blkind * blkdims[i];
			ind = (ind - blkind) / nblocks[i];

			blkdims_b[i] = MIN( dims[i] - blkpos[i], blkdims[i] );
		}
		long blkstrs[D];
		md_calc_strides( D, blkstrs, blkdims_b, CFL_SIZE );

		// Extract block and put in stack
		md_copy_block2( D, blkpos, blkdims_b, blkstrs, stk + stkstr1[stkdim] * b, dims, istrs, src, sizeof(complex float) );

	}

	long blkstrs[D];
	md_calc_strides( D, blkstrs, blkdims, CFL_SIZE );

	// Process block
	info = op( data, stkdims, stk, stk );

	// Put back block
	for (long b = 0; b < numblocks; b++)
	{
		// Get block position and actual block size
		long blkpos[D];
		long blkdims_b[D]; // actual block size
		long ind = b;
		for ( unsigned int i = 0; i < D; i++)
		{
			long blkind = ind % nblocks[i];
			blkpos[i] = blkind * blkdims[i];
			ind = (ind - blkind) / nblocks[i];

			blkdims_b[i] = MIN( dims[i] - blkpos[i], blkdims[i] );
		}
		long blkstrs[D];
		md_calc_strides( D, blkstrs, blkdims_b, CFL_SIZE );

		// Put back block
		md_copy_block2( D, blkpos, dims, ostrs, dst, blkdims_b, blkstrs, stk + stkstr1[stkdim] * b, sizeof( complex float ));
	}

	// Free stack
	md_free( stk );

	return info;
}


float stackproc( unsigned int D, const long dims[D], const long blkdims[D], unsigned int stkdim, const void* data,
		float (*op)(const void* data, const long stkdims[D], complex float* dst, const complex float* src), 
		 complex float* dst, const complex float* src )
{
	long strs[D];
	md_calc_strides( D, strs, dims, CFL_SIZE );

	return stackproc2( D, dims, blkdims, stkdim, data, op, strs, dst, strs, src );
}
