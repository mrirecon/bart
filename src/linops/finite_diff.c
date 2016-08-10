/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2014. Joseph Y Cheng.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2014 Joseph Y Cheng <jycheng@stanford.edu>
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include <complex.h>
#include <stdbool.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"
#include "num/iovec.h"
#include "num/gpuops.h"

#include "linops/linop.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/types.h"

#include "finite_diff.h"


/**
 * Contains parameters for finite difference 
 *
 * @param D number of dimensions
 * @param dims dimensions of input to be differenced
 * @param str strides of input
 * @param tmp temporary storage for computing finite difference
 * @param tmp2 temporary storage for computing cumulative sum
 * @param flags bitmask for applying operators
 * @param order finite difference order (currently only 1)
 * @param snip TRUE to zero out first dimension
 */
struct fdiff_s {

	INTERFACE(linop_data_t);

	unsigned int D;

	long* dims;
	long* str;

	complex float* tmp;
	complex float* tmp2; // future: be less sloppy with memory

	unsigned int flags;
	int order;

	bool snip;
};

DEF_TYPEID(fdiff_s);


/*
 * Implements finite difference operator (order 1 for now)
 * using circular shift: diff(x) = x - circshift(x)
 * @param snip Keeps first entry if snip = false; clear first entry if snip = true
 *
 * optr = [iptr(1); diff(iptr)]
 */
static void md_zfinitediff_core2(unsigned int D, const long dims[D], unsigned int flags, bool snip, complex float* tmp, const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr)
{
	md_copy2(D, dims, istrs, tmp, istrs, iptr, sizeof(complex float));

	long zdims[D];
	long center[D];

	md_select_dims(D, ~0, zdims, dims);
	memset(center, 0, D * sizeof(long));

	for (unsigned int i=0; i < D; i++) {
		if (MD_IS_SET(flags, i)) {
			center[i] = 1; // order

			md_circ_shift2(D, dims, center, ostrs, optr, istrs, tmp, sizeof(complex float));

			zdims[i] = 1;

			if (!snip) // zero out first dimension before subtracting
				md_clear2(D, zdims, ostrs, optr, sizeof(complex float));

			md_zsub2(D, dims, ostrs, optr, istrs, tmp, ostrs, optr);
			md_copy2(D, dims, ostrs, tmp, ostrs, optr, sizeof(complex float));

			if (snip) // zero out first dimension after subtracting
				md_clear2(D, zdims, ostrs, optr, sizeof(complex float));

			center[i] = 0;
			zdims[i] = dims[i];
		}
	}
}


/*
 * Finite difference along dimensions specified by flags (without strides)
 * Keeps first entry so that dimensions are unchanged
 *
 * optr = [iptr(1); diff(iptr)]
 */
void md_zfinitediff(unsigned int D, const long dims[D], unsigned int flags, bool snip, complex float* optr, const complex float* iptr)
{
	long str[D];
	md_calc_strides(D, str, dims, sizeof(complex float));
	md_zfinitediff2(D, dims, flags, snip, str, optr, str, iptr);
}


/*
 * Finite difference along dimensions specified by flags (with strides)
 * Keeps first entry so that dimensions are unchanged
 *
 * optr = [iptr(1); diff(iptr)]
 */
void md_zfinitediff2(unsigned int D, const long dims[D], unsigned int flags, bool snip, const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr)
{
	complex float* tmp = md_alloc_sameplace(D, dims, sizeof(complex float), optr);

	md_zfinitediff_core2(D, dims, flags, snip, tmp, ostrs, optr, istrs, iptr);

	md_free(tmp);
}



/*
 * Implements cumulative sum operator (order 1 for now)
 * using circular shift: cumsum(x) = x + circshift(x,1) + circshift(x,2) + ...
 *
 * optr = cumsum(iptr)
 */
static void md_zcumsum_core2(unsigned int D, const long dims[D], unsigned int flags, complex float* tmp, complex float* tmp2, const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr)
{
	//out = dx
	md_copy2(D, dims, ostrs, optr, istrs, iptr, sizeof(complex float));
	md_copy2(D, dims, istrs, tmp, istrs, iptr, sizeof(complex float));

	long zdims[D];
	long center[D];

	md_select_dims(D, ~0, zdims, dims);
	memset(center, 0, D * sizeof(long));

	for (unsigned int i=0; i < D; i++) {
		if (MD_IS_SET(flags, i)) {
			for (int d=1; d < dims[i]; d++) {

				// tmp = circshift(tmp, i)
				center[i] = d;
				md_circ_shift2(D, dims, center, istrs, tmp2, istrs, tmp, sizeof(complex float));
				zdims[i] = d;

				// tmp(1:d,:) = 0
				md_clear2(D, zdims, istrs, tmp2, sizeof(complex float));
				//md_zsmul2(D, zdims, istrs, tmp2, istrs, tmp2, 0.);
				//dump_cfl("tmp2", D, dims, tmp2);

				// out = out + tmp
				md_zadd2(D, dims, ostrs, optr, istrs, tmp2, ostrs, optr);
				//md_copy2(D, dims, ostrs, tmp, ostrs, optr, sizeof(complex float));

			}
			md_copy2(D, dims, ostrs, tmp, ostrs, optr, sizeof(complex float));

			center[i] = 0;
			zdims[i] = dims[i];
		}
	}
}


/*
 * Cumulative sum along dimensions specified by flags (without strides)
 *
 * optr = cumsum(iptr)
 */
void md_zcumsum(unsigned int D, const long dims[D], unsigned int flags, complex float* optr, const complex float* iptr)
{
	long str[D];
	md_calc_strides(D, str, dims, sizeof(complex float));
	md_zcumsum2(D, dims, flags, str, optr, str, iptr);
}


/*
 * Cumulative sum along dimensions specified by flags (with strides)
 *
 * optr = cumsum(iptr)
 */
void md_zcumsum2(unsigned int D, const long dims[D], unsigned int flags, const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr)
{
	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, optr);
	complex float* tmp2 = md_alloc_sameplace(D, dims, CFL_SIZE, optr);

	md_zcumsum_core2(D, dims, flags, tmp, tmp2, ostrs, optr, istrs, iptr);

	md_free(tmp);
	md_free(tmp2);
}


/*
 * Finite difference operator along specified dimensions.
 * Keeps the original value for the first entry
 *
 * optr = [iptr(1); diff(iptr)]
 */
static void fdiff_apply(const linop_data_t* _data, complex float* optr, const complex float* iptr)
{
	const struct fdiff_s* data = CAST_DOWN(fdiff_s, _data);

	md_zfinitediff_core2(data->D, data->dims, data->flags, data->snip, data->tmp, data->str, optr, data->str, iptr);
}



/*
 * Adjoint of finite difference operator along specified dimensions.
 * Equivalent to finite difference in reverse order
 * 
 * @param snip if false: keeps the original value for the last entry;
 * if true: implements the adjoint of the difference matrix with all zero first row
 *
 * optr = [-diff(iptr); iptr(end)] = flip(fdiff_apply(flip(iptr)))
 */
static void fdiff_apply_adjoint(const linop_data_t* _data, complex float* optr, const complex float* iptr)
{
	const struct fdiff_s* data = CAST_DOWN(fdiff_s, _data);

	md_copy2(data->D, data->dims, data->str, optr, data->str, iptr, CFL_SIZE);

	for (unsigned int i=0; i < data->D; i++) {

		unsigned int single_flag = data->flags & MD_BIT(i);

		if (single_flag) {

			md_flip2(data->D, data->dims, single_flag, data->str, data->tmp2, data->str, optr, CFL_SIZE);
			md_zfinitediff_core2(data->D, data->dims, single_flag, false, data->tmp, data->str, data->tmp2, data->str, data->tmp2);
			md_flip2(data->D, data->dims, single_flag, data->str, optr, data->str, data->tmp2, CFL_SIZE);

			if (data->snip) {

				long zdims[data->D];
				md_select_dims(data->D, ~0, zdims, data->dims);

				zdims[i] = 1;
				md_zsub2(data->D, zdims, data->str, optr, data->str, optr, data->str, iptr);
			}
		}
	}
}

/*
 * Cumulative sum - inverse of finite difference operator 
 *
 * optr = cumsum(iptr);
 */
static void cumsum_apply(const linop_data_t* _data, float lambda, complex float* optr, const complex float* iptr)
{
	const struct fdiff_s* data = CAST_DOWN(fdiff_s, _data);

	assert(0. == lambda);
	md_zcumsum_core2(data->D, data->dims, data->flags, data->tmp, data->tmp2, data->str, optr, data->str, iptr);
}

static void finite_diff_del(const linop_data_t* _data)
{
	const struct fdiff_s* data = CAST_DOWN(fdiff_s, _data);

	xfree(data->dims);
	xfree(data->str);
	md_free(data->tmp);
	md_free(data->tmp2);

	xfree(data);
}


/**
 * Initialize finite difference operator
 *
 * @param D number of dimensions
 * @param dim input dimensions
 * @param flags bitmask for applying operator
 * @param snip true: clear initial entry (i.c.); false: keep initial entry (i.c.)
 * @param gpu true if using gpu, false if using cpu
 *
 * Returns a pointer to the finite difference operator
 */
extern const struct linop_s* linop_finitediff_create(unsigned int D, const long dim[D], const unsigned long flags, bool snip, bool gpu)
{
	PTR_ALLOC(struct fdiff_s, data);
	SET_TYPEID(fdiff_s, data);

	data->D = D;
	data->flags = flags;
	data->order = 1;
	data->snip = snip;

	data->dims = *TYPE_ALLOC(long[D]);
	md_copy_dims(D, data->dims, dim);

	data->str = *TYPE_ALLOC(long[D]);
	md_calc_strides(D, data->str, data->dims, CFL_SIZE);

#ifdef USE_CUDA
	data->tmp = (gpu ? md_alloc_gpu : md_alloc)(D, data->dims, CFL_SIZE);
	data->tmp2 = (gpu ? md_alloc_gpu : md_alloc)(D, data->dims, CFL_SIZE);
#else
	assert(!gpu);
	data->tmp = md_alloc(D, data->dims, CFL_SIZE);
	data->tmp2 = md_alloc(D, data->dims, CFL_SIZE);
#endif

	return linop_create(D, dim, D, dim, CAST_UP(PTR_PASS(data)), fdiff_apply, fdiff_apply_adjoint, NULL, cumsum_apply, finite_diff_del);
}


void fd_proj_noninc(const struct linop_s* o, complex float* optr, const complex float* iptr)
{

	struct fdiff_s* data = (struct fdiff_s*)linop_get_data(o);
	
	dump_cfl("impre", data->D, data->dims, iptr);
	linop_forward_unchecked(o, data->tmp2, iptr);

	long tmpdim = data->dims[0];
	long dims2[data->D];
	md_select_dims(data->D, ~0u, dims2, data->dims);
	dims2[0] *= 2; 
	dump_cfl("dxpre", data->D, data->dims, data->tmp2);

	md_smin(data->D, dims2, (float*)optr, (float*)data->tmp2, 0.);

	// add back initial value
	dims2[0] = tmpdim;
	for (unsigned int i=0; i < data->D; i++) {
		if (MD_IS_SET(data->flags, i)) {
			dims2[i] = 1;
			md_copy2(data->D, dims2, data->str, optr, data->str, data->tmp2, sizeof(complex float));
			break;
		}
	}
	dump_cfl("dxpost", data->D, data->dims, optr);
	linop_norm_inv_unchecked(o, 0., optr, optr);
	
	dump_cfl("impost", data->D, data->dims, optr);
}

complex float* get_fdiff_tmp2ptr(const struct linop_s* o)
{
	struct fdiff_s* fdata = (struct fdiff_s*)linop_get_data(o);
	return fdata->tmp2;
}



/**
 * Internal data structure used for zfinitediff operator
 */
struct zfinitediff_data {

	INTERFACE(linop_data_t);

	unsigned int D;
	long dim_diff;
	bool do_circdiff;

	long* dims_in;
	long* strides_in;
	long* dims_adj;
	long* strides_adj;

	size_t size;
};

DEF_TYPEID(zfinitediff_data);


/**
 * Originally used md_circshift, but couldn't get it right, so I just
 * wrote it out for now (also avoids extra memory)
 */
static void zfinitediff_apply(const linop_data_t* _data,
		complex float* optr, const complex float* iptr)
{
	// if (docircshift)
	//     out(..,1:(end-1),..) = in(..,1:(end-1),..) - in(..,2:end,..)
	//     out(..,end,..) = in(..,end,..) - in(..,1,..)
	// else
	//     out = in(..,1:(end-1),..) - in(..,2:end,..)

	//printf("zfinitediff_apply\n");
	const struct zfinitediff_data* data = CAST_DOWN(zfinitediff_data, _data);


	unsigned long d = data->dim_diff;
	long nx = data->dims_in[d];

	long dims_sub[data->D];
	md_copy_dims(data->D, dims_sub, data->dims_in);

	long off_in, off_adj;

	if (data->do_circdiff) {

		// out(..,1:(end-1),..) = in(..,1:(end-1),..) - in(..,2:end,..)
		dims_sub[d] = nx - 1;
		off_in = data->strides_in[d] / CFL_SIZE;
		//off_adj = data->strides_in[d]/CFL_SIZE;
		md_zsub2(data->D, dims_sub, data->strides_adj, optr,
				data->strides_in, iptr, data->strides_in, iptr + off_in);

		// out(..,end,..) = in(..,end,..) - in(..,1,..)
		dims_sub[d] = 1;
		off_in = (nx - 1) * data->strides_in[d] / CFL_SIZE;
		off_adj = (nx - 1) * data->strides_adj[d] / CFL_SIZE;
		md_zsub2(data->D, dims_sub, data->strides_adj, optr + off_adj,
				data->strides_in, iptr + off_in, data->strides_in, iptr);

	} else {
		// out(..,1:(end-1),..) = in(..,1:(end-1),..) - in(..,2:end,..)
		dims_sub[d] = nx - 1;
		off_in = data->strides_in[d] / CFL_SIZE;
		md_zsub2(data->D, dims_sub, data->strides_adj, optr,
				data->strides_in, iptr, data->strides_in, iptr + off_in);
	}

	/*
	   long i_shift, i_adj, x_orig, x_new;
	   unsigned int d = data->dim_diff;
	   for (unsigned int i = 0; i < md_calc_size(data->D, data->dims_in); i++) {
	   i_shift = i;
	   i_adj = i;

	   x_orig = (i/data->strs_in[d]) % data->dims_in[d];
	   x_new = x_orig + 1; // shift by 1

	   while (x_new >= data->dims_in[d]) x_new -= data->dims_in[d];

	   i_shift += (x_new - x_orig)*data->strs_in[d];

	   optr[i_adj] = iptr[i] - iptr[i_shift];
	 */
}

static void zfinitediff_adjoint(const linop_data_t* _data,
			  complex float* optr, const complex float* iptr)
{
	//printf("zfinitediff_adjoint\n");
	const struct zfinitediff_data* data = CAST_DOWN(zfinitediff_data, _data);

	// if (docircshift)
	//     out(..,2:end,..) = in(..,2:end,..) - in(..,1:(end-1),..)
	//     out(..,1,..) = in(..,1,..) - in(..,end,..)
	// else
	//     out(..,1,..) = in(..,1,..)
	//     out(..,2:(end-1),..) = in(..,2:end,..) - in(..,1:(end-1),..)
	//     out(..,end,..) = -in(..,end,..);

	unsigned int d = data->dim_diff;
	long nx = data->dims_adj[d];
	long off_in, off_adj;

	long dims_sub[data->D];
	md_copy_dims(data->D, dims_sub, data->dims_adj);

	if (data->do_circdiff) {

		// out(..,2:end,..) = in(..,2:end,..) - in(..,1:(end-1),..)
		dims_sub[d] = nx - 1;
		off_adj = data->strides_adj[d] / CFL_SIZE;
		off_in = data->strides_in[d] / CFL_SIZE;
		md_zsub2(data->D, dims_sub, data->strides_in, optr + off_in,
				data->strides_in, iptr + off_adj, data->strides_adj, iptr);

		// out(..,1,..) = in(..,1,..) - in(..,end,..)
		dims_sub[d] = 1;
		off_adj = (nx - 1) * data->strides_adj[d] / CFL_SIZE;
		off_in = (nx - 1) * data->strides_in[d] / CFL_SIZE;
		md_zsub2(data->D, dims_sub, data->strides_in, optr,
				data->strides_adj, iptr, data->strides_adj, iptr + off_adj);

	} else {

		// out(..,end,..) = 0
		//md_clear2(data->D, data->dims_in, data->strides_in, optr, CFL_SIZE);
		dims_sub[d] = 1;
		off_in = nx * data->strides_in[d] / CFL_SIZE;
		md_clear2(data->D, dims_sub, data->strides_in, optr + off_in, CFL_SIZE);
		// out(..,1:end-1,:) = in_adj(..,1:end,:)
		md_copy2(data->D, data->dims_adj, data->strides_in, optr,
				data->strides_adj, iptr, CFL_SIZE);
		// out(..,2:end,:) -= in_adj(..,1:end,:)
		off_in = data->strides_in[d] / CFL_SIZE;
		md_zsub2(data->D, data->dims_adj, data->strides_in, optr + off_in,
				data->strides_in, optr + off_in, data->strides_adj, iptr);

		/*
		// out(..,1,..) = in_adj(..,1,..)
		dims_sub[d] = 1;
		md_copy2(data->D, dims_sub,
		data->strides_in, optr, data->strides_adj, iptr, CFL_SIZE);

		// out(..,2:(end-1),..) = in(..,2:end,..) - in(..,1:(end-1),..)
		dims_sub[d] = nx - 1;
		off_adj = data->strides_adj[d]/CFL_SIZE;
		off_in = data->strides_in[d]/CFL_SIZE;
		md_zsub2(data->D, dims_sub, data->strides_in, optr+off_in,
		data->strides_adj, iptr+off_adj, data->strides_adj, iptr);

		// out(..,end,..) = -in(..,end,..);
		dims_sub[d] = 1;
		off_adj = (nx - 1) * data->strides_adj[d]/CFL_SIZE;
		off_in = nx * data->strides_in[d]/CFL_SIZE;
		// !!!This one operation is really really slow!!!
		md_zsmul2(data->D, dims_sub, data->strides_in, optr+off_in,
		data->strides_adj, iptr+off_adj, -1.);
		 */
	}
}

// y = 2*x - circshift(x,center_adj) - circshift(x,center)
static void zfinitediff_normal(const linop_data_t* _data,
			complex float* optr, const complex float* iptr)
{
	const struct zfinitediff_data* data = CAST_DOWN(zfinitediff_data, _data);

	// Turns out that this is faster, but this requires extra memory.
	complex float* tmp = md_alloc_sameplace(data->D, data->dims_in, CFL_SIZE, iptr);

	zfinitediff_apply(_data, tmp, iptr);
	zfinitediff_adjoint(_data, optr, tmp);

	md_free(tmp);
	return;		// FIXME: WTF?


	unsigned long d = data->dim_diff;
	long nx = data->dims_in[d];
	long offset;
	long dims_sub[data->D];
	md_copy_dims(data->D, dims_sub, data->dims_in);

	// optr and iptr same size regardless if do_circdiff true/false
	// if (data->do_circdiff)
	//    out = 2*in;
	//    out(..,1:(end-1),..) = out(..,1:(end-1),..) - in(..,2:end,..)
	//    out(..,2:end,..) = out(..,2:end,..) - in(..,1:(end-1),..)
	//    out(..,end,..) = out(..,end,..) - in(..,1,..)
	//    out(..,1,..) = out(..,1,..) - in(..,end,..)
	//
	// else
	//    out(..,1,..) = in(..,1,..)
	//    out(..,end,..) = in(..,end,..)
	//    out(..,2:(end-1),..) = 2*in(..,2:(end-1),..)
	//    out(..,1:(end-1),..) = out(..,1:(end-1),..) - in(..,2:end,..)
	//    out(..,2:end,..) = out(..,2:end,..) - in(..,1:(end-1),..)
	//

	if (data->do_circdiff) {

		md_zsmul2(data->D, data->dims_in, data->strides_in, optr,
				data->strides_in, iptr, 2.);

		dims_sub[d] = (nx - 1);
		offset = data->strides_in[d] / CFL_SIZE;
		// out(..,1:(end-1),..) = out(..,1:(end-1),..) - in(..,2:end,..)
		md_zsub2(data->D, dims_sub, data->strides_in, optr,
				data->strides_in, optr, data->strides_in, iptr + offset);
		// out(..,2:end,..) = out(..,2:end,..) - in(..,1:(end-1),..)
		md_zsub2(data->D, dims_sub, data->strides_in, optr + offset,
				data->strides_in, optr + offset, data->strides_in, iptr);

		dims_sub[d] = 1;
		offset = (nx - 1) * data->strides_in[d] / CFL_SIZE;
		// out(..,1,..) = out(..,1,..) - in(..,end,..)
		md_zsub2(data->D, dims_sub, data->strides_in, optr,
				data->strides_in, optr, data->strides_in, iptr + offset);
		// out(..,end,..) = out(..,end,..) - in(..,1,..)
		md_zsub2(data->D, dims_sub, data->strides_in, optr+offset,
				data->strides_in, optr+offset, data->strides_in, iptr);

	} else {

		dims_sub[d] = 1;
		offset = (nx - 1) * data->strides_in[d] / CFL_SIZE;
		// out(..,1,..) = in(..,1,..)
		md_copy2(data->D, dims_sub,
				data->strides_in, optr, data->strides_in, iptr, CFL_SIZE);
		// out(..,end,..) = in(..,end,..)
		md_copy2(data->D, dims_sub,
				data->strides_in, optr + offset, data->strides_in, iptr + offset,
				CFL_SIZE);

		dims_sub[d] = nx - 2;
		offset = data->strides_in[d] / CFL_SIZE;
		// out(..,2:(end-1),..) = 2*in(..,2:(end-1),..)
		md_zsmul2(data->D, dims_sub, data->strides_in, optr + offset,
				data->strides_in, iptr + offset, 2.);

		dims_sub[d] = nx - 1;
		offset = data->strides_in[d] / CFL_SIZE;
		// out(..,1:(end-1),..) = out(..,1:(end-1),..) - in(..,2:end,..)
		md_zsub2(data->D, dims_sub, data->strides_in, optr,
				data->strides_in, optr, data->strides_in, iptr + offset);
		// out(..,2:end,..) = out(..,2:end,..) - in(..,1:(end-1),..)
		md_zsub2(data->D, dims_sub, data->strides_in, optr + offset,
				data->strides_in, optr + offset, data->strides_in, iptr);

	}
}



static void zfinitediff_del(const linop_data_t* _data)
{
	const struct zfinitediff_data* data = CAST_DOWN(zfinitediff_data, _data);

	xfree(data->dims_in);
	xfree(data->strides_in);

	xfree(data->dims_adj);
	xfree(data->strides_adj);

	// FIXME free data
}

const struct linop_s* linop_zfinitediff_create(unsigned int D, const long dims[D], long diffdim, bool circular)
{
	PTR_ALLOC(struct zfinitediff_data, data);
	SET_TYPEID(zfinitediff_data, data);

	data->D = D;
	data->dim_diff = diffdim;
	data->do_circdiff = circular;

	data->dims_in = *TYPE_ALLOC(long[D]);
	data->dims_adj = *TYPE_ALLOC(long[D]);
	data->strides_in = *TYPE_ALLOC(long[D]);
	data->strides_adj = *TYPE_ALLOC(long[D]);

	md_copy_dims(D, data->dims_in, dims);
	md_copy_dims(D, data->dims_adj, dims);

	md_calc_strides(D, data->strides_in, data->dims_in, CFL_SIZE);

	if (!data->do_circdiff)
		data->dims_adj[data->dim_diff] -= 1;

	md_calc_strides(D, data->strides_adj, data->dims_adj, CFL_SIZE);

	const long* dims_adj = data->dims_adj;
	const long* dims_in = data->dims_in;

	return linop_create(D, dims_adj, D, dims_in, CAST_UP(PTR_PASS(data)),
			zfinitediff_apply, zfinitediff_adjoint,
			zfinitediff_normal, NULL, zfinitediff_del);
}


