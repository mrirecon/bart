/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 20XX-2013 Frank Ong, Martin Uecker, Pat Virtue, and Mark Murphy
 * frankong@berkeley.edu
 * 2016	     Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#define _GNU_SOURCE
#include <math.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/ops.h"

#include "linops/linop.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "wavelet.h"
#include "wavelet_impl.h"
#ifdef USE_CUDA
#include "wavelet_kernels.h"
#endif


// Header
static void fwt2_cpu(struct wavelet_plan_s* plan, data_t* out, data_t* in);
static void iwt2_cpu(struct wavelet_plan_s* plan, data_t* out, const data_t* in);
static void wavthresh2_cpu(struct wavelet_plan_s* plan, data_t* out, data_t* in, scalar_t thresh);

static void fwt3_cpu(struct wavelet_plan_s* plan, data_t* out, data_t* in);
static void iwt3_cpu(struct wavelet_plan_s* plan, data_t* out, const data_t* in);
static void wavthresh3_cpu(struct wavelet_plan_s* plan, data_t* out, data_t* in, scalar_t thresh);

static void softthresh_cpu(struct wavelet_plan_s* plan, data_t* in, scalar_t thresh);

static void conv_down_2d(data_t *out, const data_t *in, int size1, int skip1, int size2, int skip2, const scalar_t *filter, int filterLen);
static void conv_up_2d(data_t *out, const data_t *in, int size1, int skip1, int size2, int skip2, const scalar_t *filter, int filterLen);
static void conv_down_3d(data_t *out, const data_t *in, int size1, int skip1, int size2, int skip2, int size3, int skip3, const scalar_t *filter, int filterLen);
static void conv_up_3d(data_t *out, const data_t *in, int size1, int skip1, int size2, int skip2, int size3, int skip3, const scalar_t *filter, int filterLen);

static void create_numLevels(struct wavelet_plan_s* plan);
static void create_wavelet_sizes(struct wavelet_plan_s* plan);

static void wavelet_del(const linop_data_t* data);


struct wavelet_plan_s* prepare_wavelet_plan_filters(int numdims, const long imSize[numdims], unsigned int flags, const long minSize[numdims], int use_gpu, int filter_length, const float filter[4][filter_length])
{
	// Currently only accept flags=3,7
	assert( (3 == flags) || (7 == flags) );
	assert((use_gpu == 0) || (use_gpu == 1));

	struct wavelet_plan_s* plan = (struct wavelet_plan_s*)xmalloc(sizeof(struct wavelet_plan_s));
	plan->use_gpu = use_gpu;
	plan->imSize = (long*)xmalloc(sizeof(long)*numdims);
	md_singleton_dims(numdims, plan->imSize);
	// Get imSize, numPixel, numdims_tr
	// plan->numdims and flags ignores imSize[i]=1
	plan->numdims_tr = 0;
	plan->numPixel = 1;
	plan->numPixel_tr = 1;
	plan->batchSize = 1;
	plan->flags = 0;
	int i,i_tr;
	int d = 0;
	for (i = 0; i < numdims; i++)
	{
		assert(imSize[i] > 0);
		if (1 != imSize[i]) {
			plan->imSize[d] = imSize[i];
			plan->numPixel *= imSize[i];
			if (MD_IS_SET(flags, i)) {
				plan->numdims_tr++;
				plan->numPixel_tr*=imSize[i];
			} else
				plan->batchSize*=imSize[i];

			if (MD_IS_SET(flags, i))
				plan->flags = MD_SET(plan->flags, d);
			d++;
		}
	}
	plan->numdims = d;
	// Get imSize_tr, trDims (dimensions that we do wavelet transform), minSize_tr 
	plan->imSize_tr = (long*)xmalloc(sizeof(long) * plan->numdims_tr);
	plan->trDims = (long*)xmalloc(sizeof(long) * plan->numdims_tr);
	plan->minSize_tr = (long*)xmalloc(sizeof(long) * plan->numdims_tr);
	i_tr = 0;
	for (i = 0; i < numdims; i++)
	{
		if (MD_IS_SET(flags, i) && (1 != imSize[i])) {
			plan->imSize_tr[i_tr] = imSize[i];
			plan->trDims[i_tr] = i;
			assert(minSize[i_tr] > 0);
			plan->minSize_tr[i_tr] = minSize[i];
			i_tr++;
		}
	}

	plan->filterLen = filter_length;
#ifdef USE_CUDA
	if (plan->use_gpu)
	{
		prepare_wavelet_filters_gpu(plan,plan->filterLen,&(filter[0][0]));

		create_numLevels(plan);
		create_wavelet_sizes(plan);

		plan->state = 1;
		plan->randShift_tr = (long*)xmalloc(sizeof(long) * plan->numdims_tr);
		memset(plan->randShift_tr, 0, sizeof(long) * plan->numdims_tr);

		prepare_wavelet_temp_gpu(plan);
	} else
#endif
	{
		plan->lod = filter[0];
		plan->hid = filter[1];
		plan->lor = filter[2];
		plan->hir = filter[3];

		create_numLevels(plan);
		create_wavelet_sizes(plan);

		plan->state = 1;
		plan->randShift_tr = (long*)xmalloc(sizeof(long) * plan->numdims_tr);
		memset(plan->randShift_tr, 0, sizeof(long) * plan->numdims_tr);

		plan->tmp_mem_tr = (data_t*)xmalloc(sizeof(data_t)*plan->numCoeff_tr*4);
	}
	plan->lambda = 1.;

	return plan;
}

struct wavelet_plan_s* prepare_wavelet_plan(int numdims, const long imSize[numdims], unsigned int flags, const long minSize[numdims], int use_gpu)
{
	return prepare_wavelet_plan_filters(numdims, imSize, flags, minSize, use_gpu, 4, wavelet2_dau2);
}





struct wavelet_data_s {

	INTERFACE(linop_data_t);

	struct wavelet_plan_s* plan;
};

DEF_TYPEID(wavelet_data_s);


static void wavelet_normal(const linop_data_t* _data, data_t* out, const data_t* _in)
{
	struct wavelet_plan_s* plan = CAST_DOWN(wavelet_data_s, _data)->plan;

	md_copy(plan->numdims, plan->imSize, out, _in, sizeof(data_t));
}


static void wavelet_forward(const linop_data_t* _data, data_t* out, const data_t* _in)
{
	struct wavelet_plan_s* plan = CAST_DOWN(wavelet_data_s, _data)->plan;

	if (plan->randshift)
		wavelet_new_randshift(plan);	

	data_t* in = (data_t*) _in;

	int numdims_tr = plan->numdims_tr;
	int numPixel_tr = plan->numPixel_tr;
	int numCoeff_tr = plan->numCoeff_tr;
	int b;


	for (b=0; b<plan->batchSize; b++)
	{
		if(numdims_tr==2)
		{
			if(plan->use_gpu==0)
				fwt2_cpu(plan, out+b*numCoeff_tr, in+b*numPixel_tr);
#ifdef USE_CUDA
			if(plan->use_gpu==1)
				fwt2_gpu(plan, out+b*numCoeff_tr, in+b*numPixel_tr);
#endif
		}
		if (numdims_tr==3)
		{
			if(plan->use_gpu==0)
				fwt3_cpu(plan, out+b*numCoeff_tr, in+b*numPixel_tr);
#ifdef USE_CUDA
			if(plan->use_gpu==1)
				fwt3_gpu(plan, out+b*numCoeff_tr, in+b*numPixel_tr);
#endif
		}
	}
}


static void wavelet_inverse(const linop_data_t* _data, data_t* out, const data_t* _in)
{
	struct wavelet_plan_s* plan = CAST_DOWN(wavelet_data_s, _data)->plan;
	data_t* in = (data_t*) _in;

	int numdims_tr = plan->numdims_tr;
	int numPixel_tr = plan->numPixel_tr;
	int numCoeff_tr = plan->numCoeff_tr;
	int b;

	for (b=0; b<plan->batchSize; b++)
	{
		if(numdims_tr==2)
		{
			if(plan->use_gpu==0)
				iwt2_cpu(plan,out+b*numPixel_tr,in+b*numCoeff_tr);
#ifdef USE_CUDA
			if(plan->use_gpu==1)
				iwt2_gpu(plan,out+b*numPixel_tr,in+b*numCoeff_tr);
#endif
		}
		if (numdims_tr==3)
		{
			if(plan->use_gpu==0)
				iwt3_cpu(plan,out+b*numPixel_tr,in+b*numCoeff_tr);
#ifdef USE_CUDA
			if(plan->use_gpu==1)
				iwt3_gpu(plan,out+b*numPixel_tr,in+b*numCoeff_tr);
#endif
		}
	}
}


static void wavelet_del(const linop_data_t* _data)
{
	struct wavelet_data_s* data = CAST_DOWN(wavelet_data_s, _data);

	// FIXME: free plan

	free(data);
}


/**
 * Wavelet linear operator
 *
 * @param numdims number of dimensions
 * @param imSize dimensions of x
 * @param wave_flags bitmask for Wavelet transform
 * @param minSize minimium size of coarse Wavelet scale
 * @param randshift apply random shift before Wavelet transforming
 * @param use_gpu true if using gpu
 */
const struct linop_s* wavelet_create(int numdims, const long imSize[numdims], unsigned int wave_flags, const long minSize[numdims], bool randshift, bool use_gpu)
{
	PTR_ALLOC(struct wavelet_data_s, data);
	SET_TYPEID(wavelet_data_s, data);

	data->plan = prepare_wavelet_plan(numdims, imSize, wave_flags, minSize, use_gpu);

	data->plan->randshift = randshift;

	long coeff_dims[numdims];
	md_select_dims(numdims, ~wave_flags, coeff_dims, imSize);
	coeff_dims[0] = data->plan->numCoeff_tr;
	coeff_dims[1] = 1;
	coeff_dims[2] = 1;

	return linop_create(numdims, coeff_dims, numdims, imSize, CAST_UP(PTR_PASS(data)), wavelet_forward, wavelet_inverse, wavelet_normal, NULL, wavelet_del);
}



void soft_thresh(struct wavelet_plan_s* plan, data_t* in, scalar_t thresh)
{
	int numCoeff_tr = plan->numCoeff_tr;
	int b;

	for (b=0; b<plan->batchSize; b++)
	{
		if (plan->use_gpu==0)
			softthresh_cpu(plan,in+b*numCoeff_tr,thresh);
#ifdef USE_CUDA
		if (plan->use_gpu==1)
			softthresh_gpu(plan,in+b*numCoeff_tr,thresh);
#endif
	}
}


struct wave_prox_s {

	INTERFACE(operator_data_t);
	struct wavelet_plan_s* plan;
};

DEF_TYPEID(wave_prox_s);


static void wavelet_thresh(const operator_data_t* _data, scalar_t thresh,  data_t* out, const data_t* _in)
{
	struct wavelet_plan_s* plan = CAST_DOWN(wave_prox_s, _data)->plan;

	if (plan->randshift)
		wavelet_new_randshift(plan);

	data_t* in = (data_t*) _in;

	int numdims_tr = plan->numdims_tr;
	int numPixel_tr = plan->numPixel_tr;
	int b;

	for (b=0; b<plan->batchSize; b++)
	{
		if(numdims_tr==2)
		{
			if(plan->use_gpu==0)
				wavthresh2_cpu(plan,out+b*numPixel_tr,in+b*numPixel_tr, plan->lambda * thresh);
#ifdef USE_CUDA
			if(plan->use_gpu==1)
				wavthresh2_gpu(plan,out+b*numPixel_tr,in+b*numPixel_tr, plan->lambda * thresh);
#endif
		}
		if (numdims_tr==3)
		{
			if(plan->use_gpu==0)
				wavthresh3_cpu(plan,out+b*numPixel_tr,in+b*numPixel_tr, plan->lambda * thresh);
#ifdef USE_CUDA
			if(plan->use_gpu==1)
				wavthresh3_gpu(plan,out+b*numPixel_tr,in+b*numPixel_tr, plan->lambda * thresh);
#endif
		}
	}

}


static int rand_lim(unsigned int* state, int limit) {

	int divisor = RAND_MAX/(limit+1);
	int retval;

	do { 
		retval = rand_r(state) / divisor;
	} while (retval > limit);

	return retval;
}

void wavelet_new_randshift (struct wavelet_plan_s* plan) {
	int i;
	int maxShift = 1 << (plan->numLevels_tr);
	for(i = 0; i < plan->numdims_tr; i++) {
		// Generate random shift value between 0 and maxShift
		plan->randShift_tr[i] = rand_lim(&plan->state, maxShift);
	}
}

void wavelet_clear_randshift (struct wavelet_plan_s* plan) {
	memset(plan->randShift_tr, 0, plan->numdims_tr*sizeof(int));
}

void wavelet_free(const struct wavelet_plan_s* plan)
{
	free(plan->imSize);
	free(plan->trDims);
	free(plan->imSize_tr);
	free(plan->minSize_tr);
	free(plan->waveSizes_tr);
	free(plan->randShift_tr);
#ifdef USE_CUDA
	if (plan->use_gpu)
		wavelet_free_gpu(plan);
	else
#endif 
	{
		free(plan->tmp_mem_tr);
	}
}



static void wavelet_prox_del(const operator_data_t* _data)
{
	struct wave_prox_s* data = CAST_DOWN(wave_prox_s, _data);
	wavelet_free(data->plan);
	free(data);
}

/**
 * Proximal operator for l1-norm with Wavelet transform: f(x) = lambda || W x ||_1
 *
 * @param numdims number of dimensions
 * @param imSize dimensions of x
 * @param wave_flags bitmask for Wavelet transform
 * @param minSize minimium size of coarse Wavelet scale
 * @param lambda threshold parameter
 * @param randshift apply random shift before Wavelet transforming
 * @param use_gpu true if using gpu
 */
const struct operator_p_s* prox_wavethresh_create(int numdims, const long imSize[numdims], unsigned int wave_flags, const long minSize[numdims], float lambda, bool randshift, bool use_gpu)
{
	PTR_ALLOC(struct wave_prox_s, data);
	SET_TYPEID(wave_prox_s, data);

	data->plan = prepare_wavelet_plan(numdims, imSize, wave_flags, minSize, use_gpu);

	data->plan->randshift = randshift;
	data->plan->lambda = lambda;

	return operator_p_create(numdims, imSize, numdims, imSize, CAST_UP(PTR_PASS(data)), wavelet_thresh, wavelet_prox_del);

}

void fwt2_cpu(struct wavelet_plan_s* plan, data_t* coeff, data_t* inImage)
{
	circshift(plan,inImage);
	data_t* origInImage = inImage;
	data_t* HxLy = coeff + plan->waveSizes_tr[0]*plan->waveSizes_tr[1];
	int l;
	for (l = 1; l <= plan->numLevels_tr; ++l){
		HxLy += 3*plan->waveSizes_tr[0 + 2*l]*plan->waveSizes_tr[1 + 2*l];
	}

	int dx = plan->imSize_tr[0];
	int dy = plan->imSize_tr[1];
	int dxNext = plan->waveSizes_tr[0 + 2*plan->numLevels_tr];
	int dyNext = plan->waveSizes_tr[1 + 2*plan->numLevels_tr];
	int blockSize = dxNext*dyNext;

	// Allocate Memory
	data_t* LxLy = plan->tmp_mem_tr;
	data_t* tempy = LxLy+blockSize;
	data_t* tempxy = tempy+dx*dyNext;

	for (l = plan->numLevels_tr; l >= 1; --l)
	{
		dxNext = plan->waveSizes_tr[0 + 2*l];
		dyNext = plan->waveSizes_tr[1 + 2*l];
		blockSize = dxNext*dyNext;

		HxLy = HxLy - 3*blockSize;
		data_t* LxHy = HxLy + blockSize;
		data_t* HxHy = LxHy + blockSize;
		int newdy = (dy + plan->filterLen-1) / 2;
		// Ly
		conv_down_2d(tempy, inImage, dy, dx, dx, 1, plan->lod,plan->filterLen);
		conv_down_2d(LxLy, tempy, dx, 1, newdy, dx, plan->lod,plan->filterLen);
		conv_down_2d(HxLy, tempy, dx, 1, newdy, dx, plan->hid,plan->filterLen);
		// Hy
		conv_down_2d(tempy, inImage, dy, dx, dx, 1, plan->hid,plan->filterLen);
		conv_down_2d(LxHy, tempy, dx, 1, newdy, dx, plan->lod,plan->filterLen);
		conv_down_2d(HxHy, tempy, dx, 1, newdy, dx, plan->hid,plan->filterLen);

		memcpy(tempxy, LxLy, blockSize*sizeof(data_t));
		inImage = tempxy;
		dx = dxNext;
		dy = dyNext;
	}

	memcpy(coeff, inImage, plan->waveSizes_tr[0]*plan->waveSizes_tr[1]*sizeof(data_t));
	circunshift(plan,origInImage);
}

void iwt2_cpu(struct wavelet_plan_s* plan, data_t *outImage, const data_t *coeff)
{
	// Workspace dimensions
	int dxWork = plan->waveSizes_tr[0 + 2*plan->numLevels_tr]*2-1 + plan->filterLen-1;
	int dyWork = plan->waveSizes_tr[1 + 2*plan->numLevels_tr]*2-1 + plan->filterLen-1;
	int dyWork2 = plan->waveSizes_tr[1 + 2*(plan->numLevels_tr-1)]*2-1 + plan->filterLen-1;

	// Workspace
	data_t* tempy = plan->tmp_mem_tr;
	data_t* tempFull = tempy+dxWork*dyWork2;
	int dx = plan->waveSizes_tr[0];
	int dy = plan->waveSizes_tr[1];

	memcpy(outImage, coeff, dx*dy*sizeof(data_t));
	const data_t* HxLy = coeff + dx*dy;
	int level;
	for (level = 1; level < plan->numLevels_tr+1; ++level)
	{
		dx = plan->waveSizes_tr[0 + 2*level];
		dy = plan->waveSizes_tr[1 + 2*level];
		int blockSize = dx*dy;

		const data_t* LxHy = HxLy + blockSize;
		const data_t* HxHy = LxHy + blockSize;

		// Briefly rename the data
		data_t* LxLy = outImage;

		int newdx = 2*dx-1 + plan->filterLen-1;
		int newdy = 2*dy-1 + plan->filterLen-1;
		int newdxy = newdx*dy;
		int newnewdxy = newdx*newdy;
		memset(tempFull, 0, newnewdxy*sizeof(data_t));

		memset(tempy, 0, newdxy*sizeof(data_t));
		conv_up_2d(tempy, LxLy, dx, 1, dy, dx, plan->lor,plan->filterLen);
		conv_up_2d(tempy, HxLy, dx, 1, dy, dx, plan->hir,plan->filterLen);
		conv_up_2d(tempFull, tempy, dy, newdx, newdx, 1, plan->lor,plan->filterLen);

		memset(tempy, 0, newdxy*sizeof(data_t));
		conv_up_2d(tempy, LxHy, dx, 1, dy, dx, plan->lor,plan->filterLen);
		conv_up_2d(tempy, HxHy, dx, 1, dy, dx, plan->hir,plan->filterLen);
		conv_up_2d(tempFull, tempy, dy, newdx, newdx, 1, plan->hir,plan->filterLen);

		// Crop center of workspace
		int dxNext = plan->waveSizes_tr[0+2*(level+1)];
		int dyNext = plan->waveSizes_tr[1+2*(level+1)];
		dxWork = (2*dx-1 + plan->filterLen-1);
		dyWork = (2*dy-1 + plan->filterLen-1);
		int xOffset = (int) floor((dxWork - dxNext) / 2.0);
		int yOffset = (int) floor((dyWork - dyNext) / 2.0);

		int j;
		for (j = 0; j < dyNext; ++j){
			memcpy(outImage+j*dxNext, tempFull+xOffset + (yOffset+j)*dxWork, dxNext*sizeof(data_t));
		}
		HxLy += 3*blockSize;
	}
	circunshift(plan,outImage);
}

void wavthresh2_cpu(struct wavelet_plan_s* plan, data_t* outImage, data_t* inImage, scalar_t thresh)
{
	data_t* coeff;
	coeff = (data_t*)xmalloc(sizeof(data_t)*plan->numCoeff_tr);
	fwt2_cpu(plan, coeff, inImage);

	softthresh_cpu(plan, coeff, thresh);
	iwt2_cpu(plan, outImage, coeff);
	free( coeff );
}

void fwt3_cpu(struct wavelet_plan_s* plan, data_t* coeff, data_t* inImage)
{
	circshift(plan,inImage);
	data_t* origInImage = inImage;
	data_t* HxLyLz = coeff + plan->waveSizes_tr[0]*plan->waveSizes_tr[1]*plan->waveSizes_tr[2];
	int l;
	for (l = 1; l <= plan->numLevels_tr; ++l){
		HxLyLz += 7*plan->waveSizes_tr[0 + 3*l]*plan->waveSizes_tr[1 + 3*l]*plan->waveSizes_tr[2 + 3*l];
	}
	int dx = plan->imSize_tr[0];
	int dy = plan->imSize_tr[1];
	int dz = plan->imSize_tr[2];
	int dxNext = plan->waveSizes_tr[0 + 3*plan->numLevels_tr];
	int dyNext = plan->waveSizes_tr[1 + 3*plan->numLevels_tr];
	int dzNext = plan->waveSizes_tr[2 + 3*plan->numLevels_tr];
	int blockSize = dxNext*dyNext*dzNext;

	data_t* LxLyLz =  plan->tmp_mem_tr;
	data_t* tempz =  LxLyLz + blockSize;
	data_t* tempyz =  tempz + dx*dy*dzNext;
	data_t* tempxyz =  tempyz + dx*dyNext*dzNext;

	for (l = plan->numLevels_tr; l >= 1; --l)
	{
		dxNext = plan->waveSizes_tr[0 + 3*l];
		dyNext = plan->waveSizes_tr[1 + 3*l];
		dzNext = plan->waveSizes_tr[2 + 3*l];
		blockSize = dxNext*dyNext*dzNext;

		HxLyLz = HxLyLz - 7*blockSize;
		data_t* LxHyLz = HxLyLz + blockSize;
		data_t* HxHyLz = LxHyLz + blockSize;
		data_t* LxLyHz = HxHyLz + blockSize;
		data_t* HxLyHz = LxLyHz + blockSize;
		data_t* LxHyHz = HxLyHz + blockSize;
		data_t* HxHyHz = LxHyHz + blockSize;

		int dxy = dx*dy;
		int newdz = (dz + plan->filterLen-1) / 2;
		int newdy = (dy + plan->filterLen-1) / 2;
		int newdxy = dx*newdy;

		// Lz
		conv_down_3d(tempz, inImage, dz, dxy, dx, 1, dy, dx, plan->lod,plan->filterLen);
		// LyLz
		conv_down_3d(tempyz, tempz, dy, dx, dx, 1, newdz, dxy, plan->lod,plan->filterLen);
		conv_down_3d(LxLyLz, tempyz, dx, 1, newdy, dx, newdz, newdxy, plan->lod,plan->filterLen);
		conv_down_3d(HxLyLz, tempyz, dx, 1, newdy, dx, newdz, newdxy, plan->hid,plan->filterLen);
		// HyLz
		conv_down_3d(tempyz, tempz, dy, dx, dx, 1, newdz, dxy, plan->hid,plan->filterLen);
		conv_down_3d(LxHyLz, tempyz, dx, 1, newdy, dx, newdz, newdxy, plan->lod,plan->filterLen);
		conv_down_3d(HxHyLz, tempyz, dx, 1, newdy, dx, newdz, newdxy, plan->hid,plan->filterLen);
		// Hz
		conv_down_3d(tempz, inImage, dz, dxy, dx, 1, dy, dx, plan->hid,plan->filterLen);
		// LyHz
		conv_down_3d(tempyz, tempz, dy, dx, dx, 1, newdz, dxy, plan->lod,plan->filterLen);
		conv_down_3d(LxLyHz, tempyz, dx, 1, newdy, dx, newdz, newdxy, plan->lod,plan->filterLen);
		conv_down_3d(HxLyHz, tempyz, dx, 1, newdy, dx, newdz, newdxy, plan->hid,plan->filterLen);
		// HyHz
		conv_down_3d(tempyz, tempz, dy, dx, dx, 1, newdz, dxy, plan->hid,plan->filterLen);
		conv_down_3d(LxHyHz, tempyz, dx, 1, newdy, dx, newdz, newdxy, plan->lod,plan->filterLen);
		conv_down_3d(HxHyHz, tempyz, dx, 1, newdy, dx, newdz, newdxy, plan->hid,plan->filterLen);

		memcpy(tempxyz, LxLyLz, blockSize*sizeof(data_t));
		inImage = tempxyz;
		dx = dxNext;
		dy = dyNext;
		dz = dzNext;
	}

	// Final LxLyLz
	memcpy(coeff, inImage, plan->waveSizes_tr[0]*plan->waveSizes_tr[1]*plan->waveSizes_tr[2]*sizeof(data_t));
	circunshift(plan,origInImage);
}

void iwt3_cpu(struct wavelet_plan_s* plan, data_t* outImage, const data_t* coeff)
{
	// Workspace dimensions
	int dxWork = plan->waveSizes_tr[0 + 3*plan->numLevels_tr]*2-1 + plan->filterLen-1;
	int dyWork = plan->waveSizes_tr[1 + 3*plan->numLevels_tr]*2-1 + plan->filterLen-1;
	int dzWork = plan->waveSizes_tr[2 + 3*plan->numLevels_tr]*2-1 + plan->filterLen-1;
	int dyWork2 = plan->waveSizes_tr[1 + 3*(plan->numLevels_tr-1)]*2-1 + plan->filterLen-1;
	int dzWork2 = plan->waveSizes_tr[2 + 3*(plan->numLevels_tr-1)]*2-1 + plan->filterLen-1;
	// Workspace
	data_t* tempyz =  plan->tmp_mem_tr;
	data_t* tempz =   tempyz + dxWork*dyWork2*dzWork2;
	data_t* tempFull =   tempz + dxWork*dyWork*dzWork2;

	int dx = plan->waveSizes_tr[0];
	int dy = plan->waveSizes_tr[1];
	int dz = plan->waveSizes_tr[2];

	memcpy(outImage, coeff, dx*dy*dz*sizeof(data_t));
	const data_t* HxLyLz = coeff + dx*dy*dz;
	int level;
	for (level = 1; level < plan->numLevels_tr+1; ++level)
	{
		dx = (int) plan->waveSizes_tr[0 + 3*level];
		dy = (int) plan->waveSizes_tr[1 + 3*level];
		dz = (int) plan->waveSizes_tr[2 + 3*level];
		int blockSize = dx*dy*dz;

		const data_t* LxHyLz = HxLyLz + blockSize;
		const data_t* HxHyLz = LxHyLz + blockSize;
		const data_t* LxLyHz = HxHyLz + blockSize;
		const data_t* HxLyHz = LxLyHz + blockSize;
		const data_t* LxHyHz = HxLyHz + blockSize;
		const data_t* HxHyHz = LxHyHz + blockSize;
		data_t* LxLyLz = outImage;

		int newdx = 2*dx-1 + plan->filterLen-1;
		int newdy = 2*dy-1 + plan->filterLen-1;
		int newdz = 2*dz-1 + plan->filterLen-1;
		int dxy = dx*dy;
		int newdxy = newdx*dy;
		int newnewdxy = newdx*newdy;


		memset(tempFull, 0, newnewdxy*newdz*sizeof(data_t));
		memset(tempz, 0, newnewdxy*dz*sizeof(data_t));
		memset(tempyz, 0, newdxy*dz*sizeof(data_t));
		conv_up_3d(tempyz, LxLyLz, dx, 1, dy, dx, dz, dxy, plan->lor,plan->filterLen);
		conv_up_3d(tempyz, HxLyLz, dx, 1, dy, dx, dz, dxy, plan->hir,plan->filterLen);
		conv_up_3d(tempz, tempyz, dy, newdx, newdx, 1, dz, newdxy, plan->lor,plan->filterLen);

		memset(tempyz, 0, newdxy*dz*sizeof(data_t));
		conv_up_3d(tempyz, LxHyLz, dx, 1, dy, dx, dz, dxy, plan->lor,plan->filterLen);
		conv_up_3d(tempyz, HxHyLz, dx, 1, dy, dx, dz, dxy, plan->hir,plan->filterLen);
		conv_up_3d(tempz, tempyz, dy, newdx, newdx, 1, dz, newdxy, plan->hir,plan->filterLen);
		conv_up_3d(tempFull, tempz, dz, newnewdxy, newdx, 1, newdy, newdx, plan->lor,plan->filterLen);

		memset(tempz, 0, newnewdxy*dz*sizeof(data_t));
		memset(tempyz, 0, newdxy*dz*sizeof(data_t));
		conv_up_3d(tempyz, LxLyHz, dx, 1, dy, dx, dz, dxy, plan->lor,plan->filterLen);
		conv_up_3d(tempyz, HxLyHz, dx, 1, dy, dx, dz, dxy, plan->hir,plan->filterLen);
		conv_up_3d(tempz, tempyz, dy, newdx, newdx, 1, dz, newdxy, plan->lor,plan->filterLen);

		memset(tempyz, 0, newdxy*dz*sizeof(data_t));
		conv_up_3d(tempyz, LxHyHz, dx, 1, dy, dx, dz, dxy, plan->lor,plan->filterLen);
		conv_up_3d(tempyz, HxHyHz, dx, 1, dy, dx, dz, dxy, plan->hir,plan->filterLen);
		conv_up_3d(tempz, tempyz, dy, newdx, newdx, 1, dz, newdxy, plan->hir,plan->filterLen);

		conv_up_3d(tempFull, tempz, dz, newnewdxy, newdx, 1, newdy, newdx, plan->hir,plan->filterLen);

		// Crop center of workspace
		int dxNext = plan->waveSizes_tr[0+3*(level+1)];
		int dyNext = plan->waveSizes_tr[1+3*(level+1)];
		int dzNext = plan->waveSizes_tr[2+3*(level+1)];
		int dxyNext = dxNext*dyNext;
		dxWork = (2*dx-1 + plan->filterLen-1);
		dyWork = (2*dy-1 + plan->filterLen-1);
		dzWork = (2*dz-1 + plan->filterLen-1);
		int dxyWork = dxWork*dyWork;
		int xOffset = (int) floor((dxWork - dxNext) / 2.0);
		int yOffset = (int) floor((dyWork - dyNext) / 2.0);
		int zOffset = (int) floor((dzWork - dzNext) / 2.0);
		int k,j;
		for (k = 0; k < dzNext; ++k){
			for (j = 0; j < dyNext; ++j){
				memcpy(outImage+j*dxNext + k*dxyNext, tempFull+xOffset + (yOffset+j)*dxWork + (zOffset+k)*dxyWork, dxNext*sizeof(data_t));
			}
		}
		HxLyLz += 7*blockSize;
	}
	circunshift(plan,outImage);
}

void wavthresh3_cpu(struct wavelet_plan_s* plan, data_t* outImage, data_t* inImage, scalar_t thresh)
{
	data_t* coeff;
	coeff = (data_t*)xmalloc(sizeof(data_t)*plan->numCoeff_tr);
	fwt3_cpu(plan, coeff, inImage);
	softthresh_cpu(plan, coeff, thresh);
	iwt3_cpu(plan, outImage, coeff);
	free( coeff );
}


void softthresh_cpu(struct wavelet_plan_s* plan, data_t* coeff, scalar_t thresh)
{
	int numMax = plan->numCoeff_tr;
	int i;
#pragma omp parallel for
//	for(i = plan->numCoarse_tr; i < numMax; i++)
	for(i = 0; i < numMax; i++)
	{
		scalar_t norm = cabsf(coeff[i]);
		scalar_t red = norm - thresh;
		coeff[i] = (red > 0.) ? ((red / norm) * (coeff[i])) : 0.;
	}
}




/********** Helper Function *********/
void conv_down_2d(data_t *out, const data_t *in,
		  int size1, int skip1, int size2, int skip2, const scalar_t *filter, int filterLen)
{
	int outSize1 = (size1 + filterLen-1) / 2;
	// Adjust out skip 2 if needed
	int outSkip2;
	if(skip2 == size1) {
		outSkip2 = outSize1;
	}
	else {
		outSkip2 = skip2;
	}

	int i2, i1, k;
#pragma omp parallel for private(i1,k)
	for (i2 = 0; i2 < size2; ++i2){
		for (i1 = 0; i1 < outSize1; ++i1)
		{
			out[i2*outSkip2 + i1*skip1] = 0.0f;

			for (k = 0; k < filterLen; ++k)
			{
				int new_i1 = 2*i1+1 - (filterLen-1) + k;
				if (new_i1 < 0) new_i1 = -new_i1-1;
				if (new_i1 >= size1) new_i1 = size1-1 - (new_i1-size1);
				out[i2*outSkip2 + i1*skip1] += in[i2*skip2 + new_i1*skip1] * filter[filterLen-1-k];
			}
		}
	}
}

void conv_up_2d(data_t *out, const data_t *in,
		int size1, int skip1, int size2, int skip2, const scalar_t *filter, int filterLen)
{
	int outSize1 = 2*size1-1 + filterLen-1;
	// Adjust out skip 2 if needed
	int outSkip2;
	if(skip2 == size1) {
		outSkip2 = outSize1;
	}
	else {
		outSkip2 = skip2;
	}

	int i2,i1,k;
#pragma omp parallel for private(i1,k)
	for (i2 = 0; i2 < size2; ++i2) {
		for (i1 = 0; i1 < outSize1; ++i1){
			for (k = (i1 - (filterLen-1)) & 1; k < filterLen; k += 2){
				int in_i1 = (i1 - (filterLen-1) + k) >> 1;
				if (in_i1 >= 0 && in_i1 < size1) {
					out[i2*outSkip2 + i1*skip1] += in[i2*skip2 + in_i1*skip1] * filter[filterLen-1-k];
				}
			}
		}
	}
}

void conv_down_3d(data_t *out, const data_t *in,
		  int size1, int skip1, int size2, int skip2, int size3, int skip3,
		  const scalar_t *filter, int filterLen)
{
	int outSize1 = (size1 + filterLen-1) / 2;

	// Adjust out skip 2 and 3 if needed
	int outSkip2;
	if(skip2 > skip1) {
		outSkip2 = outSize1*skip2/size1;
	}
	else {
		outSkip2 = skip2;
	}
	int outSkip3;
	if(skip3 > skip1) {
		outSkip3 = outSize1*skip3/size1;
	}
	else {
		outSkip3 = skip3;
	}
	int i32;
#pragma omp parallel for
	for (i32 = 0; i32 < size2*size3; ++i32)
	{
		int i2 = i32 % size2;
		int i3 = i32 / size2;
		int i1;
		for (i1 = 0; i1 < outSize1; ++i1)
		{
			out[i3*outSkip3 + i2*outSkip2 + i1*skip1] = 0.0f;
			int k;
			for (k = 0; k < filterLen; ++k)
			{
				int out_i1 = 2*i1+1 - (filterLen-1) + k;
				if (out_i1 < 0) out_i1 = -out_i1-1;
				if (out_i1 >= size1) out_i1 = size1-1 - (out_i1-size1);

				out[i3*outSkip3 + i2*outSkip2 + i1*skip1] += in[i3*skip3 + i2*skip2 + out_i1*skip1] * filter[filterLen-1-k];
			}
		}
	}
}

void conv_up_3d(data_t *out, const data_t *in,
		int size1, int skip1, int size2, int skip2, int size3, int skip3,
		const scalar_t *filter, int filterLen)
{
	int outSize1 = 2*size1-1 + filterLen-1;

	// Adjust out skip 2 and 3 if needed
	int outSkip2;
	if(skip2 > skip1) {
		outSkip2 = outSize1*skip2/size1;
	}
	else {
		outSkip2 = skip2;
	}
	int outSkip3;
	if(skip3 > skip1) {
		outSkip3 = outSize1*skip3/size1;
	}
	else {
		outSkip3 = skip3;
	}
	int i32;
#pragma omp parallel for
	for (i32 = 0; i32 < size2*size3; ++i32)
	{
		int i2 = i32 % size2;
		int i3 = i32 / size2;
		int i1;
		for (i1 = 0; i1 < outSize1; ++i1) {
			int k;
			for (k = (i1 - (filterLen-1)) & 1; k < filterLen; k += 2){
				int in_i1 = (i1 - (filterLen-1) + k) >> 1;
				if (in_i1 >= 0 && in_i1 < size1)
					out[i3*outSkip3 + i2*outSkip2 + i1*skip1] += in[i3*skip3 + i2*skip2 + in_i1*skip1] * filter[filterLen-1-k];
			}
		}
	}
}

void create_numLevels(struct wavelet_plan_s* plan)
{
	int numdims_tr = plan->numdims_tr;
	int filterLen = plan->filterLen;
	int bandSize, l, minSize;
	plan->numLevels_tr = 10000000;
	int d;
	for (d = 0; d < numdims_tr; d++)
	{
		bandSize = plan->imSize_tr[d];
		minSize = plan->minSize_tr[d];
		l = 0;
		while (bandSize > minSize)
		{
			++l;
			bandSize = (bandSize + filterLen - 1) / 2;
		}
		l--;
		plan->numLevels_tr = (l < plan->numLevels_tr) ? l : plan->numLevels_tr;
	}
}

void create_wavelet_sizes(struct wavelet_plan_s* plan)
{
	int numdims_tr = plan->numdims_tr;
	int filterLen = plan->filterLen;
	int numLevels_tr = plan->numLevels_tr;
	int numSubCoef;
	plan->waveSizes_tr = (long*)xmalloc(sizeof(long) * numdims_tr * (numLevels_tr + 2));

	// Get number of subband per level, (3 for 2d, 7 for 3d)
	// Set the last bandSize to be imSize
	int d,l;
	int numSubband = 1;
	for (d = 0; d<numdims_tr; d++)
	{
		plan->waveSizes_tr[d + numdims_tr*(numLevels_tr+1)] = plan->imSize_tr[d];
		numSubband <<= 1;
	}
	numSubband--;

	// Get numCoeff and waveSizes
	// Each bandSize[l] is (bandSize[l+1] + filterLen - 1)/2
	plan->numCoeff_tr = 0;
	for (l = plan->numLevels_tr; l >= 1; --l) {
		numSubCoef = 1;
		for (d = 0; d < numdims_tr; d++)
		{
			plan->waveSizes_tr[d + numdims_tr*l] = (plan->waveSizes_tr[d + numdims_tr*(l+1)] + filterLen - 1) / 2;
			numSubCoef *= plan->waveSizes_tr[d + numdims_tr*l];
		}
		plan->numCoeff_tr += numSubband*numSubCoef;
		if (l==1)
			plan->numCoarse_tr = numSubCoef;
	}

	numSubCoef = 1;
	for (d = 0; d < numdims_tr; d++)
	{
		plan->waveSizes_tr[d] = plan->waveSizes_tr[numdims_tr+d];
		numSubCoef *= plan->waveSizes_tr[d];
	}
	plan->numCoeff_tr += numSubCoef;

	// Get Actual numCoeff
	plan->numCoeff = plan->numCoeff_tr;
	for (d = 0; d<plan->numdims; d++)
	{
		if (!MD_IS_SET(plan->flags, d))
			plan->numCoeff *= plan->imSize[d];
	}
}


extern long get_numCoeff_tr(struct wavelet_plan_s* plan)
{
	return plan->numCoeff_tr;
}

const float wavelet2_haar[4][2] = {
	{	0.7071067812,	0.7071067812	},
	{	-0.7071067812,	0.7071067812	},
	{	0.7071067812,	0.7071067812	},
	{	0.7071067812,	-0.7071067812	}
};

const float wavelet2_dau2[4][4] = {
	{ -0.129410,  0.224144,  0.836516,  0.482963 },
	{ -0.482963,  0.836516, -0.224144, -0.129410 },
	{ 0.482963,  0.836516,  0.224144, -0.129410 },
	{ -0.129410, -0.224144,  0.836516, -0.482963 }
};

const float wavelet2_cdf44[4][10] = {
	{ 0.0, 0.03782845550726404, -0.023849465019556843, -0.11062440441843718, 0.37740285561283066, 0.85269867900889385, 
	  0.37740285561283066, -0.11062440441843718, -0.023849465019556843, 0.03782845550726404 },
	{ 0.0, -0.064538882628697058, 0.040689417609164058, 0.41809227322161724, -0.7884856164055829, 0.41809227322161724, 
	  0.040689417609164058, -0.064538882628697058, 0.0, 0.0 },
	{ 0.0, -0.064538882628697058, -0.040689417609164058, 0.41809227322161724, 0.7884856164055829, 0.41809227322161724,
	  -0.040689417609164058, -0.064538882628697058, 0.0, 0.0 },
	{ 0.0, -0.03782845550726404, -0.023849465019556843, 0.11062440441843718, 0.37740285561283066, -0.85269867900889385, 
	  0.37740285561283066, 0.11062440441843718, -0.023849465019556843, -0.03782845550726404 }
};


void circshift(struct wavelet_plan_s* plan, data_t* data) 
{
	bool shift = false;
	int N = plan->numdims_tr;

	for (int i = 0; i < N; i++)
		shift |= (0 != plan->randShift_tr[i]);

	if (shift) {

		void* data_copy = md_alloc_sameplace(N, plan->imSize_tr, sizeof(data_t), data);
		md_copy(N, plan->imSize_tr, data_copy, data, sizeof(data_t));
		md_circ_shift(N, plan->imSize_tr, plan->randShift_tr, data, data_copy, sizeof(data_t));
		md_free(data_copy);

	}
}

void circunshift(struct wavelet_plan_s* plan, data_t* data) 
{
	bool shift = false;
	int N = plan->numdims_tr;
	long rand_shifts[N];

	for (int i = 0; i < N; i++) {

		shift |= (0 != plan->randShift_tr[i]);
		rand_shifts[i] = -plan->randShift_tr[i];
	}

	if (shift) {

		void* data_copy = md_alloc_sameplace(N, plan->imSize_tr, sizeof(data_t), data);
		md_copy(N, plan->imSize_tr, data_copy, data, sizeof(data_t));
		md_circ_shift(N, plan->imSize_tr, rand_shifts, data, data_copy, sizeof(data_t));
		md_free(data_copy);

	}
}


