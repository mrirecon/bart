/* 
 * Copyright 2013-2015 The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015 Frank Ong <frankong@berkeley.edu>
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 * Ong F, Uecker M, Tariq U, Hsiao A, Alley MT, Vasanawala SS, Lustig M.
 * Robust 4D Flow Denoising using Divergence-free Wavelet Transform, 
 * Magn Reson Med 2015; 73: 828-842.
 */

#define _GNU_SOURCE
#include <math.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mri.h"

#include "linops/linop.h"
#include "linops/waveop.h"

#include "iter/thresh.h"

#include "dfwavelet.h"
#include "prox_dfwavelet.h"


static void prox_dfwavelet_del(const operator_data_t* _data);
static void prox_dfwavelet_thresh(const operator_data_t* _data, float thresh, complex float* out, const complex float* in);
static struct prox_dfwavelet_data* prepare_prox_dfwavelet_data(const long im_dims[DIMS], const long min_size[3], const complex float res[3], unsigned int flow_dim, float lambda, bool use_gpu);


static void prox_4pt_dfwavelet_del(const operator_data_t* _data);
static void prox_4pt_dfwavelet_thresh(const operator_data_t* _data, float thresh, complex float* out, const complex float* in);
static struct prox_4pt_dfwavelet_data*  prepare_prox_4pt_dfwavelet_data(const long im_dims[DIMS], const long min_size[3], const complex float res[3], unsigned int flow_dim, float lambda, bool use_gpu);




struct prox_dfwavelet_data {

	INTERFACE(operator_data_t);

	bool use_gpu;
        unsigned int slice_flag;
        unsigned int flow_dim;
        float lambda;

	long im_dims[DIMS];
	long tim_dims[DIMS];
	long im_strs[DIMS];

	complex float* vx;
	complex float* vy;
	complex float* vz;

        struct dfwavelet_plan_s* plan;
};

DEF_TYPEID(prox_dfwavelet_data);


const struct operator_p_s* prox_dfwavelet_create(const long im_dims[DIMS], const long min_size[3], const complex float res[3], unsigned int flow_dim, float lambda, bool use_gpu)
{
	struct prox_dfwavelet_data* data = prepare_prox_dfwavelet_data(im_dims, min_size, res, flow_dim, lambda, use_gpu);
        
	return operator_p_create(DIMS, im_dims, DIMS, im_dims, CAST_UP(data), prox_dfwavelet_thresh, prox_dfwavelet_del);

}

struct prox_dfwavelet_data* prepare_prox_dfwavelet_data(const long im_dims[DIMS], const long min_size[3], const complex float res[3], unsigned int flow_dim, float lambda, bool use_gpu)
{
        // get dimension
        PTR_ALLOC(struct prox_dfwavelet_data, data);
	SET_TYPEID(prox_dfwavelet_data, data);

        md_copy_dims(DIMS, data->im_dims, im_dims);
        md_select_dims(DIMS, FFT_FLAGS, data->tim_dims, im_dims);
        md_calc_strides(DIMS, data->im_strs, im_dims, CFL_SIZE);

        // initialize temp
        
#ifdef USE_CUDA
        if (use_gpu) {

                data->vx = md_alloc_gpu(DIMS, data->tim_dims, CFL_SIZE);
                data->vy = md_alloc_gpu(DIMS, data->tim_dims, CFL_SIZE);
                data->vz = md_alloc_gpu(DIMS, data->tim_dims, CFL_SIZE);

        } else
 #endif
        {
                data->vx = md_alloc(DIMS, data->tim_dims, CFL_SIZE);
                data->vy = md_alloc(DIMS, data->tim_dims, CFL_SIZE);
                data->vz = md_alloc(DIMS, data->tim_dims, CFL_SIZE);
        }

        data->flow_dim = flow_dim;
        data->slice_flag = ~FFT_FLAGS;
        data->lambda = lambda;

        data->plan = prepare_dfwavelet_plan(3, data->tim_dims, (long*) min_size, (complex float*) res, use_gpu);
        
        return PTR_PASS(data);
}


static void prox_dfwavelet_del(const operator_data_t* _data)
{
	struct prox_dfwavelet_data* data = CAST_DOWN(prox_dfwavelet_data, _data);

        md_free(data->vx);
        md_free(data->vy);
        md_free(data->vz);
        dfwavelet_free(data->plan);

	free(data);
}



static void prox_dfwavelet_thresh(const operator_data_t* _data, float thresh, complex float* out, const complex float* in)
{
	struct prox_dfwavelet_data* data = CAST_DOWN(prox_dfwavelet_data, _data);

        bool done = false;
        long pos[DIMS];
        md_set_dims(DIMS, pos, 0);
        
        while (!done) {

                // copy vx, vy, vz
                md_slice(DIMS, data->slice_flag, pos, data->im_dims, data->vx, in, CFL_SIZE);
                pos[data->flow_dim]++;
                md_slice(DIMS, data->slice_flag, pos, data->im_dims, data->vy, in, CFL_SIZE);
                pos[data->flow_dim]++;
                md_slice(DIMS, data->slice_flag, pos, data->im_dims, data->vz, in, CFL_SIZE);
                pos[data->flow_dim]=0;

                // threshold
                dfwavelet_thresh(data->plan, thresh * data->lambda, thresh* data->lambda, data->vx, data->vy, data->vz, data->vx, data->vy, data->vz);

                // copy vx, vy, vz
                md_copy_block(DIMS, pos, data->im_dims, out, data->tim_dims, data->vx, CFL_SIZE);
                pos[data->flow_dim]++;
                md_copy_block(DIMS, pos, data->im_dims, out, data->tim_dims, data->vy, CFL_SIZE);
                pos[data->flow_dim]++;
                md_copy_block(DIMS, pos, data->im_dims, out, data->tim_dims, data->vz, CFL_SIZE);
                pos[data->flow_dim]=0;

                // increment pos
                long carryon = 1;

                for (unsigned int i = 0; i < DIMS; i++) {

                        if (MD_IS_SET(data->slice_flag & ~MD_BIT(data->flow_dim), i)) {

                                pos[i] += carryon;

                                if (pos[i] < data->im_dims[i]) {

                                        carryon = 0;
                                        break;

                                } else {

                                        carryon = 1;
                                        pos[i] = 0;
                                }
                        }
                }

                done = carryon;
        }
}


struct prox_4pt_dfwavelet_data {

	INTERFACE(operator_data_t);

	bool use_gpu;
        unsigned int slice_flag;
        unsigned int flow_dim;
        float lambda;

	long im_dims[DIMS];
	long tim_dims[DIMS];
	long im_strs[DIMS];
        
	complex float* vx;
	complex float* vy;
	complex float* vz;
        complex float* ph0;
        
	complex float* pc0;
	complex float* pc1;
	complex float* pc2;
        complex float* pc3;

        struct dfwavelet_plan_s* plan;
        const struct linop_s* w_op;
        const struct operator_p_s* wthresh_op;
};

DEF_TYPEID(prox_4pt_dfwavelet_data);



const struct operator_p_s* prox_4pt_dfwavelet_create(const long im_dims[DIMS], const long min_size[3], const complex float res[3], unsigned int flow_dim, float lambda, bool use_gpu)
{
	struct prox_4pt_dfwavelet_data* data = prepare_prox_4pt_dfwavelet_data(im_dims, min_size, res, flow_dim, lambda, use_gpu);
        
	return operator_p_create(DIMS, im_dims, DIMS, im_dims, CAST_UP(data), prox_4pt_dfwavelet_thresh, prox_4pt_dfwavelet_del);
}

struct prox_4pt_dfwavelet_data* prepare_prox_4pt_dfwavelet_data(const long im_dims[DIMS], const long min_size[3], const complex float res[3], unsigned int flow_dim, float lambda, bool use_gpu)
{
	PTR_ALLOC(struct prox_4pt_dfwavelet_data, data);
	SET_TYPEID(prox_4pt_dfwavelet_data, data);

        md_copy_dims(DIMS, data->im_dims, im_dims);
        md_select_dims(DIMS, FFT_FLAGS, data->tim_dims, im_dims);
        md_calc_strides(DIMS, data->im_strs, im_dims, CFL_SIZE);

        assert(4 == im_dims[flow_dim]);

        // initialize temp
        
#ifdef USE_CUDA
        if (use_gpu) {

                data->vx = md_alloc_gpu(DIMS, data->tim_dims, CFL_SIZE);
                data->vy = md_alloc_gpu(DIMS, data->tim_dims, CFL_SIZE);
                data->vz = md_alloc_gpu(DIMS, data->tim_dims, CFL_SIZE);
                data->ph0 = md_alloc_gpu(DIMS, data->tim_dims, CFL_SIZE);
                data->pc0 = md_alloc_gpu(DIMS, data->tim_dims, CFL_SIZE);
                data->pc1 = md_alloc_gpu(DIMS, data->tim_dims, CFL_SIZE);
                data->pc2 = md_alloc_gpu(DIMS, data->tim_dims, CFL_SIZE);
                data->pc3 = md_alloc_gpu(DIMS, data->tim_dims, CFL_SIZE);

        } else
 #endif
        {
                data->vx = md_alloc(DIMS, data->tim_dims, CFL_SIZE);
                data->vy = md_alloc(DIMS, data->tim_dims, CFL_SIZE);
                data->vz = md_alloc(DIMS, data->tim_dims, CFL_SIZE);
                data->ph0 = md_alloc(DIMS, data->tim_dims, CFL_SIZE);
                data->pc0 = md_alloc(DIMS, data->tim_dims, CFL_SIZE);
                data->pc1 = md_alloc(DIMS, data->tim_dims, CFL_SIZE);
                data->pc2 = md_alloc(DIMS, data->tim_dims, CFL_SIZE);
                data->pc3 = md_alloc(DIMS, data->tim_dims, CFL_SIZE);
        }

        data->flow_dim = flow_dim;
        data->slice_flag = ~FFT_FLAGS;
        data->lambda = lambda;

        data->plan = prepare_dfwavelet_plan(3, data->tim_dims, (long*) min_size, (complex float*) res, use_gpu);

	long strs[DIMS];
	md_calc_strides(DIMS, strs, data->tim_dims, CFL_SIZE);

	data->w_op = linop_wavelet3_create(DIMS, FFT_FLAGS, data->tim_dims, strs, min_size);
        data->wthresh_op = prox_unithresh_create(DIMS, data->w_op, lambda, MD_BIT(data->flow_dim), use_gpu);

        return PTR_PASS(data);
}


static void prox_4pt_dfwavelet_del(const operator_data_t* _data)
{
	struct prox_4pt_dfwavelet_data* data = CAST_DOWN(prox_4pt_dfwavelet_data, _data);

        md_free(data->vx);
        md_free(data->vy);
        md_free(data->vz);
        md_free(data->ph0);
        md_free(data->pc0);
        md_free(data->pc1);
        md_free(data->pc2);
        md_free(data->pc3);

        dfwavelet_free(data->plan);
	operator_p_free(data->wthresh_op);
	linop_free(data->w_op);

        free(data);
}




static void prox_4pt_dfwavelet_thresh(const operator_data_t* _data, float thresh, complex float* out, const complex float* in)
{
	struct prox_4pt_dfwavelet_data* data = CAST_DOWN(prox_4pt_dfwavelet_data, _data);

        bool done = false;
        long pos[DIMS];
        md_set_dims(DIMS, pos, 0);
        
        while (!done) {

                // copy pc
                md_slice(DIMS, data->slice_flag, pos, data->im_dims, data->pc0, in, CFL_SIZE);
                pos[data->flow_dim]++;
                md_slice(DIMS, data->slice_flag, pos, data->im_dims, data->pc1, in, CFL_SIZE);
                pos[data->flow_dim]++;
                md_slice(DIMS, data->slice_flag, pos, data->im_dims, data->pc2, in, CFL_SIZE);
                pos[data->flow_dim]++;
                md_slice(DIMS, data->slice_flag, pos, data->im_dims, data->pc3, in, CFL_SIZE);
                pos[data->flow_dim] = 0;

                // pc to velocity
                // TODO: Make gpu 
                for (int i = 0; i < md_calc_size(DIMS, data->tim_dims); i++) {

                        data->vx[i] = (data->pc1[i] - data->pc0[i]) / 2;
                        data->vy[i] = (data->pc2[i] - data->pc1[i]) / 2;
                        data->vz[i] = (data->pc3[i] - data->pc2[i]) / 2;
                        data->ph0[i] = (data->pc0[i] + data->pc3[i]) / 2;
                }

                // threshold
                dfwavelet_thresh(data->plan, thresh * data->lambda, thresh* data->lambda, data->vx, data->vy, data->vz, data->vx, data->vy, data->vz);
                operator_p_apply(data->wthresh_op, thresh, DIMS, data->tim_dims, data->ph0, DIMS, data->tim_dims, data->ph0);

                // velocity to pc
                for (int i = 0; i < md_calc_size(DIMS, data->tim_dims ); i++) {

                        data->pc0[i] = (- data->vx[i] - data->vy[i] - data->vz[i] + data->ph0[i]);
                        data->pc1[i] = (+ data->vx[i] - data->vy[i] - data->vz[i] + data->ph0[i]);
                        data->pc2[i] = (+ data->vx[i] + data->vy[i] - data->vz[i] + data->ph0[i]);
                        data->pc3[i] = (+ data->vx[i] + data->vy[i] + data->vz[i] + data->ph0[i]);
                }
                
                
                // copy pc
                md_copy_block(DIMS, pos, data->im_dims, out, data->tim_dims, data->pc0, CFL_SIZE);
                pos[data->flow_dim]++;
                md_copy_block(DIMS, pos, data->im_dims, out, data->tim_dims, data->pc1, CFL_SIZE);
                pos[data->flow_dim]++;
                md_copy_block( DIMS, pos, data->im_dims, out, data->tim_dims, data->pc2, CFL_SIZE );
                pos[data->flow_dim]++;
                md_copy_block( DIMS, pos, data->im_dims, out, data->tim_dims, data->pc3, CFL_SIZE );
                pos[data->flow_dim] = 0;

                // increment pos
                long carryon = 1;

                for(unsigned int i = 0; i < DIMS; i++) {

                        if (MD_IS_SET(data->slice_flag & ~MD_BIT(data->flow_dim), i)) {

                                pos[i] += carryon;

                                if (pos[i] < data->im_dims[i]) {

                                        carryon = 0;
                                        break;

                                } else {

                                        carryon = 1;
                                        pos[i] = 0;
                                }
                        }
                }

                done = carryon;
        }
}
