/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2014 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2013-2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2014      Frank Ong <frankong@berkeley.edu>
 *
 *
 * Ra JB, Rim CY. Fast imaging using subencoding data sets from multiple
 * detectors. Magn Reson Med 1993; 30:142-145.
 *
 * Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: Sensitivity
 * encoding for fast MRI. Magn Reson Med 1999; 42:952-962.
 *
 * Pruessmann KP, Weiger M, Boernert P, Boesiger P. Advances in sensitivity
 * encoding with arbitrary k-space trajectories. 
 * Magn Reson Med 2001; 46:638-651.
 *
 * Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS,
 * Lustig M. ESPIRiT - An Eigenvalue Approach to Autocalibrating Parallel MRI:
 * Where SENSE meets GRAPPA. Magn Reson Med 2014; 71:990-1001.
 *
 * Boyd S, Parikh N, Chu E, Peleato B, Eckstein J. Distributed Optimization and
 * Statistical Learning via the Alternating Direction Method of Multipliers
 *
 */

#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/realval.h"
#include "linops/sampling.h"

#include "iter/iter2.h"
#include "iter/prox.h"
#include "iter/monitor.h"


#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "sense/model.h"

#include "bprecon.h"


const struct bpsense_conf bpsense_defaults = {

	.iconf = NULL,
	.rvc = false,
	.lambda = 0.,
	.eps = .01,

	.l1op_obj = NULL,
};



/**
 * Data structure for storing all relevant recon information
 *
 * @param prox_ops array of proximal operators passed to iter2 interface
 * @param linops array of linear operators passed to iter2 interface
 * @param kspace original kspace data
 * @param conf bpsense configuration
 */
struct data {

	const struct linop_s** linops;
	const struct operator_p_s** prox_ops; 
	const complex float* kspace;

	struct bpsense_conf* conf;
};




/**
 * Computes the objective value, || T x ||_1 + lambda/2 || x ||_2^2
 * where the T linear operator corresponds to conf->linop_obj.
 * Also prints the data consistency error, || y - A x ||_2
 */
static float bpsense_objective(const void* _data, const float* _x)
{
	const struct data* data = _data;
	const complex float* x = (const complex float*)_x;

	const struct linop_s* Aop = data->linops[0];

	complex float* tmp = md_alloc_sameplace(linop_codomain(data->conf->l1op_obj)->N, 
				linop_codomain(data->conf->l1op_obj)->dims, CFL_SIZE, x);

	linop_forward_unchecked(data->conf->l1op_obj, tmp, x);

	float t1 = md_z1norm(linop_codomain(data->conf->l1op_obj)->N, linop_codomain(data->conf->l1op_obj)->dims, tmp);

	float t2 = 0.;
	if (data->conf->lambda > 0.)
		t2 = data->conf->lambda * 0.5 * md_znorm(DIMS, linop_domain(Aop)->dims, x);

	complex float* tmp2 = md_alloc_sameplace(DIMS, linop_codomain(Aop)->dims, CFL_SIZE, x);
	linop_forward_unchecked(Aop, tmp2, x);
	float t3 = md_znorme(DIMS, linop_codomain(Aop)->dims, tmp2, data->kspace);

	debug_printf(DP_DEBUG4, "# || y - A x ||_2  = %f\n", t3);

	md_free(tmp);
	md_free(tmp2);
	return t1 + t2;
}


/**
 * Perform basis pursuit SENSE reconstruction
 *
 * @param conf bpsense config
 * @param dims dimensions of sensitivity maps
 * @param image image
 * @param maps sensitivity maps
 * @param pat_dims dimensions of kspace sampling pattern
 * @param pattern kspace sampling pattern mask
 * @param l1op transform operator for l1-minimization
 * @param l1prox proximal function for the l1-norm
 * @param ksp_dims dimensions of kspace
 * @param kspace kspace data
 * @param image_truth truth image to compare
 */
void bpsense_recon(struct bpsense_conf* conf, const long dims[DIMS], complex float* image, const complex float* maps,
		 const long pat_dims[DIMS], const complex float* pattern, 
		 const struct linop_s* l1op,
		 const struct operator_p_s* l1prox,
		 const long ksp_dims[DIMS], const complex float* kspace, const complex float* image_truth)
{
	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims);


	// -----------------------------------------------------------
	// forward model

	struct linop_s* sense_op = sense_init(dims, FFT_FLAGS|COIL_FLAG|MAPS_FLAG, maps);

	if (conf->rvc) {

		struct linop_s* rvc = linop_realval_create(DIMS, img_dims);
		struct linop_s* tmp_op = linop_chain(rvc, sense_op);

		linop_free(rvc);
		linop_free(sense_op);
		sense_op = tmp_op;
	}

	const struct linop_s* sample_op = linop_sampling_create(dims, pat_dims, pattern);
	const struct linop_s* Aop = linop_chain(sense_op, sample_op);
	linop_free(sense_op);
	linop_free(sample_op);

	
	// -----------------------------------------------------------
	// proximal operators

	const struct operator_p_s* l2ball_prox = prox_l2ball_create(DIMS, ksp_dims, conf->eps, kspace);
	const struct operator_p_s* leastsquares_prox = prox_leastsquares_create(DIMS, img_dims, conf->lambda, NULL);

	const struct operator_p_s* prox_ops[3] = { l2ball_prox, l1prox, leastsquares_prox };

	const struct linop_s* eye = linop_identity_create(DIMS, img_dims);
	const struct linop_s* linops[3] = { Aop, l1op, eye };

	long size = 2 * md_calc_size(DIMS, img_dims);


	// -----------------------------------------------------------
	// data storage

	PTR_ALLOC(struct data, data);
	data->kspace = kspace;
	data->linops = linops;
	data->prox_ops = prox_ops;
	data->conf = conf;


	// -----------------------------------------------------------
	// recon
	
	iter2_admm(conf->iconf, NULL, conf->lambda == 0. ? 2 : 3, prox_ops, linops, NULL, NULL, size, (float*)image, NULL, create_monitor(size, (const float*)image_truth, data, bpsense_objective));


	// -----------------------------------------------------------
	// clean up

	linop_free(Aop);
	linop_free(eye);
	operator_p_free(l2ball_prox);
	operator_p_free(leastsquares_prox);
	free(data);
}




/**
 * Wrapper for bpsense_recon on GPU
 */
#ifdef USE_CUDA
void bpsense_recon_gpu(struct bpsense_conf* conf, const long dims[DIMS], complex float* image, const complex float* maps,
		     const long dims_pat[DIMS], const complex float* pattern,
			  const struct linop_s* l1op,
			  const struct operator_p_s* l1prox,
		     const long ksp_dims[DIMS], const complex float* kspace, const complex float* image_truth)
{
	long img_dims[DIMS];

	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims);

	complex float* gpu_maps = md_gpu_move(DIMS, dims, maps, CFL_SIZE);
	complex float* gpu_pat = md_gpu_move(DIMS, dims_pat, pattern, CFL_SIZE);
	complex float* gpu_ksp = md_gpu_move(DIMS, ksp_dims, kspace, CFL_SIZE);
	complex float* gpu_img = md_gpu_move(DIMS, img_dims, image, CFL_SIZE);
	complex float* gpu_img_truth = md_gpu_move(DIMS, img_dims, image_truth, CFL_SIZE);

	bpsense_recon(conf, dims, gpu_img, gpu_maps, dims_pat, gpu_pat, l1op, l1prox, ksp_dims, gpu_ksp, gpu_img_truth);

	md_copy(DIMS, img_dims, image, gpu_img, CFL_SIZE);

	md_free(gpu_img_truth);
	md_free(gpu_img);
	md_free(gpu_pat);
	md_free(gpu_ksp);
	md_free(gpu_maps);
}
#endif

