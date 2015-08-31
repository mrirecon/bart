/* Copyright 2013-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2015 Martin Uecker <uecker@eecs.berkeley.edu>
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
 */

#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"
#include "linops/sampling.h"
#include "linops/rvc.h"

#include "iter/iter.h"
#include "iter/lsqr.h"

#ifdef BERKELEY_SVN
#include "iter/lad.h"
#endif

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "sense/model.h"

#include "recon.h"

const struct sense_conf sense_defaults = {

	.rvc = false,
	.rwiter = 1,
	.gamma = -1.,
	.cclambda = 0.,
};



/**
 * Data structure for storing all relevant recon information
 *
 * @param pattern sampling mask
 * @param transfer_data optional data to be applied to transfer function
 * @param transfer optional transfer function to apply normal equations (e.g. weights)
 * @param sense_data data structure for holding sense information
 * @param tmp temporary storage in kspace domain
 * @param conf sense configuration
 * @param img_dims dimensions of image
 */
struct data {

 	// Optional function to apply normal equations:
	//     For example, used for sampling mask, weights
/*	const */ complex float* pattern;
	  
	const struct operator_s* sense_data;
	complex float* tmp;
	const complex float* kspace;

	struct sense_conf* conf;

	long img_dims[DIMS];
	long ksp_dims[DIMS];
	long pat_dims[DIMS];
};


void debug_print_sense_conf(int level, const struct sense_conf* conf)
{
	debug_printf(level, "sense conf:\n");
	debug_printf(level, "\trvc:          %s\n", conf->rvc ? "on" : "off");
	debug_printf(level, "\trwiter:       %d\n", conf->rwiter);
	debug_printf(level, "\tgamma:        %f\n", conf->gamma);
	debug_printf(level, "\tcclambda:     %f\n", conf->cclambda);
	debug_printf(level, "\n\n");
}



// copied from flpmath.c (for md_sqrt below )
static void real_from_complex_dims(unsigned int D, long odims[D + 1], const long idims[D])
{
	odims[0] = 2;
	md_copy_dims(D, odims + 1, idims);
}

/**
 * Perform iterative, regularized sense reconstruction.
 *
 * @param conf sense config
 * @param dims dimensions of sensitivity maps
 * @param image image
 * @param op sense forward operator
 * @param pat_dims dimensions of kspace sampling pattern
 * @param pattern kspace sampling pattern mask
 * @param kspace kspace data
 */
void sense_recon(const struct sense_conf* conf, const long dims[DIMS], complex float* image,
		 const struct linop_s* sense_op,
		 const long pat_dims[DIMS], const complex float* pattern, 
		 italgo_fun_t italgo, void* iconf,
		 const struct operator_p_s* thresh_op,
		 const long ksp_dims[DIMS], const complex float* kspace, const complex float* image_truth)
{
	sense_recon2(conf, dims, image, sense_op, pat_dims, pattern,
		iter2_call_iter, &(struct iter_call_s){ italgo, iconf },
		1, MAKE_ARRAY(thresh_op), NULL, ksp_dims, kspace, image_truth);
}

void sense_recon2(const struct sense_conf* conf, const long dims[DIMS], complex float* image,
		const struct linop_s* sense_op,
		const long pat_dims[DIMS], const complex float* pattern,
		italgo_fun2_t italgo, void* iconf,
		unsigned int num_funs,
		const struct operator_p_s* thresh_op[num_funs],
		const struct linop_s* thresh_funs[num_funs],
		const long ksp_dims[DIMS], const complex float* kspace, const complex float* image_truth)
{
	UNUSED(image_truth);

	// iterative algorithm

	struct lsqr_conf lsqr_conf = { conf->cclambda };

	// initialize data as struct to hold all sense data and operators

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims);


	if (conf->rvc) {

		struct linop_s* rvc = rvc_create(DIMS, img_dims);
		struct linop_s* tmp_op = linop_chain(rvc, sense_op);

		linop_free(rvc);
		linop_free(sense_op);
		sense_op = tmp_op;
	}

	if (1 == conf->rwiter) {

		if (NULL == pattern) {

			lsqr2(DIMS, &lsqr_conf, italgo, iconf, sense_op, num_funs, thresh_op, thresh_funs,
				img_dims, image, ksp_dims, kspace, NULL, NULL, NULL);

		} else {

			complex float* weights = md_alloc_sameplace(DIMS, pat_dims, CFL_SIZE, kspace);
#if 0
			// buggy
//			md_zsqrt(DIMS, pat_dims, weights, pattern);
#else
			long dimsR[DIMS + 1];
			real_from_complex_dims(DIMS, dimsR, pat_dims);
			md_sqrt(DIMS + 1, dimsR, (float*)weights, (const float*)pattern);
#endif

			wlsqr2(DIMS, &lsqr_conf, italgo, iconf, sense_op, num_funs, thresh_op, thresh_funs,
				img_dims, image, ksp_dims, kspace, pat_dims, weights);

			md_free(weights);
		}

	} else {
#ifdef BERKELEY_SVN
		struct linop_s* sampling = sampling_create(dims, pat_dims, pattern);
		struct linop_s* tmp_op = linop_chain(sense_op, sampling);

		linop_free(sampling);
		linop_free(sense_op);
		sense_op = tmp_op;

		unsigned int flags = 0;
		for (unsigned int i = 0; i < DIMS; i++)
			if (pat_dims[i] > 1)
				flags = MD_SET(flags, i);

		const struct lad_conf lad_conf = { conf->rwiter, conf->gamma, flags, &lsqr_conf };

		lad2(DIMS, &lad_conf, italgo, iconf, sense_op, num_funs, thresh_op, thresh_funs,
				img_dims, image, ksp_dims, kspace);
#else
		assert(0);
#endif
	}

	// clean up
	//linop_free(sense_op);
}





/**
 * Wrapper for sense_recon on GPU
 *
 * @param conf sense config
 * @param dims dimensions of sensitivity maps
 * @param image image
 * @param op sense forward operator
 * @param dims_pat dimensions of kspace sampling pattern
 * @param pattern kspace sampling pattern mask
 * @param tf transfer function for applying pattern
 * @param tf_data data associated with transfer function
 * @param kspace kspace data
 */
#ifdef USE_CUDA
void sense_recon_gpu(const struct sense_conf* conf, const long dims[DIMS], complex float* image,
		 const struct linop_s* sense_op,
		 const long pat_dims[DIMS], const complex float* pattern, 
		 italgo_fun_t italgo, void* iconf,
		 const struct operator_p_s* thresh_op,
		 const long ksp_dims[DIMS], const complex float* kspace, const complex float* image_truth)
{
	sense_recon2_gpu(conf, dims, image, sense_op, pat_dims, pattern,
		iter2_call_iter, &(struct iter_call_s){ italgo, iconf },
		1, MAKE_ARRAY(thresh_op), NULL, ksp_dims, kspace, image_truth);
}

void sense_recon2_gpu(const struct sense_conf* conf, const long dims[DIMS], complex float* image,
		const struct linop_s* sense_op,
		const long dims_pat[DIMS], const complex float* pattern,
		italgo_fun2_t italgo, void* iter_conf,
		unsigned int num_funs,
		const struct operator_p_s* thresh_op[num_funs],
		const struct linop_s* thresh_funs[num_funs],
		const long ksp_dims[DIMS], const complex float* kspace, const complex float* image_truth)
{
	long dims_ksp[DIMS];
	long dims_img[DIMS];

	md_select_dims(DIMS, ~MAPS_FLAG, dims_ksp, dims);
	md_select_dims(DIMS, ~COIL_FLAG, dims_img, dims);

	complex float* gpu_pat = md_gpu_move(DIMS, dims_pat, pattern, CFL_SIZE);
	complex float* gpu_ksp = md_gpu_move(DIMS, dims_ksp, kspace, CFL_SIZE);
	complex float* gpu_img = md_gpu_move(DIMS, dims_img, image, CFL_SIZE);
	complex float* gpu_img_truth = md_gpu_move(DIMS, dims_img, image_truth, CFL_SIZE);

	sense_recon2(conf, dims, gpu_img, sense_op, dims_pat, gpu_pat, italgo, iter_conf,
				1, thresh_op, thresh_funs, ksp_dims, gpu_ksp, gpu_img_truth);

	md_copy(DIMS, dims_img, image, gpu_img, CFL_SIZE);

	md_free(gpu_img_truth);
	md_free(gpu_img);
	md_free(gpu_pat);
	md_free(gpu_ksp);
}


#endif

