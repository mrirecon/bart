/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "linops/linop.h"
#include "linops/sampling.h"
#include "linops/someops.h"
#include "linops/realval.h"

#include "iter/iter.h"
#include "iter/lsqr.h"
#include "iter/lad.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "sense/model.h"

#include "recon.h"

const struct sense_conf sense_defaults = {

	.rvc = false,
	.gpu = false,
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



const struct operator_s* sense_recon_create(const struct sense_conf* conf, const long dims[DIMS],
		  const struct linop_s* sense_op,
		  const long pat_dims[DIMS], const complex float* pattern,
		  italgo_fun2_t italgo, iter_conf* iconf,
		  const complex float* init,
		  unsigned int num_funs,
		  const struct operator_p_s* thresh_op[num_funs],
		  const struct linop_s* thresh_funs[num_funs],
		  const struct operator_s* precond_op)
{
	struct lsqr_conf lsqr_conf = { conf->cclambda, conf->gpu };

	const struct operator_s* op = NULL;

	assert(DIMS == linop_domain(sense_op)->N);

	long img_dims[DIMS];
	md_copy_dims(DIMS, img_dims, linop_domain(sense_op)->dims);

	long ksp_dims[DIMS];
	md_copy_dims(DIMS, ksp_dims, linop_codomain(sense_op)->dims);

	if (conf->rvc) {

		struct linop_s* rvc = linop_realval_create(DIMS, img_dims);
		struct linop_s* tmp_op = linop_chain(rvc, sense_op);

		linop_free(rvc);
		linop_free(sense_op);
		sense_op = tmp_op;
	}

	if (1 < conf->rwiter) {

		struct linop_s* sampling = linop_sampling_create(dims, pat_dims, pattern);
		struct linop_s* tmp_op = linop_chain(sense_op, sampling);

		linop_free(sampling);
		linop_free(sense_op);
		sense_op = tmp_op;

		unsigned int flags = 0;
		for (unsigned int i = 0; i < DIMS; i++)
			if (pat_dims[i] > 1)
				flags = MD_SET(flags, i);

		const struct lad_conf lad_conf = { conf->rwiter, conf->gamma, flags, &lsqr_conf };

		op = lad2_create(&lad_conf, italgo, iconf, (const float*)init, sense_op, num_funs, thresh_op, thresh_funs);

	} else
	if (NULL == pattern) {

		op = lsqr2_create(&lsqr_conf, italgo, iconf, (const float*)init, sense_op, precond_op,
					num_funs, thresh_op, thresh_funs);

	} else {

		complex float* weights = md_alloc(DIMS, pat_dims, CFL_SIZE);
#if 0
		// buggy
//		md_zsqrt(DIMS, pat_dims, weights, pattern);
#else
		long dimsR[DIMS + 1];
		real_from_complex_dims(DIMS, dimsR, pat_dims);
		md_sqrt(DIMS + 1, dimsR, (float*)weights, (const float*)pattern);
#endif
		// FIXME: weights is never freed

		struct linop_s* weights_op = linop_cdiag_create(DIMS, ksp_dims, ~COIL_FLAG, weights);

		op = wlsqr2_create(&lsqr_conf, italgo, iconf, (const float*)init,
						sense_op, weights_op, precond_op,
						num_funs, thresh_op, thresh_funs);
	}

	return op;
}

