/* Copyright 2024-2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Moritz Blumenthal
 *
 *
 * Zach C, Pock T, Bischof H. A Duality Based Approach for Realtime TV-L 1 Optical Flow.
 * In: Hamprecht FA, Schnörr C, Jähne B. (eds) Pattern Recognition. DAGM 2007.
 * Lecture Notes in Computer Science 2007;4713. Springer, Berlin, Heidelberg.
 *
 * Avants BB, Epstein CL, Grossman M, Gee JC.
 * Symmetric diffeomorphic image registration with cross-correlation: 
 * evaluating automated labeling of elderly and neurodegenerative brain.
 * Med Image Anal 2008;12:26-41.
 */

#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdio.h>
#include <strings.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "motion/opticalflow.h"
#include "motion/syn.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char help_str[] = "Non-rigid registration with greedy SyN or optical flow algorithm.";

int main_estmotion(int argc, char* argv[argc])
{
	unsigned long flags = 0;
	const char* ref_file = NULL;
	const char* motion_file = NULL;
	const char* imotion_file = NULL;
	const char* mov_file = NULL;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "flags"),
		ARG_INFILE(true, &ref_file, "reference"),
		ARG_INFILE(true, &mov_file, "moved"),
		ARG_OUTFILE(true, &motion_file, "motion field"),
		ARG_OUTFILE(false, &imotion_file, "inverse motion field"),
	};

	int levels = 3;
	float lambda = 0.01;
	float factors[5] = { 1., 0.5, 0.25, 0.125, 0.0625 };
	float sigmas[5] = { 0., 2., 4., 8., 16. };
	int nwarps[5] = { 10, 25, 100, 100, 100 };
	bool l1reg = true;
	bool l1dc = true;
	float maxnorm = 0.;
	bool synalgo = true;

	const struct opt_s opts[] = {

		OPT_SET('g', &bart_use_gpu, "use gpu (if available)"),
		OPT_PINT('l', &levels, "", "number of levels in Gaussian pyramide"),
		OPTL_CLEAR(0, "optical-flow", &synalgo, "use optical flow instead of greedy SyN"),
		OPT_FLOAT('r', &lambda, "lambda", "regularization strength for TV (optical flow)"),
		OPTL_FLOAT(0, "max-flow", &maxnorm, "max", "constraint on flow magnitude (optical flow)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	assert(levels <= 5);

	num_init_gpu_support();

#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = bart_use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!bart_use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	long rdims[DIMS];
	long mdims[DIMS];

	complex float* ref_ptr = load_cfl(ref_file, DIMS, rdims);
	complex float* mov_ptr = load_cfl(mov_file, DIMS, mdims);

	complex float* ref = my_alloc(DIMS, rdims, CFL_SIZE);
	complex float* mov = my_alloc(DIMS, mdims, CFL_SIZE);

	md_copy(DIMS, mdims, mov, mov_ptr, CFL_SIZE);
	md_copy(DIMS, mdims, ref, ref_ptr, CFL_SIZE);

	unmap_cfl(DIMS, mdims, mov_ptr);
	unmap_cfl(DIMS, rdims, ref_ptr);

	md_zabs(DIMS, mdims, mov, mov);
	md_zabs(DIMS, rdims, ref, ref);

	assert(md_check_equal_dims(DIMS, rdims, mdims, ~0UL));
	assert(1 == rdims[MOTION_DIM]);

	long udims[DIMS];
	md_copy_dims(DIMS, udims, rdims);
	udims[MOTION_DIM] = bitcount(flags);

	complex float* mot = my_alloc(DIMS, udims, CFL_SIZE);
	md_clear(DIMS, udims, mot, CFL_SIZE);

	complex float* imot = NULL;

	if (NULL != imotion_file) {
			
		imot = my_alloc(DIMS, udims, CFL_SIZE);

		md_clear(DIMS, udims, mot, CFL_SIZE);
	}

	if (synalgo)
		syn(levels, sigmas, factors, nwarps, MOTION_DIM, flags, DIMS, udims, mot, imot, ref, mov);
	else
		optical_flow_multiscale(l1reg, flags, lambda, maxnorm, l1dc, levels, sigmas, factors, nwarps, MOTION_DIM, flags, DIMS, udims, ref, mov, mot);

	md_free(mov);
	md_free(ref);

	complex float* mot_ptr = create_cfl(motion_file, DIMS, udims);
	md_copy(DIMS, udims, mot_ptr, mot, CFL_SIZE);
	unmap_cfl(DIMS, udims, mot_ptr);	

	md_free(mot);

	if (NULL != imotion_file) {

		complex float* imot_ptr = create_cfl(imotion_file, DIMS, udims);

		md_copy(DIMS, udims, imot_ptr, imot, CFL_SIZE);

		unmap_cfl(DIMS, udims, mot_ptr);

		md_free(imot);
	}

	return 0;
}

