/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author: Moritz Blumenthal
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/iovec.h"

#include "iter/iter6.h"
#include "iter/italgos.h"
#include "iter/batch_gen.h"

#include "grecon/opt_iter6.h"

#include "misc/opts.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "linops/someops.h"

#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/someops.h"

#include "nn/nn.h"
#include "nn/weights.h"
#include "nn/chain.h"

#include "networks/cunet.h"
#include "networks/score.h"
#include "num/ops_p.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = "Train a score network with Conditional-UNet architecture.";

int main_cunet(int argc, char* argv[argc])
{

	float sigma_min = 0.001;
	float sigma_max = 100;
	float sigma_data = 0.5;

	int batch_size = 32;

	opts_iter6_init();


	struct opt_s sigma_opts[] = {
		OPTL_FLOAT(0, "min", &sigma_min, "min", "minimum sigma for training"),
		OPTL_FLOAT(0, "max", &sigma_max, "max", "maximum sigma for training"),
	};

	struct nn_cunet_conf_s cunet_conf = cunet_defaults;

	bool realvalued = false;

	const struct opt_s opts[] = {

		OPTL_SUBOPT('T', "train-algo", "...", "configure general training parameters", N_iter6_opts, iter6_opts),
		OPTL_SUBOPT(0, "sigma", "", "select noise schedule for decreasing coise", ARRAY_SIZE(sigma_opts), sigma_opts),

		OPTL_SET('g', "gpu", &(bart_use_gpu), "run on gpu"),
		OPT_INT('l', &(cunet_conf.levels), "l", "Number of unet levels"),
		OPT_CLEAR('n', &(cunet_conf.conditional), "(don't use conditional embedding)"),
		OPT_SET('r', &realvalued, "use real-valued network (i.e. with z ~ N(0, I))"),
		OPT_INT('b', &batch_size, "b", "batch size"),
	};

	const char* filename_images;
	const char* filename_weights;

	struct arg_s args[] = {

		ARG_INFILE(true, &(filename_images), "images"),
		ARG_OUTFILE(true, &filename_weights, "weights"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init_gpu_support();

	long dims[DIMS];
	complex float* in = load_cfl(filename_images, DIMS, dims);

	int Nt = dims[BATCH_DIM];
	int Nb = MIN(batch_size, Nt);

	long bdims[5] = { 1, dims[0], dims[1], dims[2], Nb };
	long tdims[5] = { 1, dims[0], dims[1], dims[2], Nt };

	nn_t net = cunet_create(&cunet_conf, 5, bdims);

	net = nn_denoise_precond_edm(net, sigma_min, sigma_max, sigma_data, false);

	nn_t loss = nn_denoise_loss_VE(net, sigma_min, sigma_max, sigma_data);
	loss = nn_sort_args_F(loss);

	nn_debug(DP_INFO, loss);

	nn_weights_t weights = nn_weights_create_from_nn(loss);
	nn_init(loss, weights);

	if (bart_use_gpu)
		move_gpu_nn_weights(weights);

	const struct nlop_s* batch_generator = batch_gen_create(1, (int [1]){ 5 },
								(const long*[1]){ bdims },
								(const long*[1]){ tdims },
								(const complex float*[1]){ in },
								0, BATCH_GEN_SHUFFLE_DATA, 123);

	//setup for iter algorithm
	int II = nn_get_nr_in_args(loss);
	int OO = nn_get_nr_out_args(loss);

	float* src[II];
	src[0] = NULL; //reference output -> allocated and filled by training algorithm/batch generator
	src[1] = NULL; //reference input -> allocated and filled by training algorithm/batch generator
	src[2] = NULL; //reference input -> allocated and filled by training algorithm/batch generator

	for (int i = 0; i < weights->N; i++)
		src[i + 3] = (float*)weights->tensors[i];

	enum IN_TYPE in_type[II];
	enum OUT_TYPE out_type[OO];
	const struct operator_p_s* prox_ops[II];

	nn_get_in_types(loss, II, in_type);
	nn_get_out_types(loss, OO, out_type);
	nn_get_prox_ops(loss, II, prox_ops);

	iter_6_select_algo = ITER6_ADAM;
	struct iter6_conf_s* train_conf = iter6_get_conf_from_opts();
	train_conf->dump_filename = filename_weights;

	const struct nlop_s* nlop = nlop_clone(nn_get_nlop(loss));
	const struct iovec_s* dom = nlop_generic_domain(nlop, 1);

	if (!realvalued)
		nlop = nlop_prepend_FF(nlop_from_linop_F(linop_scale_create(dom->N, dom->dims, sqrtf(0.5))), nlop, 1);

	iter6_adam(train_conf, nlop, II, in_type, prox_ops, src, OO, out_type, Nb, Nt / Nb, batch_generator, NULL);

	nlop_free(nlop);
	nlop_free(batch_generator);


	dump_nn_weights(filename_weights, weights);
	nn_weights_free(weights);

	nn_free(loss);

	unmap_cfl(DIMS, dims, in);

	return 0;
}
