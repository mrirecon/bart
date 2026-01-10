/* Copyright 2024-2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/rand.h"

#include "iter/iter6.h"

#include "noncart/nufft.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"

#include "misc/mri.h"
#include "misc/mri2.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"
#include "misc/io.h"

#include "noir/recon2.h"
#include "noir/model_net.h"

#include "nn/weights.h"
#include "nn/data_list.h"

#include "grecon/opt_iter6.h"
#include "grecon/losses.h"
#include "grecon/network.h"

#include "networks/cnn.h"
#include "networks/unet.h"
#include "networks/reconet.h"
#include "networks/losses.h"
#include "networks/misc.h"
#include "networks/nlinvnet.h"

#include "noir/misc.h"


static const char help_str[] = "Perform NLINV-Net reconstruction.";



int main_nlinvnet(int argc, char* argv[argc])
{
	double start_time = timestamp();

	const char* ksp_file = NULL;
	const char* out_file = NULL;
	const char* weight_file = NULL;
	const char* sens_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ksp_file, "kspace"),
		ARG_INOUTFILE(true, &weight_file, "weights"),
		ARG_INOUTFILE(true, &out_file, "output/reference"),
		ARG_OUTFILE(false, &sens_file, "sensitivities"),
	};

	const char* pat_file = NULL;
	const char* traj_file = NULL;
	const char* basis_file = NULL;

	struct noir2_conf_s conf = noir2_defaults;
	conf.cgiter = 30;


	struct nufft_conf_s nufft_conf = nufft_conf_defaults;

	nufft_conf.lowmem = true;
	nufft_conf.precomp_fftmod = false;
	nufft_conf.precomp_roll = false;
	nufft_conf.precomp_linphase = false;

	conf.nufft_conf = &nufft_conf;
	struct nlinvnet_s nlinvnet = nlinvnet_config_opts;
	nlinvnet.conf = &conf;

	bool train = false;
	bool apply = false;


	unsigned long batch_flags = BATCH_FLAG;
	unsigned long cnstcoil_flags = 0;
	unsigned long scl_flags = 0;
	int Nb = 0;

	const char* filename_weights_load = NULL;

	const char* val_file_kspace = NULL;
	const char* val_file_reference = NULL;
	const char* val_file_pattern = NULL;
	const char* val_file_trajectory = NULL;

	const char* filename_mask = NULL;
	const char* filename_mask_val = NULL;

	const char* filename_filter = NULL;

	opts_iter6_init();

	struct opt_s valid_opts[] = {

		OPTL_INFILE('p', "pattern", &(val_file_pattern), "<file>", "validation data sampling pattern"),
		OPTL_INFILE('t', "trajectory", &(val_file_trajectory), "<file>", "validation data trajectory"),
		OPTL_INFILE('k', "kspace", &(val_file_kspace), "<file>", "validation data kspace"),
		OPTL_INFILE('r', "ref", &(val_file_reference), "<file>", "validation data reference"),
		OPTL_INFILE(0, "mask", &(filename_mask_val), "<mask>", "mask for computation of loss"),
	};

	bool unet = false;
	long im_vec[3] = {0, 0, 0};

	struct opt_s network_opts[] = {

		OPTL_SET(0, "unet", &(unet), "use U-Net"),
	};

	bool norm_max = true;

	const struct opt_s opts[] = {

		OPTL_INT(0, "iter-net", &(nlinvnet.iter_net), "iter", "number of iterations with network"),

		OPTL_SUBOPT(0, "resnet-block", "...", "configure residual block", N_res_block_opts, res_block_opts),
		OPTL_SUBOPT(0, "unet", "...", "configure U-Net block", N_unet_reco_opts, unet_reco_opts),
		//OPTL_CLEAR(0, "no-shared-weights", &(nlinvnet.share_weights), "don't share weights across iterations"),

		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		OPTL_FLOAT(0, "sens-os", &(nlinvnet.oversampling_coils), "val", "(over-sampling factor for sensitivities)"),

		//OPTL_FLOAT(0, "alpha", &(nlinvnet.conf->alpha), "val", "(minimal value for alpha)"),
		OPTL_FLOAT(0, "alpha-min", &(nlinvnet.conf->alpha_min), "val", "(minimal value for alpha)"),
		//OPTL_FLOAT(0, "alpha-redu", &(nlinvnet.conf->redu), "val", "(reduction of alpha in each Gauss-Newton step)"),

		OPTL_FLOAT(0, "lambda", &(nlinvnet.lambda), "val", "additional regularization for network part (negative means trainable)"),
		OPTL_FLOAT(0, "lambda-sens", &(nlinvnet.lambda_sens), "val", "additional regularization for sensitivities (negative means trainable)"),
		
		OPTL_SUBOPT('N', "network", "...", "select neural network", ARRAY_SIZE(network_opts), network_opts),
		OPTL_INFILE(0,"filter", &filename_filter, "<filter>", "filter output of network block"),
		OPTL_PINT(0, "conv-time", &(nlinvnet.conv_time), "w", "convolve along dimension 10 with window size w"),
		//OPTL_SELECT(0, "conv-time-causal", enum PADDING, &(nlinvnet.conv_padding), PAD_CAUSAL, "(Use causal convolution)"),
		OPTL_SET(0, "ref-img", &(nlinvnet.ref_init_img), "(Feed image after initialization in every network.)"),
		OPTL_SET(0, "ref-col-rt", &(nlinvnet.ref_init_col_rt), "(Temporal regularization for coil sensitivities.)"),
		OPTL_SET(0, "fix-sens", &(nlinvnet.fix_coils), "(Fix sensitivity maps after initialization)"),
		//OPTL_INT(0, "cgiter", &(conf.cgiter), "", "(number of cg iterations)"),
		OPTL_SET(0, "init-rtnlinv", &(nlinvnet.real_time_init), "initialize with rtnlinv recon"),
		
		OPTL_VEC3('x', "dims", &im_vec, "x:y:z", "image dimensions"),

		OPTL_SET('t', "train", &train, "train nlinvnet"),
		OPTL_SET('a', "apply", &apply, "apply nlinvnet"),
		OPTL_SET(0, "rss-norm", &(nlinvnet.normalize_rss), "scale output image to rss normalization"),

		OPTL_INFILE(0, "pattern", &pat_file, "<pattern>", "sampling pattern"),
		OPTL_INFILE(0, "trajectory", &(traj_file), "<traj>", "trajectory"),
		OPTL_INFILE('B', "basis", &(basis_file), "<basis>", "basis"),
		OPTL_FLOAT(0, "scaling", &(nlinvnet.scaling), "val", "scaling of data, negative means normalization to norm=val"),
		OPTL_ULONG(0, "scaling-flags", &scl_flags, "flags", "scaling is increased with sqrt(selected dims)"),

		OPTL_SET('g', "gpu", &bart_use_gpu, "run on gpu"),
		OPTL_PINT('b', "batch-size", &(Nb), "", "size of mini batches"),

		OPTL_SUBOPT(0, "valid-data", "...", "(provide validation data)", ARRAY_SIZE(valid_opts),valid_opts),

		OPTL_INFILE('l', "load", &(filename_weights_load), "<weights-init>", "load weights for continuing training"),

		OPTL_SUBOPT('T', "train-algo", "...", "configure general training parameters", N_iter6_opts, iter6_opts),
		OPTL_SUBOPT(0, "adam", "...", "configure adam optimizer", N_iter6_adam_opts, iter6_adam_opts),
		OPTL_SUBOPT(0, "train-loss", "...", "configure the training loss", N_loss_opts, loss_opts),
		OPTL_INFILE(0, "mask", &(filename_mask), "<mask>", "mask for computation of loss"),
		OPTL_FLOAT(0, "train-loss-l2-reg", &(nlinvnet.l2loss_reg), "l", "add l(||x||^2 + ||Wc||^2) to train loss"),

		OPTL_SET(0, "ksp-training", &(nlinvnet.ksp_training), "provide kspace as reference"),
		OPTL_FLOAT(0, "ss-ksp-split", &(nlinvnet.ksp_split), "p", "use p\% of kspace data as network input"),
		OPTL_ULONG(0, "ss-ksp-split-shared", &(nlinvnet.ksp_shared_dims), "flags", "shared dims for mask"),
		OPTL_INFILE(0, "ss-ksp-use-reco", &(nlinvnet.use_reco_file), "file", "mask should contain 0 or 1. Entries with 1 are neverused as loss"),
		OPTL_FLOAT(0, "ss-ksp-leaky", &(nlinvnet.ksp_leaky), "l", "also use network input scaled by l as reference"),
		OPTL_VEC2(0, "temporal-train-mask", &(nlinvnet.time_mask), "s:e", "Only use data in [s, e) as train reference"),
		OPTL_LONG(0, "average-coils-loss", &(nlinvnet.avg_coils_loss), "l", "Average coils along window of size l for loss computation"),
		OPT_ULONG('s', &cnstcoil_flags, "", "(dimensions with constant sensitivities)"),

		OPT_ULONG('L', &batch_flags, "flags", "loop over dims (apply only)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init_gpu_support();

	if (-1. != nlinvnet.ksp_split)
		nlinvnet.ksp_training = true;

	if (train) {

		nlinvnet.train_conf = iter6_get_conf_from_opts();

		if (NULL == nlinvnet.train_conf) {

			iter_6_select_algo = ITER6_ADAM;
			nlinvnet.train_conf = iter6_get_conf_from_opts();

		} else {

			iter6_copy_config_from_opts(nlinvnet.train_conf);
		}

		if ((0 < nlinvnet.train_conf->dump_mod) && (NULL == nlinvnet.train_conf->dump_filename))
			nlinvnet.train_conf->dump_filename = weight_file;
	}

	nlinvnet.network = get_default_network(unet ? NETWORK_UNET_RECO : NETWORK_RESBLOCK);
	
	if (norm_max)
		nlinvnet.network->norm = NORM_MAX;

	nlinvnet.network->loopdim = BATCH_DIM;
	nlinvnet.network->low_mem = true;

	long ksp_dims[DIMS];
	complex float* kspace = load_cfl(ksp_file, DIMS, ksp_dims);

	complex float* pattern = NULL;
	long pat_dims[DIMS];

	if (NULL != pat_file) {

		pattern = load_cfl(pat_file, DIMS, pat_dims);

	} else {

		md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);
		pattern = anon_cfl("", DIMS, pat_dims);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace);
	}

	assert(1 == ksp_dims[MAPS_DIM]);

	long ksp_strs[DIMS];
	md_calc_strides(DIMS, ksp_strs, ksp_dims, CFL_SIZE);

	long dims[DIMS];

	long trj_dims[DIMS];
	complex float* traj  = NULL;

	if (NULL != traj_file) {

		conf.noncart = true;

		traj = load_cfl(traj_file, DIMS, trj_dims);

		if (0 == md_calc_size(3, im_vec)) {

			estimate_im_dims(DIMS, FFT_FLAGS, dims, trj_dims, traj);
			debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", dims[0], dims[1], dims[2]);

		} else {

			md_copy_dims(3, dims, im_vec);;
		}

		md_copy_dims(DIMS - 3, dims + 3, ksp_dims + 3);

	} else {

		md_copy_dims(DIMS, dims, ksp_dims);
		md_singleton_dims(DIMS, trj_dims);
	}

	long bas_dims[DIMS];
	const complex float* basis = NULL;

	if (NULL != basis_file) {

		basis = load_cfl(basis_file, DIMS, bas_dims);

		assert(!md_check_dimensions(DIMS, bas_dims, COEFF_FLAG | TE_FLAG));

	} else {

		md_singleton_dims(DIMS, bas_dims);
	}


	dims[MAPS_DIM] = 1;

	long sens_dims[DIMS];
	md_select_dims(DIMS, ~cnstcoil_flags, sens_dims, dims);

	if (NULL != basis) {

		assert(1 == ksp_dims[COEFF_DIM]);
		assert(bas_dims[TE_DIM] == ksp_dims[TE_DIM]);

		dims[COEFF_DIM] = bas_dims[COEFF_DIM];
		dims[TE_DIM] = 1;
		md_select_dims(DIMS, ~(COEFF_FLAG | TE_FLAG), sens_dims, dims);
	}


	long scl_dims[DIMS];
	md_select_dims(DIMS, scl_flags, scl_dims, dims);
	nlinvnet.scaling *= sqrtf((float)md_calc_size(DIMS, scl_dims));

	long img_dims[DIMS];
	long cim_dims[DIMS];
	long msk_dims[DIMS];

	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims);
	md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, dims);
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, img_dims);

	long col_dims_s[DIMS];
	long img_dims_s[DIMS];
	long cim_dims_s[DIMS];
	long msk_dims_s[DIMS];
	long ksp_dims_s[DIMS];
	long pat_dims_s[DIMS];
	long trj_dims_s[DIMS];

	if (train)
		assert(BATCH_FLAG == batch_flags);

	md_select_dims(DIMS, ~batch_flags, col_dims_s, sens_dims);
	md_select_dims(DIMS, ~batch_flags, img_dims_s, img_dims);
	md_select_dims(DIMS, ~batch_flags, cim_dims_s, cim_dims);
	md_select_dims(DIMS, ~batch_flags, msk_dims_s, msk_dims);
	md_select_dims(DIMS, ~batch_flags, ksp_dims_s, ksp_dims);
	md_select_dims(DIMS, ~batch_flags, pat_dims_s, pat_dims);
	md_select_dims(DIMS, ~batch_flags, trj_dims_s, trj_dims);

	Nb = Nb ? Nb : 10;
	Nb = MIN(Nb, (int)ksp_dims[BATCH_DIM]);

	complex float one = 1.;

	nlinvnet_init(&nlinvnet, DIMS, traj ? trj_dims_s : NULL, pat_dims_s, bas_dims, basis, ksp_dims_s,	cim_dims_s, img_dims_s,	col_dims_s);

	long fil_dims[DIMS];

	if (NULL != filename_filter) {

		nlinvnet.filter = load_cfl(filename_filter, DIMS, fil_dims);
		nlinvnet.filter_flags = md_nontriv_dims(DIMS, fil_dims);
		assert(md_check_equal_dims(DIMS, fil_dims, img_dims, nlinvnet.filter_flags));
	}

	if (train) {

		if (NULL != filename_weights_load)
			nlinvnet.weights = load_nn_weights(filename_weights_load);


		long out_dims[DIMS];
		complex float* ref = load_cfl(out_file, DIMS, out_dims);

		assert(md_check_equal_dims(DIMS, nlinvnet.ksp_training ? ksp_dims : cim_dims, out_dims, ~0UL));

		auto train_data_list = named_data_list_create();
		named_data_list_append(train_data_list, DIMS, out_dims, ref, "ref");
		named_data_list_append(train_data_list, DIMS, ksp_dims, kspace, "ksp");
		named_data_list_append(train_data_list, DIMS, pat_dims, pattern, "pat");

		if (NULL != traj)
			named_data_list_append(train_data_list, DIMS, trj_dims, traj, "trj");

		complex float* mask = NULL;
		long mask_dims[DIMS];

		if (NULL != filename_mask) {

			mask = load_cfl(filename_mask, DIMS, mask_dims);
			nlinvnet.train_loss->mask_flags = md_nontriv_dims(DIMS, mask_dims);
			named_data_list_append(train_data_list, DIMS, mask_dims, mask, "loss_mask");
		}

		long ksp_dims_val[DIMS];
		long cim_dims_val[DIMS];
		long pat_dims_val[DIMS];
		long trj_dims_val[DIMS];
		long mask_dims_val[DIMS];

		complex float* val_kspace = NULL;
		complex float* val_ref = NULL;
		complex float* val_pattern = NULL;
		complex float* val_traj = NULL;
		complex float* mask_val = NULL;

		struct named_data_list_s* valid_data_list = NULL;

		if (NULL != val_file_kspace) {

			val_kspace = load_cfl(val_file_kspace, DIMS, ksp_dims_val);
			val_ref = load_cfl(val_file_reference, DIMS, cim_dims_val);

			if (NULL != val_file_pattern) {

				val_pattern = load_cfl(val_file_pattern, DIMS, pat_dims_val);

			} else {

				md_select_dims(DIMS, ~COIL_FLAG, pat_dims_val, ksp_dims_val);
				val_pattern = anon_cfl("", DIMS, pat_dims_val);
				estimate_pattern(DIMS, ksp_dims_val, COIL_FLAG, val_pattern, val_kspace);
			}


			if (NULL != val_file_trajectory)
				val_traj = load_cfl(val_file_trajectory, DIMS, trj_dims_val);
			else
				md_singleton_dims(DIMS, trj_dims_val);


			valid_data_list = named_data_list_create();
			named_data_list_append(valid_data_list, DIMS, cim_dims_val, val_ref, "ref");
			named_data_list_append(valid_data_list, DIMS, ksp_dims_val, val_kspace, "ksp");
			named_data_list_append(valid_data_list, DIMS, pat_dims_val, val_pattern, "pat");

			if (NULL != val_traj)
				named_data_list_append(valid_data_list, DIMS, trj_dims_val, val_traj, "trj");

			if (NULL != filename_mask_val) {

				mask_val = load_cfl(filename_mask_val, DIMS, mask_dims_val);
				nlinvnet.valid_loss->mask_flags = md_nontriv_dims(DIMS, mask_dims_val);
				named_data_list_append(valid_data_list, DIMS, mask_dims_val, mask_val, "loss_mask");
			}
		}

		train_nlinvnet(&nlinvnet, Nb, train_data_list, valid_data_list);

		named_data_list_free(train_data_list);

		unmap_cfl(DIMS, out_dims, ref);

		dump_nn_weights(weight_file, nlinvnet.weights);

		if (NULL != val_file_kspace) {

			named_data_list_free(valid_data_list);

			unmap_cfl(DIMS, ksp_dims_val, val_kspace);
			unmap_cfl(DIMS, cim_dims_val, val_ref);
			unmap_cfl(DIMS, pat_dims_val, val_pattern);

			if (NULL != val_traj)
				unmap_cfl(DIMS, trj_dims_val, val_traj);
		}


		if (NULL != mask)
			unmap_cfl(DIMS, mask_dims, mask);

		if (NULL != mask_val)
			unmap_cfl(DIMS, mask_dims_val, mask_val);
	}

	if (apply) {

		complex float* img = create_cfl(out_file, DIMS, img_dims);

		md_copy_dims(3, sens_dims, img_dims);

		complex float* col = (NULL != sens_file) ? create_cfl(sens_file, DIMS, sens_dims) : anon_cfl("", DIMS, sens_dims);
		nlinvnet.weights = load_nn_weights(weight_file);

		if (-1. != nlinvnet.ksp_split) {

			long sdims[DIMS];
			md_select_dims(DIMS, ~nlinvnet.ksp_shared_dims, sdims, pat_dims);
			complex float* tmp = md_alloc(DIMS, pat_dims, CFL_SIZE);

			md_rand_one(DIMS, sdims, tmp, nlinvnet.ksp_split);

			md_zmul2(DIMS, pat_dims, MD_STRIDES(DIMS, pat_dims, CFL_SIZE), pattern, MD_STRIDES(DIMS, pat_dims, CFL_SIZE), pattern, MD_STRIDES(DIMS, sdims, CFL_SIZE), tmp);

			md_free(tmp);
		}

		apply_nlinvnet(&nlinvnet, DIMS,
				img_dims, img,
				sens_dims, col,
				ksp_dims, kspace,
				pat_dims, pattern,
				trj_dims, traj ? traj : &one);

		unmap_cfl(DIMS, img_dims, img);
		unmap_cfl(DIMS, sens_dims, col);
	}

	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, ksp_dims, kspace);
	unmap_cfl(DIMS, bas_dims, basis);
	unmap_cfl(DIMS, fil_dims, nlinvnet.filter);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total time: %.2f s\n", recosecs);

	return 0;
}

