/* Copyright 2022. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * Moritz Blumenthal
 */

#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <float.h>

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"
#include "misc/mri.h"

#include "nlops/chain.h"
#include "nn/nn.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "grecon/losses.h"

#include "networks/losses.h"


#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = 
	"";

enum MEASURE {
	
	MEAS_IMAGE,
	MEAS_SSIM_RSS,
	MEAS_MSE,
	MEAS_MSE_RSS,
	MEAS_PSNR_RSS,
};			


int main_measure(int argc, char* argv[argc])
{
	const char* ref_file = NULL;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ref_file, "reference"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(false, &out_file, "output"),
	};

	bool print_name = false;

	enum MEASURE meas = MEAS_IMAGE;

	unsigned long bat_flags = BATCH_FLAG;
	
	const struct opt_s opts[] = {

		OPTL_SELECT(0, "mse", enum MEASURE, &meas, MEAS_MSE, "mse"),
		OPTL_SELECT(0, "mse-mag", enum MEASURE, &meas, MEAS_MSE_RSS, "mse of rss (over coil dim)"),
		OPTL_SELECT(0, "ssim", enum MEASURE, &meas, MEAS_SSIM_RSS, "ssim of rss (over coil dim) and mean over other dims"),
		OPTL_SELECT(0, "psnr", enum MEASURE, &meas, MEAS_PSNR_RSS, "psnr of rss (over coil dim) and mean over other dims"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long ref_dims[DIMS];
	long in_dims[DIMS];

	complex float* ref = load_cfl(ref_file, DIMS, ref_dims);
	complex float* in = load_cfl(in_file, DIMS, in_dims);

	assert(md_check_compat(DIMS, 0u, in_dims, ref_dims));

	struct loss_config_s config = val_loss_option;

	switch (meas) {
	
	case MEAS_IMAGE:

		print_name = true;
		config = loss_image_valid;
		break;
	
	case MEAS_SSIM_RSS:
		config.weighting_ssim_rss = 1.;
		break;

	case MEAS_MSE:
		config.weighting_mse = 1.;
		break;

	case MEAS_MSE_RSS:
		config.weighting_mse_rss = 1.;
		break;

	case MEAS_PSNR_RSS:
		config.weighting_psnr_rss = 1.;
		break;

	default:
		assert(0);
	}

	long op_dims[DIMS];
	long out_dims[DIMS];
	
	md_select_dims(DIMS, ~bat_flags, op_dims, ref_dims);
	md_select_dims(DIMS, bat_flags, out_dims, ref_dims);

	auto nn_measure = val_measure_create(&config, DIMS, op_dims);
	const struct nlop_s* nlop_measure = nlop_clone(nn_get_nlop(nn_measure));

	out_dims[0] = nn_get_nr_out_args(nn_measure);
	
	while (1 < nlop_get_nr_out_args(nlop_measure))
		nlop_measure = nlop_stack_outputs_F(nlop_measure, 0, 1, 0);
	
	long nlop_odims[DIMS];
	md_select_dims(DIMS, MD_BIT(0), nlop_odims, out_dims);
	nlop_measure = nlop_reshape_out_F(nlop_measure, 0, DIMS, nlop_odims);

	complex float* out = (NULL == out_file ? anon_cfl : create_cfl)(out_file, DIMS, out_dims);


	nlop_generic_apply_loop(nlop_measure, bat_flags,
				1, (int[1]) { DIMS }, (const long*[1]){ out_dims }, (complex float*[1]){ out },
				2, (int[2]) { DIMS, DIMS }, (const long*[2]){ in_dims, ref_dims }, (const complex float*[2]){ in, ref });

	complex float res[out_dims[0]];

	if (NULL == out_file) {

		md_zavg(DIMS, out_dims, bat_flags, res, out);

		if (print_name) {

			for (int i = 0; i < nn_get_nr_out_args(nn_measure); i++)
				bart_printf("%s:%*s%e\n", nn_measure->out_names[i], 24 - strlen(nn_measure->out_names[i]), "", crealf(res[i]));

		} else {

			assert(1 == nn_get_nr_out_args(nn_measure));
			bart_printf("%e\n", crealf(res[0]));
		}
	}

	nlop_free(nlop_measure);
	nn_free(nn_measure);

	unmap_cfl(DIMS, ref_dims, ref);
	unmap_cfl(DIMS, in_dims, in);
	unmap_cfl(DIMS, out_dims, out);

	return 0;
}



