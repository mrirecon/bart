/* Copyright 2025. Insitute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
#include <stdio.h>
#include <string.h>
#include <setjmp.h>

#include "misc/opts.h"
#include "misc/mri.h"
#include "misc/misc.h"

#include "seq/config.h"
#include "seq/helpers.h"

#include "opts.h"

static int error_catcher2(void fun(int* argcp, char* argv[*argcp], int m, const struct arg_s args[m], const char* help_str, int n, const struct opt_s opts[n]), 
					int* argcp, char* argv[*argcp], int m, const struct arg_s args[m], const char* help_str, int n, const struct opt_s opts[n])
{
	int ret = -1;

	error_jumper.initialized = true;

	if (0 == setjmp(error_jumper.buf)) {

		fun(argcp, argv, m, args, help_str, n, opts);
		ret = 0;
	}

	error_jumper.initialized = false;

	return ret;
}


int read_config_from_str(struct seq_config* seq, int N, const char* buffer_in)
{
	static const char help[] = "commandline for IDEA\n";

	const char* dummy = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(false, &dummy, "dummy"),
	};


	long custom_params_long[SEQ_MAX_PARAMS_LONG] = { 0 };
	double custom_params_double[SEQ_MAX_PARAMS_DOUBLE] = { 0. };


	const struct opt_s opts[] = {


		// FOV and resolution
		OPTL_DOUBLE(0, "FOV", &seq->geom.fov, "FOV", "Field Of View [mm]"),
		OPTL_PINT(0, "BR", &seq->geom.baseres, "BR", "Base Resolution"),
		OPTL_DOUBLE(0, "slice_thickness", &seq->geom.slice_thickness, "slice_thickness", "Slice thickness [mm]"),

		// basic sequence parameters
		OPTL_DOUBLE(0, "FA", &seq->phys.flip_angle, "flip angle", "Flip angle [deg]"),
		OPTL_DOUBLE(0, "TR", &seq->phys.tr, "TR", "TR [us]"),
		OPTL_DOUBLE(0, "TE", &seq->phys.te, "TE", "TE"),
		OPTL_DOUBLE(0, "TE_delta", &seq->phys.te_delta, "TE_delta", "TE_delta"),
		OPTL_DOUBLE(0, "BWTP", &seq->phys.bwtp, "BWTP", "Bandwidth Time Product"),

		// others sequence parameters
		OPTL_DOUBLE(0, "rf_duration", &seq->phys.rf_duration, "rf_duration", "RF pulse duration [us]"),
		OPTL_DOUBLE(0, "dwell", &seq->phys.dwell, "dwell", "Dwell time [us]"),

		// encoding
		OPTL_UINT(0, "pe_mode", &seq->enc.pe_mode, "pe_mode", "Phase-encoding mode"),
		OPTL_PINT(0, "tiny", &seq->enc.tiny, "tiny", "Tiny golden-ratio index"),


		OPTL_LONG('e', "echoes", &seq->loop_dims[TE_DIM], "echoes", "Number of echoes"),
		OPTL_LONG('r', "lines", &seq->loop_dims[PHS1_DIM], "lines", "Number of phase encoding lines"),
		OPTL_LONG('z', "partitions", &seq->loop_dims[PHS2_DIM], "partitions", "Number of partitions (3D) or SMS groups (2D)"),
		OPTL_LONG('t', "measurements", &seq->loop_dims[TIME_DIM], "measurements", "Number of measurements / frames"),
		OPTL_LONG('m', "slices", &seq->loop_dims[SLICE_DIM], "slices", "Number of slices of multiband factor (SMS)"),
		OPTL_LONG('i', "inversions", &seq->loop_dims[BATCH_DIM], "inversions", "Number of inversions"),
		OPTL_LONG(0, "physio", &seq->loop_dims[TIME2_DIM], "physio", "Number of physio phases"),
		OPTL_LONG(0, "averages", &seq->loop_dims[AVG_DIM], "averages", "Number of averages"),


		// sms
		OPTL_PINT(0, "mb_factor", &seq->geom.mb_factor, "mb_factor", "Multi-band factor"),

		// magnetization preparation
		OPTL_UINT(0, "mag_prep", &seq->magn.mag_prep, "mag_prep", "Magn. preparation [OFF, IR_SEL, IR_NON, SR_SEL, SR_NON, SR_ADIAB]"),
		OPTL_DOUBLE(0, "TI", &seq->magn.ti, "TI", "Inversion time [us]"),
		OPTL_UINT(0, "contrast", &seq->phys.contrast, "spoiling", "Spoiling [RF_RANDOM,RF_SPOILED,BALANCED,GSTF_RANDOM,GSTF_SPOILED]"),
		OPTL_DOUBLE(0, "inv_delay", &seq->magn.inv_delay_time, "inv_delay_time", "Inversion delay time (seconds)/Flat top test gradient (GIRF)"),
		OPTL_DOUBLE(0, "init_delay", &seq->magn.init_delay, "init_delay", "Initial delay of measurement (seconds)"),


		OPTL_VECN(0, "CUSTOM_LONG", custom_params_long, "custom long parameters"),
		OPTL_DOVECN(0, "CUSTOM_DOUBLE", custom_params_double, "custom double parameters"),
		OPTL_VECN(0, "LOOP", seq->loop_dims, "sequence loop dimensions"),
	};

	char* buffer = strdup(buffer_in);

	char *token = strtok(buffer, " \t");

	char* argv[N + 1];

	int i = 0;

	while (token != NULL) {

		argv[i++] = token;
		token = strtok(NULL, " \t");
	}

	int a = error_catcher2(cmdline, &i, argv, ARRAY_SIZE(args), args, help, ARRAY_SIZE(opts), opts);

	free(buffer);	

	if (custom_params_long[0] > 0)
		seq_ui_interface_custom_params(0, seq, SEQ_MAX_PARAMS_LONG, custom_params_long,
					       SEQ_MAX_PARAMS_DOUBLE, custom_params_double);

	if (0 > a)
		return 0;

	return 1;
}
