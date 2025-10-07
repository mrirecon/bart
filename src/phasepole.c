/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Moritz Blumenthal
 */

#include <complex.h>

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mri.h"
#include "misc/mmio.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/vec3.h"

#include "noir/pole.h"

enum mode_t {
	POLE_MODE_CORRECT,
	POLE_MODE_ESTIMATE,
	POLE_MODE_SAMPLE,
};


static const char help_str[] = "Detect and sample phase poles.";

int main_phasepole(int argc, char* argv[argc])
{
	const char* src_file = NULL;
	const char* dst_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(false, &src_file, "input (coils/singularity postions)"),
		ARG_OUTFILE(true, &dst_file, "output (sampled phase/singularity positions)"),
	};

	struct pole_config_s conf = pole_config_default;

	const char* pmap_file = NULL;
	const char* apmap_file = NULL;
	const char* wmap_file = NULL;

	float center[3] = { -1., -1., -1. };
	long mydims[3] = { 0, 0, 0 };

	enum mode_t mode = POLE_MODE_CORRECT;

	const struct opt_s opts[] = {

		OPTL_SELECT('e', "estimate", enum mode_t, &mode, POLE_MODE_ESTIMATE, "Estimate phase poles (input is sensitivity maps / output is singularity positions)"),
		OPTL_SELECT('s', "sample", enum mode_t, &mode, POLE_MODE_SAMPLE, "Sample phase poles (input is position / phase)"),
		OPTL_SET(0, "espirit", &conf.espirit, "Use ESPIRiT mode (diameter=1, no closing)"),

		OPTL_VEC3('x', "dims", &mydims, "x:y:z", "Explicitly specify image dimensions"),
		OPT_OUTFILE('c', &pmap_file, "<curl>", "output curl map"),
		OPT_OUTFILE('w', &wmap_file, "<weighting>", "(output weighting map)"),
		OPT_OUTFILE('a', &apmap_file, "<acurl>", "(output averaged curl map)"),
		OPT_FLOAT('t', &conf.thresh, "thresh", "threshold for pole detection"),
		OPT_FLOAT('d', &conf.diameter, "diameter", "diameter of curve integral to detect poles (in FoV)"),
		OPTL_FLOAT(0, "closing", &conf.closing, "radius", "(radius of closing ball (in FoV))"),
		OPTL_FLVEC3(0, "center", &center, "x:y:z", "specify position of phase pole to sample"),
		OPT_INT('n', &conf.normal, "normal", "normal direction for 2D pole detection (0,1,2)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	long dims[DIMS];
	md_singleton_dims(DIMS, dims);
	dims[0] = 0;

	struct lseg_s pos = { .N = 0, .pos = NULL };

	if (POLE_MODE_SAMPLE == mode) {

		if ((NULL != pmap_file) || (NULL != wmap_file) || (NULL != apmap_file))
			error("Cannot use --pmap, --wmap or --apmap if not estimating poles (-e).\n");


		if (NULL != src_file) {

			long pos_dims[DIMS] = { };
			md_singleton_dims(DIMS - 3, pos_dims + 3);

			complex float* src = load_cfl(src_file, 3, pos_dims);

			if ((3 != pos_dims[0]) || (2 != pos_dims[1]))
				error("Input file must have dimensions 3x2xN (3D pole positions).\n");

			pos.N = pos_dims[2];
			pos.pos = md_alloc(3, pos_dims, sizeof(float));

			md_real(3, pos_dims, pos.pos[0][0], src);

			unmap_cfl(3, pos_dims, src);

			pos.N = pos_dims[2];

		} else {

			if (-1. == center[0])
				error("Either input file or center (--center) must be specified.\n");

			pos.N = 1;
			pos.pos = md_alloc(3, MD_DIMS(1, 2, 3), sizeof(float));
			vec3_copy(pos.pos[0][0], center);
			vec3_copy(pos.pos[0][1], center);
		}

	} else {

		if (NULL == src_file)
			error("Input file must be specified for pole estimation (-e) or correction (default).\n");

		long sens_dims[DIMS];
		complex float* sens = load_cfl(src_file, DIMS, sens_dims);

		long curl_dims[DIMS];
		md_copy_dims(DIMS, curl_dims, sens_dims);
		curl_dims[ITER_DIM] = ((-1 == conf.normal) && (3 == bitcount(md_nontriv_dims(3, sens_dims)))) ? 3 : 1;

		complex float* curl_map = ((NULL != pmap_file) ? create_cfl : anon_cfl)(pmap_file, DIMS, curl_dims);

		compute_curl_map(conf, DIMS, curl_dims, ITER_DIM, curl_map, sens_dims, sens);

		complex float* wgh = ((NULL != wmap_file) ? create_cfl : anon_cfl)(wmap_file, DIMS, curl_dims);

		compute_curl_weighting(conf, DIMS, curl_dims, ITER_DIM, wgh, sens_dims, sens);

		long pmap_dims[DIMS];
		md_select_dims(DIMS, ~(conf.avg_flag | ITER_FLAG), pmap_dims, curl_dims);

		complex float* acurl_map = ((NULL != apmap_file) ? create_cfl : anon_cfl)(apmap_file, DIMS, pmap_dims);

		average_curl_map(DIMS, pmap_dims, acurl_map, curl_dims, ITER_DIM, curl_map, wgh);

		if (NULL != wgh)
			unmap_cfl(DIMS, curl_dims, wgh);

		unmap_cfl(DIMS, curl_dims, curl_map);

		pos = extract_phase_poles_2D(conf, DIMS, pmap_dims, acurl_map);

		unmap_cfl(DIMS, sens_dims, sens);
		unmap_cfl(DIMS, pmap_dims, acurl_map);

		md_copy_dims(DIMS, dims, pmap_dims);
	}



	if ((POLE_MODE_CORRECT == mode) || (POLE_MODE_SAMPLE == mode)) {

		if (0 != md_calc_size(3, mydims))
			md_copy_dims(3, dims, mydims);

		if (0 == md_calc_size(3, dims))
			error("No image dimensions specified.\n");

		complex float* out = create_cfl(dst_file, DIMS, dims);

		if (0 == pos.N) {

			debug_printf(DP_WARN, "No poles found, filling output with 1.\n");
			md_zfill(3, dims, out, 1.);

		} else {

			sample_phase_pole_2D(3, dims, out, pos.N, pos.pos);
		}

		unmap_cfl(DIMS, dims, out);
	} else {

		if (0 == pos.N) {

			debug_printf(DP_WARN, "No poles found, no output.\n");

		} else {

			complex float* out = create_cfl(dst_file, 3, MD_DIMS(3, 2, pos.N));

			md_zcmpl_real(3, MD_DIMS(3, 2, pos.N), out, (const float*)pos.pos[0][0]);

			unmap_cfl(3, MD_DIMS(3, 2, pos.N), out);
		}
	}

	if (NULL != pos.pos)
		xfree(pos.pos);

	return 0;
}

