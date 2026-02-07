/* Copyright 2024-2025. Uecker Lab. University Medical Center Göttingen
 * Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 - 2026 Martin Heide
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/init.h"


#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "stl/models.h"
#include "stl/misc.h"

static const char help_str[] = "Read and write stl files with '.stl' or cfl fileformat.";

int main_stl(int argc, char* argv[argc])
{
	const char* out_file = NULL;
	const char* in_file = NULL;

        bool stat = false;
        bool print = false;
        bool no_nc = false;
	bool vm = false; // volume measure
	bool sm = false; // surface measure
        float scale = 0.;
        float shift[3] = { 0., 0., 0. };
        enum stl_itype stl_choice = STL_NONE;

        struct arg_s args[] = {

		ARG_OUTFILE(true, &out_file, "output"),
	};

        struct opt_s model_opts[] = {

		OPTL_SELECT(0, "TET", enum stl_itype, &stl_choice, STL_TETRAHEDRON, "Tetrahedron."),
		OPTL_SELECT(0, "HEX", enum stl_itype, &stl_choice, STL_HEXAHEDRON, "Hexahedron (= Cube)."),
        };

	const struct opt_s opts[] = {

		OPTL_INFILE(0, "input", &in_file, "", "Path to input file (.stl or cfl file format)."),
                OPTL_SUBOPT2(0, "model", "<tag> ", "Generic geometric structures are available.", "Internal stl model (help: bart stl --model h).\n", ARRAY_SIZE(model_opts), model_opts),
		OPT_FLOAT('s', &scale, "scale", "Multiplicate all coordinates of model with factor."),
                OPT_FLVEC3('S', &shift, "shift", "Shift all coordinates of model by vector.\n"),
		OPTL_SET(0, "stat", &stat, "Show statistics of model."),
		OPTL_SET(0, "print", &print, "Print out model."),
		OPTL_SET(0, "vm", &vm, "Print out volume measure of model."),
		OPTL_SET(0, "sm", &sm, "Print out surface measure of model."),
		OPTL_SET(0, "no-nc", &no_nc, "(Don't recompute normal vectors with double precision.)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

        if (NULL != in_file && STL_NONE != stl_choice)
                error("Please provide either input from filesystem or use internal stl.\n");

        if (NULL == in_file && STL_NONE == stl_choice)
                stl_choice = STL_TETRAHEDRON;

        long dims[DIMS];
        // for build analyzer
        double* model = NULL;

        if (NULL != in_file) {

                if (stl_fileextension(in_file)) {

                        model = stl_read(in_file, dims);

                } else {

                        complex float* cmodel = load_cfl(in_file, DIMS, dims);

                        model = stl_cfl2d(dims, cmodel);

                        unmap_cfl(DIMS, dims, cmodel);
                }
        }

        if (STL_TETRAHEDRON == stl_choice)
                model = stl_internal_tetrahedron(dims);

        if (STL_HEXAHEDRON == stl_choice)
                model = stl_internal_hexahedron(dims);

	if (!no_nc)
		stl_compute_normals(dims, model);

        double dshift[3] = { shift[0], shift[1], shift[2] };

        if (0. != shift[0] || 0. != shift[1] || 0. != shift[2])
                stl_shift_model(dims, model, dshift);

        double sc[3] = { scale, scale, scale };

        if (0. != scale)
                stl_scale_model(dims, model, sc);

        if (stat)
                stl_stats(dims, model);

        if (print)
                stl_print(dims, model);

	if (sm || vm) {

		struct triangle_stack* ts = stl_preprocess_model(dims, model);
		struct triangle* t = ts->tri;

		double smv = 0.;
		double vmv = 0.;

		for (int i = 0; i < ts->N; i++) {

			smv += t[i].sur;
			vmv += t[i].svol;
		}

		if (sm)
			debug_printf(DP_INFO, "%f\n", smv);

		if (vm)
			debug_printf(DP_INFO, "%f\n", vmv);

		md_free(ts);
	}

	if (NULL != out_file) {

		if (stl_fileextension(out_file)) {

			stl_write_binary(out_file, dims, model);

		} else {

			complex float* cmodel = create_cfl(out_file, 3, dims);

			stl_d2cfl(dims, cmodel, model);

			unmap_cfl(3, dims, cmodel);
		}
	}

	md_free(model);

	return 0;
}

