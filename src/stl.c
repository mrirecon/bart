/* Copyright 2024. University Medical Center Göttingen
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Martin Heide 
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
		OPTL_SET(0, "stat", &stat, "Show statistics of model."),
		OPTL_SET(0, "print", &print, "Print out model."),
		OPTL_SET(0, "no-nc", &no_nc, "(Don't recompute normal vectors with double precision.)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

        if (NULL != in_file && STL_NONE != stl_choice)
                error("Please provide either input from filesystem ex or use internal stl.\n");

        if (NULL == in_file && STL_NONE == stl_choice)
                stl_choice = STL_TETRAHEDRON;

        long dims[DIMS];
        // for build analyzer
        double* model = NULL;

        if (NULL != in_file) {

                if (stl_fileextension(in_file)) {

                        model = stl_read(DIMS, dims, in_file);

                } else {
                        
                        complex float* cmodel = load_cfl(in_file, DIMS, dims);

                        model = stl_cfl2d(DIMS, dims, cmodel);

                        unmap_cfl(DIMS, dims, cmodel);
                }

                if (!no_nc)
                        stl_compute_normals(DIMS, dims, model);
        }
        
        if (STL_TETRAHEDRON == stl_choice)
                model = stl_internal_tetrahedron(DIMS, dims);

        if (STL_HEXAHEDRON == stl_choice)
                model = stl_internal_hexahedron(DIMS, dims);

        if (stat)
                stl_stats(DIMS, dims, model);

        if (print)
                stl_print(DIMS, dims, model);

        if (stl_fileextension(out_file)) {

                stl_write_binary(DIMS, dims, model, out_file);

        } else {

                complex float* cmodel = create_cfl(out_file, DIMS, dims);

                stl_d2cfl(DIMS, dims, model, cmodel);

                unmap_cfl(DIMS, dims, cmodel);
        }

        md_free(model);

        return 0;
}

