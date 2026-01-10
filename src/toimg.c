/* Copyright 2013-2018 The Regents of the University of California.
 * Copyright 2017-2021. Uecker Lab. Unversity Medical Center Göttingen.
 * Copyright 2023-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013, 2015 Martin Uecker
 * 2015, 2018 Jon Tamir
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <strings.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"
#ifndef NO_PNG
#include "misc/png.h"
#endif
#include "misc/dicom.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char help_str[] = "Create magnitude images as png or proto-dicom.\n"
				"The first two non-singleton dimensions will\n"
				"be used for the image, and the other dimensions\n"
				"will be looped over.";

// from view:src/draw.c
static double clamp(double a, double b, double x)
{
	return (x < a) ? a : ((x > b) ? b : x);
}

static double windowing(double g, double a, double b, double x)
{
	return pow(clamp(0., 1., (x - a) / (b - a)), g);
}

static void toimg(bool dicom, bool use_windowing, const char* name, long inum, float gamma, float contrast, float window, float scale, long h, long w, const complex float* data)
{
	int len = strlen(name);
	assert(len >= 1);

	int nr_bytes = dicom ? 2 : 3;
	unsigned char (*buf)[h][w][nr_bytes] = TYPE_ALLOC(unsigned char[h][w][nr_bytes]);

	float max_val = dicom ? 65535. : 255.;

	for (int i = 0; i < h; i++) {

		for (int j = 0; j < w; j++) {

			double val = cabsf(data[j * h + i]) / scale;
			unsigned int value = (unsigned int)(max_val * (use_windowing ? windowing(gamma, contrast, window, val) : val));

			if (!dicom) {

				(*buf)[i][j][0] = value;
				(*buf)[i][j][1] = value;
				(*buf)[i][j][2] = value;

			} else {

				(*buf)[i][j][0] = (value >> 0) & 0xFF;
				(*buf)[i][j][1] = (value >> 8) & 0xFF;
			}
		}
	}

#ifdef NO_PNG
	assert(dicom);
	dicom_write(name, w, h, inum, &(*buf)[0][0][0]);
#else
	(dicom  ? dicom_write : png_write_rgb24)(name, w, h, inum, &(*buf)[0][0][0]);
#endif
	free(buf);
}


static void toimg_stack(const char* name, bool dicom, bool dim_names, bool single_scale, bool use_windowing, float gamma, float contrast, float window, const long dims[DIMS], const complex float* data)
{
	long data_size = md_calc_size(DIMS, dims); 

	long sq_dims[DIMS] = { [0 ... DIMS - 1] = 1 };

	int l = 0;
	unsigned long im_flags = 0UL;

	for (int i = 0; i < DIMS; i++) {

		if (1 == dims[i])
			continue;

		if (2 > l)
			im_flags |= MD_BIT(i);

		sq_dims[l++] = dims[i];
	}

	float max = 0.;
	for (long i = 0; i < data_size; i++)
		max = MAX(cabsf(data[i]), max);

	int len = strlen(name);
	assert(len >= 1);

	long num_imgs = md_calc_size(DIMS - 2, sq_dims + 2);
	long img_size = md_calc_size(2, sq_dims);

	long loop_dims[DIMS];
	md_select_dims(DIMS, ~im_flags, loop_dims, dims);

	assert(md_calc_size(DIMS, loop_dims) == num_imgs);

	debug_printf(DP_INFO, "Writing %ld image(s)...", num_imgs);

#pragma omp parallel for
	for (long i = 0; i < num_imgs; i++) {

		float scale = 0.;

		if (use_windowing)
			scale = md_znorm(2, sq_dims, data + i * img_size) / md_calc_size(2, sq_dims);
		else if (single_scale)
			scale = max;
		else 
			for (long j = 0; j < md_calc_size(2, sq_dims); j++)
				scale = MAX(cabsf(data[i * img_size + j]), scale);

		if (0. == scale)
			scale = 1.;


		char *name_i = NULL;

		if (!dim_names) {

			name_i = xmalloc((size_t)(len + 10));

			if (num_imgs == 1)
				sprintf(name_i, "%s.%s", name, dicom ? "dcm" : "png");
			else
				sprintf(name_i, "%s-%04ld.%s", name, i, dicom ? "dcm" : "png");

		} else {

			long pos[DIMS] = { [0 ... DIMS - 1] = 0  };
			md_unravel_index(DIMS, pos, ~0UL, loop_dims, i);

			name_i = construct_filename(DIMS, loop_dims, pos, name, dicom ? "dcm" : "png");
		}

		toimg(dicom, use_windowing, name_i, i, gamma, contrast, window, scale, sq_dims[0], sq_dims[1], data + i * img_size);
		xfree(name_i);
	}

	debug_printf(DP_INFO, "done.\n");
}


int main_toimg(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* out_prefix = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_prefix, "output prefix"),
	};

	float gamma = 1.;
	float contrast = 0.;
	float window = 750.;
	bool use_windowing = false;
	bool single_scale = true;
	bool dicom = false;
	bool dim_names = false;

	const struct opt_s opts[] = {

		OPT_FLOAT('g', &gamma, "gamma", "gamma level"),
		OPT_FLOAT('c', &contrast, "contrast", "contrast level"),
		OPT_FLOAT('w', &window, "window", "window level"),
		OPT_SET('d', &dicom, "write to dicom format (deprecated, use extension .dcm)"),
		OPT_CLEAR('m', &single_scale, "re-scale each image"),
		OPT_SET('W', &use_windowing, "use dynamic windowing"),
		OPT_SET('D', &dim_names, "Include dimensions in output filenames"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	char* ext = strrchr(out_prefix, '.');

	if (NULL != ext) {

		assert(!dicom);

		if (0 == strcmp(ext, ".dcm"))
			dicom = true;
		else if (0 != strcmp(ext, ".png"))
			error("Unknown file extension.\n");

		*ext = '\0';
	}

	long dims[DIMS];
	complex float* data = load_cfl(in_file, DIMS, dims);

	toimg_stack(out_prefix, dicom, dim_names, single_scale, use_windowing, gamma, contrast, window, dims, data);

	unmap_cfl(DIMS, dims, data);

	return 0;
}

