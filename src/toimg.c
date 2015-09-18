/* Copyright 2013-2015 The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013, 2015 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <complex.h>
#include <stdbool.h>

#include "num/multind.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/png.h"
#include "misc/dicom.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif



struct image_data_s {

	bool dicom;
	float max;
	float max_val;
	long h;
	long w;
};


const char* usage_str = "[-b] <input> <output_prefix>";
const char* help_str =	"-b\toutput dicom\n\n"
			"Create png or prototype dicom images.\n"
			"The first two non-singleton dimensions will\n"
			"be used for the image, and the other dimensions\n"
			"will be looped over.\n"
			"\nIssues:\n1. Magnitude only images.\n"
			"\n2. There must be at least 2 non-singleton dimensions.\n\n";


static void toimg(const struct image_data_s* img_data, const char* name, unsigned char* buf, const complex float* data)
{
	int len = strlen(name);
	assert(len >= 1);

	long h = img_data->h;
	long w = img_data->w;

	for (int i = 0; i < h; i++) {

		for (int j = 0; j < w; j++) {

			unsigned int value = img_data->max_val * (cabsf(data[j * h + i]) / img_data->max);

			if (!img_data->dicom) {

				buf[(i * w + j) * 3 + 0] = value;
				buf[(i * w + j) * 3 + 1] = value;
				buf[(i * w + j) * 3 + 2] = value;

			} else {

				buf[(i * w + j) * 2 + 0] = (value >> 0) & 0xFF;
				buf[(i * w + j) * 2 + 1] = (value >> 8) & 0xFF;
			}
		}
	}

	(img_data->dicom  ? dicom_write : png_write_rgb24)(name, w, h, buf);
}


static void toimg_stack(const char* name, bool dicom, const long dims[DIMS], const complex float* data)
{
	long data_size = md_calc_size(DIMS, dims); 

	long h = 1;
	long h_dim = -1;

	long w = 1;
	long w_dim = -1;

	int l = 0;

	for (int i = 0; i < DIMS; i++) {

		if (1 != dims[i]) {

			switch (l++) {

			case 0:
				h = dims[i];
				h_dim = i;
				break;

			case 1:
				w = dims[i];
				w_dim = i;
				break;

			default:
				break;
			}
		}
	}

	assert((h_dim >= 0) && (w_dim > 0) && (h_dim != w_dim));

	int nr_bytes = dicom ? 2 : 3;
	float max_val = dicom ? 65535. : 255.;

	float max = 0.;
	for (long i = 0; i < data_size; i++)
		max = MAX(cabsf(data[i]), max);

	if (0. == max)
		max = 1.;


	struct image_data_s img_data = {

		.dicom = dicom,
		.max_val = max_val,
		.max = max,
		.h = h,
		.w = w
	};

	int len = strlen(name);
	assert(len >= 1);

	long num_imgs = md_calc_size(DIMS - w_dim - 1, dims + w_dim + 1);

	debug_printf(DP_INFO, "Writing %d images...\n", num_imgs);

#pragma omp parallel for
	for (long i = 0; i < num_imgs; i++) {

		char* name_i = xmalloc( (len + 10) * sizeof(char)); // extra space for ".0000.png"
		if (num_imgs > 1)
			sprintf(name_i, "%s.%04ld.%s", name, i, dicom ? "dcm" : "png");
		else
			sprintf(name_i, "%s.%s", name, dicom ? "dcm" : "png");

		unsigned char* buf = xmalloc(h * w * nr_bytes);
		complex float* dat = xmalloc(h * w * CFL_SIZE);

		md_copy_block(1, MD_DIMS(h * w * i), MD_DIMS(h * w), dat, MD_DIMS(data_size), data, CFL_SIZE);
		toimg(&img_data, name_i, buf, dat);

		free(name_i);
		free(buf);
		free(dat);
	}

	debug_printf(DP_INFO, "...Done\n", num_imgs);
}


int main_toimg(int argc, char* argv[])
{
	bool dicom = mini_cmdline_bool(argc, argv, 'b', 2, usage_str, help_str);

	long dims[DIMS];

	complex float* data = load_cfl(argv[1], DIMS, dims);

	toimg_stack(argv[2], dicom, dims, data);

	unmap_cfl(DIMS, dims, data);

	exit(0);
}


