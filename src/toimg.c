/* Copyright 2013, 2015 The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013, 2015 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <complex.h>
#include <stdbool.h>

#include "num/multind.h"

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/png.h"
#include "misc/dicom.h"

#ifndef DIMS
#define DIMS 16
#endif




const char* usage_str = "<input> <output.(png|dcm)>";
const char* help_str = "Create png or proto dicom image.\n";


int main_toimg(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 2, usage_str, help_str);

	long dims[DIMS];
	
	complex float* data = load_cfl(argv[1], DIMS, dims);

	long sqdims[DIMS] = { [0 ... DIMS - 1] = 1 };

	int l = 0;
	for (int i = 0; i < DIMS; i++)
		if (1 != dims[i]) 
			sqdims[l++] = dims[i];

	assert(2 == l);

	int h = sqdims[0];
	int w = sqdims[1];

	int len = strlen(argv[2]);
	assert(len >= 4);
	bool dicom = (0 == strcmp(argv[2] + (len - 4), ".dcm"));

	int nr_bytes = dicom ? 2 : 3;
	float max_val = dicom ? 65535. : 255.;

	unsigned char* buf = xmalloc(h * w * nr_bytes);
	
	float max = 0.;
	for (int i = 0; i < h * w; i++)
		max = MAX(cabsf(data[i]), max);

	if (0. == max)
		max = 1.;

	for (int i = 0; i < h; i++) {

		for (int j = 0; j < w; j++) {

			unsigned int value = max_val * (cabsf(data[j * h + i]) / max);

			if (!dicom) {

				buf[(i * w + j) * 3 + 0] = value;
				buf[(i * w + j) * 3 + 1] = value;
				buf[(i * w + j) * 3 + 2] = value;

			} else {

				buf[(i * w + j) * 2 + 0] = (value >> 0) & 0xFF;
				buf[(i * w + j) * 2 + 1] = (value >> 8) & 0xFF;
			}
		}
	}

	(dicom  ? dicom_write : png_write_rgb24)(argv[2], w, h, buf);

	free(buf);

	unmap_cfl(DIMS, dims, data);

	exit(0);
}


