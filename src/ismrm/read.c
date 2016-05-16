/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2012, 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <stdbool.h>
#include <complex.h>

#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/dataset.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "read.h"

// FIXME: does not deal correctly with repetitions (and others stuff)



int ismrm_read(const char* datafile, long dims[DIMS], complex float* buf)
{
	ISMRMRD_Dataset d;
	ismrmrd_init_dataset(&d, datafile, "/dataset");
	ismrmrd_open_dataset(&d, false);

	assert(DIMS > 5);

	unsigned int number_of_acquisitions = ismrmrd_get_number_of_acquisitions(&d);

	long pos[DIMS];
	long channels = -1;
	long slices = 0;
	long samples = -1;

	for (unsigned int i = 0; i < DIMS; i++)
		pos[i] = 0;

	long strs[DIMS];
	long adc_dims[DIMS];
	long adc_strs[DIMS];

	if (NULL == buf) {

		md_singleton_dims(DIMS, dims);

	} else {

		md_calc_strides(DIMS, strs, dims, CFL_SIZE);

		md_select_dims(DIMS, READ_FLAG|COIL_FLAG, adc_dims, dims);
		md_calc_strides(DIMS, adc_strs, adc_dims, CFL_SIZE);
	}

	ISMRMRD_Acquisition acq;

	for (unsigned int i = 0; i < number_of_acquisitions; i++) {

		ismrmrd_init_acquisition(&acq);
		ismrmrd_read_acquisition(&d, i, &acq);

		if (acq.head.flags & (1 << (ISMRMRD_ACQ_IS_NOISE_MEASUREMENT - 1)))
			continue;

		if (-1 == channels) {

			channels = acq.head.available_channels;
			samples = acq.head.number_of_samples;
		}

		pos[1] = acq.head.idx.kspace_encode_step_1;
		pos[2] = acq.head.idx.kspace_encode_step_2;
		pos[4] = slices; // acq.head.idx.slice;

		if (buf != NULL) {

			assert(pos[1] < dims[1]);
			assert(pos[2] < dims[2]);
			assert(pos[4] < dims[4]);

			assert(dims[0] == acq.head.number_of_samples);
			assert(dims[3] == acq.head.active_channels);
			assert(dims[3] == acq.head.available_channels);

			debug_printf(DP_DEBUG3, ":/%ld %ld/%ld %ld/%ld :/%ld %ld/%ld %d\n",
				dims[0], pos[1], dims[1], pos[2], dims[2], dims[3], pos[4], dims[4], number_of_acquisitions);

			md_copy_block2(DIMS, pos, dims, strs, buf, adc_dims, adc_strs, acq.data, CFL_SIZE);

		} else {

			dims[1] = MAX(dims[1], pos[1] + 1);
			dims[2] = MAX(dims[2], pos[2] + 1);
		}

		if (acq.head.flags & (1 << (ISMRMRD_ACQ_LAST_IN_SLICE - 1)))
			slices++;

	//	ismrmrd_free_acquisition(&acq);
	}


	if (NULL == buf) {

		dims[0] = samples;
		dims[3] = channels;
		dims[4] = slices;

	} else {

		assert(dims[3] == channels);
		assert(dims[4] == slices);
	}

//	printf("Done.\n");

	return 0;
}


