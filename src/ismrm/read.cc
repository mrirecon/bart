/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2012-12-17 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 * based on example code by Michael S. Hansen <michael.hansen@nih.gov>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "ismrmrd.h"
#include "ismrmrd.hxx"
#include "ismrmrd_hdf5.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "read.h"

// FIXME: does not deal correctly with repetitions (and others stuff)

const char ismrm_default_path[] = "/usr/local/ismrmrd";


int ismrm_read(const char* datafile, long dims[DIMS], _Complex float* buf)
{
	const char* env;

	if (NULL == (env = getenv("ISMRMRD_HOME"))) {

		debug_printf(DP_INFO, "ISMRMRD: Environment variable ISMRMRD_HOME not set.\n"
				"ISMRMRD: Using default path: %s\n", ismrm_default_path);

		env = ismrm_default_path;
	}

	int len = strlen(env) + 40;
	char* schema = (char*)xmalloc(len);
	snprintf(schema, len, "%s/schema/ismrmrd.xsd", env);

	ISMRMRD::IsmrmrdDataset d(datafile, "dataset");

	xml_schema::properties props;
	props.schema_location("http://www.ismrm.org/ISMRMRD", schema);

	free(schema);

	boost::shared_ptr< std::string > xml = d.readHeader();
	std::istringstream str_stream(*xml, std::stringstream::in);

	boost::shared_ptr< ISMRMRD::ismrmrdHeader > cfg;

	try {

		cfg = boost::shared_ptr< ISMRMRD::ismrmrdHeader >(ISMRMRD::ismrmrdHeader_(str_stream, 0, props));

	} catch (const xml_schema::exception& e) {

		debug_printf(DP_ERROR, "ISMRMRD: XML error: %s\n", e.what());
		return -1;	
	}

	ISMRMRD::ismrmrdHeader::encoding_sequence e_seq = cfg->encoding();

	if (e_seq.size() != 1) {

		debug_printf(DP_ERROR, "ISMRMRD: ERROR: More than one encoding space.\n");
		return -1;
	}

	ISMRMRD::encodingSpaceType e_space = (*e_seq.begin()).encodedSpace();
	ISMRMRD::encodingLimitsType e_limits = (*e_seq.begin()).encodingLimits();

	assert(DIMS > 5);

	if (NULL == buf) {

		dims[0] = e_space.matrixSize().x();
		dims[1] = e_space.matrixSize().y();
		dims[2] = e_space.matrixSize().z();
		dims[3] = -1;
		dims[4] = 1;
	
		for (unsigned int i = 5; i < DIMS; i++)
			dims[i] = 1;
			
		assert(dims[0] > 0);
		assert(dims[1] > 0);
		assert(dims[2] > 0);

	} else {
	
		assert(dims[0] == e_space.matrixSize().x());
		assert(dims[1] == e_space.matrixSize().y());
		assert(dims[2] == e_space.matrixSize().z());

		for (unsigned int i = 5; i < DIMS; i++)
			assert(1 == dims[i]);
	}

	unsigned int number_of_acquisitions = d.getNumberOfAcquisitions();

	long pos[DIMS];
	long channels = -1;
	long slices = 0;

	for (unsigned int i = 0; i < DIMS; i++)
		pos[i] = 0;

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, CFL_SIZE);

	long adc_dims[DIMS];
	md_select_dims(DIMS, READ_FLAG|COIL_FLAG, adc_dims, dims);

	long adc_strs[DIMS];
	md_calc_strides(DIMS, adc_strs, adc_dims, CFL_SIZE);

	for (unsigned int i = 0; i < number_of_acquisitions; i++) {

		boost::shared_ptr< ISMRMRD::Acquisition > acq = d.readAcquisition(i);

#ifdef ISMRMRD_OLD
		if (acq->head_.flags & (1 << (ISMRMRD::ACQ_IS_NOISE_MEASUREMENT - 1)))
			continue;

		if (-1 == channels) 
			channels = acq->head_.available_channels; // or active channels?


		pos[1] = acq->head_.idx.kspace_encode_step_1;
		pos[2] = acq->head_.idx.kspace_encode_step_2;
		pos[4] = acq->head_.isx.slice;
#else
		if (acq->getFlags() & (1 << (ISMRMRD::ACQ_IS_NOISE_MEASUREMENT - 1)))
			continue;

		if (-1 == channels) 
			channels = acq->getAvailableChannels(); // or active channels?

		pos[1] = acq->getIdx().kspace_encode_step_1;
		pos[2] = acq->getIdx().kspace_encode_step_2;
		pos[4] = acq->getIdx().slice;
#endif

		assert(pos[1] < dims[1]);
		assert(pos[2] < dims[2]);
		assert(pos[4] < dims[4]);

		if (buf != NULL) {

#ifdef ISMRMRD_OLD
			assert(dims[0] == acq->head_.number_of_samples);
			assert(dims[3] == acq->head_.active_channels);
			assert(dims[3] == acq->head_.available_channels);
#else
			assert(dims[0] == acq->getNumberOfSamples());
			assert(dims[3] == acq->getActiveChannels());
			assert(dims[3] == acq->getAvailableChannels());
#endif

			debug_printf(DP_DEBUG3, "%ld/%ld %ld/%ld %ld/%ld %ld/%ld :/%ld %d\n", 
				pos[4], dims[4], pos[3], dims[3], pos[2], dims[2], pos[1], dims[1], dims[0], number_of_acquisitions);

#ifdef ISMRMRD_OLD	
			md_copy_block2(DIMS, pos, dims, strs, buf, adc_dims, adc_strs, (_Complex float*)acq->data_, CFL_SIZE);
#else
			md_copy_block2(DIMS, pos, dims, strs, buf, adc_dims, adc_strs, (_Complex float*)&acq->getData()[0], CFL_SIZE);
#endif
		}

#ifdef ISMRMRD_OLD
		if (acq->head_.flags & (1 << (ISMRMRD::ACQ_LAST_IN_SLICE - 1)))
			slices++;
#else
		if (acq->getFlags() & (1 << (ISMRMRD::ACQ_LAST_IN_SLICE - 1)))
			slices++;
#endif
	}

	if (NULL == buf) {

		dims[3] = channels;
		dims[4] = slices;

	} else {

		assert(dims[3] == channels);
		assert(dims[4] == slices);
	}

//	printf("Done.\n");

	return 0;
}


