
#include <stdbool.h>
#include "misc/misc.h"
#include "misc/debug.h"

#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/dataset.h"
#include "ismrmrd/xml.h"

#include "xml_wrapper.h"
#include <cstdio>

#include "ismrm/read.h"

namespace ISMRMRD
{


static struct limit_s get_limit(Optional<Limit> src)
{
	struct limit_s ret;
	ret.size = 1;
	ret.center = -1;

	ret.max_hdr = -1;
	ret.min_hdr = -1;

	ret.max_hdr = -1;
	ret.min_hdr = -1;

	if (src.is_present()) {
		
		ret.center = src.get().center;
		ret.min_hdr = src.get().minimum;
		ret.max_hdr = src.get().maximum;

		ret.size = ret.max_hdr + 1;
		ret.size_hdr = ret.max_hdr + 1;
	}

	return ret;
}

extern "C" void ismrm_read_encoding_limits(const char* filename, struct isrmrm_config_s* config)
{
	ISMRMRD_Dataset d;
	ismrmrd_init_dataset(&d, filename, "/dataset");
	ismrmrd_open_dataset(&d, false);

	const char* xml =  ismrmrd_read_header(&d);

	IsmrmrdHeader h;
	deserialize(xml, h);

	xfree(xml);

	if (config->idx_encoding >= (int)h.encoding.size())
		error("ISMRMD inconsistent number of encodings!");

	Encoding encoding = h.encoding[config->idx_encoding];

	config->limits[ISMRMRD_READ_DIM].size		= encoding.encodedSpace.matrixSize.x;
	config->limits[ISMRMRD_COIL_DIM].size		= 1;

	if (h.acquisitionSystemInformation.is_present())
		if (h.acquisitionSystemInformation.get().receiverChannels.is_present()) {

			config->limits[ISMRMRD_COIL_DIM].size = h.acquisitionSystemInformation.get().receiverChannels.get();
			config->limits[ISMRMRD_COIL_DIM].size_hdr = h.acquisitionSystemInformation.get().receiverChannels.get();
		}

	config->limits[ISMRMRD_PHS1_DIM] 		= get_limit(encoding.encodingLimits.kspace_encoding_step_1);
	config->limits[ISMRMRD_PHS1_DIM].size		= encoding.encodedSpace.matrixSize.y;
	config->limits[ISMRMRD_PHS1_DIM].size_hdr	= encoding.encodedSpace.matrixSize.y;

	config->limits[ISMRMRD_PHS2_DIM] 		= get_limit(encoding.encodingLimits.kspace_encoding_step_2);
	config->limits[ISMRMRD_PHS2_DIM].size 		= encoding.encodedSpace.matrixSize.z;
	config->limits[ISMRMRD_PHS2_DIM].size_hdr	= encoding.encodedSpace.matrixSize.z;
	
	config->limits[ISMRMRD_AVERAGE_DIM] 		= get_limit(encoding.encodingLimits.average);
	config->limits[ISMRMRD_SLICE_DIM] 		= get_limit(encoding.encodingLimits.slice);
	config->limits[ISMRMRD_CONTRAST_DIM] 		= get_limit(encoding.encodingLimits.contrast);
	config->limits[ISMRMRD_PHASE_DIM] 		= get_limit(encoding.encodingLimits.phase);
	config->limits[ISMRMRD_REPETITION_DIM]		= get_limit(encoding.encodingLimits.repetition);
	config->limits[ISMRMRD_SET_DIM] 		= get_limit(encoding.encodingLimits.set);
	config->limits[ISMRMRD_SEGMENT_DIM] 		= get_limit(encoding.encodingLimits.segment);

	ismrmrd_close_dataset(&d);
}

}