#include <stdbool.h>
#include <stdexcept>

#include <cassert>
#include <climits>
#include <ext/stdio_filebuf.h>
#include <iostream>
#include <fstream>

#include "misc/misc.h"
#include "misc/debug.h"

#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/dataset.h"
#ifdef USE_ISMRMRD_STREAM
#include "ismrmrd/serialization.h"
#include "ismrmrd/serialization_iostream.h"
#endif
#include "ismrmrd/xml.h"

#include "xml_wrapper.h"
#include "read.h"



static struct limit_s get_limit(ISMRMRD::Optional<ISMRMRD::Limit>& src)
{
	struct limit_s ret = ismrmrd_default_limit;

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
	ISMRMRD::ISMRMRD_Dataset d;
	ismrmrd_init_dataset(&d, filename, "/dataset");
	ismrmrd_open_dataset(&d, false);

	const char* xml =  ismrmrd_read_header(&d);

	ismrm_read_encoding_limits_from_xml(xml, config);

	xfree(xml);

	ismrmrd_close_dataset(&d);
}

static void ismrm_read_encoding_limits_from_hdr(ISMRMRD::IsmrmrdHeader& h, struct isrmrm_config_s* config);

extern "C" void ismrm_read_encoding_limits_from_xml(const char* xml, struct isrmrm_config_s* config)
{
	ISMRMRD::IsmrmrdHeader h;
	deserialize(xml, h);
    ismrm_read_encoding_limits_from_hdr(h, config);
}

static void ismrm_read_encoding_limits_from_hdr(ISMRMRD::IsmrmrdHeader& h, struct isrmrm_config_s* config)
{
	if (config->idx_encoding >= (int)h.encoding.size())
		error("ISMRMD inconsistent number of encodings!\n");

	ISMRMRD::Encoding& encoding = h.encoding[config->idx_encoding];

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

}

struct ismrm_cpp_state {
#ifdef USE_ISMRMRD_STREAM
	std::istream* is;
	ISMRMRD::IStreamView* rs;
	ISMRMRD::ProtocolDeserializer* deserializer;
#endif
};

extern "C" struct ismrm_cpp_state* ismrm_stream_open(const char* file)
{
#ifdef USE_ISMRMRD_STREAM
	struct ismrm_cpp_state* ret = (struct ismrm_cpp_state*) malloc(sizeof *ret);
	ret->deserializer = NULL;
	ret->rs = NULL;
	ret->is = NULL;

	std::istream* is;
	if (0 != strcmp("-", file)) {

		ret->is = new std::ifstream(file, std::ifstream::binary | std::ios::binary);
		is = ret->is;
	} else {

		is = &std::cin;
	}

	ret->rs = new ISMRMRD::IStreamView(*is);
	ret->deserializer = new ISMRMRD::ProtocolDeserializer(*ret->rs);

	return ret;
#else
	error("Compiled without ISMRMRD_STREAM not enabled.\n");
	return NULL;
#endif
}

extern "C" void ismrm_stream_close(struct ismrm_cpp_state* s)
{
#ifdef USE_ISMRMRD_STREAM
	if (NULL != s->deserializer)
		delete s->deserializer;
	// idk
	if (NULL != s->rs)
		delete s->rs;
	if (NULL != s->is)
		delete s->is;
#else
	error("Compiled without ISMRMRD_STREAM not enabled.\n");
#endif
}

extern "C" void ismrm_stream_read_meta(struct isrmrm_config_s* config)
{
#ifdef USE_ISMRMRD_STREAM
	struct ismrm_cpp_state* s = config->ismrm_cpp_state;

	try {

		uint16_t type;

		while (ISMRMRD::ISMRMRD_MESSAGE_HEADER != (type = s->deserializer->peek())) {

			if (ISMRMRD::ISMRMRD_MESSAGE_CONFIG_FILE == type) {

				ISMRMRD::ConfigFile conf;
				s->deserializer->deserialize(conf);
			} else if (ISMRMRD::ISMRMRD_MESSAGE_CONFIG_TEXT == type) {

				ISMRMRD::ConfigText conf;
				s->deserializer->deserialize(conf);
			} else if (ISMRMRD::ISMRMRD_MESSAGE_TEXT == type) {

				ISMRMRD::TextMessage tm;
				s->deserializer->deserialize(tm);
				debug_printf(DP_WARN, "Message from ISMRM stream: %s", tm.message.c_str());
			} else {

				error("Unexpected message type: %d.\n", type);
			}
		}

		assert (type == ISMRMRD::ISMRMRD_MESSAGE_HEADER);

		ISMRMRD::IsmrmrdHeader hdr;
		s->deserializer->deserialize(hdr);

		ismrm_read_encoding_limits_from_hdr(hdr, config);
	}
	catch(std::runtime_error& e) {
		error("BART ISMRMRD Wrapper: Exception thrown: %s\n", e.what());
	}
#else
	error("Compiled without ISMRMRD_STREAM not enabled.\n");
#endif
}

extern "C" long ismrm_stream_read_acquisition(struct isrmrm_config_s* config, ISMRMRD::ISMRMRD_Acquisition* c_acq)
{
	struct ismrm_cpp_state* s = config->ismrm_cpp_state;


#ifdef USE_ISMRMRD_STREAM
	try {

		uint16_t type;

		while (ISMRMRD::ISMRMRD_MESSAGE_ACQUISITION != (type = s->deserializer->peek())) {

			if (ISMRMRD::ISMRMRD_MESSAGE_CLOSE == type)
				return 0;

			if (ISMRMRD::ISMRMRD_MESSAGE_TEXT == type) {

				ISMRMRD::TextMessage tm;
				s->deserializer->deserialize(tm);
				debug_printf(DP_WARN, "Message from ISMRM stream: %s", tm.message.c_str());
				continue;
			}

			error("BART ISMRMRD Wrapper: Unexpected Non-Acquisition message.\n");
		}

		assert(ISMRMRD::ISMRMRD_MESSAGE_ACQUISITION == type);
		ISMRMRD::Acquisition a;
		s->deserializer->deserialize(a);

		c_acq->head = a.getHead();

		size_t data_size = a.getDataSize();
		c_acq->data = (complex_float_t*)xmalloc(data_size);
		memcpy(c_acq->data, a.getDataPtr(), data_size);

		if (LONG_MAX < data_size)
			error("BART ISMRMRD Wrapper: Too large acquisition.\n");

		return (long)data_size;
	}
	catch(std::runtime_error& e) {
		error("BART ISMRMRD Wrapper: Exception thrown: %s\n", e.what());
	}
#else
	error("Compiled without ISMRMRD_STREAM not enabled.\n");
	return 0;
#endif
}
