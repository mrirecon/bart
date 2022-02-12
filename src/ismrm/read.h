
#include "misc/mri.h"

#include "ismrmrd/ismrmrd.h"

#ifdef __cplusplus
namespace ISMRMRD {
extern "C" {
#endif

struct limit_s {

	long size;

	long size_hdr;	//size derived from xml-header
	long center;	//center derived from xml-header

	long min_hdr;	//minimum derived from xml-header
	long max_hdr;	//maximum derived from xml-header

	long min_idx;	//minimum found in acquisitions
	long max_idx;	//maximum found in acquisitions
};

enum ISMRMRD_mri_dims {
	
	ISMRMRD_READ_DIM,
	ISMRMRD_COIL_DIM,

	ISMRMRD_PHS1_DIM,	/**< e.g. phase encoding line number */
	ISMRMRD_PHS2_DIM,	/**< e.g. partition encoding number */

	ISMRMRD_AVERAGE_DIM,	/**< e.g. signal average number */
	ISMRMRD_SLICE_DIM,	/**< e.g. imaging slice number */
	ISMRMRD_CONTRAST_DIM,	/**< e.g. echo number in multi-echo */
	ISMRMRD_PHASE_DIM,	/**< e.g. cardiac phase number */
	ISMRMRD_REPETITION_DIM,	/**< e.g. dynamic number for dynamic scanning */
	ISMRMRD_SET_DIM,	/**< e.g. flow encoding set */
	ISMRMRD_SEGMENT_DIM,	/**< e.g. segment number for segmented acquisition */

	ISMRMRD_NAMED_DIMS,
};

enum ISMRMRD_SLICE_ORDERING {

	ISMRMRD_SLICE_ASCENDING,
	ISMRMRD_SLICE_INTERLEAVED,
	ISMRMRD_SLICE_INTERLEAVED_SIEMENS,
};

struct isrmrm_config_s {

	int idx_encoding;
	int dim_mapping[ISMRMRD_NAMED_DIMS + ISMRMRD_USER_INTS];

	struct limit_s limits[ISMRMRD_NAMED_DIMS + ISMRMRD_USER_INTS];

	enum ISMRMRD_SLICE_ORDERING slice_ord;

	_Bool check_dims_with_acquisition;

	unsigned long merge_dims;
	unsigned long shift;

	int measurement;
	int repetition;
	int overwriting_idx;
};

extern struct isrmrm_config_s ismrm_default_config;

extern void ismrm_print_xml(const char* filename);
extern void ismrm_read_dims(const char* datafile, struct isrmrm_config_s* config, int N, long dims[__VLA(N)]);
extern void ismrm_read(const char* datafile, struct isrmrm_config_s* config, int N, long dims[__VLA(N)], _Complex float* buf);

#ifdef __cplusplus
}
}
#endif

