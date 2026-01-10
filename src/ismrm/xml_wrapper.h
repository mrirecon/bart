
struct isrmrm_config_s;

#include "misc/cppwrap.h"

#include "ismrmrd/ismrmrd.h"

#ifdef __cplusplus
#define ISMRMRD_NS(x) ISMRMRD::x
#else
#define ISMRMRD_NS(x) x
#endif

struct ismrm_cpp_state;

extern struct ismrm_cpp_state* ismrm_stream_open(const char* file);
extern void ismrm_stream_close(struct ismrm_cpp_state* s);

extern void ismrm_read_encoding_limits(const char* filename, struct isrmrm_config_s* encoding);
extern void ismrm_read_encoding_limits_from_xml(const char* xml, struct isrmrm_config_s* config);

extern void ismrm_stream_read_meta(struct isrmrm_config_s* config);
extern long ismrm_stream_read_acquisition(struct isrmrm_config_s* config, ISMRMRD_NS(ISMRMRD_Acquisition)* c_acq);

#include "misc/cppwrap.h"
