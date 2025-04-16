struct isrmrm_config_s;

#include "misc/cppwrap.h"


extern void ismrm_read_encoding_limits(const char* filename, struct isrmrm_config_s* encoding);
extern void ismrm_read_encoding_limits_from_xml(const char* xml, struct isrmrm_config_s* config);

#include "misc/cppwrap.h"
