#ifdef __cplusplus
namespace ISMRMRD {
extern "C" {
#endif


struct isrmrm_config_s;

extern void ismrm_read_encoding_limits(const char* filename, struct isrmrm_config_s* encoding);

#ifdef __cplusplus
}
}
#endif