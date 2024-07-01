
#ifdef __cplusplus
extern "C" {
#endif

#include "misc/mri.h"

struct linop_s;

extern struct linop_s* linop_sampling_create(const long dims[DIMS], const long pat_dims[DIMS], const _Complex float* pattern);

extern struct linop_s* sense_init(unsigned long shared_img_flags, const long max_dims[DIMS], unsigned long sens_flags, const _Complex float* sens);
extern struct linop_s* maps_create(unsigned long shared_img_flags, const long max_dims[DIMS], 
			unsigned long sens_flags, const _Complex float* sens);
extern struct linop_s* maps2_create(const long coilim_dims[DIMS], const long maps_dims[DIMS], const long img_dims[DIMS], const _Complex float* maps);


#ifdef __cplusplus
}
#endif


