
#include <complex.h>

#include "misc/mri.h"

struct nufft_conf_s;
extern const struct linop_s* sense_nc_init(const long max_dims[DIMS], const long map_dims[DIMS], const complex float* maps,
		const long ksp_dims[DIMS],
		const long traj_dims[DIMS], const complex float* traj, const struct nufft_conf_s* conf,
		const long wgs_dims[DIMS], const complex float* weights,
		const long basis_dims[DIMS], const complex float* basis,
		const struct linop_s** fft_opp, unsigned long shared_img_dims, unsigned long lowmem_stack);




