
#include <complex.h>
#include <stdbool.h>

#include "misc/mri.h"

struct nufft_conf_s;

struct pics_config {

	struct nufft_conf_s* nuconf;

	bool gpu;
	bool gpu_gridding;
	bool real_value_constraint;
	bool time_encoded_asl;

	unsigned long shared_img_flags;
	unsigned long motion_flags;
	unsigned long lowmem_flags;
};

struct linop_s;

extern const struct linop_s* pics_model(const struct pics_config* conf,
				const long img_dims[DIMS], const long ksp_dims[DIMS],
				const long traj_dims[DIMS], const complex float* traj,
				const long basis_dims[DIMS], const complex float* basis,
				const long map_dims[DIMS], const complex float* maps,
				const long pat_dims[DIMS], const complex float* pattern,
				const long motion_dims[DIMS], complex float* motion,
				const struct linop_s** nufft_op);

