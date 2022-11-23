#include "misc/cppwrap.h"

struct linop_s;
struct grid_conf_s;

extern struct linop_s* linop_kb_rolloff_create(int N, const long dims[__VLA(N)], unsigned long flags, struct grid_conf_s* conf);
extern struct linop_s* linop_interpolate_create(int N, unsigned long flags, const long ksp_dims[__VLA(N)], const long grd_dims[__VLA(N)], const long trj_dims[__VLA(N)], const _Complex float* traj, struct grid_conf_s* conf);

extern struct linop_s* nufft_create_chain(int N,
			     const long ksp_dims[N],
			     const long cim_dims[N],
			     const long traj_dims[N],
			     const _Complex float* traj,
			     const long wgh_dims[N],
			     const _Complex float* weights,
			     const long bas_dims[N],
			     const _Complex float* basis,
			     struct grid_conf_s* conf);

#include "misc/cppwrap.h"

