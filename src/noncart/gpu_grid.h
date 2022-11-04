#include "misc/cppwrap.h"

extern void cuda_apply_linphases_3D(int N, const long img_dims[__VLA(N)], const float shifts[3], _Complex float* dst, const _Complex float* src, _Bool conj, _Bool fmac, float scale);
extern void cuda_apply_rolloff_correction(float os, float width, float beta, int N, const long dims[__VLA(N)], _Complex float* dst, const _Complex float* src);


struct grid_conf_s;
extern void cuda_grid(const struct grid_conf_s* conf, int N, const long traj_dims[__VLA(N)], const _Complex float* traj, const long grid_dims[__VLA(N)], _Complex float* grid, const long ksp_dims[__VLA(N)], const _Complex float* src);
extern void cuda_gridH(const struct grid_conf_s* conf, int N, const long traj_dims[__VLA(N)], const _Complex float* traj, const long ksp_dims[__VLA(N)], _Complex float* dst, const long grid_dims[__VLA(N)], const _Complex float* grid);

#include "misc/cppwrap.h"


