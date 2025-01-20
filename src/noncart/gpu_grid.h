#include "misc/cppwrap.h"

extern void cuda_apply_linphases_3D(int N, const long img_dims[__VLA(N)], const float shifts[3], _Complex float* dst, const _Complex float* src, _Bool conj, _Bool fmac, _Bool fftm, float scale);
extern void cuda_apply_rolloff_correction2(float os, float width, float beta, int N, const long dims[4], const long ostrs[4], _Complex float* dst, const long istrs[4], const _Complex float* src);


struct grid_conf_s;
extern void cuda_grid(const struct grid_conf_s* conf, const long ksp_dims[4], const long trj_strs[4], const float* traj, const long grid_dims[4], const long grid_strs[4], _Complex float* grid, const long ksp_strs[4], const _Complex float* src);
extern void cuda_gridH(const struct grid_conf_s* conf, const long ksp_dims[4], const long trj_strs[4], const float* traj, const long ksp_strs[4], _Complex float* dst, const long grid_dims[4], const long grid_strs[4], const _Complex float* grid);

#include "misc/cppwrap.h"


