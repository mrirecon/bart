
#ifndef _GRID_H
#define _GRID_H

#include "misc/cppwrap.h"


struct grid_conf_s {

	float os;
	float width;
	_Bool periodic;
	double beta;

	float shift[3];
};

extern int kb_size;
extern double bessel_kb_beta; // = bessel_i0(beta);
extern const struct multiplace_array_s* kb_get_table(double beta);

extern void grid(const struct grid_conf_s* conf, const long ksp_dims[4], const long trj_strs[4], const _Complex float* traj, const long grid_dims[4], const long grid_strs[4], _Complex float* grid, const long ksp_strs[4], const _Complex float* src);

extern void gridH(const struct grid_conf_s* conf, const long ksp_dims[4], const long trj_strs[4], const _Complex float* traj, const long ksp_strs[4], _Complex float* dst, const long grid_dims[4], const long grid_strs[4], const _Complex float* grid);


extern void grid2(const struct grid_conf_s* conf, int D, const long trj_dims[__VLA(D)], const _Complex float* traj, const long grid_dims[__VLA(D)], _Complex float* grid, const long ksp_dims[__VLA(D)], const _Complex float* src);

extern void grid2H(const struct grid_conf_s* conf, int D, const long trj_dims[__VLA(D)], const _Complex float* traj, const long ksp_dims[__VLA(D)], _Complex float* dst, const long grid_dims[__VLA(D)], const _Complex float* grid);


extern void grid_pointH(int ch, int N, const long dims[__VLA(N)], const long strs[__VLA(N)], const float pos[__VLA(N)], _Complex float val[__VLA(ch)], const _Complex float* src, _Bool periodic, float width, int kb_size, const float kb_table[__VLA(kb_size + 1)]);
extern void grid_point(int ch, int N, const long dims[__VLA(N)], const long strs[__VLA(N)], const float pos[__VLA(N)], _Complex float* dst, const _Complex float val[__VLA(ch)], _Bool periodic, float width, int kb_size, const float kb_table[__VLA(kb_size + 1)]);

extern void kb_init(double beta);
extern double calc_beta(float os, float width);
extern void kb_precompute(double beta, int n, float table[__VLA(n + 1)]);

extern void rolloff_correction(float os, float width, float beta, const long dim[3], _Complex float* dst);
extern void apply_rolloff_correction2(float os, float width, float beta, int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], _Complex float* dst, const long istrs[__VLA(N)], const _Complex float* src);
extern void apply_rolloff_correction(float os, float width, float beta, int N, const long dimensions[__VLA(N)], _Complex float* dst, const _Complex float* src);


#include "misc/cppwrap.h"

#endif // _GRID_H

