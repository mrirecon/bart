
#ifndef _SENS_H
#define _SENS_H

#include "simu/grid.h"

enum coil_type { COIL_NONE, HEAD_2D_8CH, HEAD_3D_64CH };

struct coil_opts;

typedef _Complex double (*sim_fun_t)(const void*, const long C, const float pos[]);
typedef void (*co_dstr_t)(void* v);

struct coil_opts {

	_Bool kspace;
	enum coil_type ctype;
	unsigned long flags; // flags for channel selection
	long N; // chosen number of coil channels
	void* data;
	co_dstr_t dstr;
	sim_fun_t fun;
};

extern struct coil_opts coil_opts_defaults;
extern struct coil_opts coil_opts_pha_defaults;

extern void get_position(long D, float p[4], const long pos[D], const long gdims[D], const float* grid);
extern void sample_coils(long D, const long odims[D], _Complex double* optr, const long gdims[D], const float* grid, const struct coil_opts* copts);
extern void cnstr_coils(long D, struct coil_opts* copts, bool legacy_fov);
extern float* create_senstraj(long D, long gdims[D], struct grid_opts* gopts, struct coil_opts* copts);

extern complex float* sens_internal_H2D8CH(long D, long dims[D], unsigned long flags);
extern complex float* sens_internal_H3D64CH(long D, long dims[D], unsigned long flags);

extern const _Complex float sens_coeff[8][5][5];
extern const _Complex float sens64_coeff[64][5][5][5];

#endif
