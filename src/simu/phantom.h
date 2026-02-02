
#ifndef _PHANTOM_H
#define _PHANTOM_H

#include "stl/misc.h"
#include "simu/sens.h"

enum phantom_type { SHEPPLOGAN, CIRC, TIME, SENS, GEOM, STAR, BART, BRAIN, TUBES, RAND_TUBES, NIST, SONAR, GEOMFILE, ELLIPSOID0, STL };

struct phantom_opts {

	int D;
	_Bool kspace;
	enum phantom_type ptype;
	long Nc; // number of coefficients for COEFF_DIM
	void* data;
	co_dstr_t dstr;
	sim_fun_t fun;
};

extern struct phantom_opts phantom_opts_defaults;

extern void phantom_stl_init(struct phantom_opts* popts, int D, long dims[D], double* model);

extern void calc_ellipsoid(int D, long dims[D], _Complex float* out, _Bool d3, _Bool kspace, long tdims[D], long tstrs[D], _Complex float* traj, float ax[3], long center[3], float rot, struct coil_opts* copts);

extern void calc_sens(const long dims[DIMS], complex float* sens, struct coil_opts* copts);

extern void calc_geo_phantom(const long dims[DIMS], complex float* out, _Bool ksp, int phtype, const long tstrs[DIMS], const _Complex float* traj, struct coil_opts* copts);

extern void calc_phantom_noncart(const long dims[3], complex float* out, const complex float* traj, struct coil_opts* copts);
extern void calc_geo_phantom_noncart(const long dims[3], complex float* out, const complex float* traj, int phtype, struct coil_opts* copts);

extern void calc_phantom(const long dims[DIMS], _Complex float* out, _Bool d3, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj, struct coil_opts* copts);
extern void calc_circ(const long dims[DIMS], _Complex float* img, _Bool d3, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj, struct coil_opts* copts);
extern void calc_ring(const long dims[DIMS], _Complex float* img, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj, struct coil_opts* copts);

extern void calc_moving_circ(const long dims[DIMS], _Complex float* out, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj, struct coil_opts* copts);
extern void calc_heart(const long dims[DIMS], _Complex float* out, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj, struct coil_opts* copts);

extern void calc_phantom_tubes(const long dims[DIMS], _Complex float* out, bool kspace, bool random, float rotation_angle, int N, const long tstrs[DIMS], const complex float* traj, struct coil_opts* copts);


struct ellipsis_s;
extern void calc_phantom_arb(int N, const struct ellipsis_s* data /*[N]*/, const long dims[DIMS], _Complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, float rotation_angle, struct coil_opts* copts);

extern void calc_star(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct coil_opts* copts);
extern void calc_star3d(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct coil_opts* copts);
extern void calc_bart(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct coil_opts* copts);
extern void calc_brain(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct coil_opts* copts);

extern void calc_cfl_geom(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, int D_max, long hdims[2][D_max], complex float* x[2], struct coil_opts* copts);
extern _Complex double* sample_signal(int D, long odims[D], const long gdims[D], const float* grid, const long sgdims[D], const float* sgrid, const struct phantom_opts* popts, const struct coil_opts* copts);
extern _Complex double stl_fun_k(const void* v, const long C, const float k1[]);

#endif

