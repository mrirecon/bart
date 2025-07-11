
#ifndef _PHANTOM_H
#define _PHANTOM_H

enum coil_type { COIL_NONE, HEAD_2D_8CH, HEAD_3D_64CH };

struct pha_opts {

	enum coil_type stype;
};

extern struct pha_opts pha_opts_defaults;

extern void calc_ellipsoid(int D, long dims[D], _Complex float* out, _Bool d3, _Bool kspace, long tdims[D], long tstrs[D], _Complex float* traj, float ax[3], long center[3], float rot, struct pha_opts* popts);

extern void calc_sens(const long dims[DIMS], complex float* sens, struct pha_opts* popts);

extern void calc_geo_phantom(const long dims[DIMS], complex float* out, _Bool ksp, int phtype, const long tstrs[DIMS], const _Complex float* traj, struct pha_opts* popts);

extern void calc_phantom_noncart(const long dims[3], complex float* out, const complex float* traj, struct pha_opts* popts);
extern void calc_geo_phantom_noncart(const long dims[3], complex float* out, const complex float* traj, int phtype, struct pha_opts* popts);

extern void calc_phantom(const long dims[DIMS], _Complex float* out, _Bool d3, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj, struct pha_opts* popts);
extern void calc_circ(const long dims[DIMS], _Complex float* img, _Bool d3, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj, struct pha_opts* popts);
extern void calc_ring(const long dims[DIMS], _Complex float* img, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj, struct pha_opts* popts);

extern void calc_moving_circ(const long dims[DIMS], _Complex float* out, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj, struct pha_opts* popts);
extern void calc_heart(const long dims[DIMS], _Complex float* out, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj, struct pha_opts* popts);

extern void calc_phantom_tubes(const long dims[DIMS], _Complex float* out, bool kspace, bool random, float rotation_angle, int N, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts);


struct ellipsis_s;
extern void calc_phantom_arb(int N, const struct ellipsis_s* data /*[N]*/, const long dims[DIMS], _Complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, float rotation_angle, struct pha_opts* popts);

extern void calc_star(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts);
extern void calc_star3d(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts);
extern void calc_bart(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts);
extern void calc_brain(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts);

extern void calc_cfl_geom(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, int D_max, long hdims[2][D_max], complex float* x[2], struct pha_opts* popts);

#endif

