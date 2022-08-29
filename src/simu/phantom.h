

extern void calc_sens(const long dims[DIMS], complex float* sens);

extern void calc_geo_phantom(const long dims[DIMS], complex float* out, _Bool ksp, int phtype, const long tstrs[DIMS], const _Complex float* traj);

extern void calc_phantom_noncart(const long dims[3], complex float* out, const complex float* traj);
extern void calc_geo_phantom_noncart(const long dims[3], complex float* out, const complex float* traj, int phtype);

extern void calc_phantom(const long dims[DIMS], _Complex float* out, _Bool d3, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj);
extern void calc_circ(const long dims[DIMS], _Complex float* img, _Bool d3, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj);
extern void calc_ring(const long dims[DIMS], _Complex float* img, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj);

extern void calc_moving_circ(const long dims[DIMS], _Complex float* out, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj);
extern void calc_heart(const long dims[DIMS], _Complex float* out, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj);

extern void calc_phantom_tubes(const long dims[DIMS], _Complex float* out, bool kspace, bool random, float rotation_angle, int N, const long tstrs[DIMS], const complex float* traj);


struct ellipsis_s;
extern void calc_phantom_arb(int N, const struct ellipsis_s* data /*[N]*/, const long dims[DIMS], _Complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, float rotation_angle);

extern void calc_star(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj);
extern void calc_star3d(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj);
extern void calc_bart(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj);

