

extern void compose_displacement(int N, int d, unsigned long flags, const long dims[__VLA(N)], _Complex float* composed, const _Complex float* d1, const _Complex float* d2);
extern void invert_displacement(int N, int d, unsigned long flags, const long dims[__VLA(N)], _Complex float* inv_disp, const _Complex float* disp);

struct linop_s;
extern const struct linop_s* linop_interpolate_displacement_create(int d, unsigned long flags, int ord, int N, const long idims[__VLA(N)], const long mdims[__VLA(N)], const _Complex float* motion, const long gdims[__VLA(N)]);


