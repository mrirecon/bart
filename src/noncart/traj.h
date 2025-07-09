
struct traj_conf {

	_Bool radial;
	_Bool golden;
	_Bool aligned;
	_Bool full_circle;
	_Bool half_circle_gold;
	_Bool golden_partition;
	_Bool d3d;
	_Bool transverse;
	_Bool asym_traj;
	_Bool mems_traj;
	_Bool rational;
	_Bool double_base;
	int accel;
	int tiny_gold;
	int Y;
	int raga_inc;
	int turns;
	int mb;
};

extern const struct traj_conf traj_defaults;
extern const struct traj_conf rmfreq_defaults;

#ifndef DIMS
#define DIMS 16
#endif

extern void traj_read_dir(float dir[3], float phi, float psi);
extern void gradient_delay(float d[3], float coeff[2][3], float phi, float psi);
extern void calc_base_angles(double base_angle[DIMS], int Y, int E, struct traj_conf conf);
extern void indices_from_position(long ind[DIMS], const long pos[DIMS], struct traj_conf conf);
extern _Bool zpartition_skip(long partitions, long z_usamp[2], long partition, long frame);
extern int gen_fibonacci(int n, int ind);
extern int recover_gen_fib_ind(int Y, int inc);
extern int raga_find_index(int Y, int n);
extern int raga_increment(int Y, int n);
extern int raga_spokes(int baseresolution, int tiny_ga);

