
extern void traj_radial_angles(int N, const long adims[__VLA(N)], float* angles, const long tdims[__VLA(N)], const _Complex float* traj);
extern float traj_radial_dcshift(int N, const long tdims[__VLA(N)], const _Complex float* traj);
extern float traj_radial_deltak(int N, const long tdims[__VLA(N)], const _Complex float* traj);
