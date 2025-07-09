
#ifndef DIMS
#define DIMS 16
#endif

void traj_radial_angles(int N, float angles[N], const long tdims[DIMS], const _Complex float* traj);
float traj_radial_dcshift(const long tdims[DIMS], const _Complex float* traj);
float traj_radial_dk(const long tdims[DIMS], const _Complex float* traj);
