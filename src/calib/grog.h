
extern void grog_calib(int D, const long lnG_dims[D], complex float* lnG, const long tdims[D], const complex float* traj, const long ddims[D], const complex float* data);

extern void grog_grid(int D, const long tdims[D], const complex float* traj_shift, const long ddims[D], complex float* data_grid, const complex float* data, const long lnG_dims[D], complex float* lnG);
