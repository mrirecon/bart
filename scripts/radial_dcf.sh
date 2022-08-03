

# oversampled radial trajectory
bart traj -r -y55 -x256 traj_tmp
bart scale 0.5 traj_tmp traj

# simulate k-space
bart phantom -t traj ksp

# compute Ram-Lak filter
bart rss 1 traj ramlak

# apply to data
bart fmac ksp ramlak ksp_filt

# adjoint nufft
bart nufft -a traj ksp img
bart nufft -a traj ksp_filt img_filt

# grid and degrid ones
bart ones 3 1 256 55 ones
bart nufft -a traj ones dens_tmp
bart nufft traj dens_tmp density

# sqrt
bart spow -- -1. density dcf

# inv sqrt
bart spow -- -0.5 density sqdcf

# adjoint nufft
bart fmac dcf ksp ksp_filt2
bart nufft -a traj ksp_filt2 img_filt2

# one channel all ones sensititty
bart ones 3 256 256 1 sens

# without dcf
bart pics -i30 -t traj ksp sens img_pics_i30
bart pics -i3 -t traj ksp sens img_pics_i3

# with dcf
bart pics -i30 -t traj -p sqdcf ksp sens img_pics_dcf_i30
bart pics -i3 -t traj -p sqdcf ksp sens img_pics_dcf_i3


