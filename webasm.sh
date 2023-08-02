emcc -O3 -Wall bart.o -s EXPORTED_FUNCTIONS="['__Block_object_dispose','_malloc','_free','_bart_version', '_calc_phantom', '_calc_bart', '_calc_circ', '_fftc','_ifftc','_num_init', '_pha_opts_defaults', '_memcfl_create', '_load_cfl', '_main_ecalib', '_main_pics', '_main_phantom', '_main_fft']" -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4GB -o ./web/wwwroot/bart.js /home/ich/fftw/lib/libfftw3f.a /home/ich/openblas/lib/libopenblas.a /home/ich/blocksruntime/libBlocksRuntime.a
emcc -O3 -Wall bart.o -s EXPORTED_FUNCTIONS="['__Block_object_dispose','_malloc','_free','_bart_version', \
'_memcfl_create', '_load_cfl', '_memcfl_list_all', '_memcfl_unlink', \
'_main_avg', '_main_bench', '_main_bin', '_main_bitmask', '_main_cabs', '_main_caldir', '_main_calmat', '_main_carg', '_main_casorati', \
'_main_cc', '_main_ccapply', '_main_cdf97', '_main_circshift', '_main_conj', '_main_conv', '_main_conway', '_main_copy', '_main_cpyphs', \
'_main_creal', '_main_crop', '_main_delta', '_main_ecalib', '_main_ecaltwo', '_main_estdelay', '_main_estdims', '_main_estshift', \
'_main_estvar', '_main_extract', '_main_fakeksp', '_main_fft', '_main_fftmod', '_main_fftrot', '_main_fftshift', '_main_filter', \
'_main_flatten', '_main_flip', '_main_fmac', '_main_fovshift', '_main_homodyne', '_main_ictv', '_main_index', '_main_invert', \
'_main_itsense', '_main_join', '_main_looklocker', '_main_lrmatrix', '_main_mandelbrot', '_main_measure', '_main_mip', \
'_main_mnist', '_main_moba', '_main_mobafit', '_main_morphop', '_main_multicfl', '_main_nlinv', '_main_nnet', '_main_noise', '_main_normalize', \
'_main_nrmse', '_main_nufft', '_main_nufftbase', '_main_onehotenc', '_main_ones', '_main_pattern', '_main_phantom', '_main_pics', \
'_main_pocsense', '_main_poisson', '_main_pol2mask', '_main_poly', '_main_reconet', '_main_repmat', '_main_reshape', '_main_resize', \
'_main_rmfreq', '_main_rof', '_main_roistat', '_main_rss', '_main_rtnlinv', '_main_sake', '_main_saxpy', '_main_scale', '_main_sdot', '_main_show', \
'_main_signal', '_main_sim', '_main_slice', '_main_spow', '_main_sqpics', '_main_squeeze', '_main_ssa', '_main_std', '_main_svd', '_main_tgv', \
'_main_threshold', '_main_toimg', '_main_traj', '_main_transpose', '_main_twixread', '_main_upat', '_main_var', '_main_vec', '_main_version', \
'_main_walsh', '_main_wave', '_main_wavelet', '_main_wavepsf', '_main_whiten', '_main_window', '_main_wshfl', '_main_zeros', '_main_zexp' \
 ]" -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4GB -o ./web/wwwroot/bart_cmd.js \
 /home/ich/fftw/lib/libfftw3f.a /home/ich/openblas/lib/libopenblas.a /home/ich/blocksruntime/libBlocksRuntime.a