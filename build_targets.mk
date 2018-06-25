# Main build targets
#
TBASE=show slice crop resize join transpose squeeze flatten zeros ones flip circshift extract repmat bitmask reshape version delta copy casorati vec poly
TFLP=scale invert conj fmac saxpy sdot spow cpyphs creal carg normalize cdf97 pattern nrmse mip avg cabs zexpj
TNUM=fft fftmod fftshift noise bench threshold conv rss filter mandelbrot wavelet window var std
TRECO=pics pocsense sqpics itsense nlinv nufft rof sake wave lrmatrix estdims estshift estdelay wavepsf wshfl
TCALIB=ecalib ecaltwo caldir walsh cc ccapply calmat svd estvar whiten
TMRI=homodyne poisson twixread fakeksp
TSIM=phantom traj
TIO=toimg
