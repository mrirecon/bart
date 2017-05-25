# Main build targets
#
TBASE=show slice crop resize join transpose squeeze flatten zeros ones flip circshift extract repmat bitmask reshape version delta copy casorati
TFLP=scale invert conj fmac saxpy sdot spow cpyphs creal carg normalize cdf97 pattern nrmse mip avg cabs zexpj
TNUM=fft fftmod fftshift noise bench threshold conv rss filter mandelbrot wavelet
TRECO=pics pocsense sqpics bpsense itsense nlinv nufft rof sake wave lrmatrix estdims estshift estdelay
TCALIB=ecalib ecaltwo caldir walsh cc ccapply calmat svd estvar
TMRI=homodyne poisson twixread fakeksp
TSIM=phantom traj
TIO=toimg
