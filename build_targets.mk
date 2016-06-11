# Main build targets
#
TBASE=show slice crop resize join transpose zeros ones flip circshift extract repmat bitmask reshape version
TFLP=scale invert conj fmac saxpy sdot spow cpyphs creal carg normalize cdf97 pattern nrmse mip avg cabs
TNUM=fft fftmod fftshift noise bench threshold conv rss filter
TRECO=pics pocsense sqpics bpsense itsense nlinv nufft rof sake wave lrmatrix estdims
TCALIB=ecalib ecaltwo caldir walsh cc calmat svd estvar
TMRI=homodyne poisson twixread fakeksp
TSIM=phantom traj
TIO=toimg
