

extern void na_fft(unsigned int flags, struct na_s* dst, struct na_s* src);
extern void na_ifft(unsigned int flags, struct na_s* dst, struct na_s* src);
extern void na_fftc(unsigned int flags, struct na_s* dst, struct na_s* src);
extern void na_ifftc(unsigned int flags, struct na_s* dst, struct na_s* src);
extern void na_fftu(unsigned int flags, struct na_s* dst, struct na_s* src);
extern void na_ifftu(unsigned int flags, struct na_s* dst, struct na_s* src);
extern void na_fftuc(unsigned int flags, struct na_s* dst, struct na_s* src);
extern void na_ifftuc(unsigned int flags, struct na_s* dst, struct na_s* src);





extern void na_fmac(struct na_s* dst, struct na_s* src1, struct na_s* src2);

