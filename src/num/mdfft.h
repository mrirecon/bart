
#ifndef _MD_FFT_H
#define _MD_FFT_H	1

#define MD_FFT_FORWARD 0u
#define MD_FFT_INVERSE (~0u)


extern void md_fft2(unsigned int N, const long dims[N],
	unsigned long flags, unsigned long dirs,
	const long ostr[N], complex float* dst,
	const long istr[N], const complex float* in);

extern void md_fft(unsigned int N, const long dims[N],
		unsigned long flags, unsigned long dirs,
		complex float* dst, const complex float* in);


#endif		// _MD_FFT_H

