
#define GWIDTH( M, N, B) ( (sqrtf( (float)M ) + sqrtf( (float)N )) + sqrtf( logf( (float)B * (float) ((M > N) ? N : M )) ))

//#define GWIDTH( M, N, B) ( sqrtf( M ) + sqrtf( N ) )

//#define GWIDTH( M, N, B) sqrtf( ((M > N) ? M : N) )


#include <complex.h>

// Singular value thresholding for matrix
extern float svthresh(long M, long N, float lambda, complex float* dst, complex float* src);

extern float svthresh2(long M, long N, float lambda, complex float* dst, const complex float* src, complex float* U, float* S, complex float* VT);

extern float svthresh_nomeanu(int M, int N, float lambda, complex float* dst, const complex float* src);

extern float svthresh_nomeanv(int M, int N, float lambda, complex float* dst, const complex float* src);


// Singular value analysis (maybe useful to help determining regularization parameter for min nuclear norm)
extern float nuclearnorm(long M, long N, /* const */ complex float* d);
extern float maxsingular(long M, long N, /* const */ complex float* d);



extern struct svthresh_blockproc_data* svthresh_blockproc_create(unsigned long mflags, float lambda, int remove_mean);
extern float svthresh_blockproc(const void* _data, const long blkdims[DIMS], complex float* dst, const complex float* src);
extern float nucnorm_blockproc(const void* _data, const long blkdims[DIMS], complex float* dst, const complex float* src);

