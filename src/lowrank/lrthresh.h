
#include <complex.h>


#include "misc/mri.h"

#ifndef MAX_LEV
#define MAX_LEV 100
#endif

struct operator_p_s;


// Low rank thresholding for arbitrary block sizes
extern const struct operator_p_s* lrthresh_create(const long dims_lev[DIMS], _Bool randshift, unsigned long mflags, const long blkdims[MAX_LEV][DIMS], float lambda, _Bool noise, int remove_mean, _Bool overlapping_blocks);

// Returns nuclear norm using lrthresh operator
extern float lrnucnorm(const struct operator_p_s* op, const complex float* src);

// Generates multiscale block sizes
extern int multilr_blkdims(long blkdims[MAX_LEV][DIMS], unsigned long flags, const long dims[DIMS], int blkskip, int initblk);

// Generates locally low rank block size
extern int llr_blkdims(long blkdims[MAX_LEV][DIMS], unsigned long flags, const long dims[DIMS], int llrblk);

// Generates low rank plus sparse block size
extern int ls_blkdims(long blkdims[MAX_LEV][DIMS], const long dims[DIMS]);


extern void add_lrnoiseblk(int* level, long blkdims[MAX_LEV][DIMS], const long dims[DIMS]);

// Return the regularization parameter
extern float get_lrthresh_lambda(const struct operator_p_s* o);
