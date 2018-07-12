/*
 * Martin Uecker 2012-01-18
 * uecker@eecs.berkeley.edu
 *
 * Damien Nguyen 2018-04-19
 * damien.nguyen@alumni.epfl.ch
 */

#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <complex.h>

// MATLAB
#include "mat.h"
#include "matrix.h"

#include "num/multind.h"
#include "misc/mmio.h"






int main(int argc, char *argv[])
{
	MATFile *mat;
	const char* name = NULL;
	const mwSize* dims;

	if (argc != 2) {

		fprintf(stderr, "Usage: %s file.mat\n", argv[0]);
		return 1;
	}


	if (NULL == (mat = matOpen(argv[1], "r")))
		return 1;

	mxArray* ar;

	while (NULL != (ar = matGetNextVariable(mat, &name))) {

		int ndim = (int)mxGetNumberOfDimensions(ar);
		dims = mxGetDimensions(ar);

		bool cmp = mxIsComplex(ar);
		bool dbl = mxIsDouble(ar);

		printf("%s: [ ", name);

		if ((!cmp) || (!dbl)) {

			printf("not complex double\n");
			mxDestroyArray(ar);
			continue;
		}

		long ldims[ndim];
	        for (int i = 0; i < ndim; i++)
			ldims[i] = dims[i];	

		for (int i = 0; i < ndim; i++)
			printf("%ld ", ldims[i]);

		char outname[256];
		snprintf(outname, 256, "%s_%s", strtok(argv[1], "."), name);

		complex float* buf = create_cfl(outname, ndim, ldims);
		size_t size = md_calc_size(ndim, ldims);

#if MX_HAS_INTERLEAVED_COMPLEX
		mxComplexDouble* x = mxGetComplexDoubles(ar);

		for (unsigned long i = 0; i < size; i++) 
			buf[i] = x[i].real + 1.i * x[i].imag;
#else
		double* re = mxGetPr(ar);
		double* im = mxGetPi(ar);

		for (unsigned long i = 0; i < size; i++) 
			buf[i] = re[i] + 1.i * im[i];
#endif /* MX_HAS_INTERLEAVED_COMPLEX */

		printf("] -> %s\n", outname);

		unmap_cfl(ndim, ldims, buf);
		mxDestroyArray(ar);
	}

	matClose(mat);
}
