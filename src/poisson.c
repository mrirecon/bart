/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2013, 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/init.h"

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/pd.h"
#include "misc/opts.h"


static void random_point(int D, float p[static D])
{
	for (int i = 0; i < D; i++)
		p[i] = uniform_rand();
}

static float dist(int D, const float a[static D], const float b[static D])
{
	float r = 0.;

	for (int i = 0; i < D; i++)
		r += powf(a[i] - b[i], 2.);

	return sqrtf(r);
}



static float maxn(int D, const float a[static D], const float b[static D])
{
	float r = 0.;

	for (int i = 0; i < D; i++)
		r = MAX(fabsf(a[i] - b[i]), r);

	return r;
}



static const char usage_str[] = "<outfile>";
static const char help_str[] = "Computes Poisson-disc sampling pattern.";


int main_poisson(int argc, char* argv[])
{
	int yy = 128;
	int zz = 128;
	bool cutcorners = false;
	float vardensity = 0.;
	bool vd_def = false;
	int T = 1;
	int rnd = 0;
	int randseed = 11235;
	bool msk = true;
	int points = -1;
	float mindist = 1. / 1.275;
	float yscale = 1.;
	float zscale = 1.;
	unsigned int calreg = 0;

	const struct opt_s opts[] = {

		OPT_INT('Y', &yy, "size", "size dimension 1"),
		OPT_INT('Z', &zz, "size", "size dimension 2"),
		OPT_FLOAT('y', &yscale, "acc", "acceleration dim 1"),
		OPT_FLOAT('z', &zscale, "acc", "acceleration dim 2"),
		OPT_UINT('C', &calreg, "size", "size of calibration region"),
		OPT_SET('v', &vd_def, "variable density"),
		OPT_FLOAT('V', &vardensity, "", "(variable density)"),
		OPT_SET('e', &cutcorners, "elliptical scanning"),
		OPT_FLOAT('D', &mindist, "", "()"),
		OPT_INT('T', &T, "", "()"),
		OPT_CLEAR('m', &msk, "()"),
		OPT_INT('R', &points, "", "()"),
		OPT_INT('s', &randseed, "seed", "random seed"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();
	num_rand_init(randseed);

	if (vd_def && (0. == vardensity))
		vardensity = 20.;

	if (-1 != points)
		rnd = 1;


	assert((yscale >= 1.) && (zscale >= 1.));

	// compute mindest and scaling

	float kspext = MAX(yy, zz);

	int Pest = T * (int)(1.2 * powf(kspext, 2.) / (yscale * zscale));

	mindist /= kspext;
	yscale *= (float)kspext / (float)yy;
	zscale *= (float)kspext / (float)zz;

	if (vardensity != 0.) {

		// TODO
	}


	long dims[5] = { 1, yy, zz, T, 1 };
	complex float (*mask)[T][zz][yy] = NULL;

	if (msk) {
		
		mask = MD_CAST_ARRAY3_PTR(complex float, 5, dims, create_cfl(argv[1], 5, dims), 1, 2, 3);
		md_clear(5, dims, &(*mask)[0][0][0], sizeof(complex float));
	}

	int M = rnd ? (points + 1) : Pest;
	int P;
	
	while (true) {

		PTR_ALLOC(float[M][2], points);
		PTR_ALLOC(int[M], kind);
//		int (*kind)[M] = TYPE_ALLOC(int[M]);
		(*kind)[0] = 0;

		if (!rnd) {

			(*points)[0][0] = 0.5;
			(*points)[0][1] = 0.5;

			if (1 == T) {

				P = poissondisc(2, M, 1, vardensity, mindist, *points);

			} else {

				float (*delta)[T][T] = TYPE_ALLOC(float[T][T]);
				float dd[T];
				for (int i = 0; i < T; i++)
					dd[i] = mindist;

				mc_poisson_rmatrix(2, T, *delta, dd);
				P = poissondisc_mc(2, T, M, 1, vardensity, *delta, *points, *kind);
			}

		} else { // random pattern

			P = M - 1;
			for (int i = 0; i < P; i++)
				random_point(2, (*points)[i]);
		}

		if (P < M) {

			for (int i = 0; i < P; i++) {

				(*points)[i][0] = ((*points)[i][0] - 0.5) * yscale + 0.5;
				(*points)[i][1] = ((*points)[i][1] - 0.5) * zscale + 0.5;
			}

			// throw away points outside 
	
			float center[2] = { 0.5, 0.5 };

			int j = 0;
			for (int i = 0; i < P; i++) {

				if ((cutcorners ? dist : maxn)(2, center, (*points)[i]) <= 0.5) {

					(*points)[j][0] = (*points)[i][0];
					(*points)[j][1] = (*points)[i][1];
					j++;
				}
			}

			P = j;


			if (msk) {

				// rethink module here
				for (int i = 0; i < P; i++) {

					int yy = (int)floorf((*points)[i][0] * dims[1]);
					int zz = (int)floorf((*points)[i][1] * dims[2]);

					if ((yy < 0) || (yy >= dims[1]) || (zz < 0) || (zz >= dims[2]))
						continue;

					if (1 == T)
						(*mask)[0][zz][yy] = 1.;//cexpf(2.i * M_PI * (float)(*kind)[i] / (float)T);
					else
						(*mask)[(*kind)[i]][zz][yy] = 1.;//cexpf(2.i * M_PI * (float)(*kind)[i] / (float)T);
				}

			} else {

#if 1
				long sdims[2] = { 3, P };
				//complex float (*samples)[P][3] = (void*)create_cfl(argv[1], 2, sdims);
				complex float (*samples)[P][3] =
					MD_CAST_ARRAY2_PTR(complex float, 2, sdims, create_cfl(argv[1], 2, sdims), 0, 1);

				for (int i = 0; i < P; i++) {

					(*samples)[i][0] = 0.;
					(*samples)[i][1] = ((*points)[i][0] - 0.5) * dims[1];
					(*samples)[i][2] = ((*points)[i][1] - 0.5) * dims[2];
					//	printf("%f %f\n", creal(samples[3 * i + 0]), creal(samples[3 * i + 1]));
				}
				unmap_cfl(2, sdims, &(*samples)[0][0]);
#endif
			}

			PTR_FREE(points);
			PTR_FREE(kind);
			break;
		}

		// repeat with more points
		M *= 2;
		PTR_FREE(points);
		PTR_FREE(kind);
	}

	// calibration region

	assert((mask != NULL) || (0 == calreg));
	assert((calreg <= dims[1]) && (calreg <= dims[2]));

	for (unsigned int i = 0; i < calreg; i++) {
		for (unsigned int j = 0; j < calreg; j++) {

			int y = (dims[1] - calreg) / 2 + i;
			int z = (dims[2] - calreg) / 2 + j;

			for (int k = 0; k < T; k++) {

				if (0. == (*mask)[k][z][y]) {

					(*mask)[k][z][y] = 1.;
					P++;
				}
			}
		}
	}


	printf("points: %d", P);

	if (1 != T)
		printf(", classes: %d", T);

	if (NULL != mask) {

		float f = cutcorners ? (M_PI / 4.) : 1.;
		printf(", grid size: %ldx%ld%s = %ld (R = %f)", dims[1], dims[2], cutcorners ? "x(pi/4)" : "",
				(long)(f * dims[1] * dims[2]), f * T * dims[1] * dims[2] / (float)P);

		unmap_cfl(5, dims, &(*mask)[0][0][0]);
	}

	printf("\n");
	exit(0);
}



