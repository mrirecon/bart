/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 * Wei, L. Multi-Class Blue Noise Sampling. ACM Trans. Graph. 29:79 (2010)
 *
 */ 

#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>

#if 1
#define GRID
// #define CHECK
#endif

#ifdef GRID
#include "num/multind.h"
#endif

#include "num/rand.h"

#include "misc/misc.h"

#include "pd.h"


static float dist(int D, const float a[D], const float b[D])
{
	float r = 0.;

	for (int i = 0; i < D; i++)
		r += powf(a[i] - b[i], 2.);

	return sqrtf(r);
}

#ifdef GRID
static void grid_pos(int D, long pos[D], float delta, const float fpos[D])
{
	for (int i = 0; i < D; i++)
		pos[i] = (int)floorf(fpos[i] / delta);
}
#endif

static float vard_scale(int D, const float p[D], float vard)
{
	float cen[D];
	for (int i = 0; i < D; i++)
		cen[i] = 0.5;

	return 1. + powf(dist(D, cen, p), 2.) * vard;
}

static bool distance_check(int D, int T, int N, float vard, const float delta[T][T], /*const*/ float points[N][D], const int kind[N], int a, int b)
{
	return dist(D, points[a], points[b]) > vard_scale(D, points[a], vard) * delta[kind[a]][kind[b]];
}


int (poissondisc_mc)(int D, int T, int N, int I, float vardens, const float delta[T][T], float points[N][D], int kind[N])
{
	PTR_ALLOC(char[N], active);

	assert((0 < I) && (I < N));
	assert(vardens >= 0.); // otherwise grid granularity needs to be changed

	memset(*active, 0, N * sizeof(char));
	memset(*active, 1, I * sizeof(char));

	int k = 30;
	int p = I;
	int a = I;

#ifdef  GRID
	float mindelta = 1.;
	float maxdelta = 0.;

	for (int i = 0; i < T; i++) {
		for (int j = 0; j < T; j++) {

			if (delta[i][j] < mindelta) 
				mindelta = delta[i][j];

			if (delta[i][j] > maxdelta) 
				maxdelta = delta[i][j];
		}
	}

	float corner[D];
	for (int i = 0; i < D; i++)
		corner[i] = 0.;

	maxdelta *= vard_scale(D, corner, vardens);
	mindelta /= sqrtf((float)D);

	long patchdims[D];

	for (int i = 0; i < D; i++)
		patchdims[i] = 3 * ceilf(maxdelta / mindelta);

	long patchstrs[D];
	md_calc_strides(D, patchstrs, patchdims, 1);

	int* patch = md_alloc(D, patchdims, sizeof(int));

	long griddims[D];

	for (int i = 0; i < D; i++)
		griddims[i] = ceilf(1. / mindelta);

	long gridstrs[D];
	md_calc_strides(D, gridstrs, griddims, 1);	// element size 1!

	int* grid = md_alloc(D, griddims, sizeof(int));
	int mone = -1;
	md_fill(D, griddims, grid, &mone, sizeof(int));

	for (int i = 0; i < I; i++) {

		long pos[D];
		grid_pos(D, pos, mindelta, points[i]);
		grid[md_calc_offset(D, gridstrs, pos)] = i;
	}
#endif

	while (a > 0) {

		// pick active point randomly

		int sel = (int)floor(a * uniform_rand()) % a;
		int s2 = 0;

		while (true) {

			while (!(*active)[s2])
				s2++;

			if (0 == sel)
				break;

			sel--;
			s2++;
		}

		assert((*active)[s2]);

		// try k times to place a new point near the selected point

		bool found = false;
		int rr = 0; // ?

		for (int i = 0; i < k; i++) {

			float d;
			float dd;
	
			// create a random point between one and two times the allowed distance

			do {
	
				kind[p] = rr++ % T;
				dd = delta[kind[s2]][kind[p]];

 				dd *= vard_scale(D, points[s2], vardens);

				for (int j = 0; j < D; j++) {

					do {
						points[p][j] = points[s2][j] + (uniform_rand() - 0.5) * 4. * dd;

					} while ((points[p][j] < 0.) || (points[p][j] > 1.));
				}

				d = dist(D, points[s2], points[p]);

			} while ((d < dd) || (d > 2. * dd));

			// check if the new point is far enough from all other points

			bool accept = true;
#ifdef GRID
			long pos[D];
			grid_pos(D, pos, mindelta, points[p]);
			long index = md_calc_offset(D, gridstrs, pos);
			assert(index < md_calc_size(D, griddims));

			if (-1 != grid[index]) {

				assert(!distance_check(D, T, N, vardens, delta, points, kind, p, grid[index]));
				accept = false;
			}

			if (accept) {

				long off[D];
				for (int i = 0; i < D; i++)
					off[i] = MIN(MAX(0, pos[i] - (patchdims[i] + 1) / 2), griddims[i] - patchdims[i]);

				md_copy_block(D, off, patchdims, patch, griddims, grid, sizeof(int));

				for (int i = 0; i < md_calc_size(D, patchdims); i++)
					if (-1 != patch[i])
						accept &= distance_check(D, T, N, vardens, delta, points, kind, p, patch[i]);
			}

#endif

#ifdef CHECK
			bool accept2 = true;

			for (int j = 0; j < p; j++)
				accept2 &= distance_check(D, T, N, vardens, delta, points, kind, p, j);

			assert(accept == accept2);
#endif

#ifndef GRID
			for (int j = 0; j < p; j++)
				accept &= distance_check(D, T, N, vardens, delta, points, kind, p, j);
#endif

			if (accept) {
			
				// add new point to active list
#ifdef GRID
				assert(-1 == grid[index]); // 0 is actually the first point
				grid[index] = p;
#endif
				(*active)[p] = 1;
				a++;
				p++;

				if (N == p)
					goto out;

				found = true;
				break;
			}
		}

		// if we can not place a new point, remove point from active list

		if (!found) {

			(*active)[s2] = 0;
			a--;
		} 
	}

out:
#ifdef  GRID
	md_free(grid);
	md_free(patch);
#endif
	XFREE(active);
	return p;
}




extern int poissondisc(int D, int N, int I, float vardens, float delta, float points[N][D])
{
	PTR_ALLOC(int[N], kind);
	memset(*kind, 0, I * sizeof(int));
	const float dd[1][1] = { { delta } };
	int P = poissondisc_mc(D, 1, N, I, vardens, dd, points, *kind);
	XFREE(kind);
	return P;
}



static void compute_rmatrix(int D, int T, float rmatrix[T][T], const float delta[T], int C, const int nc[T], const int mc[T][T])
{
	unsigned long processed = 0;
	float density = 0.;

	for (int i = 0; i < T; i++)
		rmatrix[i][i] = delta[i];

	for (int k = 0; k < C; k++) {
		
		for (int i = 0; i < nc[k]; i++) {

			int ind = mc[k][i];
			processed = MD_SET(processed, ind);
			density += 1. / powf(delta[ind], (float)D);
		//	printf("%d (%f)\t", ind, density);
		}
		//printf("\n");

		for (int i = 0; i < nc[k]; i++)
			for (int j = 0; j < T; j++)
				if (MD_IS_SET(processed, j) && (i != j))
					rmatrix[i][j] = rmatrix[j][i] = powf(density, -1. / (float)D);
	}
}

struct sort_label {

	int index;
	float x;
};

static int sort_cmp(const void* _a, const void* _b)
{
	const struct sort_label* a = _a;
	const struct sort_label* b = _b;
	return ((a->x < b->x) - (a->x > b->x)); // FIXME
}

extern void mc_poisson_rmatrix(int D, int T, float rmatrix[T][T], const float delta[T])
{
	assert(T <= 32);

	struct sort_label table[T];

	for (int i = 0; i < T; i++) {

		table[i].index = i;
		table[i].x = delta[i];
	}
	
	qsort(&table, T, sizeof(struct sort_label), sort_cmp);

	int mc[T][T];
	int nc[T];
	int ind = 0;
	int i;

	for (i = 0; (i < T) && (ind < T); i++) {
	
		float val = table[ind].x;
		int j = 0;

		while ((table[ind].x == val) && (ind < T))
			mc[i][j++] = table[ind++].index;

		nc[i] = j;
	}

	compute_rmatrix(D, T, rmatrix, delta, i, nc, (const int (*)[T])mc);
}



