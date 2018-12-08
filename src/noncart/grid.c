/* Copyright 2014-2015 The Regents of the University of California.
 * Copyright 2015-2018 Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2011, 2015, 2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Frank Ong <frankong@berkeley.edu>
 */

#include <math.h>
#include <complex.h>
#include <assert.h>
#include <string.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/specfun.h"

#include "misc/nested.h"
#include "misc/misc.h"

#include "grid.h"


#define KB_BETA		13.9086
// 13.8551
// 2x oversampling

#define KB_WIDTH	3



#ifndef KB128
static double kb(double beta, double x)
{
	if (fabs(x) >= 0.5)
		return 0.;

        return bessel_i0(beta * sqrt(1. - pow(2. * x, 2.))) / bessel_i0(beta);
}

static void kb_precompute(double beta, int n, float table[n + 1])
{
	for (int i = 0; i < n + 1; i++)
		table[i] = kb(beta, (double)(i) / (double)(n - 1) / 2.);
}
#endif



static double I0_beta(double beta)
{
#ifndef KB128
	return bessel_i0(beta);
#else
	assert(KB_BETA == beta);
	return 118509.158946;
#endif
}

const float kb_table128[129] = {
	1.0000000000000000, 0.9995847139398653, 0.9983398161018390, 0.9962681840754728, 
	0.9933746024007669, 0.9896657454602714, 0.9851501536396374, 0.9798382028675283, 
	0.9737420676763386, 0.9668756779551011, 0.9592546695947362, 0.9508963292536357, 
	0.9418195334980564, 0.9320446825968820, 0.9215936292739139, 0.9104896027426631, 
	0.8987571283688524, 0.8864219433239096, 0.8735109086091393, 0.8600519178443380, 
	0.8460738032267725, 0.8316062390762707, 0.8166796433899514, 0.8013250778354499, 
	0.7855741466147803, 0.7694588946318385, 0.7530117053952898, 0.7362651990850234, 
	0.7192521312046796, 0.7020052922349470, 0.6845574086924412, 0.6669410459871208, 
	0.6491885134575039, 0.6313317719473744, 0.6134023442704702, 0.5954312288909031, 
	0.5774488171268115, 0.5594848141632452, 0.5415681641375969, 0.5237269795371914, 
	0.5059884751240438, 0.4883789065765270, 0.4709235140117633, 0.4536464705263341, 
	0.4365708358662894, 0.4197185153108639, 0.4031102238276424, 0.3867654555305848, 
	0.3707024584462233, 0.3549382145678427, 0.3394884251524746, 0.3243675011914082, 
	0.3095885589616078, 0.2951634205431575, 0.2811026191666483, 0.2674154092344597, 
	0.2541097808412017, 0.2411924786012657, 0.2286690245755740, 0.2165437450752543, 
	0.2048198011071496, 0.1934992222148726, 0.1825829434594916, 0.1720708452759937, 
	0.1619617959353290, 0.1522536963371723, 0.1429435268554672, 0.1340273959573592, 
	0.1255005903162311, 0.1173576261411828, 0.1095923014483913, 0.1021977490043101, 
	0.0951664896765205, 0.0884904859351842, 0.0821611952563688, 0.0761696231879639, 
	0.0705063758493464, 0.0651617116473479, 0.0601255920032731, 0.0553877308986640, 
	0.0509376430610617, 0.0467646906251106, 0.0428581281188566, 0.0392071456399203, 
	0.0358009101012748, 0.0326286044415178, 0.0296794647097185, 0.0269428149500251, 
	0.0244080998261697, 0.0220649149406994, 0.0199030348181235, 0.0179124385351197, 
	0.0160833329944038, 0.0144061738517878, 0.0128716841182598, 0.0114708704705587, 
	0.0101950373146533, 0.0090357986567118, 0.0079850878455423, 0.0070351652590612, 
	0.0061786240150858, 0.0054083937936397, 0.0047177428639869, 0.0041002784147792, 
	0.0035499452900106, 0.0030610232369345, 0.0026281227747268, 0.0022461797944939, 
	0.0019104490022500, 0.0016164963167613, 0.0013601903336964, 0.0011376929663821, 
	0.0009454493716780, 0.0007801772670918, 0.0006388557423128, 0.0005187136648862, 
	0.0004172177758400, 0.0003320605667526, 0.0002611480250751, 0.0002025873295356, 
	0.0001546745722200, 0.0001158825784783, 0.0000848488902175, 0.0000603639724392, 
	0.0000413596971257, 0.0000268981528101, 0.0000161608224276, 0.0000000000000000, 
	0.0000000000000000, 
};

static double ftkb(double beta, double x)
{
	double a = sqrt(pow(beta, 2.) - pow(M_PI * x, 2.));
	return ((0. == a) ? 1. : (sinh(a) / a)) / I0_beta(beta);
}

static float rolloff(float x, double beta, float width)
{
	return (float)ftkb(beta, x * width) / ftkb(beta, 0.);
}


// Linear interpolation
static float lerp(float a, float b, float c)
{
	return (1. - c) * a + c * b;
}

// Linear interpolation look up
static float intlookup(int n, const float table[n + 1], float x)
{
	float fpart;

//	fpart = modff(x * n, &ipart);
//	int index = ipart;

	int index = (int)(x * (n - 1));
	fpart = x * (n - 1) - (float)index;
#if 1
	assert(index >= 0);
	assert(index <= n);
	assert(fpart >= 0.);
	assert(fpart <= 1.);
#endif
	float l = lerp(table[index], table[index + 1], fpart);
#if 1
	assert(l <= 1.);
	assert(0 >= 0.);
#endif
	return l;
}



void gridH(const struct grid_conf_s* conf, const complex float* traj, const long ksp_dims[4], complex float* dst, const long grid_dims[4], const complex float* grid)
{
	long C = ksp_dims[3];
#ifndef KB128
	// precompute kaiser bessel table
	int kb_size = 500;
	float kb_table[kb_size + 1];
	kb_precompute(conf->beta, kb_size, kb_table);
#else
	assert(KB_BETA == beta);
	int kb_size = 128;
	const float* kb_table = kb_table128;
#endif
	assert(1 == ksp_dims[0]);
	long samples = ksp_dims[1] * ksp_dims[2];

#pragma omp parallel for
	for(int i = 0; i < samples; i++) {

		float pos[3];
		pos[0] = conf->os * (creal(traj[i * 3 + 0]));
		pos[1] = conf->os * (creal(traj[i * 3 + 1]));
		pos[2] = conf->os * (creal(traj[i * 3 + 2]));

		pos[0] += (grid_dims[0] > 1) ? ((float)grid_dims[0] / 2.) : 0.;
		pos[1] += (grid_dims[1] > 1) ? ((float)grid_dims[1] / 2.) : 0.;
		pos[2] += (grid_dims[2] > 1) ? ((float)grid_dims[2] / 2.) : 0.;

		complex float val[C];
		for (int j = 0; j < C; j++)
			val[j] = 0.0;
		
		grid_pointH(C, 3, grid_dims, pos, val, grid, conf->periodic, conf->width, kb_size, kb_table);

		for (int j = 0; j < C; j++)
			dst[j * samples + i] += val[j];
	}
}


void grid(const struct grid_conf_s* conf, const complex float* traj, const long grid_dims[4], complex float* grid, const long ksp_dims[4], const complex float* src)
{
	long C = ksp_dims[3];

#ifndef	KB128
	// precompute kaiser bessel table
	int kb_size = 500;
	float kb_table[kb_size + 1];
	kb_precompute(conf->beta, kb_size, kb_table);
#else
	assert(KB_BETA == beta);
	int kb_size = 128;
	const float* kb_table = kb_table128;
#endif
	assert(1 == ksp_dims[0]);
	long samples = ksp_dims[1] * ksp_dims[2];

	// grid
#pragma omp parallel for
	for(int i = 0; i < samples; i++) {

		float pos[3];
		pos[0] = conf->os * (creal(traj[i * 3 + 0]));
		pos[1] = conf->os * (creal(traj[i * 3 + 1]));
		pos[2] = conf->os * (creal(traj[i * 3 + 2]));

		pos[0] += (grid_dims[0] > 1) ? ((float) grid_dims[0] / 2.) : 0.;
		pos[1] += (grid_dims[1] > 1) ? ((float) grid_dims[1] / 2.) : 0.;
		pos[2] += (grid_dims[2] > 1) ? ((float) grid_dims[2] / 2.) : 0.;

		complex float val[C];
		
		for (int j = 0; j < C; j++)
			val[j] = src[j * samples + i];

		grid_point(C, 3, grid_dims, pos, grid, val, conf->periodic, conf->width, kb_size, kb_table);
	}
}


static void grid2_dims(unsigned int D, const long trj_dims[D], const long ksp_dims[D], const long grid_dims[D])
{
	assert(D >= 4);
	assert(md_check_compat(D - 3, ~0, grid_dims + 3, ksp_dims + 3));
//	assert(md_check_compat(D - 3, ~(MD_BIT(0) | MD_BIT(1)), trj_dims + 3, ksp_dims + 3));
	assert(md_check_bounds(D - 3, ~0, trj_dims + 3, ksp_dims + 3));

	assert(3 == trj_dims[0]);
	assert(1 == trj_dims[3]);
	assert(1 == ksp_dims[0]);
}


void grid2(const struct grid_conf_s* conf, unsigned int D, const long trj_dims[D], const complex float* traj, const long grid_dims[D], complex float* dst, const long ksp_dims[D], const complex float* src)
{
	grid2_dims(D, trj_dims, ksp_dims, grid_dims);

	long ksp_strs[D];
	md_calc_strides(D, ksp_strs, ksp_dims, CFL_SIZE);

	long trj_strs[D];
	md_calc_strides(D, trj_strs, trj_dims, CFL_SIZE);

	long grid_strs[D];
	md_calc_strides(D, grid_strs, grid_dims, CFL_SIZE);


	long pos[D];
	for (unsigned int i = 0; i < D; i++)
		pos[i] = 0;

	do {

		grid(conf, &MD_ACCESS(D, trj_strs, pos, traj),
			grid_dims, &MD_ACCESS(D, grid_strs, pos, dst),
			ksp_dims, &MD_ACCESS(D, ksp_strs, pos, src));

	} while(md_next(D, ksp_dims, (~0 ^ 15), pos));
}


void grid2H(const struct grid_conf_s* conf, unsigned int D, const long trj_dims[D], const complex float* traj, const long ksp_dims[D], complex float* dst, const long grid_dims[D], const complex float* src)
{
	grid2_dims(D, trj_dims, ksp_dims, grid_dims);

	long ksp_strs[D];
	md_calc_strides(D, ksp_strs, ksp_dims, CFL_SIZE);

	long trj_strs[D];
	md_calc_strides(D, trj_strs, trj_dims, CFL_SIZE);

	long grid_strs[D];
	md_calc_strides(D, grid_strs, grid_dims, CFL_SIZE);

	long pos[D];
	for (unsigned int i = 0; i < D; i++)
		pos[i] = 0;

	do {
		gridH(conf, &MD_ACCESS(D, trj_strs, pos, traj),
			ksp_dims, &MD_ACCESS(D, ksp_strs, pos, dst),
			grid_dims, &MD_ACCESS(D, grid_strs, pos, src));

	} while(md_next(D, ksp_dims, (~0 ^ 15), pos));
}


typedef void CLOSURE_TYPE(grid_update_t)(int ind, float d);

#ifndef __clang__
#define VLA(x) x
#else
// blocks extension does not play well even with arguments which
// just look like variably-modified types
#define VLA(x)
#endif

static void grid_point_gen(int N, const long dims[VLA(N)], const float pos[VLA(N)], bool periodic, float width, int kb_size, const float kb_table[VLA(kb_size + 1)], grid_update_t update)
{
#ifndef __clang__
	int sti[N];
	int eni[N];
	int off[N];
#else
	// blocks extension does not play well with variably-modified types
	int* sti = alloca(sizeof(int[N]));
	int* eni = alloca(sizeof(int[N]));
	int* off = alloca(sizeof(int[N]));
#endif
	for (int j = 0; j < N; j++) {

		sti[j] = (int)ceil(pos[j] - width);
		eni[j] = (int)floor(pos[j] + width);
		off[j] = 0;

		if (sti[j] > eni[j])
			return;

		if (!periodic) {

			sti[j] = MAX(sti[j], 0);
			eni[j] = MIN(eni[j], dims[j] - 1);

		} else {

			while (sti[j] + off[j] < 0)
				off[j] += dims[j];
		}

		if (1 == dims[j]) {

			assert(0. == pos[j]); // ==0. fails nondeterministically for test_nufft_forward bbdec08cb
			sti[j] = 0;
			eni[j] = 0;
		}
	}

	__block NESTED(void, grid_point_r, (int N, int ind, float d))	// __block for recursion
	{
		if (0 == N) {

			update(ind, d);

		} else {

			N--;

			for (int w = sti[N]; w <= eni[N]; w++) {

				float frac = fabs(((float)w - pos[N]));
				float d2 = d * intlookup(kb_size, kb_table, frac / width);
				int ind2 = (ind * dims[N] + ((w + off[N]) % dims[N]));

				grid_point_r(N, ind2, d2);
			}
		}
	};

	grid_point_r(N, 0, 1.);
}



void grid_point(unsigned int ch, int N, const long dims[VLA(N)], const float pos[VLA(N)], complex float* dst, const complex float val[VLA(ch)], bool periodic, float width, int kb_size, const float kb_table[kb_size + 1])
{
	NESTED(void, update, (int ind, float d))
	{
		for (unsigned int c = 0; c < ch; c++) {

			// we are allowed to update real and imaginary part independently which works atomically
			#pragma omp atomic
			__real(dst[ind + c * dims[0] * dims[1] * dims[2]]) += __real(val[c]) * d;
			#pragma omp atomic
			__imag(dst[ind + c * dims[0] * dims[1] * dims[2]]) += __imag(val[c]) * d;
		}
	};

	grid_point_gen(N, dims, pos, periodic, width, kb_size, kb_table, update);
}



void grid_pointH(unsigned int ch, int N, const long dims[VLA(N)], const float pos[VLA(N)], complex float val[VLA(ch)], const complex float* src, bool periodic, float width, int kb_size, const float kb_table[kb_size + 1])
{
	NESTED(void, update, (int ind, float d))
	{
		for (unsigned int c = 0; c < ch; c++) {

			// we are allowed to update real and imaginary part independently which works atomically
			#pragma omp atomic
			__real(val[c]) += __real(src[ind + c * dims[0] * dims[1] * dims[2]]) * d;
			#pragma omp atomic
			__imag(val[c]) += __imag(src[ind + c * dims[0] * dims[1] * dims[2]]) * d;
		}
	};

	grid_point_gen(N, dims, pos, periodic, width, kb_size, kb_table, update);
}



double calc_beta(float os, float width)
{
	return M_PI * sqrt(pow((width * 2. / os) * (os - 0.5), 2.) - 0.8);
}


static float pos(int d, int i)
{
	return (1 == d) ? 0. : (((float)i - (float)d / 2.) / (float)d);
}

void rolloff_correction(float os, float width, float beta, const long dimensions[3], complex float* dst)
{
	UNUSED(os);

#pragma omp parallel for collapse(3)
	for (int z = 0; z < dimensions[2]; z++) 
		for (int y = 0; y < dimensions[1]; y++) 
			for (int x = 0; x < dimensions[0]; x++)
				dst[x + dimensions[0] * (y + z * dimensions[1])] 
					= 1. / (  rolloff(pos(dimensions[0], x), beta, width)
						* rolloff(pos(dimensions[1], y), beta, width) 
						* rolloff(pos(dimensions[2], z), beta, width) );
}




