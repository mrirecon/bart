/* Copyright 2014 The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Martin Uecker, 2011-12-10
 * uecker@eecs.berkeley.edu
 * Frank Ong, 2014
 * frankong@berkeley.edu
 */

#include <math.h>
#include <complex.h>
#include <assert.h>
#include <string.h>


#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/misc.h"

#include "grid.h"

#define KB_BETA		13.9086
// 13.8551
// 2x oversampling

#define KB_WIDTH	3

#include <gsl/gsl_specfunc.h>


static double kb(double beta, double x);
static void kb_precompute(double beta, int n, float table[n+1] );
static double I0_beta(double beta);

static double kb(double beta, double x)
{
	if (fabs(x) >= 0.5)
		return 0.;

        return gsl_sf_bessel_I0(beta * sqrt(1. - pow(2. * x, 2.))) / gsl_sf_bessel_I0(beta);
}

static void kb_precompute(double beta, int n, float table[n + 1])
{
	for (int i = 0; i < n + 1; i++)
		table[i] = kb(beta, (double)(i) / (double)(n - 1) / 2.);
}


static double I0_beta(double beta)
{
	return gsl_sf_bessel_I0(beta);
}

float kb_table128[129] = {

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

	assert(index >= 0);
	assert(index <= n);
	assert(fpart >= 0.);
	assert(fpart <= 1.);

	float l = lerp(table[index], table[index + 1], fpart);


	assert(l <= 1.);
	assert(0 >= 0.);


	return l;
}



void gridH(float os, float width, double beta, float scale, const float shifts[3], const complex float* traj, const complex float* weights, const long ksp_dims[DIMS], complex float* dst, const long grid_dims[DIMS], const complex float* grid)
{
	long C = ksp_dims[COIL_DIM];

	// precompute kaiser bessel table
	int kb_size = 500;
	float kb_table[kb_size];
	kb_precompute(beta, kb_size, kb_table);

#pragma omp parallel for
	for(int i = 0; i < ksp_dims[1]; i++)
	{
		float pos[3];
		pos[0] = os * (creal(traj[i * 3 + 0]));
		pos[1] = os * (creal(traj[i * 3 + 1]));
		pos[2] = os * (creal(traj[i * 3 + 2]));

		pos[0] += (grid_dims[0] > 1) ? ((float) grid_dims[0] / 2.) : 0.;
		pos[1] += (grid_dims[1] > 1) ? ((float) grid_dims[1] / 2.) : 0.;
		pos[2] += (grid_dims[2] > 1) ? ((float) grid_dims[2] / 2.) : 0.;

		pos[0] += shifts[0];
		pos[1] += shifts[1];
		pos[2] += shifts[2];

		complex float val[C];
		for (int j = 0; j < C; j++)
			val[j] = 0.0;
		
		grid_pointH(C, grid_dims, pos, val, grid, width, kb_size, kb_table);

		for (int j = 0; j < C; j++)
		{
			if (NULL == weights)
				dst[j * ksp_dims[1] + i] += scale * val[j];
			else 
				dst[j * ksp_dims[1] + i] += scale * val[j] * weights[i];
		}
	}
}


void grid( float os, float width, double beta, float scale, const float shifts[3], const complex float* traj, const complex float* weights, const long grid_dims[DIMS], complex float* grid, const long ksp_dims[DIMS], const complex float* src)
{
	long C = ksp_dims[COIL_DIM];

	// precompute kaiser bessel table
	int kb_size = 500;
	float kb_table[kb_size];
	kb_precompute(beta, kb_size, kb_table);

	// grid
#pragma omp parallel for
	for(int i = 0; i < ksp_dims[1]; i++)
	{
		float pos[3];
		pos[0] = os * (creal(traj[i * 3 + 0]));
		pos[1] = os * (creal(traj[i * 3 + 1]));
		pos[2] = os * (creal(traj[i * 3 + 2]));

		pos[0] += (grid_dims[0] > 1) ? ((float) grid_dims[0] / 2.) : 0.;
		pos[1] += (grid_dims[1] > 1) ? ((float) grid_dims[1] / 2.) : 0.;
		pos[2] += (grid_dims[2] > 1) ? ((float) grid_dims[2] / 2.) : 0.;

		pos[0] += shifts[0];
		pos[1] += shifts[1];
		pos[2] += shifts[2];

		complex float val[C];
		
		for (int j = 0; j < C; j++)
		{
			if (NULL == weights)
				val[j] = src[j * ksp_dims[1] + i] * scale;
			else
				val[j] = src[j * ksp_dims[1] + i] * scale * weights[i];

		}

		grid_point(C, grid_dims, pos, grid, val, width, kb_size, kb_table);
	}
}




void grid_point(unsigned int ch, const long dims[3], const float pos[3], complex float* dst, const complex float val[ch], float width, int kb_size, float kb_table[kb_size+1])
{
	int sti[3];
	int eni[3];

	for (int j = 0; j < 3; j++) {

		int st = MAX((int)ceil(pos[j] - width), 0);
		int en = MIN((int)floor(pos[j] + width), dims[j] - 1);

		if (st > en)
			return;

		sti[j] = st;
		eni[j] = en;
	}

	for (int w = sti[2]; w <= eni[2]; w++) {

		float frac = fabs(((float)w - pos[2]));
		float dw = 1. * intlookup(kb_size, kb_table, frac / width);
		int indw = w * dims[1];

	for (int v = sti[1]; v <= eni[1]; v++) {

		float frac = fabs(((float)v - pos[1]));
		float dv = dw * intlookup(kb_size, kb_table, frac / width);
		int indv = (indw + v) * dims[0];

	for (int u = sti[0]; u <= eni[0]; u++) {

		float frac = fabs(((float)u - pos[0]));
		float du = dv * intlookup(kb_size, kb_table, frac / width);
		int indu = (indv + u);

	for (unsigned int c = 0; c < ch; c++) {

		// we are allowed to update real and imaginary part independently which works atomically
		#pragma omp atomic
		__real(dst[indu + c * dims[0] * dims[1] * dims[2]]) += __real(val[c]) * du;
		#pragma omp atomic
		__imag(dst[indu + c * dims[0] * dims[1] * dims[2]]) += __imag(val[c]) * du;
	}}}}
}



void grid_pointH(unsigned int ch, const long dims[3], const float pos[3], complex float val[ch], const complex float* src, float width, int kb_size, float kb_table[kb_size])
{
	int sti[3];
	int eni[3];

	for (int j = 0; j < 3; j++) {

		sti[j] = MAX((int)ceil(pos[j] - width), 0);
		eni[j] = MIN((int)floor(pos[j] + width), dims[j] - 1);
	}

	for (unsigned int i = 0; i < ch; i++)
		val[i] = 0.;


        //printf("%f %f %f: %d %d %d -- %d %d %d\n", pos[0], pos[1], pos[2], sti[0], sti[1], sti[2], eni[0], eni[1], eni[2]);

	for (int w = sti[2]; w <= eni[2]; w++) {

		float frac = fabs(((float)w - pos[2]));
		float dw = 1. * intlookup(kb_size, kb_table, frac / width);
		int indw = w * dims[1];

	for (int v = sti[1]; v <= eni[1]; v++) {

		float frac = fabs(((float)v - pos[1]));
		float dv = dw * intlookup(kb_size, kb_table, frac / width);
		int indv = (indw + v) * dims[0];

	for (int u = sti[0]; u <= eni[0]; u++) {

		float frac = fabs(((float)u - pos[0]));
		float du = dv * intlookup(kb_size, kb_table, frac / width);
		int indu = (indv + u);

	for (unsigned int c = 0; c < ch; c++) {

		// we are allowed to update real and imaginary part independently which works atomically
		#pragma omp atomic
		__real(val[c]) += __real(src[indu + c * dims[0] * dims[1] * dims[2]]) * du;
		#pragma omp atomic
		__imag(val[c]) += __imag(src[indu + c * dims[0] * dims[1] * dims[2]]) * du;
	}}}}
}


void grid_line3d(unsigned int n, unsigned int ch, const long dims[3], const float start[3], const float end[3], complex float* dst, const complex float* src, float width, int kb_size, float kb_table[kb_size+1])
{
	#pragma omp parallel for
	for (unsigned int i = 0; i < n; i++) {
		
		float pos[3];

		for (int j = 0; j < 3; j++)
			pos[j] = lerp(start[j], end[j], (float)i / (float)(n - 1));
		
		complex float val[ch];

		for (unsigned int c = 0; c < ch; c++)
			 val[c] = src[i + c * n];
		
//		printf("%f %f %f: %d %d %d -- %d %d %d\n", pos[0], pos[1], pos[2], sti[0], sti[1], sti[2], eni[0], eni[1], eni[2]);

		grid_point(ch, dims, pos, dst, val, width, kb_size, kb_table);
	}		
}




void density_compensation(unsigned int samples, unsigned int channels, unsigned int spokes, complex float* dst, const complex float* src)
{
	for (unsigned int i = 0; i < spokes; i++)
		for (unsigned int k = 0; k < channels; k++)
			for (unsigned int j = 0; j < samples; j++)
					dst[j + (i * channels + k) * samples] = src[j + (i * channels + k) * samples] * (0. + fabs((double)(j - samples / 2 + 0.5)));
}

void density_comp3d(const long dim[3], complex float* kspace)
{
	for (int c = 0; c < dim[3]; c++) 
	for (int i = 0; i < dim[2]; i++)
		for (int j = 0; j < dim[1]; j++)
			for (int k = 0; k < dim[0]; k++) {

				float dist = sqrtf(powf((float)(k - dim[0] / 2), 2.) + powf((float)(j - dim[1] / 2), 2.) + powf((float)(i - dim[2] / 2), 2.));
				kspace[k + dim[0] * (j + dim[1] * (i + c * dim[2]))] *= powf(dist + 0.5, 2.);
			}
}


static float pos(int d, int i)
{
	return (1 == d) ? 0. : (((float)i - (float)d / 2.) / (float)d);
}

void rolloff_correction(float os, float width, const long dimensions[3], complex float* dst)
{
	double beta = M_PI * sqrt( pow( (width * 2. / os ) * (os - 0.5 ), 2. ) - 0.8 );

#pragma omp parallel for
	for (int z = 0; z < dimensions[2]; z++) 
		for (int y = 0; y < dimensions[1]; y++) 
			for (int x = 0; x < dimensions[0]; x++)
				dst[x + dimensions[0] * (y + z * dimensions[1])] 
					= 1. / (  rolloff(pos(dimensions[0], x), beta, width)
						* rolloff(pos(dimensions[1], y), beta, width) 
						* rolloff(pos(dimensions[2], z), beta, width) );
}




void grid_radial(const long dimensions[3], unsigned int samples, unsigned int channels, unsigned int spokes, unsigned int sp, complex float* dst, const complex float* src)
{

	float st[3];
	float en[3];

	float center[3] = { dimensions[0] / 2., dimensions[1] / 2., dimensions[2] / 2. };

	if (1 == dimensions[2])
		center[2] = 0.;


	double x = cos(2. * M_PI * (double)sp / (double)spokes);
	double y = sin(2. * M_PI * (double)sp / (double)spokes);

	int span = MIN(dimensions[0], dimensions[1]);

	st[0] = center[0] - (span - 1) / 2. * x;
	st[1] = center[1] - (span - 1) / 2. * y;
	st[2] = center[2];

	en[0] = center[0] + (span - 1) / 2. * x;
	en[1] = center[1] + (span - 1) / 2. * y;
	en[2] = center[2];

#if 0
		double halfpixelx = (en[0] - st[0]) / (samples - 1) / 2.;
		double halfpixely = (en[1] - st[1]) / (samples - 1) / 2.;

		st[0] += halfpixelx;
		st[1] += halfpixely;

		en[0] += halfpixelx;
		en[1] += halfpixely;
#endif

		grid_line3d(samples, channels, dimensions, st, en, dst, src, KB_WIDTH, 128, kb_table128); 
}





	
void grid_line(const long dimensions[3], unsigned int samples, unsigned int channels, unsigned int phencs, unsigned int pe, complex float* dst, const complex float* src)
{

	float st[3];
	float en[3];

	st[0] = 0.;
	st[1] = (float)pe  / (float)phencs * (float)dimensions[1] ;
	st[2] = dimensions[2] / 2.;

	en[0] = ((float)samples - 1.) / (float)samples * (float)dimensions[0];
	en[1] = (float)pe / (float)phencs * (float)dimensions[1] ;
	en[2] = dimensions[2] / 2.;

#if 0
		double halfpixelx = (en[0] - st[0]) / (samples - 1) / 2.;
		double halfpixely = (en[1] - st[1]) / (samples - 1) / 2.;

		st[0] += halfpixelx;
		st[1] += halfpixely;

		en[0] += halfpixelx;
		en[1] += halfpixely;
#endif

		grid_line3d(samples, channels, dimensions, st, en, dst, src, KB_WIDTH, 128, kb_table128); 
//	grid_line3d(samples, channels, dimensions, st, en, dst, src, 1.); 
}

