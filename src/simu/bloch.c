/* Copyright 2022-2023. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include <assert.h>
#include <math.h>

#include "misc/misc.h"

#include "num/vec3.h"
#include "num/linalg.h"
#include "num/matexp.h"

#include "bloch.h"


// Rotations in RIGHT-handed coordinate system with CLOCKWISE rotation for angle > 0
// Keep it consistent with clockwise rotation of Bloch equations
//       z
//       |
//       |
//       |
//       |_ _ _ _ _ _ _y
//      /
//     /
//    /
//   x

void rotx(float out[3], const float in[3], float angle)
{
	out[0] = in[0];
	out[1] = in[1] * cosf(angle) + in[2] * sinf(angle);
	out[2] = -in[1] * sinf(angle) + in[2] * cosf(angle);
}

void roty(float out[3], const float in[3], float angle)
{
	out[0] = in[0] * cosf(angle) - in[2] * sinf(angle);
	out[1] = in[1];
	out[2] = in[0] * sinf(angle) + in[2] * cosf(angle);
}

void rotz(float out[3], const float in[3], float angle)
{
	out[0] = in[0] * cosf(angle) + in[1] * sinf(angle);
	out[1] = -in[0] * sinf(angle) + in[1] * cosf(angle);
	out[2] = in[2];
}

// RIGHT-handed coordinate system with CLOCKWISE rotation
// dM/dt = MxB - ...
void bloch_ode(float out[3], const float in[3], float r1, float r2, const float gb[3])
{
	float m0 = 1.;
	vec3_rot(out, in, gb);
	out[0] -= in[0] * r2;
	out[1] -= in[1] * r2;
	out[2] -= (in[2] - m0) * r1;
}

void bloch_pdy(float out[3][3], const float in[3], float r1, float r2, const float gb[3])
{
	(void)in;

	vec3_rot(out[0], (float[3]){ 1., 0., 0. }, gb);
	out[0][0] -= r2;

	vec3_rot(out[1], (float[3]){ 0., 1., 0. }, gb);
	out[1][1] -= r2;

	vec3_rot(out[2], (float[3]){ 0., 0., 1. }, gb);
	out[2][2] -= r1;
}

void bloch_pdp(float out[2][3], const float in[3], float r1, float r2, const float gb[3])
{
	(void)r1; (void)r2; (void)gb;

	float m0 = 1.;
	out[0][0] = 0.;
	out[0][1] = 0.;
	out[0][2] = -(in[2] - m0);
	out[1][0] = -in[0];
	out[1][1] = -in[1];
	out[1][2] = 0.;
}

void bloch_b1_pdp(float out[3][3], const float in[3], float r1, float r2, const float gb[3], float phase, float b1)
{
	(void)r1; (void)r2; (void)gb;

	float m0 = 1.;
	out[0][0] = 0.;
	out[0][1] = 0.;
	out[0][2] = -(in[2] - m0);
	out[1][0] = -in[0];
	out[1][1] = -in[1];
	out[1][2] = 0.;
	out[2][0] = sinf(phase) * in[2] * b1;
	out[2][1] = cosf(phase) * in[2] * b1;
	out[2][2] = (-sinf(phase) * in[0] - cosf(phase) * in[1]) * b1;
}


void bloch_relaxation(float out[3], float t, const float in[3], float r1, float r2, const float gb[3])
{
	float m0 = 1.;
	assert((0. == gb[0]) && (0. == gb[1])); // no B1(t)

        rotz(out, in, gb[2] * t);

	out[0] *= expf(-t * r2);
	out[1] *= expf(-t * r2);
	out[2] += (m0 - in[2]) * (1. - expf(-t * r1));
}


void bloch_excitation(float out[3], float t, const float in[3], float r1, float r2, const float gb[3])
{
	(void)r1; (void)r2;
	assert(0. == gb[2]); // no gradient, rotating frame

	rotx(out, in, gb[0] * t);
}

// Rotation effects of RF fields only
void bloch_excitation2(float out[3], const float in[3], float angle, float phase)
{
	float tmp[3] = { 0. };
	float tmp2[3] = { 0. };

	rotz(tmp, in, -phase);
	rotx(tmp2, tmp, angle);
	rotz(out, tmp2, phase);
}



void bloch_matrix_ode(float matrix[4][4], float r1, float r2, const float gb[3])
{
	float m0 = 1.;
	float m[4][4] = {
		{	-r2,		gb[2],		-gb[1],		0.	},
		{	-gb[2],		-r2,		gb[0],		0.	},
		{	gb[1],		-gb[0],		-r1,		m0 * r1 },
		{	0.,		0.,		0.,		0.	},
	};

	matf_copy(4, 4, matrix, m);
}


void bloch_matrix_int(float matrix[4][4], float t, float r1, float r2, const float gb[3])
{
	float blm[4][4];
	bloch_matrix_ode(blm, r1, r2, gb);

	mat_exp(4, t, matrix, blm);
}

void bloch_matrix_ode_sa(float matrix[10][10], float r1, float r2, const float gb[3])
{
	float m0 = 1.;
	float m[10][10] = {
		{	-r2,	gb[2],	-gb[1],	0.,	0.,	0.,	0.,	0.,	0.,	0.	},
		{	-gb[2],	-r2,	gb[0],	0.,	0.,	0.,	0.,	0.,	0.,	0.	},
		{	gb[1],	-gb[0],	-r1,	0.,	0.,	0.,	0.,	0.,	0.,	m0 * r1 },
		{	0.,	0.,	0.,	-r2,	gb[2],	-gb[1],	0.,	0.,	0.,	0.	},
		{	0.,	0.,	0.,	-gb[2],	-r2,	gb[0],	0.,	0.,	0.,	0.	},
		{	0.,	0.,	-1.,	gb[1],	-gb[0],	-r1,	0.,	0.,	0.,	m0	},
		{	-1.,	0.,	0.,	0.,	0.,	0.,	-r2,	gb[2],	-gb[1],	0.	},
		{	0.,	-1.,	0.,	0.,	0.,	0.,	-gb[2],	-r2,	gb[0],	0.	},
		{	0.,	0.,	0.,	0.,	0.,	0.,	gb[1],	-gb[0],	-r1,	0.	},
		{	0.,	0.,	0.,	0.,	0.,	0.,	0.,	0.,	0.,	0.	},
	};

	matf_copy(10, 10, matrix, m);
}

void bloch_matrix_int_sa(float matrix[10][10], float t, float r1, float r2, const float gb[3])
{
	float blm[10][10];
	bloch_matrix_ode_sa(blm, r1, r2, gb);

	mat_exp(10, t, matrix, blm);
}

void bloch_matrix_ode_sa2(float matrix[13][13], float r1, float r2, const float gb[3], float phase, float b1)
{
	float m0 = 1.;
	float m[13][13] = {
		{	-r2,		gb[2],		-gb[1],		0.,	0.,	0.,	0.,	0.,	0.,	0.,	0.,	0.,	0.	},
		{	-gb[2],		-r2,		gb[0],		0.,	0.,	0.,	0.,	0.,	0.,	0.,	0.,	0.,	0.	},
		{	gb[1],		-gb[0],		-r1,		0.,	0.,	0.,	0.,	0.,	0.,	0.,	0.,	0.,	m0 * r1 },
		{	0.,		0.,		0.,		-r2,	gb[2],	-gb[1],	0.,	0.,	0.,	0.,	0.,	0.,	0.	},
		{	0.,		0.,		0.,		-gb[2],	-r2,	gb[0],	0.,	0.,	0.,	0.,	0.,	0.,	0.	},
		{	0.,		0.,		-1.,		gb[1],	-gb[0],	-r1,	0.,	0.,	0.,	0.,	0.,	0.,	m0	},
		{	-1.,		0.,		0.,		0.,	0.,	0.,	-r2,	gb[2],	-gb[1],	0.,	0.,	0.,	0.	},
		{	0.,		-1.,		0.,		0.,	0.,	0.,	-gb[2],	-r2,	gb[0],	0.,	0.,	0.,	0.	},
		{	0.,		0.,		0.,		0.,	0.,	0.,	gb[1],	-gb[0],	-r1,	0.,	0.,	0.,	0.	},
		{	0.,		0.,		sinf(phase) *b1,0.,	0.,	0.,	0.,	0.,	0.,	-r2,	gb[2],	-gb[1],	0.	},
		{	0.,		0.,		cosf(phase)*b1,	0.,	0.,	0.,	0.,	0.,	0.,	-gb[2],	-r2,	gb[0],	0.	},
		{	-sinf(phase)*b1,	-cosf(phase)*b1,0.,		0.,	0.,	0.,	0.,	0.,	0.,	gb[1],	-gb[0],	-r1,	0.	},
		{	0.,		0.,		0.,		0.,	0.,	0.,	0.,	0.,	0.,	0.,	0.,	0.,	0.	},
	};

	matf_copy(13, 13, matrix, m);
}

void bloch_matrix_int_sa2(float matrix[13][13], float t, float r1, float r2, const float gb[3], float phase, float b1)
{
	float blm[13][13];
	bloch_matrix_ode_sa2(blm, r1, r2, gb, phase, b1);

	mat_exp(13, t, matrix, blm);
}



void bloch_mcconnel_matrix_ode(int P, float matrix[1 + P * 3][1 + P * 3], const float r1[P], const float r2[P], const float k[P - 1], const float m0[P], const float Om[P], const float gb[3])
{
	// +1 for T1 relaxation term in homogeneous ODE representation
	int N = 1 + P * 3;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			matrix[i][j] = 0.;

	// copy 3 x 3 Bloch matrix

	for (int p = 0; p < P; p++) {

		float g[3];
		for (int i = 0; i < 3; i++)
			g[i] = gb[i];

		g[2] += Om[p];

		float m[4][4];
		bloch_matrix_ode(m, r1[p], r2[p], g);

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				matrix[3 * p + i][3 * p + j] = m[i][j];
	}

	// equilibrium

	for (int p = 0; p < P; p++)
		matrix[3 * p + 2][N - 1] = m0[p] * r1[p];

	// exchange  FIX ME?: Additional pools only exchange with water pool
	for (int p = 0; p < P - 1; p++)
		for (int i = 0; i < 3; i++)
			matrix[i][i] -= k[p] * m0[p + 1];

	for (int p = 1; p < P; p++)
		for (int i = 0; i < 3; i++) {

			matrix[p * 3 + i][p * 3 + i] -= k[p - 1] * m0[0];
			matrix[i][p * 3 + i] += k[p - 1] * m0[0];
			matrix[p * 3 + i][i] += k[p - 1] * m0[p];
		}

}


void bloch_mcconnell_ode(int P, float out[P * 3], const float in[P * 3] , float r1[P], float r2[P], float k[P - 1], float m0[P], float Om[P], float gb[3])
{
	int N = 1 + P * 3;
	float m[N][N];

	// create Bloch-McConnell matrix
	bloch_mcconnel_matrix_ode(P, m, r1, r2, k, m0, Om, gb);

	float out_tmp[N];
	float in_tmp[N];

	for (int i = 0; i < N - 1; i++)
		in_tmp[i] = in[i];

	in_tmp[N-1] = 1.;

	// multiply matrix with input vector
	for (int i = 0; i < N; i++) {

		out_tmp[i] = 0.;

		for (int j = 0; j < N; j++)
			out_tmp[i] += m[i][j] * in_tmp[j];
	}

	for (int i = 0; i < N - 1; i++)
		out[i] = out_tmp[i];
}

void bloch_mcc_pdy(int P, float out[P * 3][P * 3], const float in[P * 3], float r1[P], float r2[P], const float k[P - 1], const float m0[P], const float Om[P], const float gb[3])
{
	UNUSED(in);

	float m[P * 3 + 1][P * 3 + 1];

	bloch_mcconnel_matrix_ode(P, m, r1, r2, k, m0, Om, gb);

	for (int i = 0; i < P * 3; i++)
		for (int j = 0; j < P * 3; j++)
			out[j][i] = m[i][j]; //transposition

}

void bloch_mcc_b1_pdp(int P, float out[P * 5 - 1][P * 3], const float in[P * 3], float r1[P], float r2[P], const float k[P - 1], const float m0[P], const float gb[3], float phase, float b1)
{
	UNUSED(r2); UNUSED(gb); 

	for (int i = 0; i < P * 5 - 1; i++)
		for (int j = 0; j < P * 3; j++)
			out[i][j] = 0.;

	assert(P > 1);

	for (int p = 0; p < P; p++) {

		// R1
		out[0 + p][2 + p * 3] = - (in[2 + p * 3] - m0[p]);
		// R2
		out[P + p][0 + p * 3] = - in[p * 3];
		out[P + p][1 + p * 3] = - in[1 + p * 3];
		// B1
		out[2 * P][0 + p * 3] = sinf(phase) * in[2 + p * 3] * b1;
		out[2 * P][1 + p * 3] = cosf(phase) * in[2 + p * 3] * b1;
		out[2 * P][2 + p * 3] = (-sinf(phase) * in[0 + p * 3] - cosf(phase) * in[1 + p * 3]) * b1;
	}

	for (int p = 0; p < P - 1; p++) {

		// M0 water; special treatment because of exchange with all other pools
		out[2 * P + 1][0] += in[3 + p * 3] * k[p];
		out[2 * P + 1][1] += in[4 + p * 3] * k[p];
		out[2 * P + 1][2] += in[5 + p * 3] * k[p];
		out[2 * P + 1][3 + p * 3] = -in[3 + p * 3] * k[p];
		out[2 * P + 1][4 + p * 3] = -in[4 + p * 3] * k[p];
		out[2 * P + 1][5 + p * 3] = -in[5 + p * 3] * k[p];

		// M0 remaining
		out[2 * P + 2 + p][0] = -in[0] * k[p];
		out[2 * P + 2 + p][1] = -in[1] * k[p];
		out[2 * P + 2 + p][2] = -in[2] * k[p];
		out[2 * P + 2 + p][3 + p * 3] = in[0] * k[p];
		out[2 * P + 2 + p][4 + p * 3] = in[1] * k[p];
		out[2 * P + 2 + p][5 + p * 3] = r1[p + 1] + in[2] * k[p];

		// k; Pools only exchange with water pool
		out[3 * P + 1 + p][0] = -in[0] * m0[p + 1] + in[3 + p * 3] * m0[0];
		out[3 * P + 1 + p][1] = -in[1] * m0[p + 1] + in[4 + p * 3] * m0[0];
		out[3 * P + 1 + p][2] = -in[2] * m0[p + 1] + in[5 + p * 3] * m0[0];
		out[3 * P + 1 + p][3 + p * 3] = in[0] * m0[p + 1] - in[3 + p * 3] * m0[0];
		out[3 * P + 1 + p][4 + p * 3] = in[1] * m0[p + 1] - in[4 + p * 3] * m0[0];
		out[3 * P + 1 + p][5 + p * 3] = in[2] * m0[p + 1] - in[5 + p * 3] * m0[0];

		// Om
		out[4 * P + p][3 + p * 3] = in[4 + p * 3];
		out[4 * P + p][4 + p * 3] = -in[3 + p * 3];

	}

	// Missing term for M0 water
	out[2 * P + 1][2] += r1[0];

}
