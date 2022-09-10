/* Copyright 2022. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include <assert.h>
#include <math.h>

#include "num/vec3.h"

#include "num/matexp.h"

#include "bloch.h"


// Rotations in RIGHT-handed coordinate system with CLOCKWISE rotation for angle > 0
// Keep it consitent with clockwise rotation of Bloch equations
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

        rotz(out, in, gb[2]*t);

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


static void matf_copy(int N, int M, float out[N][M], /*const*/ float in[N][M])
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			out[i][j] = in[i][j];
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
		{	0.,		0.,		sinf(phase)*b1,0.,	0.,	0.,	0.,	0.,	0.,	-r2,	gb[2],	-gb[1],	0.	},
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

