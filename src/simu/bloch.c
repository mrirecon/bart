
#include <assert.h>
#include <math.h>

#include "num/vec3.h"

#include "bloch.h"



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


void bloch_relaxation(float out[3], float t, const float in[3], float r1, float r2, const float gb[3])
{
	float m0 = 1.;
	assert((0. == gb[0]) && (0. == gb[1])); // no B1(t)

	out[0] =  (in[0] * cosf(gb[2] * t) - in[1] * sinf(gb[2] * t)) * expf(-t * r2);
	out[1] = -(in[0] * sinf(gb[2] * t) + in[1] * cosf(gb[2] * t)) * expf(-t * r2);
	out[2] = in[2] + (m0 - in[2]) * (1. - expf(-t * r1));
}


void bloch_excitation(float out[3], float t, const float in[3], float r1, float r2, const float gb[3])
{
	(void)r1; (void)r2;
	assert(0. == gb[2]); // no gradient, rotating frame

	out[0] = in[0];
	out[1] = (in[2] * sinf(gb[0] * t) + in[0] * cosf(gb[0] * t));
	out[2] = (in[2] * cosf(gb[0] * t) - in[0] * sinf(gb[0] * t));
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



