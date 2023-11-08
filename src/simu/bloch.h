
#include <complex.h>

// rad s^-1 T^-1
#define GAMMA_H1 267.513e6

//  @3T
#define WATER_T1 3.0
#define WATER_T2 0.3

#define SKYRA_B0 3.

// T/m
#define SKYRA_GRADIENT 0.045

// T/m/s
#define SKYRA_RAMP 200.

extern void rotx(float out[3], const float in[3], float angle);
extern void roty(float out[3], const float in[3], float angle);
extern void rotz(float out[3], const float in[3], float angle);

extern void bloch_ode(float out[3], const float in[3], float r1, float r2, const float gb[3]);
extern void bloch_relaxation(float out[3], float t, const float in[3], float r1, float r2, const float gb[3]);
extern void bloch_excitation(float out[3], float t, const float in[3], float r1, float r2, const float gb[3]);
extern void bloch_excitation2(float out[3], const float in[3], float angle, float phase);

extern void bloch_matrix_ode(float matrix[4][4], float r1, float r2, const float gb[3]);
extern void bloch_matrix_int(float matrix[4][4], float t, float r1, float r2, const float gb[3]);

extern void bloch_matrix_ode_sa(float matrix[10][10], float r1, float r2, const float gb[3]);
extern void bloch_matrix_int_sa(float matrix[10][10], float t, float r1, float r2, const float gb[3]);

extern void bloch_matrix_ode_sa2(float matrix[13][13], float r1, float r2, const float gb[3], complex float b1);
extern void bloch_matrix_int_sa2(float matrix[13][13], float t, float r1, float r2, const float gb[3], complex float b1);

extern void bloch_pdy(float out[3][3], const float in[3], float r1, float r2, const float gb[3]);
extern void bloch_pdp(float out[2][3], const float in[3], float r1, float r2, const float gb[3]);
extern void bloch_b1_pdp(float out[3][3], const float in[3], float r1, float r2, const float gb[3], complex float b1);

extern void bloch_mcconnel_matrix_ode(int P, float matrix[1 + P * 3][1 + P * 3], const float r1[P], const float r2[P], const float k[P - 1], const float m0[P], const float Om[P], const float gb[3]);
extern void bloch_mcconnell_ode(int P, float out[P * 3], const float in[P  *3] , float r1[P], float r2[P], float k[P - 1], float m0[P], float Om[P], float gb[3]);

extern void bloch_mcc_pdy(int P, float out[P * 3][P * 3], const float in[P * 3], float r1[P], float r2[P], const float k[P - 1], const float m0[P], const float Om[P], const float gb[3]);
extern void bloch_mcc_b1_pdp(int P, float out[P * 5 - 1][P * 3], const float in[P * 3], float r1[P], float r2[P], const float k[P - 1], const float m0[P], const float gb[3], complex float b1);

extern void bloch_mcc_matrix_ode_sa(int P, float matrix[15 * P * P - 3 * P + 1][15 * P * P - 3 * P + 1], float r1[P], float r2[P], float k[P - 1], float m0[P], float Om[P], const float gb[3]);
extern void bloch_mcc_matrix_ode_sa2(int P, float matrix[15 * P * P + 1][15 * P * P + 1], float r1[P], float r2[P], float k[P - 1], float m0[P], float Om[P],  const float gb[3], complex float b1);

