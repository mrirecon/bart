
#include <complex.h>
#include <stdbool.h>

#ifndef DIMS
#define DIMS 16
#endif

extern void radial_self_delays(int N, float shifts[N], const float phi[N], const long dims[DIMS], const complex float* in);



struct ring_conf {

	int pad_factor;
	float size;
	int no_intersec_sp;
	float crop_factor;
	bool b0;
};

extern struct ring_conf ring_defaults;

extern void ring(const struct ring_conf* conf, float S[3], int N, const float angles[N], const long dims[DIMS], const complex float* in);


