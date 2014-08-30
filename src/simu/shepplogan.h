
#include <complex.h>


struct ellipsis_s {

	double intensity;
	double axis[2];
	double center[2];
	double angle;
};

extern const struct ellipsis_s shepplogan[10];
extern const struct ellipsis_s shepplogan_mod[10];
extern const struct ellipsis_s phantom_disc[1];


extern complex double xellipsis(const double center[2], const double axis[2], double angle, const double p[2]);
extern complex double kellipsis(const double center[2], const double axis[2], double angle, const double p[2]);
    


extern complex double phantom(unsigned int N, const struct ellipsis_s arr[N], const double pos[2], _Bool ksp);

