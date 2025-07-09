
#ifndef _GRADIENT_H
#define _GRADIENT_H

#include <stdbool.h>

struct grad_limits {

	double inv_slew_rate;
	double max_amplitude;
};

struct grad_trapezoid {

	double rampup;
	double flat;
	double rampdown;

	double ampl;
};

extern double grad_duration(const struct grad_trapezoid* grad);
extern double grad_total_time(const struct grad_trapezoid* grad);
extern double grad_momentum(struct grad_trapezoid* grad);

extern bool grad_soft(struct grad_trapezoid* grad, double dur, double moment, struct grad_limits sys);
extern bool grad_hard(struct grad_trapezoid* grad, double moment, struct grad_limits sys);

#endif // _GRADIENT_H

