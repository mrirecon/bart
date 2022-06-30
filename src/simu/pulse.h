
#ifndef __PULSE_H
#define __PULSE_H

#include <stdbool.h>

struct hs_pulse {

	float a0;
	float beta;
	float mu;
	float duration; /*[s]*/
        bool on;
};

extern const struct hs_pulse hs_pulse_defaults;

struct simdata_pulse {

	float rf_start;
	float rf_end;
	float flipangle;
	float phase;
	float bwtp;		/* BWTP=2*n for n=nl=nr */
	float nl;		/* number of zero crossings to the left of the main lope */
	float nr; 		/* number of zero crossings to the right of the main lope */
	float n;		/* max(nl, nr) */
	float t0;		/* time of main lope: t0 = pulse_len / ( 2 + (nl-1)  + (nr-1)) */
	float alpha; 	/* windows of pulse ( 0: normal sinc, 0.5: Hanning, 0.46: Hamming) */
	float A;		/* offset */

        struct hs_pulse hs;
};

extern const struct simdata_pulse simdata_pulse_defaults;

extern float pulse_sinc(const struct simdata_pulse* pulse, float t);

extern float sinc_integral(const struct simdata_pulse* pulse);

extern void sinc_pulse_init(struct simdata_pulse* pulse, float rf_start, float rf_end, float angle, float phase, float bwtp, float alpha);

extern float pulse_hypsec_am(const struct hs_pulse* pulse, float t);
extern float pulse_hypsec_fm(const struct hs_pulse* pulse, float t);
extern float pulse_hypsec_phase(const struct hs_pulse* pulse, float t);

#endif

