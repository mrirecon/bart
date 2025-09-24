
#ifndef _PULSE_H
#define _PULSE_H

#include <stdbool.h>
#include <complex.h>

#include "misc/types.h"

enum pulse_t { PULSE_SINC, PULSE_SINC_SMS, PULSE_HS, PULSE_REC, PULSE_ARB, PULSE_GAUSS };

struct pulse {

	TYPEID* TYPEID;

	float flipangle;
	float duration;		/* pulse duration */

	_Complex float (*eval)(const struct pulse *p, float t);
};

struct pulse_sinc {

	struct pulse super;

	float alpha;		/* windows of pulse (0: normal sinc, 0.5: Hanning, 0.46: Hamming) */
	float A;		/* amplitude */
	float bwtp;		/* BWTP=2*n for n=nl=nr */
};

extern const struct pulse_sinc pulse_sinc_defaults;

extern void pulse_sinc_init(struct pulse_sinc* ps, float duration, float angle /*[deg]*/, float phase, float bwtp, float alpha);
extern float pulse_sinc_integral(const struct pulse_sinc* ps);


struct pulse_sms {

	struct pulse super;

	float alpha;
	float A;
	float bwtp;
	int mb_factor;
	int mb_part;		/* partition to evaluate */
	float SMS_dist;		/* center-to-center slice distance defined by phase modulation */
	float slice_th;		/* slice thickness of one partition*/
};

extern const struct pulse_sms pulse_sms_defaults;

extern void pulse_sms_init(struct pulse_sms* ps, float duration, float angle /*[deg]*/, float phase, float bwtp, float alpha, 
		int mb, int part, float dist, float th);

extern float pulse_sms_integral(const struct pulse_sms* ps);

inline _Complex float pulse_eval(const struct pulse* p, float t)
{
	return p->eval(p, t);
}

struct pulse_rect {

	struct pulse super;

	float A;		/* amplitude */
};

extern const struct pulse_rect pulse_rect_defaults;

extern void pulse_rect_init(struct pulse_rect* pr, float duration, float angle /*[deg]*/, float phase);


struct pulse_hypsec {

	struct pulse super;

	float a0;
	float beta;
	float mu;
	float gamma;
	float A;
};

extern const struct pulse_hypsec pulse_hypsec_defaults;

extern float pulse_hypsec_phase(const struct pulse_hypsec* pr, float t);
extern void pulse_hypsec_init(float gamma, struct pulse_hypsec* pr);
extern float pulse_hypsec_integral(const struct pulse_hypsec* hs);



struct pulse_arb {

	struct pulse super;

	int samples;
	_Complex float* values;
	float gamma;
	float A;		/* amplitude */
};

extern const struct pulse_arb pulse_arb_oc_cest_sat_defaults;

extern void pulse_arb_init(struct pulse_arb* pa, float gamma);
extern float pulse_arb_integral(const struct pulse_arb* pa);


struct pulse_gauss {

	struct pulse super;

	float alpha;
	float A;
	float bwtp;
};

extern const struct pulse_gauss pulse_gauss_defaults;

extern void pulse_gauss_init(struct pulse_gauss* pg, float duration, float angle /*[deg]*/, float phase, float bwtp, float alpha);
extern float pulse_gauss_integral(const struct pulse_gauss* pg);

#endif		// _PULSE_H

