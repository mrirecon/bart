
#ifndef _SEQ_EVENT_H
#define _SEQ_EVENT_H

#include "misc/cppwrap.h"

#include "misc/mri.h"

#define MAX_RF_SAMPLES 8192

enum seq_event_type { SEQ_EVENT_PULSE, SEQ_EVENT_GRADIENT, SEQ_EVENT_ADC, SEQ_EVENT_WAIT , SEQ_EVENT_TRIGGER, SEQ_EVENT_OUTPUT };

enum rf_type_t { UNDEFINED, EXCITATION, REFOCUSSING, STORE };

enum adc_flags {

	ADC_FLAG_ADJ		= (1u << 0),
	ADC_FLAG_PHASEFT	= (1u << 1),
	ADC_FLAG_PARTFT		= (1u << 2),
	ADC_FLAG_FIRSTSLI	= (1u << 3),
	ADC_FLAG_LASTSLI	= (1u << 4),
	ADC_FLAG_LASTCON	= (1u << 5),
	ADC_FLAG_LASTMEAS	= (1u << 6),
	ADC_FLAG_MEASTIME	= (1u << 7),
};


struct seq_pulse {

	int shape_id;
	enum rf_type_t type;
	double fa;

	double freq;
	double phase;
};

struct seq_gradient {

	double ampl[3];
};

struct seq_adc {

	long dwell_ns;
	long columns;
	long pos[DIMS];
	unsigned long flags;

	double os;

	double freq;
	double phase;
};

struct seq_event {

	double start;
	double mid;
	double end;

	enum seq_event_type type;

	const struct seq_event* dependency;

	union {

		struct seq_pulse pulse;
		struct seq_gradient grad;
		struct seq_adc adc;
	};
};

struct rf_shape {

	double sar_calls;
	double sar_dur; // for SAR calculation we prepare the RF shape without seq_event

	double fa_prep;

	double max; // can be scaled by ev->pulse.fa / fa_prep
	double integral;

	long samples;	// shape defined in rad / s
#ifdef __cplusplus
	float  shape[MAX_RF_SAMPLES][2];
#else
	_Complex float shape[MAX_RF_SAMPLES];
#endif

};


int events_counter(enum seq_event_type type, int N, const struct seq_event ev[__VLA(N)]);
int events_idx(int n, enum seq_event_type type, int N, const struct seq_event ev[__VLA(N)]);

struct grad_trapezoid;
int seq_grad_to_event(struct seq_event ev[2], double start, const struct grad_trapezoid* grad, double proj[3]);

int wait_time_to_event(struct seq_event* ev, double start, double dur);

void events_get_te(int E, double te[__VLA(E)], int N, const struct seq_event ev[__VLA(N)]);

double events_end_time(int N, const struct seq_event ev[__VLA(N)], int gradients_only, int flat_end);

void moment(double m0[3], double t, const struct seq_event* ev);
void moment_sum(double m0[3], double t, int N, const struct seq_event ev[__VLA(N)]);

#include "misc/cppwrap.h"

#endif // _SEQ_EVENT_H

