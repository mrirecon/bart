
#include <stdio.h>
#include <stdint.h>
#include <complex.h>
#include "misc/mri.h"

#include "seq/event.h"
#include "seq/config.h"

struct ps_block {

	int num;
	uint64_t dur;
	int rf;
	int g[3];
	int adc;
	int ext;
};

struct gradient {
	int id;
	double amp;
	int shape_id;
	int time_id;
	uint64_t delay;
};

struct trapezoid {

	int id;
	double amp;
	uint64_t rise;
	uint64_t flat;
	uint64_t fall;
	uint64_t delay;
};

struct rfpulse {

	int id;
	double mag;
	int mag_id;
	int ph_id;
	int time_id;
	uint64_t delay;
	double freq;
	double phase;
};

struct adc {

	int id;
	uint64_t num;
	uint64_t dwell;
	uint64_t delay;
	double freq;
	double phase;
};

#define VEC(T) struct { int len; typeof(T) data[]; }

struct shape {

	int id;
	int num;
	VEC(double) *values;
};


struct pulseq {

	int version[3];
	double adc_raster_time;
	double gradient_raster_time;
	double block_raster_time;
	double rf_raster_time;
	double total_duration;

	VEC(struct ps_block) *ps_blocks;
	VEC(struct gradient) *gradients;
	VEC(struct trapezoid) *trapezoids;
	VEC(struct adc) *adcs;
	VEC(struct rfpulse) *rfpulses;
	VEC(struct shape) *shapes;
};

#undef VEC

extern void pulseq_init(struct pulseq *ps);
extern void pulseq_free(struct pulseq *ps);

extern void pulse_shapes_to_pulseq(struct pulseq *ps, int N, const struct rf_shape rf_shapes[__VLA(N)]);

extern void events_to_pulseq(struct pulseq *ps, enum block mode, long tr, struct seq_sys sys, int M, const struct rf_shape rf_shapes[__VLA(M)], int N, const struct seq_event ev[__VLA(N)]);

