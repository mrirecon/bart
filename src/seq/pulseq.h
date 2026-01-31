
#include <stdio.h>
#include <stdint.h>
#include <complex.h>

#include "misc/mri.h"
#include "misc/misc.h"

#include "seq/event.h"
#include "seq/config.h"

struct ps_block {

	int num;
	unsigned long dur;
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
	unsigned long delay;
};

struct trapezoid {

	int id;
	double amp;
	unsigned long rise;
	unsigned long flat;
	unsigned long fall;
	unsigned long delay;
};

struct rfpulse {

	int id;
	double mag;
	int mag_id;
	int ph_id;
	int time_id;
	unsigned long delay;
	double freq;
	double phase;
};

struct adc {

	int id;
	unsigned long num;
	unsigned long dwell;
	unsigned long delay;
	double freq;
	double phase;
};

struct extension {

	int id;
	int type;
	int ref;
	int next;
};

#define VEC(T) struct { int len; typeof(T) data[]; }

struct shape {

	int id;
	int num;
	VEC(double) *values;
};

struct ext {

	int val;
	int dim;
};

struct extension_spec {

	const char* string_id;
	int type;
	VEC(struct ext) *values;
};

struct pulseq {

	int version[3];
	double adc_raster_time;
	double gradient_raster_time;
	double block_raster_time;
	double rf_raster_time;
	double fov[3];
	double total_duration;

	unsigned long label_flags;

	VEC(struct ps_block) *ps_blocks;
	VEC(struct gradient) *gradients;
	VEC(struct trapezoid) *trapezoids;
	VEC(struct adc) *adcs;
	VEC(struct rfpulse) *rfpulses;
	VEC(struct shape) *shapes;
	VEC(struct extension) *extensions;
	VEC(struct extension_spec) *extension_spec;
};

#undef VEC


extern void pulseq_init(struct pulseq *ps, const struct seq_config* seq);
extern void pulseq_free(struct pulseq *ps);

extern void pulse_shapes_to_pulseq(struct pulseq *ps, int N, const struct rf_shape rf_shapes[__VLA(N)]);

extern void events_to_pulseq(struct pulseq *ps, enum seq_block mode, double tr, struct seq_sys sys, int M, const struct rf_shape rf_shapes[__VLA(M)], int N, const struct seq_event ev[__VLA(N)]);

extern void pulseq_writef(FILE *fp, struct pulseq *ps);

