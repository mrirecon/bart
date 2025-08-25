
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include <math.h>

#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/misc.h"
#include "misc/version.h"
#include "misc/debug.h"
#include "misc/mri.h"

#include "pulseq.h"

#define MAX_LINE_LENGTH 100

#define VEC(T) struct { int len; typeof(T) data[]; }

static void* vec_init(void)
{
	struct {int len;} * vec;
	vec = xmalloc(sizeof *vec);
	vec->len = 0;
	return vec;
}


#define	VEC_ADD(v, o) 							\
do {									\
	auto _p = &(v);							\
	typedef typeof((*_p)->data[0]) eltype_t;			\
	int n2 = (*_p) ? (++(*_p)->len) : 1;				\
	/* fix fanalyzer leak detection if realloc fails */		\
	auto _q = *_p;							\
	*_p = realloc(*_p, sizeof(**_p) + (unsigned long)n2 * sizeof(eltype_t));\
	if (!*_p) error("memory out");					\
	(void)_q;							\
	(*_p)->len = n2;						\
	(*_p)->data[n2 - 1] = (o);					\
} while (0)


#define SECTIONS(X) X(VERSION) X(DEFINITIONS) X(BLOCKS) X(GRADIENTS) X(TRAP) X(RF) X(ADC) X(SHAPES) X(SIGNATURE)
#define BLOCKS_FORMAT "%4d %ld %d %d %d %d %d %d"
#define BLOCKS_ACCESS(X) X(num) X(dur) X(rf) X(g[0]) X(g[1]) X(g[2]) X(adc) X(ext)
#define GRADIENTS_FORMAT "%d %lf %d %d %ld"
#define GRADIENTS_ACCESS(X) X(id) X(amp) X(shape_id) X(time_id) X(delay)
#define TRAP_FORMAT "%d %lf %ld %ld %ld %ld"
#define TRAP_ACCESS(X) X(id) X(amp) X(rise) X(flat) X(fall) X(delay)
#define ADC_FORMAT "%d %ld %ld %ld %lf %lf"
#define ADC_ACCESS(X) X(id) X(num) X(dwell) X(delay) X(freq) X(phase)
#define RF_FORMAT "%d %lf %d %d %d %ld %lf %lf"
#define RF_ACCESS(X) X(id) X(mag) X(mag_id) X(ph_id) X(time_id) X(delay) X(freq) X(phase)


static struct shape make_shape(int id, int num, int len, const double val[len])
{
	struct shape shape = { id, num, vec_init() };

	bool compression = (num != len);
	double value = 0.;

	for (int i = 0; i < len; i++) {

		int count = 1;

		double update = val[i];
		if (compression && (0 < i) && (i < len - 1) && (val[i - 1] == val[i]))
			count += val[++i];

		while (count--)
			VEC_ADD(shape.values, (compression) ? value += update : update);
	}

	assert(shape.values->len == num);
	return shape;
}

static struct shape make_compressed_shape(int id, int len, const double val[len])
{
	struct shape shape = { id, len, vec_init() };

	double der = 0.;

	int count = 0;
	
	for(int i = 0; i < len; i++) {

		double der_i  = val[i] - ((0 == i) ? 0. : val[i - 1]);
		der_i = (1.e-8 > fabs(der_i)) ? 0. : der_i;

		if (1.e-12 > fabs(der - der_i)) {

			if (count < 2)
				VEC_ADD(shape.values, der_i);

			count++;
		}
		else {

			if (count > 2)
				VEC_ADD(shape.values, count - 2);

			VEC_ADD(shape.values, der_i);

			der = der_i;
			count = 1;
		}
	}

	if (count > 2)
		VEC_ADD(shape.values, count - 2);

	if ((int)(0.75 * len) < shape.values->len) {

		debug_printf(DP_DEBUG3, "insufficient forced compression (%d/%d), return uncompressed shape\n", shape.values->len, len);
		xfree(shape.values);
		return make_shape(id, len, len, val); // uncompressed shape because num == len
	}

	return shape;
}

void pulseq_init(struct pulseq *ps)
{
	ps->version[0] = 1;
	ps->version[1] = 4;
	ps->version[2] = 2;
	ps->adc_raster_time = 1.e-7;
	ps->gradient_raster_time = 1.e-5;
	ps->block_raster_time = 1.e-5;
	ps->rf_raster_time = 1.e-6;
	ps->total_duration = 0.;


	ps->ps_blocks = vec_init();
	ps->gradients = vec_init();
	ps->trapezoids = vec_init();
	ps->adcs = vec_init();
	ps->rfpulses = vec_init();
	ps->shapes = vec_init();
}


void pulseq_free(struct pulseq *ps)
{
	xfree(ps->ps_blocks);
	xfree(ps->gradients);
	xfree(ps->trapezoids);
	xfree(ps->adcs);
	xfree(ps->rfpulses);

	for (int i = 0; i < ps->shapes->len; i++)
		xfree(ps->shapes->data[i].values);

	xfree(ps->shapes);
}
