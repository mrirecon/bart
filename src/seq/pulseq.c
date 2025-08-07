
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


#define	VEC_ADD(v, o) 							\
do {									\
	auto _p = &(v);							\
	typedef typeof((*_p)->data[0]) eltype_t;			\
	int n2 = (*_p) ? (++(*_p)->len) : 1;				\
	/* fix fanalyzer leak detection if realloc fails */		\
	auto _q = *_p;							\
	*_p = realloc(*_p, sizeof(*_p) + (unsigned long)n2 * sizeof(eltype_t));	\
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


	ps->ps_blocks = xmalloc(sizeof(int));
	ps->gradients = xmalloc(sizeof(int));
	ps->trapezoids = xmalloc(sizeof(int));
	ps->adcs = xmalloc(sizeof(int));
	ps->rfpulses = xmalloc(sizeof(int));
	ps->shapes = xmalloc(sizeof(struct shape));

	ps->ps_blocks->len = 0;
	ps->gradients->len = 0;
	ps->trapezoids->len = 0;
	ps->adcs->len = 0;
	ps->rfpulses->len = 0;
	ps->shapes->len = 0;
}

