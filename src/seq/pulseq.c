
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/misc.h"
#include "misc/version.h"
#include "misc/debug.h"
#include "misc/mri.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/seq.h"

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


#define SECTIONS(X) X(VERSION) X(DEFINITIONS) X(BLOCKS) X(GRADIENTS) X(TRAP) X(RF) X(ADC) X(EXTENSIONS) X(SHAPES) X(SIGNATURE)
#define BLOCKS_FORMAT "%4d %lu %d %d %d %d %d %d"
#define BLOCKS_ACCESS(X) X(num) X(dur) X(rf) X(g[0]) X(g[1]) X(g[2]) X(adc) X(ext)
#define GRADIENTS_FORMAT "%d %lf %d %d %lu"
#define GRADIENTS_ACCESS(X) X(id) X(amp) X(shape_id) X(time_id) X(delay)
#define TRAP_FORMAT "%d %lf %lu %lu %lu %lu"
#define TRAP_ACCESS(X) X(id) X(amp) X(rise) X(flat) X(fall) X(delay)
#define ADC_FORMAT "%d %lu %lu %lu %lf %lf"
#define ADC_ACCESS(X) X(id) X(num) X(dwell) X(delay) X(freq) X(phase)
#define EXTENSIONS_FORMAT "%d %d %d %d"
#define EXTENSIONS_ACCESS(X) X(id) X(type) X(ref) X(next)
#define RF_FORMAT "%d %lf %d %d %d %lu %lf %lf"
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

			if (count >= 2)
				VEC_ADD(shape.values, count - 2);

			VEC_ADD(shape.values, der_i);

			der = der_i;
			count = 1;
		}
	}

	if (count >= 2)
		VEC_ADD(shape.values, count - 2);

	if ((int)(0.75 * len) < shape.values->len) {

		debug_printf(DP_DEBUG3, "insufficient forced compression (%d/%d), return uncompressed shape\n", shape.values->len, len);
		xfree(shape.values);
		return make_shape(id, len, len, val); // uncompressed shape because num == len
	}

	return shape;
}

void pulseq_init(struct pulseq *ps, const struct seq_config* seq)
{
	ps->version[0] = 1;
	ps->version[1] = 4;
	ps->version[2] = 2;
	ps->adc_raster_time = 1.e-7;
	ps->gradient_raster_time = 1.e-5;
	ps->block_raster_time = 1.e-5;
	ps->rf_raster_time = 1.e-6;
	ps->fov[0] = 1.e-3 * seq->geom.fov;
	ps->fov[1] = 1.e-3 * seq->geom.fov;
	ps->fov[2] = 1.e-3 * seq->geom.slice_thickness * seq->loop_dims[SLICE_DIM];
	ps->total_duration = 0.;

	ps->label_flags = md_nontriv_dims(DIMS, seq->loop_dims) & SEQ_FLAGS & ~(COEFF_FLAG | COEFF2_FLAG| ITER_FLAG); // MDH dimension to write

	ps->ps_blocks = vec_init();
	ps->gradients = vec_init();
	ps->trapezoids = vec_init();
	ps->adcs = vec_init();
	ps->rfpulses = vec_init();
	ps->shapes = vec_init();
	ps->extensions = vec_init();
	ps->extension_spec = vec_init();
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

	xfree(ps->extensions);

	for (int i = 0; i < ps->extension_spec->len; i++)
		xfree(ps->extension_spec->data[i].values);

	xfree(ps->extension_spec);
}


void pulse_shapes_to_pulseq(struct pulseq *ps, int N, const struct rf_shape rf_shapes[N])
{
	assert(0 == ps->shapes->len);
	double tmp[2];
	for (int i = 0; i < N; i++) {

		long samples = rf_shapes[i].samples;
		double mag[samples];
		double pha[samples];
		double time[samples];

		int mag_id = 3 * i + 1;

		float m;
		float p;

		for (int j = 0; j < samples; j++) {

			idea_cfl_to_sample(&rf_shapes[i], j, &m, &p);
			mag[j] = (double)m;
			pha[j] = round(1.e5 * (double)p / (2 * M_PI)) * 1.e-5;
			time[j] = round(j * rf_shapes[i].sar_dur / rf_shapes[i].samples / (1.e6 * ps->rf_raster_time));
		}

		VEC_ADD(ps->shapes, make_compressed_shape(mag_id, samples, mag));
		VEC_ADD(ps->shapes, make_compressed_shape(mag_id + 1, samples, pha));

		if (rf_shapes[i].samples != (1.e-6 / ps->rf_raster_time) * rf_shapes[i].sar_dur)
			VEC_ADD(ps->shapes, make_compressed_shape(mag_id + 2, samples, time));
		else
		 	VEC_ADD(ps->shapes, make_compressed_shape(mag_id + 2, 2, tmp));
	}
}


static int check_existing_shape(const struct pulseq* ps, const struct shape* shape2)
{
	for (int i = 0; i < ps->shapes->len; i++) {

		struct shape sh = ps->shapes->data[i];

		if (sh.values->len != shape2->values->len)
			continue;

		bool equal = true;

		for (int j = 0; j < sh.values->len; j++) {

			if (1.e-12 < fabs(sh.values->data[j] - shape2->values->data[j])) {

				equal = false;
				break;
			}
		}

		if (equal)
			return sh.id;
	}

	return -1;
}

static void grad_to_pulseq(int grad_id[3], struct pulseq *ps, struct seq_sys sys, int N, const struct seq_event ev[N])
{
	double g[MAX_GRAD_POINTS][3];
	seq_compute_gradients(MAX_GRAD_POINTS, g, 10., N, ev);
	long grad_len = (seq_block_end_flat(N, ev) + seq_block_rdt(N, ev)) / GRAD_RASTER_TIME;

	if (0 == grad_len)
		grad_len = 2; // dummy

	double g_axis[grad_len];

	for (int a = 0; a < 3; a++) {

		for (int i = 0; i < grad_len; i++)
			g_axis[i] = - g[i][a] / sys.grad.max_amplitude; // -1. for constistency

		int sid = ps->shapes->len + 1;
		struct shape tmp_shape = make_compressed_shape(sid, grad_len, g_axis);

		int sid2 = check_existing_shape(ps, &tmp_shape);

		if (0 < sid2) {

			debug_printf(DP_DEBUG3, "re-using existing gradient shape %d instead of %d\n", sid2, sid);
			xfree(tmp_shape.values);
			sid = sid2;
		}
		else {

			auto _tmp = tmp_shape.values;
			VEC_ADD(ps->shapes, tmp_shape);
			(void)_tmp;
		}

		grad_id[a] = ps->gradients->len + 1;
		struct gradient g = {

			.id = grad_id[a],
			.amp = sys.grad.max_amplitude * sys.gamma * 1.e3,
			.shape_id = sid
		};

		VEC_ADD(ps->gradients, g);
	}
}

static double phase_pulseq(const struct seq_event* ev)
{
	double phase_mid = (SEQ_EVENT_PULSE == ev->type) ? ev->pulse.phase : ev->adc.phase;
	double freq = (SEQ_EVENT_PULSE == ev->type) ? ev->pulse.freq : ev->adc.freq;
	double ret = fmod(DEG2RAD(- freq * 0.000360 * (ev->mid - ev->start) + phase_mid), 2. * M_PI);
	return (ret < 0.) ? (ret + 2. * M_PI) : ret;
}

static int adc_to_pulseq(struct pulseq *ps, int N, const struct seq_event ev[N])
{
	int adc_idx = events_idx(0, SEQ_EVENT_ADC, N, ev);
	if (0 > adc_idx)
		return 0;

	assert(SEQ_EVENT_ADC == ev[adc_idx].type);
	int adc_id = ps->adcs->len + 1;
	struct adc a = {

		.id = adc_id,
		.num = (uint64_t)lround(ev[adc_idx].adc.columns * ev[adc_idx].adc.os),
		.dwell = (uint64_t)lround(ev[adc_idx].adc.dwell_ns / ev[adc_idx].adc.os),
		.delay = ev[adc_idx].start,
		.freq = ev[adc_idx].adc.freq,
		.phase = phase_pulseq(&ev[adc_idx])
	};

	VEC_ADD(ps->adcs, a);
	return adc_id;
}

static int label_check_spec(struct pulseq* ps, struct extension_spec es)
{
	for (int i = 0; i < ps->extension_spec->len; i++)
		if (0 == strcmp(ps->extension_spec->data[i].string_id, es.string_id))
			return i;

	return -1;
}

static const char* dim_to_string(int dim)
{
	switch (dim) {

	case PHS1_DIM: return "LIN";
	case TIME_DIM: return "REP";
	case SLICE_DIM: return "SLC";
	case TE_DIM: return "ECO";
	case PHS2_DIM: return "PAR";
	case CSHIFT_DIM: return "SET";
	case BATCH_DIM: return "SEG";
	case TIME2_DIM: return "PHS";
	case AVG_DIM: return "AVG";
	default: assert(0);
	}
}


static int label_set(struct pulseq* ps, const char* set, struct seq_event adc_ev)
{
	struct extension_spec es;
	es.string_id = set;

	int idx = label_check_spec(ps, es);

	for (int i = 0; i < DIMS; i++) {

		if (MD_IS_SET(ps->label_flags, i)) {

			struct ext e = { (int)adc_ev.adc.pos[i], i };

			if (0 > idx) {

				idx = ps->extension_spec->len;
				es.type = idx + 1;
				es.values = vec_init();
				VEC_ADD(es.values, e);
				VEC_ADD(ps->extension_spec, es);
				continue;
			}

			VEC_ADD(ps->extension_spec->data[idx].values, e);
		}
	}

	return idx + 1;
}

static int ext_to_pulseq(struct pulseq *ps, int N, const struct seq_event ev[N])
{
	int adc_idx = events_idx(0, SEQ_EVENT_ADC, N, ev);
	if (0 > adc_idx)
		return 0;

	assert(SEQ_EVENT_ADC == ev[adc_idx].type);

	int labels = bitcount(ps->label_flags);
	int es_type = label_set(ps, "LABELSET", ev[adc_idx]);

	int ext_id = ps->extensions->len + 1;

	for (int i = 0; i < 3; i++) {

		struct extension e = {

			.id = ext_id + i,
			.type = es_type,
			.ref = ext_id + i,
			.next = ((i + 1) == labels) ? 0 : ext_id + i + 1,
		};

		VEC_ADD(ps->extensions, e);
	}

	return ext_id;
}

static int rf_to_pulseq(struct pulseq *ps, int M, const struct rf_shape rf_shapes[M], int N, const struct seq_event ev[N])
{
	int rf_idx = events_idx(0, SEQ_EVENT_PULSE, N, ev);
	if (0 > rf_idx)
		return 0;

	assert(SEQ_EVENT_PULSE == ev[rf_idx].type);
	int rf_id = ps->rfpulses->len + 1;

	int pulse_id = ev[rf_idx].pulse.shape_id;
	int mag_id = 3 * pulse_id + 1;

	double ampl = rf_shapes[pulse_id].max / (2. * M_PI);

	int time_id = 0;

	if (rf_shapes[pulse_id].samples != (1.e-6 / ps->rf_raster_time) * rf_shapes[pulse_id].sar_dur)
		time_id = mag_id + 2;

	struct rfpulse rf = {

		.id = rf_id,
		.mag = ampl,
		.mag_id = mag_id,
		.ph_id = mag_id + 1,
		.time_id = time_id,
		.delay = (uint64_t)(ev[rf_idx].start),
		.freq = ev[rf_idx].pulse.freq,
		.phase = phase_pulseq(&ev[rf_idx])
	};

	VEC_ADD(ps->rfpulses, rf);
	return rf_id;
}

void events_to_pulseq(struct pulseq *ps, enum block mode, long tr, struct seq_sys sys, int M, const struct rf_shape rf_shapes[M], int N, const struct seq_event ev[N])
{
	unsigned long dur = seq_block_end(N, ev, mode, tr) / (1.e6 * ps->block_raster_time);
	ps->total_duration += dur * ps->block_raster_time;


	if (0 < events_counter(SEQ_EVENT_ADC, N, ev))
		error("Multiple ADCs per block not supported\n");

	int adc_id = adc_to_pulseq(ps, N, ev);

	int ext_id = 0;
	if (adc_id)
		ext_id = ext_to_pulseq(ps, N, ev);

	int rf_id = rf_to_pulseq(ps, M, rf_shapes, N, ev);

	int g_id[3] = { 0, 0, 0 };
	grad_to_pulseq(g_id, ps, sys, N, ev);

	struct ps_block b = {

		.num = ps->ps_blocks->len + 1,
		.dur = dur,
		.rf = rf_id,
		.g = { g_id[0], g_id[1], g_id[2] },
		.adc = adc_id,
		.ext = ext_id
	};

	VEC_ADD(ps->ps_blocks, b);
}



void pulseq_writef(FILE *fp, struct pulseq *ps)
{
	fprintf(fp, "# Pulseq sequence file\n"
		    "# Created by BART %s\n", bart_version);

	fprintf(fp, "# bart %s\n", command_line);

	fprintf(fp, "\n[VERSION]\nmajor %d\nminor %d\nrevision %d\n",
		ps->version[0], ps->version[1], ps->version[2]);

	fprintf(fp, "\n[DEFINITIONS]\n");
	fprintf(fp, "AdcRasterTime %.e\n", ps->adc_raster_time);
	fprintf(fp, "BlockDurationRaster %.e\n", ps->block_raster_time);
	fprintf(fp, "GradientRasterTime %.e\n", ps->gradient_raster_time);
	fprintf(fp, "RadiofrequencyRasterTime %.e\n", ps->rf_raster_time);
	fprintf(fp, "FOV %.3f %.3f %.3f \n", ps->fov[0], ps->fov[1], ps->fov[2]);
	fprintf(fp, "PhaseResolution 1.0 \n"); // we have to set this for the interpreter [ otherwise small LIN counter --> crash]
	fprintf(fp, "TotalDuration %.5f\n", ps->total_duration);

	fprintf(fp, "\n\n# Format of blocks:\n");
	fprintf(fp, "# NUM DUR RF  GX  GY  GZ  ADC  EXT");
	fprintf(fp, "\n[BLOCKS]\n");
#define ACCESS(X) , ps->ps_blocks->data[i].X
	for (int i = 0; i < ps->ps_blocks->len; i++)
		fprintf(fp, BLOCKS_FORMAT "\n" BLOCKS_ACCESS(ACCESS));
#undef	ACCESS

	fprintf(fp, "\n[ADC]\n");
#define ACCESS(X) , ps->adcs->data[i].X
	for (int i = 0; i < ps->adcs->len; i++)
		fprintf(fp, ADC_FORMAT "\n" ADC_ACCESS(ACCESS));
#undef	ACCESS

	if (0 < ps->extensions->len) {

		fprintf(fp, "\n[EXTENSIONS]\n");
#define ACCESS(X) , ps->extensions->data[i].X
		for (int i = 0; i < ps->extensions->len; i++)
			fprintf(fp, EXTENSIONS_FORMAT "\n" EXTENSIONS_ACCESS(ACCESS));
#undef	ACCESS

		fprintf(fp, "\n#id set labelstring");

		struct extension_spec es = ps->extension_spec->data[0];
		fprintf(fp, "\nextension %s %d\n", es.string_id, es.type);

		for (int j = 0; j < es.values->len; j++)
			fprintf(fp, "%d %d %s\n", j + 1, es.values->data[j].val, dim_to_string(es.values->data[j].dim));
	}


	fprintf(fp, "\n[GRADIENTS]\n");
#define ACCESS(X) , ps->gradients->data[i].X
	for (int i = 0; i < ps->gradients->len; i++)
		fprintf(fp, GRADIENTS_FORMAT "\n" GRADIENTS_ACCESS(ACCESS));
#undef	ACCESS

	fprintf(fp, "\n[TRAP]\n");
#define ACCESS(X) , ps->trapezoids->data[i].X
	for (int i = 0; i < ps->trapezoids->len; i++)
		fprintf(fp, TRAP_FORMAT "\n" TRAP_ACCESS(ACCESS));
#undef	ACCESS

	fprintf(fp, "\n[RF]\n");
#define ACCESS(X) , ps->rfpulses->data[i].X
	for (int i = 0; i < ps->rfpulses->len; i++)
		fprintf(fp, RF_FORMAT "\n" RF_ACCESS(ACCESS));
#undef	ACCESS

	fprintf(fp, "\n[SHAPES]\n");
	for (int i = 0; i < ps->shapes->len; i++) {

		fprintf(fp, "\nshape_id %d\n", i + 1);
		struct shape sh = ps->shapes->data[i];

		fprintf(fp, "num_samples %d\n", sh.num);

		for (int j = 0; j < sh.values->len; j++) {

			if (1 < sh.values->data[j])
				fprintf(fp, "%d\n", (int)sh.values->data[j]);
			else
				fprintf(fp, "%.10f\n", sh.values->data[j]);
		}
	}


	fprintf(fp, "\n[SIGNATURE]\n");
	fprintf(fp, "# TODO\n");

	pulseq_free(ps);
}



