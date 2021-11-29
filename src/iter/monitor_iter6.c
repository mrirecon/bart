/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <stdbool.h>
#include <complex.h>

#include "misc/debug.h"
#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mmio.h"

#include "num/iovec.h"
#include "num/flpmath.h"
#include "num/multind.h"

#include "nlops/nlop.h"

#include "monitor_iter6.h"
#include "monitor.h"




typedef struct monitor_iter6_value_data_s {

	const struct typeid_s* TYPEID;
	unsigned int N_vals;

} monitor_iter6_value_data_t;

typedef void (*monitor_iter6_value_fun_t)(const monitor_iter6_value_data_t* data, unsigned int N, complex float vals[N], long NI, const float* args[NI]);
typedef bool (*monitor_iter6_value_eval_t)(const monitor_iter6_value_data_t* data, long epoch, long batch, long numbatches);
typedef const char* (*monitor_iter6_value_print_string_t)(const monitor_iter6_value_data_t* data);
typedef void (*monitor_iter6_value_free_t)(const monitor_iter6_value_data_t* data);

struct monitor_value_s {

	struct monitor_iter6_value_data_s* data;

	monitor_iter6_value_fun_t fun;
	monitor_iter6_value_eval_t eval;
	monitor_iter6_value_print_string_t print;
	monitor_iter6_value_free_t free;
};


struct monitor_iter6_default_s {

	INTERFACE(struct monitor_iter6_s);

	bool print_time;
	bool print_progress;
	bool print_overwrite;
	bool print_average_obj;

	float average_obj;

	double start_time;

	int num_val_monitors;
	const struct monitor_value_s** val_monitors;

	bool use_record;
	long epochs_written;
	long epochs_created;
	long num_batches;
	long record_dim;
	complex float* record;
};

static DEF_TYPEID(monitor_iter6_default_s);

static const char* print_progress_bar(int length, int done, int total)
{
	const char* result = ptr_printf(" [");

	for (int i = 0; i < length; i++) {

		auto tmp = result;

		result = ptr_printf("%s%c", tmp, ((float)i <= (float)(done * length) / (float)(total)) ? '=' : ' ');

		xfree(tmp);
	}

	auto tmp = result;

	result = ptr_printf("%s];", tmp);

	xfree(tmp);

	return result;
}

static const char* print_time_string(double time, double est_time)
{
	return ptr_printf(" time: %d:%02d:%02d/%d:%02d:%02d;",
		(int)time / 3600, ((int)time %3600)/60, ((int)time % 3600) % 60,
		(int)est_time / 3600, ((int)est_time %3600)/60, ((int)est_time % 3600) % 60);
}

static void create_record(struct monitor_iter6_default_s* monitor, long epoch, long num_batches)
{
	if (!monitor->use_record)
		return;

	monitor->num_batches = num_batches;

	if (epoch >= monitor->epochs_created) {

		long new_epochs_created = epoch + 1;
		long new_record_dims[4] = { new_epochs_created, num_batches, 2, monitor->record_dim };

		complex float* new_record = md_calloc(4, new_record_dims, CFL_SIZE);

		if (NULL != monitor->record) {

			long old_record_dims[4] = { monitor->epochs_created, num_batches, 2, monitor->record_dim };

			md_copy2(4, old_record_dims, MD_STRIDES(4, new_record_dims, CFL_SIZE), new_record, MD_STRIDES(4, old_record_dims, CFL_SIZE), monitor->record, CFL_SIZE);

			md_free(monitor->record);
		}

		monitor->record = new_record;
		monitor->epochs_created = new_epochs_created;
	}

	monitor->epochs_written = epoch + 1;
}

/* *
 * Saves train history to file
 * The resulting file has dimensions [epochs, batches, 2, no. value monitors]
 * If no extra value monitors are specified, time and loss are saved.
 * We store if a value is acquired in the 0-index of dim 3 and the actual value in the 1-index
 * */
void monitor_iter6_dump_record(struct monitor_iter6_s* _monitor, const char* filename)
{
	auto monitor = CAST_DOWN(monitor_iter6_default_s, _monitor);

	if (NULL == monitor->record)
		error("Record not available!\n");

	long rdims_write[4] = { monitor->epochs_written, monitor->num_batches, 2, monitor->record_dim };
	long rdims_read[4] = { monitor->epochs_created, monitor->num_batches, 2, monitor->record_dim };

	complex float* file = create_cfl(filename, 4, rdims_write);

	md_copy2(4, rdims_write, MD_STRIDES(4, rdims_write, CFL_SIZE), file, MD_STRIDES(4, rdims_read, CFL_SIZE), monitor->record, CFL_SIZE);

	unmap_cfl(4, rdims_write, file);
}

static const char* compute_val_monitors(struct monitor_iter6_default_s* monitor, long epoch, long batch, long num_batches, long NI, const float* x[NI])
{
	create_record(monitor, epoch, num_batches);

	const char* result = ptr_printf("");

	long rpos[4] = { epoch, batch, 0, 2 };

	for (int i = 0; i < monitor->num_val_monitors; i++) {

		unsigned int N_vals = monitor->val_monitors[i]->data->N_vals;

		complex float eval[N_vals];
		complex float vals[N_vals];

		for (unsigned int i = 0; i < N_vals; i++) {

			eval[i] = 0;
			vals[i] = 0;
		}

		if (monitor->val_monitors[i]->eval(monitor->val_monitors[i]->data, epoch, batch, num_batches)) {

			for (unsigned int i = 0; i < N_vals; i++)
				eval[i] = 1;

			monitor->val_monitors[i]->fun(monitor->val_monitors[i]->data, N_vals, vals, NI, x);
			const char* tmp = monitor->val_monitors[i]->print(monitor->val_monitors[i]->data);

			if (NULL != tmp) {

				const char* tmp2 = result;
				result = ptr_printf("%s%s", tmp2, tmp);

				xfree(tmp);
				xfree(tmp2);
			}
		}

		if (NULL != monitor->record) {

			long rstrs[4];
			md_calc_strides(4, rstrs, MD_DIMS(monitor->epochs_created, num_batches, 2, monitor->record_dim), CFL_SIZE);

			rpos[2] = 0;
			md_copy2(4, MD_DIMS(1, 1, 1, N_vals), rstrs, &(MD_ACCESS(4, rstrs, rpos, monitor->record)), MD_STRIDES(4, MD_DIMS(1, 1, 1, N_vals), CFL_SIZE), eval, CFL_SIZE);

			rpos[2] = 1;
			md_copy2(4, MD_DIMS(1, 1, 1, N_vals), rstrs, &(MD_ACCESS(4, rstrs, rpos, monitor->record)), MD_STRIDES(4, MD_DIMS(1, 1, 1, N_vals), CFL_SIZE), vals, CFL_SIZE);

			rpos[3] += N_vals;
		}
	}

	return result;
}



static void monitor6_default_fun(struct monitor_iter6_s* _monitor, long epoch, long batch, long numbatches, float objective, long NI, const float* x[NI], char* post_string)
{
	auto monitor = CAST_DOWN(monitor_iter6_default_s, _monitor);

	bool print_progress = monitor->print_progress;
	bool print_time = monitor->print_time;
	bool print_loss = true;
	bool print_overwrite = true;

	const char* str_progress = (print_progress) ? print_progress_bar(10, batch, numbatches) :  ptr_printf("");

	double time = timestamp() - monitor->start_time;
	double est_time = time + (double)(numbatches - batch - 1) * time / (double)(batch + 1);
	const char* str_time = (print_time) ? print_time_string(time, est_time) : ptr_printf("");

	monitor->average_obj = ((batch) * monitor->average_obj + objective) / (batch + 1);

	const char* str_loss = (print_loss) ? ptr_printf(" loss: %e;", monitor->print_average_obj ? monitor->average_obj: objective) :  ptr_printf("");

	const char* str_val_monitor = compute_val_monitors(monitor, epoch, batch, numbatches, NI, x);

	const char* str_overwrite = (print_overwrite) ? ptr_printf("\33[2K\r") : ptr_printf("");


	debug_printf(DP_INFO, "%s#%d->%d/%d;%s%s%s%s%s", str_overwrite, epoch + 1, batch + 1, numbatches,
		     str_progress, str_time, str_loss, str_val_monitor, (NULL == post_string) ? "" : post_string);

	xfree(str_progress);
	xfree(str_time);
	xfree(str_loss);
	xfree(str_val_monitor);
	xfree(str_overwrite);

	if (!print_overwrite || batch + 1 == numbatches)
		debug_printf(DP_INFO, "\n");

	if (NULL != monitor->record) {

		long dims[4] = { monitor->epochs_created, numbatches, 2, monitor->record_dim };
		long pos[4] = { epoch, batch, 0, 0 };

		MD_ACCESS(4, MD_STRIDES(4, dims, sizeof(complex float)), pos, monitor->record) = 1;

		pos[2] = 1;

		MD_ACCESS(4, MD_STRIDES(4, dims, sizeof(complex float)), pos, monitor->record) = time;

		pos[2] = 0;
		pos[3] = 1;

		MD_ACCESS(4, MD_STRIDES(4, dims, sizeof(complex float)), pos, monitor->record) = 1;

		pos[2] = 1;

		MD_ACCESS(4, MD_STRIDES(4, dims, sizeof(complex float)), pos, monitor->record) = monitor->print_average_obj ? monitor->average_obj: objective;
	}

	if (batch == numbatches - 1)
		monitor->start_time = timestamp();
}

void monitor6_average_objective(struct monitor_iter6_s* _monitor)
{
	auto monitor = CAST_DOWN(monitor_iter6_default_s, _monitor);
	monitor->print_average_obj = true;
}

static void monitor6_default_free(const struct monitor_iter6_s* _monitor)
{
	auto monitor = CAST_DOWN(monitor_iter6_default_s, _monitor);

	for (int i = 0; i < monitor->num_val_monitors; i++) {

		monitor->val_monitors[i]->free(monitor->val_monitors[i]->data);

		xfree(monitor->val_monitors[i]);
	}

	xfree(monitor->val_monitors);

	md_free(monitor->record);

	xfree(_monitor);
}

void monitor_iter6_free(const struct monitor_iter6_s* monitor)
{
	monitor->free(monitor);
}


struct monitor_iter6_s* monitor_iter6_create(bool progressbar, bool record, int M, const struct monitor_value_s* val_monitors[M])
{
	PTR_ALLOC(struct monitor_iter6_default_s, monitor);
	SET_TYPEID(monitor_iter6_default_s, monitor);

	monitor->INTERFACE.fun = monitor6_default_fun;
	monitor->INTERFACE.free = monitor6_default_free;

	monitor->print_time = true;
	monitor->print_progress = progressbar;
	monitor->print_overwrite = progressbar;
	monitor->print_average_obj = false;

	monitor->start_time = timestamp();

	monitor->num_val_monitors = M;

	PTR_ALLOC(const struct monitor_value_s*[M], nval_monitors);

	for (int i = 0; i < M; i++)
		(*nval_monitors)[i] = val_monitors[i];

	monitor->val_monitors = *PTR_PASS(nval_monitors);

	monitor->use_record = record;
	monitor->epochs_written = 0;
	monitor->epochs_created = 0;
	monitor->num_batches = 0;
	monitor->record_dim = 2;

	for (int i = 0; i < M; i++)
		monitor->record_dim += val_monitors[i]->data->N_vals;

	monitor->record = NULL;

	monitor->average_obj = 0.;

	return CAST_UP(PTR_PASS(monitor));
}

struct monitor_iter6_nlop_s {

	INTERFACE(monitor_iter6_value_data_t);

	const struct nlop_s* nlop;
	bool eval_each_batch;
	const char** names;
	complex float* last_result;
};

static DEF_TYPEID(monitor_iter6_nlop_s);

static void monitor_iter6_nlop_fun(const monitor_iter6_value_data_t* data, unsigned int N, complex float vals[N], long NI, const float* args[NI])
{
        const auto d = CAST_DOWN(monitor_iter6_nlop_s, data);

	assert(nlop_get_nr_in_args(d->nlop) == NI);

	unsigned int NO = nlop_get_nr_out_args(d->nlop);

	assert(N == d->INTERFACE.N_vals);

	void* tmp_args[NI + NO];

	tmp_args[0] = md_alloc_sameplace(1, MD_DIMS(d->INTERFACE.N_vals), CFL_SIZE, args[0]);

	for(unsigned int o = 1; o < NO; o++) {

		auto iov = nlop_generic_codomain(d->nlop, o - 1);

		tmp_args[o] = ((complex float*)tmp_args[o - 1]) + md_calc_size(iov->N, iov->dims);
	}

	for (int i = 0; i < NI; i++)
		tmp_args[NO + i] = (void*)args[i];

	nlop_generic_apply_select_derivative_unchecked(d->nlop, NI + NO, tmp_args, 0, 0);

	md_copy(1, MD_DIMS(d->INTERFACE.N_vals), d->last_result, tmp_args[0], CFL_SIZE);
	md_copy(1, MD_DIMS(d->INTERFACE.N_vals), vals, tmp_args[0], CFL_SIZE);

	md_free(tmp_args[0]);
}

static bool monitor_iter6_nlop_eval(const monitor_iter6_value_data_t* _data, long epoch, long batch, long num_batches)
{
	const auto d = CAST_DOWN(monitor_iter6_nlop_s, _data);
	UNUSED(epoch);

	return d->eval_each_batch || (num_batches == batch + 1);
}

static const char* monitor_iter6_nlop_print(const monitor_iter6_value_data_t* _data)
{
	const auto d = CAST_DOWN(monitor_iter6_nlop_s, _data);

	const char* result = ptr_printf("");

	if (NULL == d->names)
		return result;

	for (unsigned int i = 0; i < d->INTERFACE.N_vals; i++) {

		auto tmp = result;

		if (0. == cimagf(d->last_result[i]))
			result = ptr_printf("%s %s: %.3e;", tmp, d->names[i], crealf(d->last_result[i]));
		else
			result = ptr_printf("%s %s: %.3e + %.3ei;", tmp, d->names[i], crealf(d->last_result[i]), cimagf(d->last_result[i]));

		xfree(tmp);
	}

	return result;
}

static void monitor_iter6_nlop_free(const monitor_iter6_value_data_t* _data)
{
	const auto d = CAST_DOWN(monitor_iter6_nlop_s, _data);

	nlop_free(d->nlop);

	for (unsigned int i = 0; i < d->INTERFACE.N_vals; i++)
		xfree(d->names[i]);

	xfree(d->names);
	xfree(d->last_result);
	xfree(d);
}


struct monitor_value_s* monitor_iter6_nlop_create(const struct nlop_s* nlop, _Bool eval_each_batch, unsigned int N, const char* print_name[N])
{
	PTR_ALLOC(struct monitor_iter6_nlop_s, data);
	SET_TYPEID(monitor_iter6_nlop_s, data);

	data->nlop = nlop_clone(nlop);
	data->eval_each_batch = eval_each_batch;

	data->INTERFACE.N_vals = 0;

	for (int i = 0; i < nlop_get_nr_out_args(nlop); i++)
		data->INTERFACE.N_vals += md_calc_size(nlop_generic_codomain(nlop, i)->N, nlop_generic_codomain(nlop, i)->dims);

	data->last_result = md_alloc(1, MD_DIMS(data->INTERFACE.N_vals), CFL_SIZE);
	data->names = NULL;

	if (NULL != print_name) {

		assert(N == data->INTERFACE.N_vals);

		PTR_ALLOC(const char*[N], tmp_names);

		for (unsigned int i = 0; i < N; i++) {

			if ((NULL != print_name[i]) && (0 < strlen(print_name[i]))) {

				PTR_ALLOC(char[strlen(print_name[i]) + 1], tmp_name);

				strcpy(*tmp_name, print_name[i]);
				(*tmp_names)[i] = *PTR_PASS(tmp_name);

			} else {

				(*tmp_names)[i] = NULL;
			}
		}

		data->names = *PTR_PASS(tmp_names);
	}

	PTR_ALLOC(struct monitor_value_s, monitor);

	monitor->data = CAST_UP(PTR_PASS(data));
	monitor->fun = monitor_iter6_nlop_fun;
	monitor->eval = monitor_iter6_nlop_eval;
	monitor->print = monitor_iter6_nlop_print;
	monitor->free = monitor_iter6_nlop_free;

	return PTR_PASS(monitor);
}

struct monitor_iter6_function_s {

	INTERFACE(monitor_iter6_value_data_t);

	monitor_iter6_value_by_function_t fun;

	bool eval_each_batch;
	const char* name;
	complex float last_result;
};

static DEF_TYPEID(monitor_iter6_function_s);

static void monitor_iter6_function_fun(const monitor_iter6_value_data_t* data, unsigned int N, complex float vals[N], long NI, const float* args[NI])
{
	const auto d = CAST_DOWN(monitor_iter6_function_s, data);
	assert(1 == N);

	d->last_result = d->fun(NI, args);

	vals[0] = d->last_result;
}

static bool monitor_iter6_function_eval(const monitor_iter6_value_data_t* _data, long epoch, long batch, long num_batches)
{
	const auto d = CAST_DOWN(monitor_iter6_function_s, _data);
	UNUSED(epoch);

	return d->eval_each_batch || (num_batches == batch + 1);
}

static const char* monitor_iter6_function_print(const monitor_iter6_value_data_t* _data)
{
	const auto d = CAST_DOWN(monitor_iter6_function_s, _data);

	if (NULL == d->name)
		return ptr_printf("");

	if (0. == cimagf(d->last_result))
		return ptr_printf(" %s: %.3e;", d->name, crealf(d->last_result));
	else
		return ptr_printf(" %s: %.3e + %.3ei;", d->name, crealf(d->last_result), cimagf(d->last_result));
}

static void monitor_iter6_function_free(const monitor_iter6_value_data_t* _data)
{
	const auto d = CAST_DOWN(monitor_iter6_function_s, _data);

	xfree(d->name);
	xfree(d);
}


struct monitor_value_s* monitor_iter6_function_create(monitor_iter6_value_by_function_t fun, _Bool eval_each_batch, const char* print_name)
{
	PTR_ALLOC(struct monitor_iter6_function_s, data);
	SET_TYPEID(monitor_iter6_function_s, data);

	data->INTERFACE.N_vals = 1;

	data->fun = fun;
	data->eval_each_batch = eval_each_batch;
	data->name = NULL;

	if ((NULL != print_name) && (0 < strlen(print_name))) {

		PTR_ALLOC(char[strlen(print_name) + 1], tmp_name);

		strcpy(*tmp_name, print_name);

		data->name = *PTR_PASS(tmp_name);
	}

	PTR_ALLOC(struct monitor_value_s, monitor);

	monitor->data = CAST_UP(PTR_PASS(data));
	monitor->fun = monitor_iter6_function_fun;
	monitor->eval = monitor_iter6_function_eval;
	monitor->print = monitor_iter6_function_print;
	monitor->free = monitor_iter6_function_free;

	return PTR_PASS(monitor);
}

