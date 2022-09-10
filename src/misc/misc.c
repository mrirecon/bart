/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2015-2018. Martin Uecker.
 * Copyright 2017. University of Oxford.
 * Copyright 2017-2018. Damien Nguyen
 * Copyright 2019-2022. Uecker Lab, University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2017 Sofia Dimoudi <sofia.dimoudi@cardiov.ox.ac.uk>
 * 2017-2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
 * Christian Holme <christian.holme@med.uni-goettingen.de>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <setjmp.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <limits.h>

#ifdef BART_WITH_PYTHON
#include <Python.h>
#endif

#include "misc/debug.h"
#include "misc/nested.h"
#include "misc/opts.h"
#include "misc.h"


#ifndef isnanf
#define isnanf(X) isnan(X)
#endif



struct error_jumper_s {

	bool initialized;
	jmp_buf buf;
};

extern struct error_jumper_s error_jumper;	// FIXME should not be extern

struct error_jumper_s error_jumper = { .initialized = false };






void* xmalloc(size_t s)
{
	void* p = malloc(s);

	if (NULL == p)
		error("Could not allocate memory.\n");

	return p;
}



void xfree(const void* x)
{
	free((void*)x);
}


void warn_nonnull_ptr(void* p)
{
	void** p2 = p;

	if (NULL != *p2) {

		debug_printf(DP_WARN, "pointer not cleared: ");
		debug_backtrace(1);
	}
}


void error(const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);

#ifndef BART_WITH_PYTHON
#ifdef USE_LOG_BACKEND
	debug_vprintf_trace("error", __FILE__, __LINE__, DP_ERROR, fmt, ap);
#else
	debug_vprintf(DP_ERROR, fmt, ap);
#endif
#else
	char err[1024] = { 0 };

	if (NULL == PyErr_Occurred()) {

		vsnprintf(err, 1023, fmt, ap);
		PyErr_SetString(PyExc_RuntimeError, err);
	}
	// No else required as the error indicator has already been set elsewhere
#endif /* !BART_WITH_PYTHON */

	va_end(ap);

	if (error_jumper.initialized)
		longjmp(error_jumper.buf, 1);

	abort();
}


int error_catcher(int fun(int argc, char* argv[argc]), int argc, char* argv[argc])
{
	int ret = -1;

	error_jumper.initialized = true;

	if (0 == setjmp(error_jumper.buf))
		ret = fun(argc, argv);

	error_jumper.initialized = false;

	return ret;
}


extern FILE* bart_output;
FILE* bart_output = NULL;

int bart_printf(const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);

	FILE* out = bart_output;

	if (NULL == bart_output)
		out = stdout;

	int ret = vfprintf(out, fmt, ap);
	va_end(ap);

	return ret;
}


void print_dims(int D, const long dims[D])
{
	printf("[");

	for (int i = 0; i < D; i++)
		printf("%3ld ", dims[i]);

	printf("]\n");
}



void debug_print_bits(int dblevel, int D, unsigned long bitmask)
{
	bool dbl = debug_logging;
	debug_logging = false;
	debug_printf(dblevel, "[");

	unsigned long curr_bit = 1;

	while (curr_bit < (1ul << (D - 1))) {

		debug_printf(dblevel, "%u", bitmask & curr_bit ? 1 : 0);
		curr_bit = curr_bit << 1;
	}

	debug_printf(dblevel, "]\n");
	debug_logging = dbl;
}


#ifdef REDEFINE_PRINTF_FOR_TRACE
#undef debug_print_dims
#endif

void debug_print_dims(int dblevel, int D, const long dims[D])
{
	bool dbl = debug_logging;
	debug_logging = false;
	debug_printf(dblevel, "[");

	for (int i = 0; i < D; i++)
		debug_printf(dblevel, "%3ld ", dims[i]);

	debug_printf(dblevel, "]\n");
	debug_logging = dbl;
}




void debug_print_dims_trace(const char* func_name,
			    const char* file,
			    int line,
			    int dblevel,
			    int D,
			    const long dims[D])
{
	bool dbl = debug_logging;
	debug_logging = false;
	debug_printf_trace(func_name, file, line, dblevel, "[");

	for (int i = 0; i < D; i++)
		debug_printf_trace(func_name, file, line, dblevel, "%3ld ", dims[i]);

	debug_printf_trace(func_name, file, line, dblevel, "]\n");
	debug_logging = dbl;
}



int parse_cfl(complex float res[1], const char* str)
{
	char* tail;
	float re = strtof(str, &tail);
	float im = 0.;

	if (str == tail)
		return -1;

	if ('\0' == tail[0])
		goto ok;

	if (('i' == tail[0]) && ('\0' == tail[1])) {

		im = re;
		re = 0;
		goto ok;
	}

	str = tail;
	im = strtof(str, &tail);

	if (('i' != tail[0]) || ('\0' != tail[1]))
		return -1;
ok:
	res[0] = re + 1.i * im;
	return 0;
}


int parse_long(long res[1], const char* str)
{
	char* tail;
	long l = strtol(str, &tail, 10);

	if ('\0' != tail[0])
		return -1;
	res[0] = l;
	return 0;
}


int parse_int(int res[1], const char* str)
{
	long val;
	if (0 != parse_long(&val, str))
		return -1;

	if (val < INT_MIN || val > INT_MAX)
		error("Value %ld too large for int!\n", val);

	res[0] = (int) val;
	return 0;
}

void quicksort(int N, int ord[N], quicksort_cmp_t cmp)
{
	if (N < 2)
		return;

	int pivot = ord[N / 2];
	int l = 0;
	int h = N - 1;

	while (l <= h) {

		if (cmp(ord[l], pivot) < 0) {

			l++;
			continue;
		}

		if (cmp(ord[h], pivot) > 0) {

			h--;
			continue;
		}

		int swap = ord[l];
		ord[l] = ord[h];
		ord[h] = swap;

		l++;
		h--;
	}

	if (h + 1 > 0)
		quicksort(h + 1, ord, cmp);

	if (N > l)
		quicksort(N - l, ord + l, cmp);
}



/**
 * Quickselect adapted from §8.5 in Numerical Recipes in C,
 * The Art of Scientific Computing
 * Second Edition, William H. Press, 1992.
 * Sort descending of an array of floats, stop at k largest element.
 *
 * @param arr array of floats, input
 * @param n total number of elements in input array
 * @param k the rank of the element to be selected in the sort
 *
 * @returns the k-th largest float in the array
 *
 * @note In-place sort. The input array contents are not preserved in their original order.
 */
float quickselect(float *arr, int n, int k)
{
	long i, ir, j, l, mid;
	float a;

	l = 0;
	ir = n - 1;

	for(;;) {

		if (ir <= l + 1) {

			if ((ir == l + 1) && (arr[ir] > arr[l]))
				SWAP(arr[l], arr[ir]);

			return arr[k];
		}

		mid = (l + ir) / 2;

		SWAP(arr[mid], arr[l + 1]);

		if (arr[l] < arr[ir])
			SWAP(arr[l], arr[ir]);

		if (arr[l + 1] < arr[ir])
			SWAP(arr[l + 1], arr[ir]);

		if (arr[l] < arr[l + 1])
			SWAP(arr[l], arr[l + 1]);

		i = l + 1;
		j = ir;
		a = arr[l + 1];

		for (;;) {

			do i++; while (arr[i] > a);
			do j--; while (arr[j] < a);

			if (j < i)
				break;

			SWAP(arr[i], arr[j]);
		}

		arr[l + 1] = arr[j];
		arr[j] = a;

		if (j >= k)
			ir = j - 1;

		if (j <= k)
			l = i;
	}
}

/**
 *
 * Same as quickselect, but the input is a complex array
 * and the absolute value of the k-th largest element is returned.
 * Possibly faster for application to complex arrays.
 *
 */
float quickselect_complex(complex float* arr, int n, int k)
{
	long i, ir, j, l, mid;
	float a;
	complex float ca;

	l = 0;
	ir = n - 1;

	for(;;) {

		if (ir <= l + 1) {

			if ((ir == l + 1) && (cabsf(arr[ir]) > cabsf(arr[l])))
				SWAP(arr[l], arr[ir]);

			return cabsf(arr[k]);
		}

		mid = (l + ir) / 2;

		SWAP(arr[mid], arr[l + 1]);

		if (cabsf(arr[l]) < cabsf(arr[ir]))
			SWAP(arr[l], arr[ir]);

		if (cabsf(arr[l + 1]) < cabsf(arr[ir]))
			SWAP(arr[l + 1], arr[ir]);

		if (cabsf(arr[l]) < cabsf(arr[l + 1]))
			SWAP(arr[l], arr[l + 1]);

		i = l + 1;
		j = ir;
		a = cabsf(arr[l + 1]);
		ca = arr[l + 1];

		for (;;) {

			do i++; while (cabsf(arr[i]) > a);
			do j--; while (cabsf(arr[j]) < a);

			if (j < i)
				break;

			SWAP(arr[i], arr[j]);
		}

		arr[l + 1] = arr[j];
		arr[j] = ca;

		if (j >= k)
			ir = j - 1;

		if (j <= k)
			l = i;
	}
}

static const char* quote(const char* str)
{
	int i = 0;
	int j = 0;
	int c;
	bool flag = false;

	while ('\0' != (c = str[i++])) {

		if (isspace(c))
			flag = true;

		switch (c) {
		case '\\':
		case '\'':
		case '"':
		case '$':
			j++;
			/* fall through */
		default:
			break;
		}
	}

	if ((!flag) && (0 == j))
		return strdup(str);

	int len = strlen(str);
	char (*qstr)[len + j + 3] = TYPE_ALLOC(char[len + j + 3]);

	i = 0;
	j = 0;

	(*qstr)[j++] = '\"';

	while ('\0' != (c = str[i++])) {

		switch (c) {
		case '\\':
		case '\'':
		case '"':
		case '$':
			(*qstr)[j++] = '\'';
			/* fall through */
		default:
			(*qstr)[j++] = c;
		}
	}

	(*qstr)[j++] = '\"';
	(*qstr)[j++] = '\0';

	return *qstr;
}

const char* command_line = NULL;

void save_command_line(int argc, char* argv[static argc])
{
	size_t len = 0;
	const char* qargv[argc];

	for (int i = 0; i < argc; i++) {

		qargv[i] = quote(argv[i]);
		len += strlen(qargv[i]) + 1;
	}

	char (*buf)[len + 1] = TYPE_ALLOC(char[len + 1]);

	size_t pos = 0;

	for (int i = 0; i < argc; i++) {

		strcpy((*buf) + pos, qargv[i]);
		pos += strlen(qargv[i]);
		xfree(qargv[i]);
		(*buf)[pos++] = ' ';
	}

	(*buf)[pos] = '\0';

	XFREE(command_line);

	command_line = (*buf);
}



void print_long(int D, const long arr[D])
{
	for (int i = 0; i < D; i++)
		printf("arr[%i] = %ld\n", i, arr[i]);
}

void print_float(int D, const float arr[D])
{
	for (int i = 0; i < D; i++)
		printf("arr[%i] = %f\n", i, arr[i]);
}

void print_int(int D, const int arr[D])
{
	for (int i = 0; i < D; i++)
		printf("arr[%i] = %i\n", i, arr[i]);
}

void print_complex(int D, const complex float arr[D])
{
	for (int i = 0; i < D; i++)
		printf("arr[%i]: real = %f, imag = %f\n", i, crealf(arr[i]), cimagf(arr[i]));
}


int bitcount(unsigned long flags)
{
	int N = 0;

	for (; flags > 0; N++)
		flags &= (flags - 1);

	return N;
}

 __attribute__((optimize("-fno-finite-math-only")))
bool safe_isnanf(float x)
{
	return isnanf(x);
}

__attribute__((optimize("-fno-finite-math-only")))
bool safe_isfinite(float x)
{
	return (!isnan(x) && !isinf(x));
	// return isfinite(x); <- is sometimes true when x is NaN.
}


static bool long_mul_overflow_p(long a, long b)
{
	bool of = false;

	of |= (a > 0) && (b > 0) && (a > LONG_MAX / b);
	of |= (a > 0) && (b < 0) && (b < LONG_MIN / a);
	of |= (a < 0) && (b > 0) && (a < LONG_MIN / b);
	of |= (a < 0) && (b < 0) && (b < LONG_MAX / a);

	return of;
}

long io_calc_size(int D, const long dims[D?:1], size_t size)
{
	if (0 == D)
		return size;

	long a = io_calc_size(D - 1, dims + 1, size);
	long b = dims[0];

	if ((a < 0) || (b < 0))
		return -1;

	if (long_mul_overflow_p(a, b))
		return -1;

	return a * b;
}

const char* ptr_vprintf(const char* fmt, va_list ap)
{
	va_list ap1;
	va_copy(ap1, ap);

	ssize_t len = vsnprintf(NULL, 0, fmt, ap1);

	va_end(ap1);

	if (len < 0)
		abort();

	PTR_ALLOC(char[len + 1], result);

	vsprintf(*result, fmt, ap);

	return *PTR_PASS(result);
}

const char* ptr_printf(const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);

	auto result = ptr_vprintf(fmt, ap);

	va_end(ap);

	return result;
}

const char* ptr_print_dims(int D, const long dims[D])
{
	const char* result = ptr_printf("[");

	for (int i = 0; i < D; i++) {

		const char* tmp = ptr_printf("%s%3ld ", result, dims[i]);

		xfree(result);

		result = tmp;
	}

	const char* tmp = ptr_printf("%s]", result);

	xfree(result);

	return tmp;
}
