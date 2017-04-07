/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc.h"


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

	debug_vprintf(DP_ERROR, fmt, ap);

	va_end(ap);
	abort();
}



void print_dims(int D, const long dims[D])
{
	printf("[");

	for (int i = 0; i < D; i++)
		printf("%3ld ", dims[i]);

	printf("]\n");
}



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




void quicksort(unsigned int N, unsigned int ord[N], const void* data, quicksort_cmp_t cmp)
{
	if (N < 2)
		return;

	unsigned int pivot = ord[N / 2];
	unsigned int l = 0;
	unsigned int h = N - 1;

	while (l <= h) {

		if (cmp(data, ord[l], pivot) < 0) {

			l++;
			continue;
		}

		if (cmp(data, ord[h], pivot) > 0) {

			h--;
			continue;
		}

		unsigned int swap = ord[l];
		ord[l] = ord[h];
		ord[h] = swap;

		l++;
		h--;
	}

	if (h + 1 > 0)
		quicksort(h + 1, ord, data, cmp);

	if (N > l)
		quicksort(N - l, ord + l, data, cmp);
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

void save_command_line(int argc, char* argv[])
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
		free((void*)qargv[i]);
		(*buf)[pos++] = ' ';
	}

	(*buf)[pos] = '\0';

	command_line = (*buf);
}



void mini_cmdline(int argc, char* argv[], int expected_args, const char* usage_str, const char* help_str)
{
	mini_cmdline_bool(argc, argv, '\0', expected_args, usage_str, help_str);
}


bool mini_cmdline_bool(int argc, char* argv[], char flag_char, int expected_args, const char* usage_str, const char* help_str)
{
	bool flag = false;
	struct opt_s opts[1] = { { flag_char, false, opt_set, &flag, NULL } };

	char* help = strdup(help_str);

	int hlen = strlen(help);

	if ((hlen > 1) && ('\n' == help[hlen - 1]))
		help[hlen - 1] = '\0';

	int min_args = expected_args;
	int max_args = expected_args;

	if (expected_args < 0) {

		min_args = -expected_args;
		max_args = 1000;
	}

	cmdline(&argc, argv, min_args, max_args, usage_str, help, 1, opts);

	free(help);

	return flag;
}


void print_long(unsigned int D, const long arr[D])
{
	for (unsigned int i = 0; i < D; i++)
		printf("arr[%i] = %ld\n", i, arr[i]);
}

void print_float(unsigned int D, const float arr[D])
{
	for (unsigned int i = 0; i < D; i++)
		printf("arr[%i] = %f\n", i, arr[i]);
}

void print_int(unsigned int D, const int arr[D])
{
	for (unsigned int i = 0; i < D; i++)
		printf("arr[%i] = %i\n", i, arr[i]);
}

void print_complex(unsigned int D, const complex float arr[D])
{
	for (unsigned int i = 0; i < D; i++)
		printf("arr[%i]: real = %f, imag = %f\n", i, crealf(arr[i]), cimagf(arr[i]));
}


unsigned int bitcount(unsigned int flags)
{
	unsigned int N = 0;

	for (; flags > 0; N++)
		flags &= (flags - 1);

	return N;
}
