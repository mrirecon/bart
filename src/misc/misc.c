/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2011-2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdarg.h>
#include <getopt.h>

#include "misc/debug.h"
#include "misc.h"


void* xmalloc(size_t s)
{
	void* p = malloc(s);

	if (NULL == p)
		error("Could not allocate memory.\n");

	return p;
}




void error(const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	if (!debug_logging)
		debug_printf(DP_ERROR, "Error: ");
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
	debug_printf(dblevel, "[");

	for (int i = 0; i < D; i++)
		debug_printf(dblevel, "%3ld ", dims[i]);

	debug_printf(dblevel, "]\n");
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



void mini_cmdline(int argc, char* argv[], int expected_args, const char* usage_str, const char* help_str)
{
	mini_cmdline_bool(argc, argv, '\0', expected_args, usage_str, help_str);
}


bool mini_cmdline_bool(int argc, char* argv[], char flag_char, int expected_args, const char* usage_str, const char* help_str)
{
	bool flag = false;
	char opts[3] = { 'h', flag_char, '\0' };

	int c;
	while (-1 != (c = getopt(argc, argv, opts))) {

		if (c == flag_char) {

			flag = true;

		}  else
		switch (c) {

		case 'h':
			printf("Usage: %s %s\n\n%s", argv[0], usage_str, help_str);
			exit(0);

		default:
			fprintf(stderr, "Usage: %s %s\n", argv[0], usage_str);
			exit(1);
		}
	}

	if (!(     ((expected_args >= 0) && (argc - optind == expected_args))
		|| ((expected_args <  0) && (argc - optind >= -expected_args)))) {

		fprintf(stderr, "Usage %s: %s\n", argv[0], usage_str);
		exit(1);
	}

	int i;
	for (i = optind; i < argc; i++)
		argv[i - optind + 1] = argv[i];

	argv[i] = NULL;

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
