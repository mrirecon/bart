/* Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#define _GNU_SOURCE
#include <getopt.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "misc/misc.h"

#include "opts.h"


static void print_usage(const char* name, const char* usage_str)
{
	fprintf(stderr, "Usage: %s %s\n", name, usage_str);
}

static void print_help(const char* name, const char* usage_str, const char* help_str, int n, const struct opt_s opts[n])
{
	printf("Usage: %s %s\n\n%s\n", name, usage_str, help_str);

	for (int i = 0; i < n; i++)
		if (NULL != opts[i].descr)
			printf("-%c\t%s\n", opts[i].c, opts[i].descr);
}


void cmdline(int argc, char* argv[], int expected_args, const char* usage_str, const char* help_str, int n, const struct opt_s opts[n])
{
	char optstr[2 * n + 2];

	int l = 0;
	optstr[l++] = 'h';

	for (int i = 0; i < n; i++) {

		optstr[l++] = opts[i].c;

		if (opts[i].arg)
			optstr[l++] = ':';
	}

	optstr[l] = '\0';

	int c;
	while (-1 != (c = getopt(argc, argv, optstr))) {

		if ('h' == c) {

			print_help(argv[0], usage_str, help_str, n, opts);
			exit(0);
		}

		for (int i = 0; i < n; i++) {

			if (opts[i].c == c) {

				if (!opts[i].conv(opts[i].ptr, c, optarg)) {

					print_usage(argv[0], usage_str);
					exit(1);
				}

				goto out;
			}
		}

		print_usage(argv[0], usage_str);
		exit(1);

	out:	continue;
	}

	if (!(     ((expected_args >= 0) && (argc - optind == expected_args))
		|| ((expected_args <  0) && (argc - optind >= -expected_args)))) {

		print_usage(argv[0], usage_str);
		exit(1);
	}

	int i;
	for (i = optind; i < argc; i++)
		argv[i - optind + 1] = argv[i];

	argv[i] = NULL;
}


extern bool opt_set(void* ptr, char c, const char* optarg)
{
	UNUSED(c); UNUSED(optarg);
	*(bool*)ptr = true;
	return false;
}

extern bool opt_clear(void* ptr, char c, const char* optarg)
{
	UNUSED(c); UNUSED(optarg);
	*(bool*)ptr = false;
	return false;
}

extern bool opt_int(void* ptr, char c, const char* optarg)
{
	UNUSED(c); UNUSED(optarg);
	*(int*)ptr = atoi(optarg);
	return false;
}

extern bool opt_float(void* ptr, char c, const char* optarg)
{
	UNUSED(c); UNUSED(optarg);
	*(float*)ptr = atof(optarg);
	return false;
}

extern bool opt_string(void* ptr, char c, const char* optarg)
{
	UNUSED(c); UNUSED(optarg);
	*(char**)ptr = strdup(optarg);
	return false;
}


