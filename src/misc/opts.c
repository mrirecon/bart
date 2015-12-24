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

enum OPT_ARG_TYPE { OPT_SPECIAL, OPT_SET, OPT_CLEAR, OPT_INT, OPT_LONG, OPT_FLOAT, OPT_STRING };

static const char* opt_arg_types[] = { " ...", "", "", " d", " d", " f", " <string>" };

static enum OPT_ARG_TYPE opt_arg_type(opt_conv_f fun)
{
	if (opt_set == fun)
		return OPT_SET;

	if (opt_clear == fun)
		return OPT_CLEAR;

	if (opt_int == fun)
		return OPT_INT;

	if (opt_long == fun)
		return OPT_LONG;

	if (opt_float == fun)
		return OPT_FLOAT;

	if (opt_string == fun)
		return OPT_STRING;

	return OPT_SPECIAL;
}


static void print_usage(FILE* fp, const char* name, const char* usage_str, int n, const struct opt_s opts[static n])
{
	fprintf(fp, "Usage: %s ", name);

	for (int i = 0; i < n; i++)
		if (NULL != opts[i].descr)
			fprintf(fp, "[-%c%s] ", opts[i].c, opt_arg_types[opt_arg_type(opts[i].conv)]);

	fprintf(fp, "%s\n", usage_str);
}

static void print_help(const char* help_str, int n, const struct opt_s opts[n])
{
	printf("\n%s\n\n",  help_str);

	for (int i = 0; i < n; i++)
		if (NULL != opts[i].descr)
			printf("-%c%s\n", opts[i].c, opts[i].descr);

	printf("-h\thelp\n");
}


void cmdline(int* argcp, char* argv[], int min_args, int max_args, const char* usage_str, const char* help_str, int n, const struct opt_s opts[n])
{
	int argc = *argcp;
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

			print_usage(stdout, argv[0], usage_str, n, opts);
			print_help(help_str, n, opts);
			exit(0);
		}

		for (int i = 0; i < n; i++) {

			if (opts[i].c == c) {

				if (opts[i].conv(opts[i].ptr, c, optarg)) {

					print_usage(stderr, argv[0], usage_str, n, opts);
					exit(1);
				}

				goto out;
			}
		}

		print_usage(stderr, argv[0], usage_str, n, opts);
		exit(1);

	out:	continue;
	}

	if (	   (argc - optind < min_args)
		|| (argc - optind > max_args)) {

		print_usage(stderr, argv[0], usage_str, n, opts);
		exit(1);
	}

	int i;
	for (i = optind; i < argc; i++)
		argv[i - optind + 1] = argv[i];

	argv[i] = NULL;
	*argcp = argc - optind + 1;
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

extern bool opt_long(void* ptr, char c, const char* optarg)
{
	UNUSED(c); UNUSED(optarg);
	*(long*)ptr = atoi(optarg);
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
	assert(NULL != ptr);
	return false;
}


