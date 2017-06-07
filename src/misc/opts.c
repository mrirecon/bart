/* Copyright 2015-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#define _GNU_SOURCE
#include <getopt.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "opts.h"

enum OPT_ARG_TYPE { OPT_SPECIAL, OPT_SET, OPT_CLEAR, OPT_INT, OPT_UINT, OPT_LONG, OPT_FLOAT, OPT_STRING };

static const char* opt_arg_types[] = { " ...", "", "", " d", " d", " d", " f", " <string>" };

static enum OPT_ARG_TYPE opt_arg_type(opt_conv_f fun)
{
	if (opt_set == fun)
		return OPT_SET;

	if (opt_clear == fun)
		return OPT_CLEAR;

	if (opt_int == fun)
		return OPT_INT;

	if (opt_uint == fun)
		return OPT_UINT;

	if (opt_long == fun)
		return OPT_LONG;

	if (opt_float == fun)
		return OPT_FLOAT;

	if (opt_string == fun)
		return OPT_STRING;

	return OPT_SPECIAL;
}

static const char* trim_space(const char* str)
{
	while (isspace(*str))
		str++;

	return str;
}

static bool show_option_p(const struct opt_s opt)
{
	return (NULL != opt.descr) && (')' != opt.descr[strlen(opt.descr) - 1]);
}

static void print_usage(FILE* fp, const char* name, const char* usage_str, int n, const struct opt_s opts[static n ?: 1])
{
	fprintf(fp, "Usage: %s ", name);

	for (int i = 0; i < n; i++)
		if (show_option_p(opts[i]))
			fprintf(fp, "[-%c%s] ", opts[i].c, opt_arg_types[opt_arg_type(opts[i].conv)]);

	fprintf(fp, "%s\n", usage_str);
}


static const char* add_space(bool has_arg, bool has_space)
{
	const char* space = "\t\t";

	if (has_arg)
		space = " ";

	if (!has_space)
		space = "";

	return space;
}


static void print_help(const char* help_str, int n, const struct opt_s opts[n ?: 1])
{
	printf("\n%s\n\n",  help_str);

	for (int i = 0; i < n; i++)
		if (show_option_p(opts[i]))
			printf("-%c%s%s\n", opts[i].c,
					add_space(opts[i].arg, isspace(opts[i].descr[0])),
					trim_space(opts[i].descr));

	printf("-h\t\thelp\n");
}


static void check_options(int n, const struct opt_s opts[n ?: 1])
{
	bool f[256] = { false };

	for (int i = 0; i < n; i++) {

		assert(256 > (unsigned int)opts[i].c);

		if (f[(unsigned int)opts[i].c])
			error("duplicate option: %c\n", opts[i].c);

		f[(unsigned int)opts[i].c] = true;
	}
}


static void process_option(char c, const char* optarg, const char* name, const char* usage_str, const char* help_str, int n, const struct opt_s opts[n ?: 1])
{
	if ('h' == c) {

		print_usage(stdout, name, usage_str, n, opts);
		print_help(help_str, n, opts);
		exit(0);
	}

	for (int i = 0; i < n; i++) {

		if (opts[i].c == c) {

			if (opts[i].conv(opts[i].ptr, c, optarg)) {

				print_usage(stderr, name, usage_str, n, opts);
				exit(1);
			}

			return;
		}
	}

	print_usage(stderr, name, usage_str, n, opts);
	exit(1);
}

void cmdline(int* argcp, char* argv[], int min_args, int max_args, const char* usage_str, const char* help_str, int n, const struct opt_s opts[n ?: 1])
{
	int argc = *argcp;
	char optstr[2 * n + 2];

	check_options(n, opts);

	save_command_line(argc, argv);

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
#if 0
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
#else
		process_option(c, optarg, argv[0], usage_str, help_str, n, opts);
#endif
	}

	if (	   (argc - optind < min_args)
		|| (argc - optind > max_args)) {

		print_usage(stderr, argv[0], usage_str, n, opts);
		exit(1);
	}

	int i;
	for (i = optind; i < argc; i++)
		argv[i - optind + 1] = argv[i];

	*argcp = argc - optind + 1;
}


bool opt_set(void* ptr, char c, const char* optarg)
{
	UNUSED(c); UNUSED(optarg);
	*(bool*)ptr = true;
	return false;
}

bool opt_clear(void* ptr, char c, const char* optarg)
{
	UNUSED(c); UNUSED(optarg);
	*(bool*)ptr = false;
	return false;
}

bool opt_int(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	*(int*)ptr = atoi(optarg);
	return false;
}

bool opt_uint(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	*(unsigned int*)ptr = atoi(optarg);
	return false;
}

bool opt_long(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	*(long*)ptr = atoi(optarg);
	return false;
}

bool opt_float(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	*(float*)ptr = atof(optarg);
	return false;
}

bool opt_string(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	*(char**)ptr = strdup(optarg);
	assert(NULL != ptr);
	return false;
}


bool opt_float_vec3(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	int r = sscanf(optarg, "%f:%f:%f", &(*(float(*)[3])ptr)[0], &(*(float(*)[3])ptr)[1], &(*(float(*)[3])ptr)[2]);

	assert(3 == r);

	return false;
}


bool opt_vec3(void* ptr, char c, const char* optarg)
{
	if (islower(c)) {

		if (3 != sscanf(optarg, "%ld:%ld:%ld", &(*(long(*)[3])ptr)[0], &(*(long(*)[3])ptr)[1], &(*(long(*)[3])ptr)[2])) {

			(*(long(*)[3])ptr)[0] = atol(optarg);
			(*(long(*)[3])ptr)[1] = atol(optarg);
			(*(long(*)[3])ptr)[2] = atol(optarg);
		}

	} else {

		debug_printf(DP_WARN, "the upper-case options for specifying dimensions are deprecated.\n");
		int r = sscanf(optarg, "%ld:%ld:%ld", &(*(long(*)[3])ptr)[0], &(*(long(*)[3])ptr)[1], &(*(long(*)[3])ptr)[2]);
		assert(3 == r);
	}

	return false;
}

bool opt_select(void* ptr, char c, const char* optarg)
{
	UNUSED(c); UNUSED(optarg);
	struct opt_select_s* sel = ptr;

	if (0 != memcmp(sel->ptr, sel->default_value, sel->size))
		return true;

	memcpy(sel->ptr, sel->value, sel->size);
	return false;
}

bool opt_subopt(void* _ptr, char c, const char* optarg)
{
	UNUSED(c);
	struct opt_subopt_s* ptr = _ptr;

	process_option(optarg[0], optarg + 1, "", "", "", ptr->n, ptr->opts);
	return false;
}


