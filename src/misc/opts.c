/* Copyright 2018-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2015-2017. Martin Uecker.
 * Copyright 2017-2018. Damien Nguyen.
 * Copyright 2017-2018. Francesco Santini.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2017-2018 Nguyen Damien <damien.nguyen@alumni.epfl.ch>
 * 2017-2018 Francesco Santini <francesco.santini@unibas.ch>
 */

#define _GNU_SOURCE
#include "ya_getopt.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

#include <complex.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/io.h"
#include "misc/list.h"

#include "opts.h"

list_t str_list = NULL;

void opt_free_strdup(void)
{
	if (NULL == str_list)
		return;

	const char* str = list_pop(str_list);
	while (NULL != str) {

		xfree(str);
		str = list_pop(str_list);
	}

	list_free(str_list);
	str_list = NULL;
}

opt_conv_f opt_set;
opt_conv_f opt_clear;
opt_conv_f opt_uint;
opt_conv_f opt_int;
opt_conv_f opt_ulong;
opt_conv_f opt_long;
opt_conv_f opt_float;
opt_conv_f opt_cfl;
opt_conv_f opt_string;
opt_conv_f opt_infile;
opt_conv_f opt_outfile;
opt_conv_f opt_inoutfile;
opt_conv_f opt_vec2;
opt_conv_f opt_float_vec2;
opt_conv_f opt_vec3;
opt_conv_f opt_float_vec3;
opt_conv_f opt_float_vec4;
opt_conv_f opt_select;
opt_conv_f opt_subopt;

static const char* opt_arg_str(enum OPT_TYPE type)
{
	switch(type) {

	case OPT_SPECIAL:
	case OPT_SUBOPT:
		return "...";

	case OPT_SELECT:
		return "";

	case OPT_SET:
	case OPT_CLEAR:
		return "";

	case OPT_INT:
	case OPT_UINT:
	case OPT_ULONG:
	case OPT_LONG:
		return "d";

	case OPT_FLOAT:
		return "f";

	case OPT_CFL:
		return "cfl";

	case OPT_VEC2:
		return "d:d";

	case OPT_VEC3:
		return "d:d:d";

	case OPT_FLOAT_VEC2:
		return "f:f";

	case OPT_FLOAT_VEC3:
		return "f:f:f";

        case OPT_FLOAT_VEC4:
		return "f:f:f:f";

	case OPT_STRING:
		return "<string>";

	case OPT_INFILE:
	case OPT_OUTFILE:
	case OPT_INOUTFILE:
		return "<file>";
	}
	error("Invalid OPT_ARG_TYPE!\n");
}

#define OPT_ARG_TYPE_CASE(X) 	case X: return #X;
static const char* opt_type_str(enum OPT_TYPE type)
{
	switch(type) {

	OPT_ARG_TYPE_CASE(OPT_SPECIAL)
	OPT_ARG_TYPE_CASE(OPT_SET)
	OPT_ARG_TYPE_CASE(OPT_CLEAR)
	OPT_ARG_TYPE_CASE(OPT_INT)
	OPT_ARG_TYPE_CASE(OPT_UINT)
	OPT_ARG_TYPE_CASE(OPT_ULONG)
	OPT_ARG_TYPE_CASE(OPT_LONG)
	OPT_ARG_TYPE_CASE(OPT_FLOAT)
	OPT_ARG_TYPE_CASE(OPT_CFL)
	OPT_ARG_TYPE_CASE(OPT_VEC2)
	OPT_ARG_TYPE_CASE(OPT_VEC3)
	OPT_ARG_TYPE_CASE(OPT_FLOAT_VEC2)
	OPT_ARG_TYPE_CASE(OPT_FLOAT_VEC3)
        OPT_ARG_TYPE_CASE(OPT_FLOAT_VEC4)
	OPT_ARG_TYPE_CASE(OPT_STRING)
	OPT_ARG_TYPE_CASE(OPT_INFILE)
	OPT_ARG_TYPE_CASE(OPT_OUTFILE)
	OPT_ARG_TYPE_CASE(OPT_INOUTFILE)
	OPT_ARG_TYPE_CASE(OPT_SELECT)
	OPT_ARG_TYPE_CASE(OPT_SUBOPT)
	}

	error("Invalid OPT_ARG_TYPE!\n");
}
#undef OPT_ARG_TYPE_CASE


static bool opt_dispatch(enum OPT_TYPE type, void* ptr, opt_conv_f* conv, char c, const char*  optarg)
{
	switch(type) {

	case OPT_SPECIAL:
		return conv(ptr, c, optarg);
	case OPT_SET:
		return opt_set(ptr, c, optarg);
	case OPT_CLEAR:
		return opt_clear(ptr, c, optarg);
	case OPT_INT:
		return opt_int(ptr, c, optarg);
	case OPT_UINT:
		return opt_uint(ptr, c, optarg);
	case OPT_ULONG:
		return opt_ulong(ptr, c, optarg);
	case OPT_LONG:
		return opt_long(ptr, c, optarg);
	case OPT_FLOAT:
		return opt_float(ptr, c, optarg);
	case OPT_CFL:
		return opt_cfl(ptr, c, optarg);
	case OPT_VEC2:
		return opt_vec2(ptr, c, optarg);
	case OPT_VEC3:
		return opt_vec3(ptr, c, optarg);
	case OPT_FLOAT_VEC2:
		return opt_float_vec2(ptr, c, optarg);
	case OPT_FLOAT_VEC3:
		return opt_float_vec3(ptr, c, optarg);
        case OPT_FLOAT_VEC4:
		return opt_float_vec4(ptr, c, optarg);
	case OPT_STRING:
		return opt_string(ptr, c, optarg);
	case OPT_INFILE:
		return opt_infile(ptr, c, optarg);
	case OPT_OUTFILE:
		return opt_outfile(ptr, c, optarg);
	case OPT_INOUTFILE:
		return opt_inoutfile(ptr, c, optarg);
	case OPT_SELECT:
		return opt_select(ptr, c, optarg);
	case OPT_SUBOPT:
		return opt_subopt(ptr, c, optarg);
	}

	error("Invalid OPT_ARG_TYPE!\n");
}

static const char* trim_space(const char* str)
{
	while (isspace(*str))
		str++;

	return str;
}

static bool show_option_p(const struct opt_s opt)
{
	return     (NULL != opt.descr)
		&& !(   ('(' == trim_space(opt.descr)[0])
		     && (')' == opt.descr[strlen(opt.descr) - 1]));
}


static const char* add_sep(const char* sep, bool has_arg)
{
	return has_arg ? sep : "";
}

static void print_usage(FILE* fp, const char* name, const char* usage_str, int n, const struct opt_s opts[static n ?: 1])
{
	fprintf(fp, "Usage: %s ", name);

	for (int i = 0; i < n; i++) {

		if (show_option_p(opts[i])) {

			if (NULL == opts[i].s) {

				fprintf(fp, "[-%c%s%s] ", opts[i].c, add_sep(" ", opts[i].arg), opt_arg_str(opts[i].type));

			} else {

				if (!isprint(opts[i].c))
					fprintf(fp, "[--%s%s%s] ", opts[i].s, add_sep(" ", opts[i].arg), opt_arg_str(opts[i].type));
				else
					fprintf(fp, "[-%c,--%s%s%s] ", opts[i].c, opts[i].s, add_sep(" ", opts[i].arg), opt_arg_str(opts[i].type));
			}
		}
	}

	fprintf(fp, "%s\n", usage_str);
}

static void print_usage_subopts(FILE* fp, char c, const char* arg_name, const char* usage_str, int n, const struct opt_s opts[static n ?: 1])
{
	fprintf(fp, "Usage of sub-option: ");

	if (0 != c)
		fprintf(fp, "-%c", c);

	if (NULL != arg_name)
		fprintf(fp, (0 == c) ? "--%s " : ",--%s ", arg_name);
	else
		fprintf(fp, " ");

	for (int i = 0; i < n; i++) {

		if (show_option_p(opts[i])) {

			if (NULL == opts[i].s) {

				fprintf(fp, "[%c%s%s]%s", opts[i].c, strlen(opt_arg_str(opts[i].type)) ? "=" : "", opt_arg_str(opts[i].type), i < (n - 1) ? "," : "");
			} else {

				if (opts[i].c < (int) ' ')
					fprintf(fp, "[%s%s%s]%s", opts[i].s, strlen(opt_arg_str(opts[i].type)) ? "=" : "", opt_arg_str(opts[i].type), i < (n - 1) ? "," : "");
				else
					fprintf(fp, "[%c%s%s,%s%s%s]%s", opts[i].c, strlen(opt_arg_str(opts[i].type)) ? "=" : "", opt_arg_str(opts[i].type), opts[i].s, strlen(opt_arg_str(opts[i].type)) ? "=" : "", opt_arg_str(opts[i].type), i < (n - 1) ? "," : "");
			}
		}
	}

	fprintf(fp, "%s\n", usage_str);
}



static void print_help(const char* help_str_prefix, const char* help_str, bool dash_prefix, const char* sep, int n, const struct opt_s opts[n ?: 1])
{
	if (NULL != help_str)
		printf("\n%s%s\n\n", help_str_prefix, help_str);
	else
		printf("\n");

	const char* short_only_format = NULL;
	const char* long_only_format = NULL;
	const char* short_long_format = NULL;
	if (dash_prefix) {

		short_only_format = "-%c%s%s";
		long_only_format = "--%s%s%s";
		short_long_format = "-%c,--%s%s%s";
	} else {

		short_only_format = "%c%s%s";
		long_only_format = "%s%s%s";
		short_long_format = "%c,%s%s%s";
	}

	int max_len = 0;

	// get needed padding lengths
	for (int i = 0; i < n; i++) {

		if (show_option_p(opts[i])) {

			int len = 0;

			if (NULL == opts[i].s) {

				len = snprintf(NULL, 0, short_only_format, opts[i].c, add_sep(sep, opts[i].arg), opts[i].argname);

			} else {

				if (!isprint(opts[i].c))
					len = snprintf(NULL, 0, long_only_format, opts[i].s, add_sep(sep, opts[i].arg), opts[i].argname);
				else
					len = snprintf(NULL, 0, short_long_format, opts[i].c, opts[i].s, add_sep(sep, opts[i].arg), opts[i].argname);
			}

			max_len = MAX(max_len, len);
		}
	}

	const int pad_len = max_len + 4;

	// print help
	for (int i = 0; i < n; i++) {

		if (show_option_p(opts[i])) {

			int written = 0;

			if (NULL == opts[i].s) {

				written = fprintf(stdout, short_only_format, opts[i].c, add_sep(sep, opts[i].arg), opts[i].argname);

			} else {

				if (!isprint(opts[i].c))
					written = fprintf(stdout, long_only_format, opts[i].s, add_sep(sep, opts[i].arg), opts[i].argname);
				else
					written = fprintf(stdout, short_long_format, opts[i].c, opts[i].s, add_sep(sep, opts[i].arg), opts[i].argname);
			}

			assert(pad_len > written);

			fprintf(stdout, "%*c%s\n", pad_len - written, ' ', opts[i].descr);
		}
	}

	if (dash_prefix)
		printf("-h%*chelp\n", pad_len - 2, ' ');
	else
		printf("h%*chelp\n", pad_len - 1, ' ');
}


static const char* arg_type_str(enum ARG_TYPE type);

static void print_interface(FILE* fp, const char* name, const char* usage_str, const char* help_str, int n, const struct opt_s opts[static n ?: 1], int m, struct arg_s args[m ?: 1])
{
	fprintf(fp, "name: %s, usage_str: \"%s\", help_str: \"%s\"\n", name, usage_str, help_str);

	fprintf(fp, "positional arguments:\n");

	for (int i = 0; i < m; i++) {

		fprintf(fp, "{ %s, \"%s\", %d, ", args[i].required ? "true" : "false", arg_type_str(args[i].arg_type), args[i].nargs);

		for (int j = 0; j < args[i].nargs; j++) {

			if (1 != args[i].nargs)
				fprintf(fp, "\n\t");

			fprintf(fp, "{ %s, %zd, \"%s\" } ", opt_type_str(args[i].arg[j].opt_type), args[i].arg[j].size, args[i].arg[j].argname);
		}

		if (1 != args[i].nargs)
			fprintf(fp, "\n");

		fprintf(fp, "}\n");
	}

	fprintf(fp, "options:\n");

	for (int i = 0; i < n; i++) {

		char cs[] =  "n/a";

		if (isprint(opts[i].c)) {

			cs[0] = opts[i].c;
			cs[1] = '\0';
		}

		fprintf(fp, "{ \"%s\", \"%s\", %s, %s, \"%s\", \"%s\" }\n", cs, opts[i].s, opts[i].arg ? "true" : "false", opt_type_str(opts[i].type), opt_arg_str(opts[i].type), opts[i].descr);
	}
}

static void check_options(int n, const struct opt_s opts[n ?: 1])
{
	bool f[256] = { 0 };
	f[0] = true; // interface
	f['h'] = true; // help

	for (int i = 0; i < n; i++) {

		int c = opts[i].c;

		assert(256 > c);

		if (f[c])
			error("duplicate option: %c\n", opts[i].c);

		f[c] = true;

		if ((OPT_SPECIAL != opts[i].type) && (NULL != opts[i].conv))
			error("Custom conversion functions are only allowed in OPT_SPECIAL\n");

		if ((OPT_SPECIAL == opts[i].type) && (NULL == opts[i].conv))
			error("Custom conversion function is required for OPT_SPECIAL\n");
	}
}


static void add_argnames(int n, struct opt_s wopts[n ?: 1])
{
	for (int i = 0; i < n; i++)
		if ((NULL == wopts[i].argname) || (0 == strlen(wopts[i].argname)))
			wopts[i].argname = opt_arg_str(wopts[i].type);
}


static void process_option(char c, const char* optarg, const char* name, const char* usage_str, const char* help_str, int n, const struct opt_s opts[n ?: 1], int m, struct arg_s args[m ?: 1])
{
	if ('h' == c) {

		print_usage(stdout, name, usage_str, n, opts);
		print_help("", help_str, true, " ", n, opts);
		exit(0);
	}

	if (0 == c) {

		print_interface(stdout, name, usage_str, help_str, n, opts, m, args);
		exit(0);
	}

	for (int i = 0; i < n; i++) {

		if (opts[i].c == c) {


			if (opt_dispatch(opts[i].type, opts[i].ptr, opts[i].conv, c, optarg)) {

				print_usage(stderr, name, usage_str, n, opts);
				error("process_option: failed to convert value\n");
			}

			return;
		}
	}

	print_usage(stderr, name, usage_str, n, opts);
	error("process_option: unknown option\n");
}


static void options(int* argcp, char* argv[], int min_args, int max_args, const char* usage_str, const char* help_str, int n, const struct opt_s opts[n ?: 1], int m, struct arg_s args[m ?: 1])
{
	int argc = *argcp;

	// create writable copy of opts

	struct opt_s wopts[n ?: 1];

	if ((n > 0) && (NULL != opts))
		memcpy(wopts, opts, sizeof wopts);


	add_argnames(n, wopts);

	int max_num_long_opts = 256;
	struct option longopts[max_num_long_opts];

	// According to documentation, the last element of the longopts array has to be filled with zeros.
	// So we fill it entirely before using it

	memset(longopts, 0, sizeof longopts);


	char lc = 0;
	int nlong = 0;

	// add (hidden) interface option
	longopts[nlong++] = (struct option){ "interface", false, NULL, lc++ };

	for (int i = 0; i < n; i++) {

		if (NULL != wopts[i].s) {

			// if it is only longopt, overwrite c with an unprintable char
			if (0 == wopts[i].c) {

				while (isprint(++lc)) // increment and skip over printable chars
					;

				wopts[i].c = lc;
			}

			longopts[nlong++] = (struct option){ wopts[i].s, wopts[i].arg, NULL, wopts[i].c };

			// Ensure that we only used unprintable characters
			// and that the last entry of the array is only zeros
			if ((nlong >= max_num_long_opts) || (lc >= max_num_long_opts))
				error("Too many long options specified, aborting...");
		}
	}

#if 0
	for (int i = 0; i < n; ++i)
		debug_printf(DP_INFO, "opt: %d: %s: %d\n",i, wopts[i].descr, (int) wopts[i].c);
#endif



	char optstr[2 * n + 2];
	ya_getopt_reset(); // reset getopt variables to process multiple argc/argv pairs

	check_options(n, wopts);

	save_command_line(argc, argv);

	int l = 0;
	optstr[l++] = 'h';

	for (int i = 0; i < n; i++) {

		optstr[l++] = wopts[i].c;

		if (wopts[i].arg)
			optstr[l++] = ':';
	}

	optstr[l] = '\0';

	int c;
	int longindex = -1;

	while (-1 != (c = ya_getopt_long(argc, argv, optstr, longopts, &longindex))) {
	//while (-1 != (c = ya_getopt(argc, argv, optstr))) {
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
					error("cmdline: failed to convert value\n");
				}

				goto out;
			}
		}

		print_usage(stderr, argv[0], usage_str, n, opts);
		exit(1);

	out:	continue;
#else
		process_option(c, optarg, argv[0], usage_str, help_str, n, wopts, m, args);
#endif
	}

	if (   (argc - optind < min_args)
	    || (argc - optind > max_args)) {

		print_usage(stderr, argv[0], usage_str, n, wopts);
		error("cmdline: too few or too many arguments\n");
	}

	int i;
	for (i = optind; i < argc; i++)
		argv[i - optind + 1] = argv[i];

	*argcp = argc - optind + 1;
	argv[*argcp] = NULL;
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
	int val;

	if (0 != parse_int(&val, optarg))
		error("Could not parse argument to opt_int: %s!\n", optarg);

	*(int*)ptr = val;
	return false;
}

bool opt_uint(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	int val;
	if (0 != parse_int(&val, optarg))
		error("Could not parse argument to opt_uint: %s!\n", optarg);

	if (0 > val)
		error("Argument \"%s\" to opt_uint is not unsigned!\n", optarg);

	*(unsigned int*)ptr = (unsigned int) val;
	return false;
}

bool opt_long(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	long val;

	if (0 != parse_long(&val, optarg))
		error("Could not parse argument to opt_long: %s!\n", optarg);

	*(long*)ptr = val;

	return false;
}

bool opt_ulong(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	long val;
	if (0 != parse_long(&val, optarg))
		error("Could not parse argument to opt_ulong: %s!\n", optarg);

	if (0 > val)
		error("Argument \"%s\" to opt_ulong is not unsigned!\n", optarg);

	*(unsigned long*)ptr = (unsigned long) val;
	return false;
}

bool opt_float(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	complex float val;

	if (0 != parse_cfl(&val, optarg))
		error("Could not parse argument to opt_float: %s!\n", optarg);

	if (0.f != cimagf(val))
		error("Argument \"%s\" to opt_float is not real\n", optarg);

	*(float*)ptr = crealf(val);

	return false;
}


bool opt_cfl(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	return 0 != parse_cfl((complex float*) ptr, optarg);
}


bool opt_string(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	*(const char**)ptr = strdup(optarg);
	list_append(str_list, *(char**)ptr);

	assert(NULL != ptr);

	return false;
}


static bool opt_file(void* ptr, char c, const char* optarg, bool out, bool in)
{
	UNUSED(c);

	*(const char**)ptr = strdup(optarg);
	list_append(str_list, *(char**) ptr);

	if (out)
		io_reserve_output(*(char**)ptr);

	if (in)
		io_reserve_input(*(char**)ptr);

	assert(NULL != ptr);
	return false;
}

bool opt_infile(void* ptr, char c, const char* optarg)
{
	return opt_file(ptr, c, optarg, false, true);
}

bool opt_outfile(void* ptr, char c, const char* optarg)
{
	return opt_file(ptr, c, optarg, true, false);
}

bool opt_inoutfile(void* ptr, char c, const char* optarg)
{
	return opt_file(ptr, c, optarg, true, true);
}

bool opt_float_vec2(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	int r = sscanf(optarg, "%f:%f", &(*(float(*)[2])ptr)[0], &(*(float(*)[2])ptr)[1]);

	assert(2 == r);

	return false;
}

bool opt_vec2(void* ptr, char c, const char* optarg)
{
	if (islower(c) || !isprint(c)) {

		if (2 != sscanf(optarg, "%ld:%ld", &(*(long(*)[2])ptr)[0], &(*(long(*)[2])ptr)[1])) {

			(*(long(*)[3])ptr)[0] = atol(optarg);
			(*(long(*)[3])ptr)[1] = atol(optarg);
		}

	} else {

		debug_printf(DP_WARN, "the upper-case options for specifying dimensions are deprecated.\n");

		int r = sscanf(optarg, "%ld:%ld", &(*(long(*)[2])ptr)[0], &(*(long(*)[2])ptr)[1]);

		assert(2 == r);
	}

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
	if (islower(c) || !isprint(c)) {

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


bool opt_float_vec4(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	int r = sscanf(optarg, "%f:%f:%f:%f", &(*(float(*)[3])ptr)[0], &(*(float(*)[3])ptr)[1], &(*(float(*)[3])ptr)[2], &(*(float(*)[3])ptr)[3]);

	assert(4 == r);

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

	int n = ptr->n;
	auto opts = ptr->opts;

	struct opt_s wopts[n ?: 1];

	if ((n > 0) && (NULL != opts))
		memcpy(wopts, opts, sizeof wopts);

	char lc = 1;

	for (int i = 0; i < n; i++) {

		if (NULL != wopts[i].s) {

			// if it is only longopt, overwrite c with an unprintable char
			if (0 == wopts[i].c)
				wopts[i].c = lc++;

			assert(lc < ' ');
		}
	}

	const char* tokens[2 * ptr->n + 2];

	for (int i = 0; i < ptr->n; i++) {

		tokens[2 * i] = ptr_printf("%c", wopts[i].c);

		if (NULL == wopts[i].s)
			tokens[2 * i + 1] = ptr_printf("char_only_token_%c", wopts[i].c);
		else
			tokens[2 * i + 1] = ptr_printf("%s", wopts[i].s);
	}

	tokens[2 * ptr->n] = ptr_printf("h");
	tokens[2 * ptr->n + 1] = NULL;


	char* tmpoptionp = strdup(optarg);
	char* option = tmpoptionp;
	char* value = NULL;

	int i = -1;

	while ('\0' != *option) {

		i = getsubopt(&option, (char *const *)tokens, &value);

		if ((i == 2 * n) || (-1 == i)) {

			print_usage_subopts(stdout, ptr->calling_c, ptr->calling_s, "", n, opts);
			print_help("Sub-options: ", ptr->calling_desc, false, "=", ptr->n, ptr->opts);
		}

		if (-1 == i)
			error("Sub-option could not be parsed: %s", value);

		if (i == 2 * n)
			exit(0);

		assert(i < 2 * n);

		process_option(wopts[i / 2].c, value, "", "", "", n, wopts, 0, NULL);
	}

	for (int i = 0; i < 2 * n + 1; i++)
		xfree(tokens[i]);

	xfree(tmpoptionp);

	return false;
}


static const char* arg_type_str(enum ARG_TYPE type)
{
	switch (type) {

	case ARG: return "ARG";
	case ARG_TUPLE: return "ARG_TUPLE";

	}

	error("Invalid ARG_TYPE!\n");
}





static void check_args(int N, const struct arg_s args[N])
{
	int num_tuples = 0;

	for (int i = 0; i < N; ++i) {

		if ((0 < num_tuples) && !args[i].required)
			error("Cannot have an optional argument after a tuple!\n");

		if (ARG_TUPLE == args[i].arg_type)
			num_tuples++;
		else if (ARG == args[i].arg_type)
			assert(1 == args[i].nargs);
	}

	if (num_tuples > 1)
		error("Cannot have more than one tuple argument!\n");
}



static int xsnprintf(int size, char buf[static size], const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);

	int rv = vsnprintf((size > 0) ? buf : NULL, size, fmt, ap);

	va_end(ap);

	return rv;
}

static int add_arg(int bufsize, char buf[static bufsize], const char* argname, bool required, bool file)
{
	const char* fstring;

	if (file)
		fstring = required ? "<%s>" : "[<%s>]";
	else
		fstring = required ? "%s" : "[%s]";

	return xsnprintf(bufsize, buf, fstring, argname);
}


static int add_tuple_args(int bufsize, char buf[static bufsize], const struct arg_s* arg, bool file)
{
	int pos = 0;

	if (!arg->required)
		pos += xsnprintf(bufsize - pos, buf + pos, "[");

	for (int k = 0; k < arg->nargs; ++k) {

		pos += add_arg(bufsize - pos, buf + pos, arg->arg[k].argname, true, file);

		if (file)
			pos += xsnprintf(bufsize - pos, buf + pos, "%c", '\b');

		pos += xsnprintf(bufsize - pos, buf + pos, "1");

		if (file)
			pos += xsnprintf(bufsize - pos, buf + pos, "%c", '>');

		pos += xsnprintf(bufsize - pos, buf + pos, " ");
	}

	pos += xsnprintf(bufsize - pos, buf + pos, "... ");

	for (int k = 0; k < arg->nargs; ++k) {

		pos += add_arg(bufsize - pos, buf + pos, arg->arg[k].argname, true, file);

		if (file)
			pos += xsnprintf(bufsize - pos, buf + pos, "%c", '\b');

		pos += xsnprintf(bufsize - pos, buf + pos, "N");

		if (file)
			pos += xsnprintf(bufsize - pos, buf + pos, "%c", '>');

		pos += xsnprintf(bufsize - pos, buf + pos, " ");
	}

	if (!arg->required)
		pos += xsnprintf(bufsize - pos, buf + pos, "\b] ");

	return pos;
}




void cmdline(int* argc, char* argv[*argc], int m, struct arg_s args[m], const char* help_str, int n, const struct opt_s opts[n])
{
	check_args(m, args);

	if (NULL == str_list)
		str_list = list_create();

	long min_args = 0;
	long max_args = 0;

	enum { bufsize = 1024 };
	char buf[bufsize] = { 0 };

	int pos = 0;

	for (int i = 0; i < m; ++i) {

		if (ARG_TUPLE == args[i].arg_type)
			max_args = 1000; // should be plenty for most use cases, but not overflow long
		else
			max_args += args[i].nargs;

		if (args[i].required)
			min_args += args[i].nargs;

		bool file = false;

		switch (args[i].arg->opt_type) {

		case OPT_INFILE:
		case OPT_OUTFILE:
		case OPT_INOUTFILE:

			file = true;
			break;

		default:
			file = false;
			break;
		}

		switch (args[i].arg_type) {

		case ARG:
			pos += add_arg(bufsize - pos, buf + pos, args[i].arg->argname, args[i].required, file);
			pos += xsnprintf(bufsize - pos, buf + pos, " ");
			break;

		case ARG_TUPLE:
			pos += add_tuple_args(bufsize - pos, buf + pos, &args[i], file);
			break;
		}
	}

	options(argc, argv, min_args, max_args, buf, help_str, n, opts, m, args);

	int req_args_remaining = min_args;

	for (int i = 0, j = 1; (i < m) && (j < *argc); ++i) {


		int given_args_following = *argc - j; // number of following args given on the command line, NOT in the args-array
		int declared_args_following = m - i - 1; // number of following arguments in args-array, NOT on the command line

#if 0
		debug_printf(DP_INFO, "j: %d, arg: %d, given_args_following: %d, req_args_remaining: %d, argstr: %s\n", j, i, given_args_following, req_args_remaining, argv[j]);
#endif

		switch (args[i].arg_type) {

		case ARG:

			// Skip optional arguments if the number of given command-line arguments is the number of still required arguments.
			// This is just for fmac, which has an optional arg in the middle

			if (!args[i].required && (given_args_following == req_args_remaining))
				continue;

			if (opt_dispatch(args[i].arg->opt_type, args[i].arg->ptr, NULL, '\0', argv[j++]))
				error("failed to convert value\n");

			break;

		case ARG_TUPLE:

			;

			// Consume as many arguments as possible, except for possible args following the tuple
			// As we can only have one tuple, a tuple consuming multiple arguments cannot follow.
			// Further, as we cannot have an optional arg following, all declared args afer the tuple
			// are required and take exactly one argument.
			int tuple_end = *argc - declared_args_following;
			int num_tuple_args = tuple_end - j;

			if (0 != (num_tuple_args % args[i].nargs))
				error("Incorrect number of arguments!\n");

			*args[i].count = num_tuple_args / args[i].nargs;

			if (0 == *args[i].count)
				continue;
#if 1
			for (int k = 0; k < args[i].nargs; ++k)
				*(void**)args[i].arg[k].ptr = calloc(args[i].arg[k].size, *args[i].count);
#endif
			int c = 0;

			while (j < tuple_end) {

				for (int k = 0; k < args[i].nargs; ++k)	// FIXME ????
					if (opt_dispatch(args[i].arg[k].opt_type, (*(void**)args[i].arg[k].ptr) + c * args[i].arg[k].size, NULL, '\0', argv[j++]))
						error("failed to convert value\n");

				c++;
			}

			break;
		}

		// This does not accurately count how many arguments are consumed, but does accurately count
		// how many required arguments are, at minimum, still needed.
		if (args[i].required)
			req_args_remaining -= args[i].nargs;

	}

	assert(0 == req_args_remaining);

#if 0
	// for debug, make argv inaccessible
	for (int i = 0; i < *argc; ++i)
		argv[i] = NULL;
	// and set argc to something that is likely to break anything relying on it
	*argc = -1;
#endif
}
