#ifndef _OPTS_H
#define _OPTS_H

#include <stdbool.h>

// FILE
#include <stdio.h>

#include "misc/cppwrap.h"
#include "misc/types.h"
#include "misc/misc.h"

enum OPT_TYPE {

	OPT_SPECIAL,
	OPT_SET, OPT_CLEAR,
	OPT_INT, OPT_UINT, OPT_LONG, OPT_ULONG, OPT_ULLONG,
	OPT_PINT,
	OPT_FLOAT,
	OPT_DOUBLE,
	OPT_CFL,
	OPT_VEC2, OPT_VEC3, OPT_VECN,
	OPT_FLOAT_VEC2, OPT_FLOAT_VEC3, OPT_FLOAT_VEC4, OPT_FLOAT_VECN,
	OPT_DOUBLE_VEC3, OPT_DOUBLE_VECN,
	OPT_STRING,
	OPT_INFILE, OPT_OUTFILE, OPT_INOUTFILE,
	OPT_SELECT,
	OPT_SUBOPT,
};

typedef bool opt_conv_f(void* ptr, char c, const char* optarg);

struct opt_s {

	char c;
	const char* s;
	bool arg;
	enum OPT_TYPE type;
	opt_conv_f* conv;
	void* ptr;
	const char* argname;
	const char* descr;
};

struct opt_select_s {

	void* ptr;
	const void* value;
	const void* default_value;
	size_t size;
};

struct opt_vec_s {

	int max;
	int required;
	int* count;
	void* ptr;
};

struct opt_subopt_s {

	int n;
	struct opt_s* opts;

	char calling_c;
	const char* calling_s;
	const char* calling_desc;
};

typedef long opt_vec2_t[2];
typedef float opt_fvec2_t[2];
typedef long opt_vec3_t[3];
typedef float opt_fvec3_t[3];
typedef float opt_fvec4_t[4];
typedef double opt_dvec3_t[3];

#define OPT_SEL(T, x, v)	&(struct opt_select_s){ (x), &(T){ (v) }, &(T){ *(x) }, sizeof(T) }
#define OPT_VEC(x, min, count)	&(struct opt_vec_s){ ARRAY_SIZE(x), min, count, x }
#define OPT_SUB(n, opts, c, s, descr)	&(struct opt_subopt_s){ (n), (opts), (c), (s), (descr) }

#define OPT_SET(c, ptr, descr)			{ (c), NULL, false, OPT_SET, NULL, TYPE_CHECK(bool*, (ptr)), "", descr }
#define OPT_CLEAR(c, ptr, descr)		{ (c), NULL, false, OPT_CLEAR, NULL, TYPE_CHECK(bool*, (ptr)), "", descr }
#define OPT_ARG(c, type, T, ptr, argname, descr) { (c), NULL, true, type, NULL, TYPE_CHECK(T*, (ptr)), argname, descr }
#define OPT_STRING(c, ptr, argname, descr)	OPT_ARG(c, OPT_STRING, const char*, ptr, argname, descr)
#define OPT_INFILE(c, ptr, argname, descr)	OPT_ARG(c, OPT_INFILE, const char*, ptr, argname, descr)
#define OPT_OUTFILE(c, ptr, argname, descr)	OPT_ARG(c, OPT_OUTFILE, const char*, ptr, argname, descr)
#define OPT_INOUTFILE(c, ptr, argname, descr)	OPT_ARG(c, OPT_INOUTFILE, const char*, ptr, argname, descr)

#define OPT_PINT(c, ptr, argname, descr)	OPT_ARG(c, OPT_PINT, int, ptr, argname, descr)
#define OPT_UINT(c, ptr, argname, descr)	OPT_ARG(c, OPT_UINT, unsigned int, ptr, argname, descr)
#define OPT_INT(c, ptr, argname, descr)		OPT_ARG(c, OPT_INT, int, ptr, argname, descr)
#define OPT_ULONG(c, ptr, argname, descr)	OPT_ARG(c, OPT_ULONG, unsigned long, ptr, argname, descr)
#define OPT_ULLONG(c, ptr, argname, descr)	OPT_ARG(c, OPT_ULLONG, unsigned long long, ptr, argname, descr)
#define OPT_LONG(c, ptr, argname, descr)	OPT_ARG(c, OPT_LONG, long, ptr, argname, descr)
#define OPT_FLOAT(c, ptr, argname, descr)	OPT_ARG(c, OPT_FLOAT, float, ptr, argname, descr)
#define OPT_DOUBLE(c, ptr, argname, descr)	OPT_ARG(c, OPT_DOUBLE, double, ptr, argname, descr)
#define OPT_VEC2(c, ptr, argname, descr)	OPT_ARG(c, OPT_VEC2, opt_vec2_t, ptr, argname, descr)
#define OPT_FLVEC2(c, ptr, argname, descr)	OPT_ARG(c, OPT_FLOAT_VEC2, opt_fvec2_t, ptr, argname, descr)
#define OPT_VEC3(c, ptr, argname, descr)	OPT_ARG(c, OPT_VEC3, opt_vec3_t, ptr, argname, descr)
#define OPT_FLVEC3(c, ptr, argname, descr)	OPT_ARG(c, OPT_FLOAT_VEC3, opt_fvec3_t, ptr, argname, descr)
#define OPT_FLVEC4(c, ptr, argname, descr)	OPT_ARG(c, OPT_FLOAT_VEC4, opt_fvec4_t, ptr, argname, descr)
#define OPT_FLVECN(c, ptr, descr)		{ (c), NULL, true, OPT_FLOAT_VECN, NULL, OPT_VEC(ptr, 0, NULL), "", descr }
#define OPT_DOVEC3(c, ptr, argname, descr)	OPT_ARG(c, OPT_DOUBLE_VEC3, opt_dvec3_t, ptr, argname, descr)
#define OPT_DOVECN(c, ptr, descr)		{ (c), NULL, true, OPT_DOUBLE_VECN, NULL, OPT_VEC(ptr, 0, NULL), "", descr }

#define OPT_VECN(c, ptr, descr) 		{ (c), NULL, true, OPT_VECN, NULL, OPT_VEC(ptr, 0, NULL), "", descr }
#define OPT_VECC(c, count, ptr, descr)		{ (c), NULL, true, OPT_VECN, NULL, OPT_VEC(ptr, 0, count), "", descr }
#define OPT_SELECT(c, T, ptr, value, descr)	{ (c), NULL, false, OPT_SELECT, NULL, OPT_SEL(T, TYPE_CHECK(T*, ptr), value), "", descr }
#define OPT_SUBOPT(c, argname, descr, NR, opts)	OPT_ARG(c, OPT_SUBOPT, struct opt_subopt_s, OPT_SUB(NR, opts, c, NULL, descr), argname, descr)

// If the character in these macros is 0 (please note: NOT '0'), then it is only a long opt
// Otherwise, it is both
#define OPTL_SET(c, s, ptr, descr)			{ (c), (s), false, OPT_SET, NULL, TYPE_CHECK(bool*, (ptr)), "", descr }
#define OPTL_CLEAR(c, s, ptr, descr)			{ (c), (s), false, OPT_CLEAR, NULL, TYPE_CHECK(bool*, (ptr)), "", descr }
#define OPTL_ARG(c, s, type, T, ptr, argname, descr) { (c), (s), true, type, NULL, TYPE_CHECK(T*, (ptr)), argname, descr }
#define OPTL_STRING(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_STRING, const char*, ptr, argname, descr)
#define OPTL_INFILE(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_INFILE, const char*, ptr, argname, descr)
#define OPTL_OUTFILE(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_OUTFILE, const char*, ptr, argname, descr)
#define OPTL_INOUTFILE(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_INOUTFILE, const char*, ptr, argname, descr)
#define OPTL_UINT(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_UINT, unsigned int, ptr, argname, descr)
#define OPTL_PINT(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_INT, int, ptr, argname, descr)
#define OPTL_INT(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_INT, int, ptr, argname, descr)
#define OPTL_ULONG(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_ULONG, unsigned long, ptr, argname, descr)
#define OPTL_ULLONG(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_ULLONG, unsigned long long, ptr, argname, descr)
#define OPTL_LONG(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_LONG, long, ptr, argname, descr)
#define OPTL_FLOAT(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_FLOAT, float, ptr, argname, descr)
#define OPTL_DOUBLE(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_DOUBLE, double, ptr, argname, descr)
#define OPTL_VEC2(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_VEC2, opt_vec2_t, ptr, argname, descr)
#define OPTL_FLVEC2(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_FLOAT_VEC2, opt_fvec2_t, ptr, argname, descr)
#define OPTL_VEC3(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_VEC3, opt_vec3_t, ptr, argname, descr)
#define OPTL_VECN(c, s, ptr, descr)	{ (c), (s), true, OPT_VECN, NULL, OPT_VEC(ptr, 0, NULL), "", descr }
#define OPTL_VECC(c, s, count, ptr, descr)	{ (c), (s), true, OPT_VECN, NULL, OPT_VEC(ptr, 0, count), "", descr }
#define OPTL_FLVEC3(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_FLOAT_VEC3, opt_fvec3_t, ptr, argname, descr)
#define OPTL_FLVEC4(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_FLOAT_VEC4, opt_fvec4_t, ptr, argname, descr)
#define OPTL_FLVECN(c, s, ptr, descr)	{ (c), (s), true, OPT_FLOAT_VECN, NULL, OPT_VEC(ptr, 0, NULL), "", descr }
#define OPTL_DOVEC3(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_DOUBLE_VEC3, opt_dvec3_t, ptr, argname, descr)
#define OPTL_DOVECN(c, s, ptr, descr)	{ (c), (s), true, OPT_DOUBLE_VECN, NULL, OPT_VEC(ptr, 0, NULL), "", descr }
#define OPTL_SELECT(c, s, T, ptr, value, descr)	{ (c), (s), false, OPT_SELECT, NULL, OPT_SEL(T, TYPE_CHECK(T*, ptr), value), "", descr }
#define OPTL_SELECT_DEF(c, s, T, ptr, value, def, descr)	{ (c), (s), false, OPT_SELECT, NULL, &(struct opt_select_s){ (TYPE_CHECK(T*, ptr)), &(T){ TYPE_CHECK(T, value) }, &(T){ (TYPE_CHECK(T, def)) }, sizeof(T) }, "", descr }
#define OPTL_SUBOPT(c, s, argname, descr, NR, opts)	OPTL_ARG(c, s, OPT_SUBOPT, struct opt_subopt_s, OPT_SUB(NR, opts, c, s, descr), argname, descr)
#define OPTL_SUBOPT2(c, s, argname, descr, descr1, NR, opts)	OPTL_ARG(c, s, OPT_SUBOPT, struct opt_subopt_s, OPT_SUB(NR, opts, c, s, descr), argname, descr1)


enum ARG_TYPE {

	ARG,
	ARG_TUPLE,
};

struct arg_single_s {

	enum OPT_TYPE opt_type;
	size_t size;
	void* ptr;
	const char* argname;
};


struct arg_s {

	bool required;
	enum ARG_TYPE arg_type;
	int* count;
	int nargs;
	struct arg_single_s* arg;
};


#define ARG_SINGLE(type, T, ptr, argname)			&(struct arg_single_s){ (type), sizeof(T), (ptr), (argname) }

#define ARG_CHECKED(required, type, T, ptr, argname)		{ required, ARG, NULL, 1, ARG_SINGLE(type, T, TYPE_CHECK(T*, ptr), argname) }

#define ARG_UINT(required, ptr, argname) 		ARG_CHECKED(required, OPT_UINT,  unsigned int, ptr, argname)
#define ARG_INT(required, ptr, argname) 		ARG_CHECKED(required, OPT_INT,  int, ptr, argname)
#define ARG_ULONG(required, ptr, argname) 		ARG_CHECKED(required, OPT_ULONG, unsigned long, ptr, argname)
#define ARG_LONG(required, ptr, argname) 		ARG_CHECKED(required, OPT_LONG,  long, ptr, argname)
#define ARG_CFL(required, ptr, argname) 		ARG_CHECKED(required, OPT_CFL,  _Complex float, ptr, argname)
#define ARG_INFILE(required, ptr, argname) 		ARG_CHECKED(required, OPT_INFILE, const char*, ptr, argname)
#define ARG_OUTFILE(required, ptr, argname) 		ARG_CHECKED(required, OPT_OUTFILE,  const char*, ptr, argname)
#define ARG_INOUTFILE(required, ptr, argname) 		ARG_CHECKED(required, OPT_INOUTFILE,  const char*, ptr, argname)
#define ARG_STRING(required, ptr, argname)		ARG_CHECKED(required, OPT_STRING,  const char*, ptr, argname)
#define ARG_FLOAT(required, ptr, argname)		ARG_CHECKED(required, OPT_FLOAT,  float, ptr, argname)
#define ARG_VEC2(required, ptr, argname)		ARG_CHECKED(required, OPT_VEC2,  opt_vec2_t, ptr, argname)
#define ARG_FLVEC2(required, ptr, argname)		ARG_CHECKED(required, OPT_FLOAT_VEC2,  opt_fvec2_t, ptr, argname)
#define ARG_VEC3(required, ptr, argname)		ARG_CHECKED(required, OPT_VEC3,  opt_vec3_t, ptr, argname)
#define ARG_FLVEC3(required, ptr, argname)		ARG_CHECKED(required, OPT_FLOAT_VEC3,  opt_fvec3_t, ptr, argname)
#define ARG_FLVEC4(required, ptr, argname)		ARG_CHECKED(required, OPT_FLOAT_VEC4,  opt_fvec4_t, ptr, argname)
#define ARG_DOVEC3(required, ptr, argname)		ARG_CHECKED(required, OPT_DOUBLE_VEC3,  opt_dvec3_t, ptr, argname)

#define ARG_TUPLE(required, count, n, ...)		{ (required), ARG_TUPLE, (count), (n), (struct arg_single_s[(n)]){ __VA_ARGS__ } }
#define TUPLE_LONG(ptr, argname)			(struct arg_single_s){ OPT_LONG, sizeof(long), TYPE_CHECK(long**, ptr), argname }
#define TUPLE_ULONG(ptr, argname)			(struct arg_single_s){ OPT_ULONG, sizeof(unsigned long), TYPE_CHECK(unsigned long**, ptr), argname }



extern void print_usage(FILE* fp, const char* name, const char* usage_str, int n, const struct opt_s opts[n ?: 1]);
extern int options(int* argcp, char* argv[*argcp], const char* usage_str, const char* help_str, int n, const struct opt_s opts[n], int m, const struct arg_s args[m], bool stop_at_nonopt);
extern void cmdline(int* argcp, char* argv[*argcp], int m, const struct arg_s args[m], const char* help_str, int n, const struct opt_s opts[n]);
extern void opt_free_strdup(void);
extern void cmdline_synth(void (*print)(const char *fmt, ...),  int n, const struct opt_s opts[static n ?: 1]);


#include "misc/cppwrap.h"
#endif // _OPTS_H

