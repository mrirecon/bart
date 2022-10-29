#ifndef OPTS_H
#define OPTS_H

#include <stdbool.h>


#include "misc/cppwrap.h"
#include "misc/types.h"
#include "misc/misc.h"

enum OPT_TYPE {

	OPT_SPECIAL,
	OPT_SET, OPT_CLEAR,
	OPT_INT, OPT_UINT, OPT_LONG, OPT_ULONG,
	OPT_FLOAT,
	OPT_CFL,
	OPT_VEC2, OPT_VEC3,
	OPT_FLOAT_VEC2, OPT_FLOAT_VEC3, OPT_FLOAT_VEC4,
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

#define OPT_SEL(T, x, v)	&(struct opt_select_s){ (x), &(T){ (v) }, &(T){ *(x) }, sizeof(T) }
#define OPT_SUB(n, opts, c, s, descr)	&(struct opt_subopt_s){ (n), (opts), (c), (s), (descr) }

#define OPT_SET(c, ptr, descr)			{ (c), NULL, false, OPT_SET, NULL, TYPE_CHECK(bool*, (ptr)), "", descr }
#define OPT_CLEAR(c, ptr, descr)		{ (c), NULL, false, OPT_CLEAR, NULL, TYPE_CHECK(bool*, (ptr)), "", descr }
#define OPT_ARG(c, type, T, ptr, argname, descr) { (c), NULL, true, type, NULL, TYPE_CHECK(T*, (ptr)), argname, descr }
#define OPT_STRING(c, ptr, argname, descr)	OPT_ARG(c, OPT_STRING, const char*, ptr, argname, descr)
#define OPT_INFILE(c, ptr, argname, descr)	OPT_ARG(c, OPT_INFILE, const char*, ptr, argname, descr)
#define OPT_OUTFILE(c, ptr, argname, descr)	OPT_ARG(c, OPT_OUTFILE, const char*, ptr, argname, descr)
#define OPT_INOUTFILE(c, ptr, argname, descr)	OPT_ARG(c, OPT_INOUTFILE, const char*, ptr, argname, descr)
#define OPT_UINT(c, ptr, argname, descr)	OPT_ARG(c, OPT_UINT, unsigned int, ptr, argname, descr)
#define OPT_INT(c, ptr, argname, descr)		OPT_ARG(c, OPT_INT, int, ptr, argname, descr)
#define OPT_ULONG(c, ptr, argname, descr)	OPT_ARG(c, OPT_ULONG, unsigned long, ptr, argname, descr)
#define OPT_LONG(c, ptr, argname, descr)	OPT_ARG(c, OPT_LONG, long, ptr, argname, descr)
#define OPT_FLOAT(c, ptr, argname, descr)	OPT_ARG(c, OPT_FLOAT, float, ptr, argname, descr)
#define OPT_VEC2(c, ptr, argname, descr)	OPT_ARG(c, OPT_VEC2, opt_vec2_t, ptr, argname, descr)
#define OPT_FLVEC2(c, ptr, argname, descr)	OPT_ARG(c, OPT_FLOAT_VEC2, opt_fvec2_t, ptr, argname, descr)
#define OPT_VEC3(c, ptr, argname, descr)	OPT_ARG(c, OPT_VEC3, opt_vec3_t, ptr, argname, descr)
#define OPT_FLVEC3(c, ptr, argname, descr)	OPT_ARG(c, OPT_FLOAT_VEC3, opt_fvec3_t, ptr, argname, descr)
#define OPT_FLVEC4(c, ptr, argname, descr)	OPT_ARG(c, OPT_FLOAT_VEC4, opt_fvec4_t, ptr, argname, descr)
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
#define OPTL_INT(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_INT, int, ptr, argname, descr)
#define OPTL_ULONG(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_ULONG, unsigned long, ptr, argname, descr)
#define OPTL_LONG(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_LONG, long, ptr, argname, descr)
#define OPTL_FLOAT(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_FLOAT, float, ptr, argname, descr)
#define OPTL_VEC2(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_VEC2, opt_vec2_t, ptr, argname, descr)
#define OPTL_FLVEC2(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_FLOAT_VEC2, opt_fvec2_t, ptr, argname, descr)
#define OPTL_VEC3(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_VEC3, opt_vec3_t, ptr, argname, descr)
#define OPTL_FLVEC3(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_FLOAT_VEC3, opt_fvec3_t, ptr, argname, descr)
#define OPTL_FLVEC4(c, s, ptr, argname, descr)	OPTL_ARG(c, s, OPT_FLOAT_VEC4, opt_fvec4_t, ptr, argname, descr)
#define OPTL_SELECT(c, s, T, ptr, value, descr)	{ (c), (s), false, OPT_SELECT, NULL, OPT_SEL(T, TYPE_CHECK(T*, ptr), value), "", descr }
#define OPTL_SELECT_DEF(c, s, T, ptr, value, def, descr)	{ (c), (s), false, OPT_SELECT, NULL, &(struct opt_select_s){ (TYPE_CHECK(T*, ptr)), &(T){ TYPE_CHECK(T, value) }, &(T){ (TYPE_CHECK(T, def)) }, sizeof(T) }, "", descr }
#define OPTL_SUBOPT(c, s, argname, descr, NR, opts)	OPTL_ARG(c, s, OPT_SUBOPT, struct opt_subopt_s, OPT_SUB(NR, opts, c, s, descr), argname, descr)


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
	long* count;
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

#define ARG_TUPLE(required, count, n, ...)		{ (required), ARG_TUPLE, (count), (n), (struct arg_single_s[(n)]){ __VA_ARGS__ } }
#define TUPLE_LONG(ptr, argname)			(struct arg_single_s){ OPT_LONG, sizeof(long), TYPE_CHECK(long**, ptr), argname }
#define TUPLE_ULONG(ptr, argname)			(struct arg_single_s){ OPT_ULONG, sizeof(unsigned long), TYPE_CHECK(unsigned long**, ptr), argname }

extern void cmdline(int* argc, char* argv[*argc], int m, struct arg_s args[m], const char* help_str, int n, const struct opt_s opts[n]);
extern void opt_free_strdup(void);

#include "misc/cppwrap.h"
#endif //OPTS_H

