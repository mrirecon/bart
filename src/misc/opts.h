
#include <stdbool.h>

typedef bool opt_conv_f(void* ptr, char c, const char* optarg);

struct opt_s {

	char c;
	bool arg;
	opt_conv_f* conv;
	void* ptr;
	const char* descr;
};

extern opt_conv_f opt_set;
extern opt_conv_f opt_clear;
extern opt_conv_f opt_int;
extern opt_conv_f opt_uint;
extern opt_conv_f opt_long;
extern opt_conv_f opt_float;
extern opt_conv_f opt_string;
extern opt_conv_f opt_vec3;
extern opt_conv_f opt_select;
extern opt_conv_f opt_subopt;

struct opt_select_s {

	void* ptr;
	const void* value;
	const void* default_value;
	size_t size;
};

struct opt_subopt_s {

	int n;
	struct opt_s* opts;
};

typedef long opt_vec3_t[3];
#define OPT_SEL(T, x, v)	&(struct opt_select_s){ (x), &(T){ (v) }, &(T){ *(x) }, sizeof(T) }

//#define TYPE_CHECK(T, x)	({ T _val = (x); _val; })
#define TYPE_CHECK(T, x)	({ T* _ptr1 = 0; __typeof(x)* _ptr2 = _ptr1; (void)_ptr2; (x);  })
#define OPT_SET(c, ptr, descr)			{ (c), false, opt_set, TYPE_CHECK(bool*, (ptr)), "\t" descr }
#define OPT_CLEAR(c, ptr, descr)		{ (c), false, opt_clear, TYPE_CHECK(bool*, (ptr)), "\t" descr }
#define OPT_ARG(c, _fun, T, ptr, argname, descr) { (c), true, _fun, TYPE_CHECK(T*, (ptr)), " " argname "      \t" descr }
#define OPT_STRING(c, ptr, argname, descr)	OPT_ARG(c, opt_string, const char*, ptr, argname, descr)
#define OPT_UINT(c, ptr, argname, descr)	OPT_ARG(c, opt_uint, unsigned int, ptr, argname, descr)
#define OPT_INT(c, ptr, argname, descr)		OPT_ARG(c, opt_int, int, ptr, argname, descr)
#define OPT_LONG(c, ptr, argname, descr)	OPT_ARG(c, opt_long, long, ptr, argname, descr)
#define OPT_FLOAT(c, ptr, argname, descr)	OPT_ARG(c, opt_float, float, ptr, argname, descr)
#define OPT_VEC3(c, ptr, argname, descr)	OPT_ARG(c, opt_vec3, opt_vec3_t, ptr, argname, descr)
#define OPT_SELECT(c, T, ptr, value, descr)	{ (c), false, opt_select, OPT_SEL(T, TYPE_CHECK(T*, ptr), value), descr }
#define OPT_SUB(c, argname, descr, NR, opts)	OPT_ARG(c, opt_subopt, &(struct opt_subopt_s){ NR, opts }, argname, descr)

extern void cmdline(int* argc, char* argv[], int min_args, int max_args, const char* usage_str, const char* help_str, int n, const struct opt_s opts[n]);


