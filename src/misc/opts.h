
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
extern opt_conv_f opt_long;
extern opt_conv_f opt_float;
extern opt_conv_f opt_string;

extern void cmdline(int* argc, char* argv[], int min_args, int max_args, const char* usage_str, const char* help_str, int n, const struct opt_s opts[n]);


