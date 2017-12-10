
#if defined(__clang__) && !defined(__CUDACC__)
#define NESTED(RET, NAME, ARGS) \
	RET (^NAME)ARGS = ^ARGS
#define CLOSURE_TYPE(x) (^x)
#else
#define NESTED(RET, NAME, ARGS) \
	RET NAME ARGS
#define CLOSURE_TYPE(x) x
#define __block
#endif


