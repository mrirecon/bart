
#ifndef __clang__
#define NESTED(RET, NAME, ARGS) \
	RET NAME ARGS
#define CLOSURE_TYPE(x) x
#define __block
#else
#define NESTED(RET, NAME, ARGS) \
	RET (^NAME)ARGS = ^ARGS
#define CLOSURE_TYPE(x) (^x)
#endif


