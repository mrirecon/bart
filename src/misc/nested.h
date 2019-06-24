
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

#if defined(__clang__) || !defined(NOEXEC_STACK)
#define NESTED_CALL(x, args)	((x)args)
#else
#ifndef __x86_64__
#error NOEXEC_STACK only supported on x86_64
#endif
#include <stdio.h>
#if __GNUC__ >= 10
#define NESTED_CALL(p, args) ({												\
		__auto_type __p = (p);											\
		struct { unsigned short mov1; unsigned int addr; unsigned short mov2; void* chain; unsigned int jmp; } 	\
			__attribute__((packed))* __t = (void*)p;							\
		assert((0xbb41 == __t->mov1) && (0xba49 == __t->mov2) && (0x90e3ff49 == __t->jmp));			\
		__builtin_call_with_static_chain(((__typeof__(__p))((unsigned long)__t->addr))args, (void*)__t->chain);	\
	})
#else
#define NESTED_CALL(p, args) ({												\
		__auto_type __p = (p);											\
		struct { unsigned short mov1; void* addr; unsigned short mov2; void* chain; unsigned int jmp; } 	\
			__attribute__((packed))* __t = (void*)p;							\
		assert((0xbb49 == __t->mov1) && (0xba49 == __t->mov2) && (0x90e3ff49 == __t->jmp));			\
		__builtin_call_with_static_chain(((__typeof__(__p))(__t->addr))args, __t->chain);			\
	})
#endif
#endif

