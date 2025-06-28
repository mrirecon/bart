
#ifndef _BART_CPP_WRAP
#define _BART_CPP_WRAP

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __VLA
#ifdef __cplusplus
#define __VLA(x)
#else
#define __VLA(x) static x
#endif
#endif

#ifndef __VLA2
#ifdef __cplusplus
#define __VLA2(x)
#else
#define __VLA2(x) x
#endif
#endif

#else
#undef _BART_CPP_WRAP

#ifdef __cplusplus
}
#endif
#endif

