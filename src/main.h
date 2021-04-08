
#ifndef __cplusplus

#include "misc/cppmap.h"


#define DECLMAIN(x) \
extern int main_ ## x(int argc, char* argv[argc]);
MAP(DECLMAIN, MAIN_LIST)
#undef	DECLMAIN

extern int main_bart(int argc, char* argv[argc]);
extern int main_bbox(int argc, char* argv[argc]);


// for use as a library
extern int bart_command(int len, char* buf, int argc, char* argv[]);


#endif

