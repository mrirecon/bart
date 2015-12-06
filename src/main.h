
#include "misc/cppmap.h"


#define DECLMAIN(x) \
extern int main_ ## x(int argc, char* argv[]);
MAP(DECLMAIN, MAIN_LIST)
#undef	DECLMAIN

extern int main_bart(int argc, char* argv[]);
extern int main_bbox(int argc, char* argv[]);

