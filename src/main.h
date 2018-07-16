
#include "misc/cppmap.h"

#define DECLMAIN(x)					\
     extern int main_ ## x(int argc, char* argv[]);
MAP(DECLMAIN, MAIN_LIST)
#undef	DECLMAIN

// if not NULL, output should point to a memory location with at least *512* elements
extern int in_mem_bitmask_main (int argc, char* argv[], char* out);
extern int in_mem_devel_main   (int argc, char* argv[], char* out);
extern int in_mem_estdelay_main(int argc, char* argv[], char* out);
extern int in_mem_estdims_main (int argc, char* argv[], char* out);
extern int in_mem_estshift_main(int argc, char* argv[], char* out);
extern int in_mem_estvar_main  (int argc, char* argv[], char* out);
extern int in_mem_nrmse_main   (int argc, char* argv[], char* out);
extern int in_mem_sdot_main    (int argc, char* argv[], char* out);
extern int in_mem_show_main    (int argc, char* argv[], char* out);
extern int in_mem_version_main (int argc, char* argv[], char* out);

extern int main_bart(int argc, char* argv[]);
// if not NULL, output should point to a memory location with at least *512* elements
extern int in_mem_bart_main(int argc, char* argv[], char* out);
extern int main_bbox(int argc, char* argv[]);

