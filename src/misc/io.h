
#include "misc/cppwrap.h"

extern int write_ra(int fd, unsigned int n, const long dimensions[__VLA(n)]);
extern int read_ra(int fd, unsigned int n, long dimensions[__VLA(n)]);

extern int write_coo(int fd, unsigned int n, const long dimensions[__VLA(n)]);
extern int read_coo(int fd, unsigned int n, long dimensions[__VLA(n)]);

extern int write_cfl_header(int fd, unsigned int n, const long dimensions[__VLA(n)]);
extern int read_cfl_header(int fd, unsigned int D, long dimensions[__VLA(D)]);

extern void io_register_input(const char* name);
extern void io_register_output(const char* name);

#include "misc/cppwrap.h"

