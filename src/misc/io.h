
#include "misc/cppwrap.h"


enum file_types_e {
	FILE_TYPE_CFL, FILE_TYPE_RA, FILE_TYPE_COO,
#ifdef USE_MEM_CFL
	FILE_TYPE_MEM,
#endif
};

extern enum file_types_e file_type(const char* name);

extern int write_ra(int fd, unsigned int n, const long dimensions[__VLA(n)]);
extern int read_ra(int fd, unsigned int n, long dimensions[__VLA(n)]);

extern int write_coo(int fd, unsigned int n, const long dimensions[__VLA(n)]);
extern int read_coo(int fd, unsigned int n, long dimensions[__VLA(n)]);

extern int write_cfl_header(int fd, unsigned int n, const long dimensions[__VLA(n)]);
extern int read_cfl_header(int fd, unsigned int D, long dimensions[__VLA(D)]);

extern void io_register_input(const char* name);
extern void io_register_output(const char* name);
extern void io_unregister(const char* name);

extern void io_unlink_if_opened(const char* name);

extern void io_memory_cleanup(void);

#include "misc/cppwrap.h"

