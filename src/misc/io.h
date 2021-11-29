
#include "misc/cppwrap.h"


enum file_types_e {
	FILE_TYPE_CFL, FILE_TYPE_RA, FILE_TYPE_COO, FILE_TYPE_SHM, FILE_TYPE_PIPE, FILE_TYPE_MEM,
};

extern enum file_types_e file_type(const char* name);

extern int write_ra(int fd, int n, const long dimensions[__VLA(n)]);
extern int read_ra(int fd, int n, long dimensions[__VLA(n)]);

extern int write_coo(int fd, int n, const long dimensions[__VLA(n)]);
extern int read_coo(int fd, int n, long dimensions[__VLA(n)]);

extern int write_cfl_header(int fd, const char* filename, int n, const long dimensions[__VLA(n)]);
extern int read_cfl_header(int fd, char** file, int D, long dimensions[__VLA(D)]);

extern int write_multi_cfl_header(int fd, const char* filename, long num_ele, int D, int n[D], const long* dimensions[D]);
extern int read_multi_cfl_header(int fd, char** file, int D_max, int n_max, int n[D_max], long dimensions[D_max][n_max]);

extern void io_register_input(const char* name);
extern void io_register_output(const char* name);
extern void io_reserve_input(const char* name);
extern void io_reserve_output(const char* name);
extern void io_reserve_inout(const char* name);
extern void io_unregister(const char* name);
extern void io_close(const char* name);

extern void io_unlink_if_opened(const char* name);

extern void io_memory_cleanup(void);

#include "misc/cppwrap.h"
