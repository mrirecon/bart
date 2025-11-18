
#ifndef _IO_H
#define _IO_H 1

#include "misc/cppwrap.h"

#define IO_MAX_HDR_SIZE 4096
#define IO_MIN_HDR_SIZE  256

#define BART_MAX_DIR_PATH_SIZE 4096

enum file_types_e {
	FILE_TYPE_CFL, FILE_TYPE_RA, FILE_TYPE_COO, FILE_TYPE_SHM, FILE_TYPE_PIPE, FILE_TYPE_MEM,
};

extern int xread(int fd, int n, char buf[n]);
extern int xwrite(int fd, int n, const char buf[n]);

extern enum file_types_e file_type(const char* name);

extern int write_ra(int fd, int n, const long dimensions[__VLA(n)]);
extern int read_ra(int fd, int n, long dimensions[__VLA(n)]);

extern int write_coo(int fd, int n, const long dimensions[__VLA(n)]);
extern int read_coo(int fd, int n, long dimensions[__VLA(n)]);

extern void toolgraph_close(void);
extern void toolgraph_create(const char* tool_name, int argc, char* argv[__VLA(argc)]);

extern int write_cfl_header(int fd, const char* filename, int n, const long dimensions[__VLA(n)]);
extern int read_cfl_header(int fd, const char* hdrname, char** file, char** cmd, int D, long dimensions[__VLA(D)]);
extern int read_cfl_header2(int N, char buf[__VLA(N + 1)], int fd, const char* hdrname, char** file, char** cmd, int D, long dimensions[__VLA(D)]);
extern int parse_cfl_header(long N, const char header[__VLA(N + 1)], char** file, char** cmd, char** node, int n, long dimensions[__VLA(n)]);
extern int write_stream_header(int fd, const char* filename, int n, const long dimensions[n]);

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
extern _Bool io_check_if_opened(const char* name);

extern void io_memory_cleanup(void);

#include "misc/cppwrap.h"

#endif // _IO_H

