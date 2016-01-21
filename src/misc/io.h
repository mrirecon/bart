
#ifdef __cplusplus
extern "C" {
#define __VLA(x) u
#else
#define __VLA(x) static x
#endif

extern int write_ra(int fd, unsigned int n, const long dimensions[__VLA(n)]);
extern int read_ra(int fd, unsigned int n, long dimensions[__VLA(n)]);

extern int write_coo(int fd, unsigned int n, const long dimensions[__VLA(n)]);
extern int read_coo(int fd, unsigned int n, long dimensions[__VLA(n)]);

extern int write_cfl_header(int fd, unsigned int n, const long dimensions[__VLA(n)]);
extern int read_cfl_header(int fd, unsigned int D, long dimensions[__VLA(D)]);



#ifdef __cplusplus
}
#endif

