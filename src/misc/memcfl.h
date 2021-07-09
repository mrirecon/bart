
#include <stdbool.h>
#include <complex.h>

extern void memcfl_register(const char* name, int D, const long dims[D], complex float* data, bool managed);
extern bool memcfl_exists(const char* name);
extern complex float* memcfl_create(const char* name, int D, const long dims[D]);
extern complex float* memcfl_load(const char* name, int D, long dims[D]);
extern bool memcfl_unmap(const complex float* p);
extern void memcfl_unlink(const char* name);


