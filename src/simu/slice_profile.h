
#include <complex.h>

struct sim_data;
struct simdata_pulse;

extern void slice_profile_fourier(int N, const long dims[N], complex float* out, const struct simdata_pulse* pulse);

