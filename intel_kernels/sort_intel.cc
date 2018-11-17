
#include <complex.h>
#include <parallel/algorithm>

static int _compare_cmpl_magn(const float __complex__ & a, const float __complex__ &b)
{
	return cabsf(a) < cabsf(b);
}

extern "C" void gnu_sort_wrapper(float __complex__ * base, size_t len)
{
	__gnu_parallel::sort(base, base+len, _compare_cmpl_magn); 
}
