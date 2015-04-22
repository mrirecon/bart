
#include "num/multind.h"

#include "misc/debug.h"
#include "misc/misc.h"

#include "shuffle.h"


#if 0
void md_shuffle2(unsigned int N, const long dims[N], const long factors[N],
		const long ostrs[N], void* out, const long istrs[N], const void* in, size_t size)
{
	long dims2[2 * N];
	long ostrs2[2 * N];
	long istrs2[2 * N];

	for (unsigned int i = 0; i < N; i++) {

		assert(0 == dims[i] % factors[i]);

		long f2 = dims[i] / factors[i];

		dims2[0 * N + i] = f2;
		dims2[1 * N + i] = factors[i];

		ostrs2[1 * N + i] = ostrs[i];
		ostrs2[0 * N + i] = ostrs[i] * f2;

		istrs2[0 * N + i] = istrs[i] * factors[i];
		istrs2[1 * N + i] = istrs[i];
	}

	md_copy2(2 * N, dims2, ostrs2, out, istrs2, in, size);
}

void md_shuffle(unsigned int N, const long dims[N], const long factors[N],
		void* out, const void* in, size_t size)
{
	long strs[N];
	md_calc_strides(N, strs, dims, size);

	md_shuffle2(N, dims, factors, strs, out, strs, in, size);
}
#endif


static void decompose_dims(unsigned int N, long dims2[2 * N], long ostrs2[2 * N], long istrs2[2 * N],
		const long factors[N], const long odims[N + 1], const long ostrs[N + 1], const long idims[N], const long istrs[N])
{
	long prod = 1;

	for (unsigned int i = 0; i < N; i++) {

		long f2 = idims[i] / factors[i];

		assert(0 == idims[i] % factors[i]);
		assert(odims[i] == idims[i] / factors[i]);
		
		dims2[1 * N + i] = factors[i];
		dims2[0 * N + i] = f2;
	
		istrs2[0 * N + i] = istrs[i] * factors[i];
		istrs2[1 * N + i] = istrs[i];

		ostrs2[0 * N + i] = ostrs[i];
		ostrs2[1 * N + i] = ostrs[N] * prod;

		prod *= factors[i];
	}

	assert(odims[N] == prod);
}

void md_decompose2(unsigned int N, const long factors[N],
		const long odims[N + 1], const long ostrs[N + 1], void* out,
		const long idims[N], const long istrs[N], const void* in, size_t size)
{
	long dims2[2 * N];
	long ostrs2[2 * N];
	long istrs2[2 * N];

	decompose_dims(N, dims2, ostrs2, istrs2, factors, odims, ostrs, idims, istrs);

	md_copy2(2 * N, dims2, ostrs2, out, istrs2, in, size);
}

void md_decompose(unsigned int N, const long factors[N], const long odims[N + 1], 
		void* out, const long idims[N], const void* in, size_t size)
{
	long ostrs[N + 1];
	md_calc_strides(N + 1, ostrs, odims, size);

	long istrs[N];
	md_calc_strides(N, istrs, idims, size);

	md_decompose2(N, factors, odims, ostrs, out, idims, istrs, in, size);
}

void md_recompose2(unsigned int N, const long factors[N],
		const long odims[N], const long ostrs[N], void* out,
		const long idims[N + 1], const long istrs[N + 1], const void* in, size_t size)
{
	long dims2[2 * N];
	long ostrs2[2 * N];
	long istrs2[2 * N];

	decompose_dims(N, dims2, istrs2, ostrs2, factors, idims, istrs, odims, ostrs);

	md_copy2(2 * N, dims2, ostrs2, out, istrs2, in, size);
}

void md_recompose(unsigned int N, const long factors[N], const long odims[N], 
		void* out, const long idims[N + 1], const void* in, size_t size)
{
	long ostrs[N];
	md_calc_strides(N, ostrs, odims, size);

	long istrs[N + 1];
	md_calc_strides(N + 1, istrs, idims, size);

	md_recompose2(N, factors, odims, ostrs, out, idims, istrs, in, size);
}

