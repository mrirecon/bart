
#include "num/ops.h"
#include "num/iovec.h"
#include "num/multind.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"

#include "iter/prox.h"
#include "iter/thresh.h"

#include "tgv.h"


struct reg2 tgvreg(unsigned long flags, unsigned long jflags, float lambda, int N, const long in_dims[N])
{
	long out_dims[N];
	struct reg2 reg2;

	const struct linop_s* grad1 = linop_grad_create(N - 1, in_dims, N - 1, flags);
	const struct linop_s* grad2x = linop_grad_create(N + 0, linop_codomain(grad1)->dims, N + 0, flags);


	auto grad2a = linop_transpose_create(N + 1, N - 1, N + 0, linop_codomain(grad2x)->dims);
	auto grad2b = linop_identity_create(N + 1, linop_codomain(grad2x)->dims);
	auto grad2 = linop_chain_FF(grad2x, linop_plus_FF(grad2a, grad2b));


	long grd_dims[N];
	md_copy_dims(N, grd_dims, linop_codomain(grad1)->dims);

	md_copy_dims(N, out_dims, grd_dims);
	out_dims[N - 1]++;


	long pos1[N];

	for (int i = 0; i < N; i++)
		pos1[i] = 0;

	pos1[N - 1] = 0;

	auto grad1b = linop_extract_create(N, pos1, in_dims, out_dims);
	auto grad1c = linop_reshape_create(N - 1, linop_domain(grad1)->dims, N, in_dims);
	auto grad1d = linop_chain_FF(linop_chain_FF(grad1b, grad1c), grad1);


	long pos1b[N];

	for (int i = 0; i < N; i++)
		pos1b[i] = 0;

	pos1b[N - 1] = 1;

	auto grad1e = linop_extract_create(N, pos1b, grd_dims, out_dims);
	reg2.linop[0] = linop_plus_FF(grad1e, grad1d);


	long pos2[N];

	for (int i = 0; i < N; i++)
		pos2[i] = 0;

	pos2[N - 1] = 1;

	auto grad2e = linop_extract_create(N, pos2, grd_dims, out_dims);
	reg2.linop[1] = linop_chain_FF(grad2e, grad2);

	reg2.prox[0] = prox_thresh_create(N + 0, linop_codomain(reg2.linop[0])->dims, lambda, jflags);
	reg2.prox[1] = prox_thresh_create(N + 1, linop_codomain(reg2.linop[1])->dims, lambda, jflags);

	return reg2;
}
