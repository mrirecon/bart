#include <complex.h>
#include <stdbool.h>

#include "num/init.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/someops.h"
#include "linops/sum.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"
#include "nlops/nltest.h"

#include "nn/pytorch_wrapper.h"

#include "utest.h"

//import torch
//
//class MyModule(torch.nn.Module):
//	def __init__(self):
//		super(MyModule, self).__init__()
//
//	def forward(self, arg1, arg2):
//		out1 = arg1 * arg2
//		out1 = out1.real
//		out2 = torch.sum(out1, dim = 0, keepdim = True)    
//		return out1, out2
//
//my_module = MyModule()
//sm = torch.jit.script(my_module)
//sm.save("./test_nn_pytorch.pt")


static bool test_nn_pytorch(void)
{
	enum { N = 2 };

	long dims[N] = { 5, 3};
	long dims1[N] = { 5, 1};
	long dims2[N] = { 1, 3};
	long dims0[N] = { 1, 1};

	const struct nlop_s* nlop = nlop_tenmul_create(N, dims, dims1, dims2);
	nlop = nlop_append_FF(nlop, 0, nlop_from_linop_F(linop_zreal_create(N, dims)));
	nlop = nlop_chain2_keep_FF(nlop, 0, nlop_from_linop_F(linop_sum_create(N, dims, MD_BIT(1))), 0);
	nlop = nlop_shift_output_F(nlop, 1, 0);

	const struct nlop_s* nlop_pytorch = nlop_pytorch_create("./utests/test_nn_pytorch.pt", 2, (int[2]) { N, N }, (const long*[2]) {dims1, dims2 }, false);

	nlop_debug(DP_DEBUG1, nlop_pytorch);

	bool okay = compare_nlops(nlop, nlop_pytorch, true, false, true, UT_TOL);
	
	nlop_free(nlop_pytorch);
	nlop_free(nlop);

	UT_RETURN_ASSERT(okay);
}

UT_REGISTER_TEST(test_nn_pytorch);

