/* Copyright 2018. Massachusetts Institute of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2018 Siddharth Iyer <ssi@mit.edu>
 *
 * Tamir J, Uecker M, Chen W, Lai P, Alley MT, Vasanawala SS, Lustig M. 
 * T2 shuffling: Sharp, multicontrast, volumetric fast spin‐echo imaging. 
 * Magnetic resonance in medicine. 2017 Jan 1;77(1):180-95.
 *
 * B Bilgic, BA Gagoski, SF Cauley, AP Fan, JR Polimeni, PE Grant,
 * LL Wald, and K Setsompop, Wave-CAIPI for highly accelerated 3D
 * imaging. Magn Reson Med (2014) doi: 10.1002/mrm.25347
 *
 * Iyer S, Bilgic B, Setsompop K.
 * Faster T2 shuffling with Wave.
 * Presented in the session: "Signal Encoding and Decoding" at ISMRM 2018.
 * https://www.ismrm.org/18/program_files/O67.htm 
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/iovec.h"
#include "num/ops.h"

#include "iter/iter.h"
#include "iter/lsqr.h"
#include "iter/misc.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/someops.h"
#include "sense/model.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "wavelet/wavthresh.h"
#include "lowrank/lrthresh.h"

static const char usage_str[] = "<maps> <wave> <phi> <reorder> <table> <output>";
static const char help_str[]  = 
	"Perform a wave-shuffling reconstruction.\n\n"
	"Conventions:\n"
	"  * (sx, sy, sz) - Spatial dimensions.\n"
	"  * wx           - Extended FOV in READ_DIM due to\n"
	"                   wave's voxel spreading.\n"
	"  * (nc, md)     - Number of channels and ESPIRiT's \n"
	"                   extended-SENSE model operator\n"
	"                   dimensions (or # of maps).\n"
	"  * (tf, tk)     - Turbo-factor and the rank\n"
	"                   of the temporal basis used in\n"
	"                   shuffling.\n"
	"  * ntr          - Number of TRs, or the number of\n"
	"                   (ky, kz) points acquired of one\n"
	"                   echo image.\n"
	"  * n            - Total number of (ky, kz) points\n"
	"                   acquired. This is equal to the\n"
	"                   product of ntr and tf.\n\n"
	"Descriptions:\n"
	"  * reorder is an (n by 3) index matrix such that\n"
	"    [ky, kz, t] = reorder(i, :) represents the\n"
	"    (ky, kz) kspace position of the readout line\n" 
	"    acquired at echo number (t), and 0 <= ky < sy,\n"
	"    0 <= kz < sz, 0 <= t < tf).\n"
	"  * table is a (wx by nc by n) matrix such that\n"
	"    table(:, :, k) represents the kth multichannel\n"
	"    kspace line.\n\n"
	"Expected dimensions:\n"
	"  * maps    - (   sx, sy, sz, nc, md,  1,  1)\n"
	"  * wave    - (   wx, sy, sz,  1,  1,  1,  1)\n"
                                "  * phi     - (    1,  1,  1,  1,  1, tf, tk)\n"
	"  * output  - (   sx, sy, sz,  1, md,  1, tk)\n"
	"  * reorder - (    n,  3,  1,  1,  1,  1,  1)\n"
	"  * table   - (   wx, nc,  n,  1,  1,  1,  1)";

/* Helper function to print out operator dimensions. */
static void print_opdims(const struct linop_s* op) 
{
	const struct iovec_s* domain   = linop_domain(op);
	const struct iovec_s* codomain = linop_codomain(op);
	debug_printf(DP_INFO, "\tDomain:   ");
	debug_print_dims(DP_INFO, domain->N, domain->dims);
	debug_printf(DP_INFO, "\tCodomain: ");
	debug_print_dims(DP_INFO, codomain->N, codomain->dims);
}

/* Construct sampling mask array from reorder tables. */
static void construct_mask(
	long reorder_dims[DIMS], complex float* reorder, 
	long mask_dims[DIMS],    complex float* mask)
{
	long n  = reorder_dims[0];
	long sy = mask_dims[1];
	long sz = mask_dims[2];

	long y = 0;
	long z = 0;
	long t = 0;

	for (int i = 0; i < n; i++) {
		y = lround(creal(reorder[i]));
		z = lround(creal(reorder[i + n]));
		t = lround(creal(reorder[i + 2 * n]));
		mask[(y + z * sy) + t * sy * sz] = 1;
	}
}

static DEF_TYPEID(kern_s);

struct kern_s {
	INTERFACE(linop_data_t);

	unsigned int N;

	long* reorder_dims; // Dimension of the index table:    ( n,  3,  1,  1, 1,  1,  1,  1)
	long* phi_dims;     // Dimension of the temporal basis: ( 1,  1,  1,  1, 1, tf, tk,  1)
	long* table_dims;   // Dimension of the data table:     (wx, nc,  n,  1, 1,  1,  1,  1)
	long* kernel_dims;  // Dimension of the kernel:         ( 1, sy, sz,  1, 1,  1, tk, tk)

	complex float* reorder;
	complex float* phi;
	complex float* kernel;
};

/* Go to table from coefficient-kspace with memory efficiency. */
static void kern_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct kern_s* data = CAST_DOWN(kern_s, _data);

	long wx = data->table_dims[0];
	long sy = data->kernel_dims[1];
	long sz = data->kernel_dims[2];
	long nc = data->table_dims[1];
	long n  = data->reorder_dims[0];
	long tf = data->phi_dims[5];
	long tk = data->phi_dims[6];

	long input_dims[] = { [0 ... DIMS - 1] = 1 };
	input_dims[0] = wx;
	input_dims[1] = sy;
	input_dims[2] = sz;
	input_dims[3] = nc;
	input_dims[6] = tk;

	long perm_dims[] = { [0 ... DIMS - 1] = 1 };
	perm_dims[0] = wx;
	perm_dims[1] = nc;
	perm_dims[3] = tk;
	perm_dims[4] = sy;
	perm_dims[5] = sz;
	complex float* perm = md_alloc_sameplace(DIMS, perm_dims, CFL_SIZE, src);
	unsigned int permute_order[DIMS] = {0, 3, 5, 6, 1, 2, 4, 7};
	for (unsigned int i = 8; i < DIMS; i++)
		permute_order[i] = i;
	md_permute(DIMS, permute_order, perm_dims, perm, input_dims, src, CFL_SIZE);

	long vec_dims[]     = {wx, nc, tf,  1};
	long phi_mat_dims[] = { 1,  1, tf, tk};
	long phi_in_dims[]  = {wx, nc,  1, tk};
	long fmac_dims[]    = {wx, nc, tf, tk};
	long line_dims[]    = {wx, nc,  1,  1};

	complex float* vec = md_alloc_sameplace(4, vec_dims, CFL_SIZE, src);

	long vec_str[4];
	md_calc_strides(4, vec_str, vec_dims, CFL_SIZE);
	long phi_mat_str[4];
	md_calc_strides(4, phi_mat_str, phi_mat_dims, CFL_SIZE);
	long phi_in_str[4];
	md_calc_strides(4, phi_in_str, phi_in_dims, CFL_SIZE);
	long fmac_str[4];
	md_calc_strides(4, fmac_str, fmac_dims, CFL_SIZE);

	int y = -1;
	int z = -1;
	int t = -1;

	for (int i = 0; i < n; i ++) {

		y = lround(creal(data->reorder[i]));
		z = lround(creal(data->reorder[i + n]));
		t = lround(creal(data->reorder[i + 2 * n]));

		md_clear(4, vec_dims, vec, CFL_SIZE);
		md_zfmac2(4, fmac_dims, vec_str, vec, phi_in_str, (perm + ((wx * nc * tk) * (y + z * sy))), phi_mat_str, data->phi);
		md_copy(4, line_dims, dst + (i * wx * nc), vec + (t * wx * nc), CFL_SIZE);
	}

	md_free(perm);
	md_free(vec);
}

/* Collapse data table into the temporal basis for memory efficiency. */
static void kern_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	struct kern_s* data = CAST_DOWN(kern_s, _data);

	long wx = data->table_dims[0];
	long sy = data->kernel_dims[1];
	long sz = data->kernel_dims[2];
	long nc = data->table_dims[1];
	long n  = data->reorder_dims[0];
	long tf = data->phi_dims[5];
	long tk = data->phi_dims[6];

	long perm_dims[] = { [0 ... DIMS - 1] = 1 };
	perm_dims[0] = wx;
	perm_dims[1] = nc;
	perm_dims[3] = tk;
	perm_dims[4] = sy;
	perm_dims[5] = sz;

	complex float* perm = md_alloc_sameplace(DIMS, perm_dims, CFL_SIZE, dst);
	md_clear(DIMS, perm_dims, perm, CFL_SIZE);

	long vec_dims[]     = {wx, nc, tf,  1};
	long phi_mat_dims[] = { 1,  1, tf, tk};
	long phi_out_dims[] = {wx, nc,  1, tk};
	long fmac_dims[]    = {wx, nc, tf, tk};
	long line_dims[]    = {wx, nc,  1,  1};

	complex float* vec = md_alloc_sameplace(4, vec_dims, CFL_SIZE, src);

	long vec_str[4];
	md_calc_strides(4, vec_str, vec_dims, CFL_SIZE);
	long phi_mat_str[4];
	md_calc_strides(4, phi_mat_str, phi_mat_dims, CFL_SIZE);
	long phi_out_str[4];
	md_calc_strides(4, phi_out_str, phi_out_dims, CFL_SIZE);
	long fmac_str[4];
	md_calc_strides(4, fmac_str, fmac_dims, CFL_SIZE);

	int t = -1;
	for (int z = 0; z < sz; z ++) {
		for (int y = 0; y < sy; y ++) {

			md_clear(4, vec_dims, vec, CFL_SIZE);

			for (int i = 0; i < n; i ++) {
				if ((y == lround(creal(data->reorder[i]))) && (z == lround(creal(data->reorder[i + n])))) {
					t = lround(creal(data->reorder[i + 2 * n]));
					md_copy(4, line_dims, (vec + t * wx * nc), (src + i * wx * nc), CFL_SIZE);
				}
			}

			md_zfmacc2(4, fmac_dims, phi_out_str, perm + (y + z * sy) * (wx * nc * tk), vec_str, vec, phi_mat_str, data->phi);

		}
	}

	long out_dims[] = { [0 ... DIMS - 1] = 1 };
	out_dims[0] = wx;
	out_dims[1] = sy;
	out_dims[2] = sz;
	out_dims[3] = nc;
	out_dims[6] = tk;
	unsigned int permute_order[DIMS] = {0, 4, 5, 1, 6, 2, 3, 7};
	for (unsigned int i = 8; i < DIMS; i++)
		permute_order[i] = i;
	md_permute(DIMS, permute_order, out_dims, dst, perm_dims, perm, CFL_SIZE);

	md_free(vec);
	md_free(perm);
}

static void kern_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct kern_s* data = CAST_DOWN(kern_s, _data);

	long wx = data->table_dims[0];
	long sy = data->kernel_dims[1];
	long sz = data->kernel_dims[2];
	long nc = data->table_dims[1];
	long tk = data->phi_dims[6];

	long input_dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	input_dims[0] = wx;
	input_dims[1] = sy;
	input_dims[2] = sz;
	input_dims[3] = nc;
	input_dims[6] = tk;
	long input_str[DIMS];
	md_calc_strides(DIMS, input_str, input_dims, CFL_SIZE);

	long output_dims[DIMS];
	md_copy_dims(DIMS, output_dims, input_dims);
	output_dims[6] = 1;
	output_dims[7] = tk;
	long output_str[DIMS];
	md_calc_strides(DIMS, output_str, output_dims, CFL_SIZE);

	long kernel_str[DIMS];
	md_calc_strides(DIMS, kernel_str, data->kernel_dims, CFL_SIZE);

	long fmac_dims[DIMS];
	md_merge_dims(DIMS, fmac_dims, input_dims, data->kernel_dims);

	md_clear(DIMS, output_dims, dst, CFL_SIZE);
	md_zfmac2(DIMS, fmac_dims, output_str, dst, input_str, src, kernel_str, data->kernel);
}

static void kern_free(const linop_data_t* _data)
{
	const struct kern_s* data = CAST_DOWN(kern_s, _data);

	xfree(data->reorder_dims);
	xfree(data->phi_dims);
	xfree(data->table_dims);
	xfree(data->kernel_dims);

	xfree(data);
}

static const struct linop_s* linop_kern_create(long N,
	const long _reorder_dims[N], complex float* reorder,
	const long _phi_dims[N],     complex float* phi,
	const long _kernel_dims[N],  complex float* kernel,
	const long _table_dims[N])
{
	PTR_ALLOC(struct kern_s, data);
	SET_TYPEID(kern_s, data);

	data->N = N;

	PTR_ALLOC(long[N], reorder_dims);
	PTR_ALLOC(long[N], phi_dims);
	PTR_ALLOC(long[N], table_dims);
	PTR_ALLOC(long[N], kernel_dims);

	md_copy_dims(N, *reorder_dims, _reorder_dims);
	md_copy_dims(N, *phi_dims,     _phi_dims);
	md_copy_dims(N, *table_dims,   _table_dims);
	md_copy_dims(N, *kernel_dims,  _kernel_dims);

	data->reorder_dims = *PTR_PASS(reorder_dims);
	data->phi_dims     = *PTR_PASS(phi_dims);
	data->table_dims   = *PTR_PASS(table_dims);
	data->kernel_dims  = *PTR_PASS(kernel_dims);

	data->reorder = reorder;
	data->phi     = phi;
	data->kernel  = kernel;

	long input_dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	input_dims[0] = _table_dims[0];
	input_dims[1] = _kernel_dims[1];
	input_dims[2] = _kernel_dims[2];
	input_dims[3] = _table_dims[1];
	input_dims[6] = _phi_dims[6];

	long output_dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	output_dims[0] = _table_dims[0];
	output_dims[1] = _table_dims[1];
	output_dims[2] = _reorder_dims[0];

	const struct linop_s* K = linop_create(N, output_dims, N, input_dims, CAST_UP(PTR_PASS(data)), kern_apply, kern_adjoint, kern_normal, NULL, kern_free);
	return K;
}

/* ESPIRiT operator. */
static const struct linop_s* linop_espirit_create(long sx, long sy, long sz, long nc, long md, long tk, complex float* maps)
{
	long max_dims[] = { [0 ... DIMS - 1] = 1};
	max_dims[0] = sx;
	max_dims[1] = sy;
	max_dims[2] = sz;
	max_dims[3] = nc;
	max_dims[4] = md;
	max_dims[6] = tk;

	const struct linop_s* E = linop_fmac_create(DIMS, max_dims, MAPS_FLAG, COIL_FLAG, TE_FLAG|COEFF_FLAG, maps);
 
	return E;
}

/* Resize operator. */
static const struct linop_s* linop_reshape_create(long wx, long sx, long sy, long sz, long nc, long tk)
{
	long input_dims[] = { [0 ... DIMS - 1] = 1};
	input_dims[0] = sx;
	input_dims[1] = sy;
	input_dims[2] = sz;
	input_dims[3] = nc;
	input_dims[6] = tk;
	long output_dims[DIMS];
	md_copy_dims(DIMS, output_dims, input_dims);
	output_dims[0] = wx;
	struct linop_s* R = linop_resize_create(DIMS, output_dims, input_dims);
	return R;
}

/* Fx operator. */
static const struct linop_s* linop_fx_create(long wx, long sy, long sz, long nc, long tk)
{
	long dims[] = { [0 ... DIMS - 1] = 1};
	dims[0] = wx;
	dims[1] = sy;
	dims[2] = sz;
	dims[3] = nc;
	dims[6] = tk;
	struct linop_s* Fx = linop_fft_create(DIMS, dims, READ_FLAG);
	return Fx;
}

/* Wave operator. */
static const struct linop_s* linop_wave_create(long wx, long sy, long sz, long nc, long tk, complex float* psf)
{
	long dims[] = { [0 ... DIMS - 1] = 1};
	dims[0] = wx;
	dims[1] = sy;
	dims[2] = sz;
	dims[3] = nc;
	dims[6] = tk;
	struct linop_s* W = linop_cdiag_create(DIMS, dims, FFT_FLAGS, psf);
	return W;
}

/* Fyz operator. */
static const struct linop_s* linop_fyz_create(long wx, long sy, long sz, long nc, long tk)
{
	long dims[] = { [0 ... DIMS - 1] = 1};
	dims[0] = wx;
	dims[1] = sy;
	dims[2] = sz;
	dims[3] = nc;
	dims[6] = tk;
	struct linop_s* Fyz = linop_fft_create(DIMS, dims, PHS1_FLAG|PHS2_FLAG);
	return Fyz;
}

/* Construction sampling temporal kernel.*/
static void construct_kernel(
	long mask_dims[DIMS], complex float* mask,
	long phi_dims[DIMS],  complex float* phi, 
	long kern_dims[DIMS], complex float* kern)
{
	long sy = mask_dims[1];
	long sz = mask_dims[2];
	long tf = phi_dims[5];
	long tk = phi_dims[6];

	long cvec_dims[] = { [0 ... DIMS - 1] = 1 };
	cvec_dims[6] = tk;
	long cvec_str[DIMS];
	md_calc_strides(DIMS, cvec_str, cvec_dims, CFL_SIZE);

	complex float cvec[tk];

	long tvec_dims[] = { [0 ... DIMS - 1] = 1 };
	tvec_dims[5] = tf;
	long tvec_str[DIMS];
	md_calc_strides(DIMS, tvec_str, tvec_dims, CFL_SIZE);

	complex float mvec[tf];
	complex float tvec1[tf];
	complex float tvec2[tf];

	long phi_str[DIMS];
	md_calc_strides(DIMS, phi_str, phi_dims, CFL_SIZE);

	long out_dims[] = { [0 ... DIMS - 1] = 1 };
	out_dims[0] = tk;
	out_dims[1] = sy;
	out_dims[2] = sz;
	out_dims[3] = tk;
	complex float* out = md_calloc(DIMS, out_dims, CFL_SIZE);

	for (int y = 0; y < sy; y ++) {
		for (int z = 0; z < sz; z ++) {

			for (int t = 0; t < tf; t ++)
				mvec[t] = mask[(y + sy * z) + (sy * sz) * t];

			for (int t = 0; t < tk; t ++) {
				cvec[t] = 1;

				md_clear(DIMS, tvec_dims, tvec1, CFL_SIZE);
				md_zfmac2(DIMS, phi_dims, tvec_str, tvec1, cvec_str, cvec, phi_str, phi);

				md_clear(DIMS, tvec_dims, tvec2, CFL_SIZE);
				md_zfmac2(DIMS, tvec_dims, tvec_str, tvec2, tvec_str, tvec1, tvec_str, mvec);

				md_clear(DIMS, cvec_dims, out + y * tk + z * sy * tk + t * sy * sz * tk, CFL_SIZE);
				md_zfmacc2(DIMS, phi_dims, cvec_str, out + y * tk + z * sy * tk + t * sy * sz * tk,
				tvec_str, tvec2, phi_str, phi);

				cvec[t] = 0;
			}
		}
	}

	unsigned int permute_order[DIMS] = {4, 1, 2, 5, 6, 7, 3, 0};
	for (unsigned int i = 8; i < DIMS; i++)
		permute_order[i] = i;

	md_permute(DIMS, permute_order, kern_dims, kern, out_dims, out, CFL_SIZE);
	md_free(out);
}

static void fftmod_apply(long sy, long sz,
	long reorder_dims[DIMS], complex float* reorder, 
	long table_dims[DIMS],   complex float* table,
	long maps_dims[DIMS],    complex float* maps)
{
	long wx = table_dims[0];
	long nc = table_dims[1];

	fftmod(DIMS, table_dims, READ_FLAG, table, table);
	fftmod(DIMS, maps_dims, FFT_FLAGS, maps, maps);

	long y = -1;
	long z = -1;

	double dy = ((double) sy/2)/((double) sy);
	double dz = ((double) sz/2)/((double) sz);

	complex float py = 1;
	complex float pz = 1;

	long dims[] = { [0 ... DIMS] = 1};
	dims[0] = wx;
	dims[1] = nc;

	long n = reorder_dims[0];
	for (long k = 0; k < n; k++) {
		y = lround(creal(reorder[k]));
		z = lround(creal(reorder[k + n]));

		py = cexp(2.i * M_PI * dy * y);
		pz = cexp(2.i * M_PI * dz * z);

		md_zsmul(DIMS, dims, table + k * wx * nc, table + k * wx * nc, py * pz);
	}
}

enum algo_t { CG, IST, FISTA };

int main_wshfl(int argc, char* argv[])
{
	double start_time = timestamp();

	float lambda    = 1E-5;
	int   maxiter   = 300;
	int   blksize   = 8;
	float step      = 0.5;
	float tol       = 1.E-3;
	bool  llr       = false;
	bool  wav       = false;
	bool  fista     = false;
	bool  hgwld     = false;
	float cont      = 1;
	float eval      = -1;
	const char* fwd = NULL;

	const struct opt_s opts[] = {
		OPT_FLOAT( 'r', &lambda,  "lambda", "Soft threshold lambda for wavelet or locally low rank."),
		OPT_INT(   'b', &blksize, "blkdim", "Block size for locally low rank."),
		OPT_INT(   'i', &maxiter, "mxiter", "Maximum number of iterations."),
		OPT_FLOAT( 's', &step,    "stepsz", "Step size for iterative method."),
		OPT_FLOAT( 'c', &cont,    "cntnu",  "Continuation value for IST/FISTA."),
		OPT_FLOAT( 't', &tol,     "toler",  "Tolerance convergence condition for iterative method."),
		OPT_FLOAT( 'e', &eval,    "eigvl",  "Maximum eigenvalue of normal operator, if known."),
		OPT_STRING('F', &fwd,     "frwrd",  "Go from shfl-coeffs to data-table. Pass in coeffs path."),
		OPT_SET(   'f', &fista,             "Reconstruct using FISTA instead of IST."),
		OPT_SET(   'H', &hgwld,             "Use hogwild in IST/FISTA."),
		OPT_SET(   'w', &wav,               "Use wavelet."),
		OPT_SET(   'l', &llr,               "Use locally low rank."),
	};

	cmdline(&argc, argv, 6, 6, usage_str, help_str, ARRAY_SIZE(opts), opts);

	debug_printf(DP_INFO, "Loading data... ");

	long maps_dims[DIMS];
	complex float* maps = load_cfl(argv[1], DIMS, maps_dims);

	long wave_dims[DIMS];
	complex float* wave = load_cfl(argv[2], DIMS, wave_dims);

	long phi_dims[DIMS];
	complex float* phi = load_cfl(argv[3], DIMS, phi_dims);

	long reorder_dims[DIMS];
	complex float* reorder = load_cfl(argv[4], DIMS, reorder_dims);

	long table_dims[DIMS];
	complex float* table = load_cfl(argv[5], DIMS, table_dims);

	debug_printf(DP_INFO, "Done.\n");

	num_init();

	int wx = wave_dims[0];
	int sx = maps_dims[0];
	int sy = maps_dims[1];
	int sz = maps_dims[2];
	int nc = maps_dims[3];
	int md = maps_dims[4];
	int tf = phi_dims[5];
	int tk = phi_dims[6];

	debug_printf(DP_INFO, "Constructing sampling mask from reorder table... ");
	long mask_dims[] = { [0 ... DIMS - 1] = 1 };
	mask_dims[1] = sy;
	mask_dims[2] = sz;
	mask_dims[5] = tf;
	complex float* mask = md_calloc(DIMS, mask_dims, CFL_SIZE);
	construct_mask(reorder_dims, reorder, mask_dims, mask);
	debug_printf(DP_INFO, "Done.\n");

	debug_printf(DP_INFO, "Constructing sampling-temporal kernel... ");
	long kernel_dims[] = { [0 ... DIMS - 1] = 1 };
	kernel_dims[1] = sy;
	kernel_dims[2] = sz;
	kernel_dims[6] = tk;
	kernel_dims[7] = tk;
	complex float* kernel = md_calloc(DIMS, kernel_dims, CFL_SIZE);
	construct_kernel(mask_dims, mask, phi_dims, phi, kernel_dims, kernel);
	md_free(mask);
	debug_printf(DP_INFO, "Done.\n");

	long coeff_dims[] = { [0 ... DIMS - 1] = 1 };
	coeff_dims[0] = sx;
	coeff_dims[1] = sy;
	coeff_dims[2] = sz;
	coeff_dims[4] = md;
	coeff_dims[6] = tk;

	debug_printf(DP_INFO, "Linear operator.\n");
	const struct linop_s* E   = linop_espirit_create(sx, sy, sz, nc, md, tk, maps);
	const struct linop_s* R   = linop_reshape_create(wx, sx, sy, sz, nc, tk);
	const struct linop_s* Fx  = linop_fx_create(wx, sy, sz, nc, tk);
	const struct linop_s* W   = linop_wave_create(wx, sy, sz, nc, tk, wave);
	const struct linop_s* Fyz = linop_fyz_create(wx, sy, sz, nc, tk);
	const struct linop_s* K   = linop_kern_create(DIMS, reorder_dims, reorder, phi_dims, phi, kernel_dims, kernel, table_dims);

	struct linop_s* A = linop_chain(linop_chain(linop_chain(linop_chain(linop_chain(
		E, R), Fx), W), Fyz), K);

	linop_free(E);
	linop_free(R);
	linop_free(Fx);
	linop_free(W);
	linop_free(Fyz);
	linop_free(K);

	print_opdims(A);

	if (fwd != NULL) {
		debug_printf(DP_INFO, "Going from coefficients to data table... ");
		complex float* coeffs_to_fwd = load_cfl(fwd, DIMS, coeff_dims);
		complex float* table_forward = create_cfl(argv[6], DIMS, table_dims);
		operator_apply(A->forward, DIMS, table_dims, table_forward, DIMS, coeff_dims, coeffs_to_fwd);
		unmap_cfl(DIMS, table_dims, table_forward);
		debug_printf(DP_INFO, "Done. Output table not normalized and not centered for fft.\n");
		return 0;
	}

	if (eval < 0)	
		eval = estimate_maxeigenval(A->normal);
	debug_printf(DP_INFO, "\tMax eval: %.2e\n", eval);
	step /= eval;

	debug_printf(DP_INFO, "Normalizing data table and applying fftmod to table... ");
	float norm = md_znorm(DIMS, table_dims, table);
	md_zsmul(DIMS, table_dims, table, table, 1. / norm);
	fftmod_apply(sy, sz, reorder_dims, reorder, table_dims, table, maps_dims, maps);
	debug_printf(DP_INFO, "Done.\n");

	const struct operator_p_s* T = NULL;
	long blkdims[MAX_LEV][DIMS];
	long minsize[] = { [0 ... DIMS - 1] = 1 };
	minsize[0] = MIN(sx, 16);
	minsize[1] = MIN(sy, 16);
	minsize[2] = MIN(sz, 16);
	unsigned int WAVFLAG = (sx > 1) * READ_FLAG | (sy > 1) * PHS1_FLAG | (sz > 2) * PHS2_FLAG;

	enum algo_t algo = CG;
	if ((wav == true) || (llr == true)) {
		algo = (fista) ? FISTA : IST;
		if (wav) {
			debug_printf(DP_INFO, "Creating wavelet threshold operator... ");
			T = prox_wavelet_thresh_create(DIMS, coeff_dims, WAVFLAG, 0u, minsize, lambda, true);
		} else {
			debug_printf(DP_INFO, "Creating locally low rank threshold operator... ");
			llr_blkdims(blkdims, ~COEFF_DIM, coeff_dims, blksize);
			T = lrthresh_create(coeff_dims, true, ~COEFF_FLAG, (const long (*)[])blkdims, lambda, false, false);
		}
		debug_printf(DP_INFO, "Done.\n");
	}

	italgo_fun2_t italgo = iter2_call_iter;
	struct iter_call_s iter2_data;
	SET_TYPEID(iter_call_s, &iter2_data);
	iter_conf* iconf = CAST_UP(&iter2_data);

	struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;
	struct iter_fista_conf    fsconf = iter_fista_defaults;
	struct iter_ist_conf      isconf = iter_ist_defaults;

	switch(algo) {

		case IST:

			debug_printf(DP_INFO, "Using IST.\n");
			debug_printf(DP_INFO, "\tLambda:             %0.2e\n", lambda);
			debug_printf(DP_INFO, "\tMaximum iterations: %d\n", maxiter);
			debug_printf(DP_INFO, "\tStep size:          %0.2e\n", step);
			debug_printf(DP_INFO, "\tHogwild:            %d\n", (int) hgwld);
			debug_printf(DP_INFO, "\tTolerance:          %0.2e\n", tol);
			debug_printf(DP_INFO, "\tContinuation:       %0.2e\n", cont);

			isconf              = iter_ist_defaults;
			isconf.step         = step;
			isconf.maxiter      = maxiter;
			isconf.tol          = tol;
			isconf.continuation = cont;
			isconf.hogwild      = hgwld;

			iter2_data.fun   = iter_ist;
			iter2_data._conf = CAST_UP(&isconf);

			break;

		case FISTA:

			debug_printf(DP_INFO, "Using FISTA.\n");
			debug_printf(DP_INFO, "\tLambda:             %0.2e\n", lambda);
			debug_printf(DP_INFO, "\tMaximum iterations: %d\n", maxiter);
			debug_printf(DP_INFO, "\tStep size:          %0.2e\n", step);
			debug_printf(DP_INFO, "\tHogwild:            %d\n", (int) hgwld);
			debug_printf(DP_INFO, "\tTolerance:          %0.2e\n", tol);
			debug_printf(DP_INFO, "\tContinuation:       %0.2e\n", cont);

			fsconf              = iter_fista_defaults;
			fsconf.maxiter      = maxiter;
			fsconf.step         = step;
			fsconf.hogwild      = hgwld;
			fsconf.tol          = tol;
			fsconf.continuation = cont;

			iter2_data.fun   = iter_fista;
			iter2_data._conf = CAST_UP(&fsconf);

			break;

		default:
		case CG:

			debug_printf(DP_INFO, "Using CG.\n");
			debug_printf(DP_INFO, "\tMaximum iterations: %d\n", maxiter);
			debug_printf(DP_INFO, "\tTolerance:          %0.2e\n", tol);

			cgconf          = iter_conjgrad_defaults;
			cgconf.maxiter  = maxiter;
			cgconf.l2lambda = 0;
			cgconf.tol      = tol;

			iter2_data.fun   = iter_conjgrad;
			iter2_data._conf = CAST_UP(&cgconf);

			break;

	}

	debug_printf(DP_INFO, "Reconstruction... ");
	complex float* recon = create_cfl(argv[6], DIMS, coeff_dims);
	struct lsqr_conf lsqr_conf = { 0., false };
	double recon_start = timestamp();
	const struct operator_s* J = lsqr2_create(&lsqr_conf, italgo, iconf, NULL, A, NULL, 1, &T, NULL, NULL);
	operator_apply(J, DIMS, coeff_dims, recon, DIMS, table_dims, table);
	double recon_end = timestamp();
	debug_printf(DP_INFO, "Done.\nReconstruction time: %f seconds.\n", recon_end - recon_start);

	debug_printf(DP_INFO, "Cleaning up and saving result... ");
	operator_free(J);
	linop_free(A);
	md_free(kernel);
	unmap_cfl(DIMS, maps_dims, maps);
	unmap_cfl(DIMS, wave_dims, wave);
	unmap_cfl(DIMS, phi_dims, phi);
	unmap_cfl(DIMS, reorder_dims, reorder);
	unmap_cfl(DIMS, table_dims, table);
	unmap_cfl(DIMS, coeff_dims, recon);
	debug_printf(DP_INFO, "Done.\n");

	double end_time = timestamp();
	debug_printf(DP_INFO, "Total time: %f seconds.\n", end_time - start_time);

	return 0;
}
