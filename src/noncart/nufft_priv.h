
#include "noncart/grid.h"

struct multiplace_array_s;

/**
 *
 * NUFFT internal data structure
 *
 */
struct nufft_data {

	INTERFACE(linop_data_t);

	struct nufft_conf_s conf;	///< NUFFT configuration structure
	struct grid_conf_s grid_conf;

	int N;				///< Number of dimension
	unsigned long flags;

	struct multiplace_array_s* linphase;	///< Linear phase for pruned FFT
	struct multiplace_array_s* traj;	///< Trajectory
	struct multiplace_array_s* roll;	///< Roll-off factor
	struct multiplace_array_s* psf;	///< Point-spread function (2x size)
	struct multiplace_array_s* fftmod;	///< FFT modulation for centering
	struct multiplace_array_s* weights;	///< Weights, ex, density compensation
	struct multiplace_array_s* basis;

	float width;			///< Interpolation kernel width
	double beta;			///< Kaiser-Bessel beta parameter

	const struct linop_s* fft_op;	///< FFT operator

	long* ksp_dims;			///< Kspace dimension
	long* cim_dims;			///< Coil image dimension
	long* cml_dims;			///< Coil + linear phase dimension
	long* img_dims;			///< Image dimension
	long* trj_dims;			///< Trajectory dimension
	long* lph_dims;			///< Linear phase dimension
	long* psf_dims;			///< Point spread function dimension
	long* wgh_dims;			///< Weights dimension
	long* bas_dims;
	long* out_dims;
	long* ciT_dims;			///< Coil image dimension
	long* cmT_dims;			///< Coil + linear phase dimension

	//!
	long* cm2_dims;			///< 2x oversampled coil image dimension
	long* factors;

	long* ksp_strs;
	long* cim_strs;
	long* cml_strs;
	long* img_strs;
	long* trj_strs;
	long* lph_strs;
	long* psf_strs;
	long* wgh_strs;
	long* bas_strs;
	long* out_strs;

	const struct linop_s* cfft_op;   ///< Pcycle FFT operator
	unsigned int cycle;

	struct linop_s* lop_nufft_psf;
	struct linop_s* lop_fftuc_psf;
};




