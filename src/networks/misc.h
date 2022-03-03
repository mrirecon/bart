
#include "misc/mri.h"

struct network_data_s {

	int N;
	int ND;

	long ksp_dims[DIMS];
	long col_dims[DIMS];
	long psf_dims[DIMS + 1];
	long img_dims[DIMS];
	long max_dims[DIMS];
	long cim_dims[DIMS];
	long out_dims[DIMS];
	long pat_dims[DIMS];
	long trj_dims[DIMS];
	long bas_dims[DIMS];
	long scl_dims[DIMS];

	const char* filename_trajectory;
	const char* filename_pattern;
	const char* filename_kspace;
	const char* filename_coil;
	const char* filename_basis;
	const char* filename_out;

	_Bool export;
	const char* filename_adjoint;
	const char* filename_psf;

	_Complex float* kspace;
	_Complex float* adjoint;
	_Complex float* initialization;
	_Complex float* coil;
	_Complex float* psf;
	_Complex float* out;
	_Complex float* pattern;
	_Complex float* trajectory;
	_Complex float* basis;
	_Complex float* scale;

	struct nufft_conf_s* nufft_conf;

	_Bool create_out;
	_Bool load_mem;

	unsigned long batch_flags;
};

extern struct network_data_s network_data_empty;

extern void load_network_data(struct network_data_s* network_data);
extern void free_network_data(struct network_data_s* network_data);

extern void network_data_normalize(struct network_data_s* nd);
extern void network_data_compute_init(struct network_data_s* nd, _Complex float lambda, int cg_iter);
extern void network_data_slice_dim_to_batch_dim(struct network_data_s* nd);

extern void network_data_check_simple_dims(struct network_data_s* network_data);
extern long network_data_get_tot(struct network_data_s* network_data);

extern struct named_data_list_s* network_data_get_named_list(struct network_data_s* nd);