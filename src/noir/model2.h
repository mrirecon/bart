
struct linop_s;
struct nlop_s;


struct noir2_model_conf_s {

	_Bool noncart;

	unsigned long fft_flags;
	unsigned long wght_flags;

	_Bool rvc;
	_Bool sos;
	float a;
	float b;

	float oversampling_coils;

	struct nufft_conf_s* nufft_conf;

	_Bool asymetric;
	_Bool ret_os_coils;
};

extern struct noir2_model_conf_s noir2_model_conf_defaults;


struct noir2_s {

	struct noir2_model_conf_s model_conf;

	const struct nlop_s* model;		// nlop holding the model 
	const struct linop_s* lop_asym;		// for asymetric reconstruction
						// use adjoint to grid data

	// linops to construct model: lop_fft(tenmul(lop_im, lop_coil))
	const struct linop_s* lop_fft;		// fft/nufft from coil images to kspace
	const struct linop_s* lop_coil;		// kspace coils to img-coils
	const struct linop_s* lop_im;		// masking/resizing of image


	const struct linop_s* lop_coil2;	// kspace coils to img-coils for postptocessing
	
	// references to linops to update model parameters
	const struct linop_s* lop_nufft;	// for retrospectively changing trajectory
	const struct linop_s* lop_pattern;	// for retrospectively changing pattern
	const struct linop_s* lop_basis;	// for retrospectively changing basis (cartesian)


	int N;
	long* pat_dims;
	long* bas_dims;
	long* msk_dims;
	long* ksp_dims;
	long* cim_dims;
	long* img_dims;
	long* col_dims;
	long* col_ten_dims;	// col dims as input of tenmul
	long* trj_dims;

	struct multiplace_array_s* basis;	// this is used in nlinv-net to store basis for trajectory update
};

extern struct noir2_s noir2_noncart_create(int N,
	const long trj_dims[N], const _Complex float* traj,
	const long wgh_dims[N], const _Complex float* weights,
	const long bas_dims[N], const _Complex float* basis,
	const long msk_dims[N], const _Complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long kco_dims[N],
	const long col_dims[N],
	const struct noir2_model_conf_s* conf);

extern struct noir2_s noir2_cart_create(int N,
	const long pat_dims[N], const _Complex float* pattern,
	const long bas_dims[N], const _Complex float* basis,
	const long msk_dims[N], const _Complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long kco_dims[N],
	const long col_dims[N],
	const struct noir2_model_conf_s* conf);

extern void noir2_noncart_update(struct noir2_s* model, int N,
	const long trj_dims[N], const _Complex float* traj,
	const long wgh_dims[N], const _Complex float* weights,
	const long bas_dims[N], const _Complex float* basis);

extern void noir2_cart_update(struct noir2_s* model, int N,
	const long pat_dims[N], const _Complex float* pattern,
	const long bas_dims[N], const _Complex float* basis);

extern void noir2_free(struct noir2_s* model);

extern void noir2_orthogonalize(int N, const long col_dims[N], _Complex float* coils);

