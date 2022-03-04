

struct nn_weights_s;
struct loss_config_s;

enum BOOL_SELECT {BOOL_DEFAULT, BOOL_TRUE, BOOL_FALSE};

struct reconet_s {

	struct network_s* network;
	long Nt;

	enum BOOL_SELECT share_weights_select;
	enum BOOL_SELECT share_lambda_select;
	_Bool share_weights;
	_Bool share_lambda;

	struct config_nlop_mri_s* mri_config;
	_Bool one_channel_per_map;

	_Bool external_initialization;	//initialize network with precomputed reconstruction

	//data consistency config
	float dc_lambda_fixed;
	float dc_lambda_init;
	_Bool dc_gradient;
	_Bool dc_scale_max_eigen;
	_Bool dc_proxmap;
	int dc_max_iter;

	//network initialization
	_Bool normalize;
	_Bool sense_init;
	int init_max_iter;
	float init_lambda_fixed;
	float init_lambda_init;

	struct nn_weights_s* weights;
	struct iter6_conf_s* train_conf;

	struct loss_config_s* train_loss;
	struct loss_config_s* valid_loss;

	_Bool low_mem;
	_Bool gpu;

	const char* graph_file;

	_Bool coil_image;

	_Bool normalize_rss;
};

extern struct reconet_s reconet_config_opts;

struct named_data_list_s;

extern void reconet_init_modl_default(struct reconet_s* reconet);
extern void reconet_init_varnet_default(struct reconet_s* reconet);
extern void reconet_init_unet_default(struct reconet_s* reconet);

extern void reconet_init_modl_test_default(struct reconet_s* reconet);
extern void reconet_init_varnet_test_default(struct reconet_s* reconet);
extern void reconet_init_unet_test_default(struct reconet_s* reconet);

extern void apply_reconet(	struct reconet_s* config,
				int N, const long max_dims[N],
				int ND, const long psf_dims[ND],
				struct named_data_list_s* data);

extern void train_reconet(	struct reconet_s* config,
				int N, const long max_dims[N],
				int ND, const long psf_dims[ND],
				long Nb_train, struct named_data_list_s* train_data,
				long Nb_valid, struct named_data_list_s* valid_data);

extern void eval_reconet(	struct reconet_s* config,
				int N, const long max_dims[N],
				int ND, const long psf_dims[ND],
				struct named_data_list_s* data);