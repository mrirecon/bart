
#include "linops/someops.h"
#include <iso646.h>

struct nn_weights_s;
struct loss_config_s;

struct nlinvnet_s {

	// Training configuration
	struct iter6_conf_s* train_conf;
	struct loss_config_s* train_loss;
	struct loss_config_s* valid_loss;
	float l2loss_reg;
	long time_mask[2];
	long avg_coils_loss;

	// Self-Supervised k-Space
	_Bool ksp_training;
	float ksp_split;
	unsigned long ksp_shared_dims;
	float ksp_leaky;
	const char* use_reco_file; 

	// Network block
	struct network_s* network;
	struct nn_weights_s* weights;
	_Bool share_weights;
	float lambda;
	float lambda_sens;
	unsigned long filter_flags;
	const _Complex float* filter;
	
	int conv_time;
	enum PADDING conv_padding;
	
	// NLINV configuration
	struct noir2_conf_s* conf;
	struct noir2_net_config_s* model;
	struct iter_conjgrad_conf* iter_conf;
	struct iter_conjgrad_conf* iter_conf_net;
	float cgtol;
	int iter_net;		//# of iterations with network
	float oversampling_coils;
	long senssize;

	_Bool fix_coils;
	_Bool ref_init_img;
	_Bool ref_init_col;
	_Bool ref_init_col_rt;
	float scaling;
	_Bool real_time_init;
	float temp_damp;

	_Bool debug;

	_Bool normalize_rss;
};

extern struct nlinvnet_s nlinvnet_config_opts;

struct network_data_s;

extern void nlinvnet_init_varnet_default(struct nlinvnet_s* nlinvnet);
extern void nlinvnet_init_varnet_test_default(struct nlinvnet_s* nlinvnet);
extern void nlinvnet_init_resnet_default(struct nlinvnet_s* nlinvnet);

void nlinvnet_init(struct nlinvnet_s* nlinvnet, int N,
	const long trj_dims[__VLA2(N)],
	const long pat_dims[__VLA(N)],
	const long bas_dims[__VLA2(N)], const _Complex float* basis,
	const long ksp_dims[__VLA(N)],
	const long cim_dims[__VLA(N)],
	const long img_dims[__VLA(N)],
	const long col_dims[__VLA(N)]);


enum nlinvnet_out { NLINVNET_OUT_CIM, NLINVNET_OUT_KSP, NLINVNET_OUT_IMG_COL };

struct named_data_list_s;
void train_nlinvnet(struct nlinvnet_s* nlinvnet, int Nb, struct named_data_list_s* train_data, struct named_data_list_s* valid_data);

void apply_nlinvnet(struct nlinvnet_s* nlinvnet, int N,
	const long img_dims[N], _Complex float* img,
	const long col_dims[N], _Complex float* col,
	const long ksp_dims[N], const _Complex float* ksp,
	const long pat_dims[N], const _Complex float* pat,
	const long trj_dims[N], const _Complex float* trj);
