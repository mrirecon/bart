

struct loss_config_s;
struct nn_weights_s;
struct network_s;
struct iter6_conf_s;
struct nnet_s;

typedef unsigned int (*nnet_get_no_odims_t)(const struct nnet_s* config, unsigned int NI, const long idims[NI]);
typedef void (*nnet_get_odims_t)(const struct nnet_s* config, unsigned int NO, long odims[NO], unsigned int NI, const long idims[NI]);

struct nnet_s {

	struct network_s* network;

	struct nn_weights_s* weights;
	struct iter6_conf_s* train_conf;

	struct loss_config_s* train_loss;
	struct loss_config_s* valid_loss;

	_Bool low_mem;
	_Bool gpu;

	nnet_get_no_odims_t get_no_odims;
	nnet_get_odims_t get_odims;

	const char* graph_file;

	long N_segm_labels;
};

extern struct nnet_s nnet_init;

struct network_data_s;

extern void nnet_init_mnist_default(struct nnet_s* nnet);
extern void nnet_init_unet_segm_default(struct nnet_s* nnet, long N_segm_labels);

extern void apply_nnet(	const struct nnet_s* nnet,
			unsigned int NO, const long odims[NO], _Complex float* out,
			unsigned int NI, const long idims[NI], const _Complex float* in
			);

extern void apply_nnet_batchwise(
			const struct nnet_s* nnet,
			unsigned int NO, const long odims[NO], _Complex float* out,
			unsigned int NI, const long idims[NI], const _Complex float* in,
			long Nb
			);

extern void train_nnet(	struct nnet_s* nnet,
			unsigned int NO, const long odims[NO], const _Complex float* out,
			unsigned int NI, const long idims[NI], const _Complex float* in,
			long Nb, const struct nn_weights_s* valid_files);

extern void eval_nnet(	struct nnet_s* nnet,
			unsigned int NO, const long odims[NO], const _Complex float* out,
			unsigned int NI, const long idims[NI], const _Complex float* in,
			long Nb);