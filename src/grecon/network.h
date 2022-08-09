
#include "misc/opts.h"

enum NETWORK_SELECT {
	NETWORK_NONE,
	NETWORK_MNIST,
	NETWORK_RESBLOCK,
	NETWORK_VARNET,
	NETWORK_TENSORFLOW,
};

extern struct network_s* get_default_network(enum NETWORK_SELECT net);

extern const int N_variational_block_opts;
extern struct opt_s variational_block_opts[];

extern const int N_res_block_opts;
extern struct opt_s res_block_opts[];

extern const int N_unet_reco_opts;
extern struct opt_s unet_reco_opts[];

extern const int N_unet_segm_opts;
extern struct opt_s unet_segm_opts[];

extern const int N_tensorflow_opts;
extern struct opt_s network_tensorflow_opts[];

