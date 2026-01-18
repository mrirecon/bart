
#include "misc/mri.h"

struct nlop_s;
struct nn_cunet_conf_s;

extern const struct nlop_s* prior_cunet(const char* cunet_weights, struct nn_cunet_conf_s* cunet_conf,
				bool real_valued, const long msk_dims[DIMS], complex float* mask,
				long img_dims[DIMS]);

extern const struct nlop_s* prior_graph(const char* graph, bool real_valued, bool gpu,
		const long msk_dims[DIMS], complex float* mask, long img_dims[DIMS]);

extern const struct nlop_s* prior_gmm(const long means_dims[DIMS], const complex float* means,
				const long weights_dims0[DIMS], const complex float *weights0,
				const long vars_dims0[DIMS], const complex float *vars0,
				long img_dims[DIMS], float *min_var);


