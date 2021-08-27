
#include <complex.h>

extern int bin_quadrature(const long bins_dims[DIMS], float* bins,
			const long labels_dims[DIMS], complex float* labels,
			const long resp_labels_idx[2], const long card_labels_idx[2],
			int n_resp, int n_card,
			int mavg_window, int mavg_window_card, const char* card_out);
	
