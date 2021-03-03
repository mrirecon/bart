
#include "misc/mri.h"

struct network_data_s {

	long kdims[DIMS];
	long cdims[DIMS];
	long pdims[DIMS];
	long idims[DIMS];

	const char* filename_trajectory;
	const char* filename_pattern;
	const char* filename_kspace;
	const char* filename_coil;
	const char* filename_out;

	_Complex float* kspace;
	_Complex float* coil;
	_Complex float* pattern;
	_Complex float* out;

	_Bool create_out;
	_Bool load_mem;
};

extern struct network_data_s network_data_empty;

void load_network_data(struct network_data_s* network_data);
void free_network_data(struct network_data_s* network_data);

void network_data_check_simple_dims(struct network_data_s* network_data);