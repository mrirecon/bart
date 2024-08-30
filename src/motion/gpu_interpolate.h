#include "misc/cppwrap.h"

void cuda_positions(int N, int d, unsigned long flags, const long sdims[__VLA(N)], const long pdims[__VLA(N)], _Complex float* pos);

void cuda_interpolate2(int ord, int M, 
			const long intp_dims[__VLA(M)], const long intp_strs[__VLA(M)], _Complex float* intp,
							const long coor_strs[__VLA(M)], long coor_dir_dim_str, const _Complex float* coor,
			const long grid_dims[__VLA(M)], const long grid_strs[__VLA(M)], const _Complex float* grid);

void cuda_interpolateH2(int ord, int M, 
			const long grid_dims[__VLA(M)], const long grid_strs[__VLA(M)], _Complex float* grid,
			const long intp_dims[__VLA(M)], const long intp_strs[__VLA(M)], const _Complex float* intp,
							const long coor_strs[__VLA(M)], long coor_dir_dim_str, const _Complex float* coor);

void cuda_interpolate_adj_coor2(int ord, int M, 
			const long intp_dims[__VLA(M)], const long intp_strs[__VLA(M)], const _Complex float* dintp,
							const long coor_strs[__VLA(M)], long coor_dir_dim_str, const _Complex float* coor, _Complex float* dcoor,
			const long grid_dims[__VLA(M)], const long grid_strs[__VLA(M)], const _Complex float* grid);

void cuda_interpolate_der_coor2(int ord, int M, 
			const long intp_dims[__VLA(M)], const long intp_strs[__VLA(M)], _Complex float* dintp,
							const long coor_strs[__VLA(M)], long coor_dir_dim_str, const _Complex float* coor, const _Complex float* dcoor,
			const long grid_dims[__VLA(M)], const long grid_strs[__VLA(M)], const _Complex float* grid);

#include "misc/cppwrap.h"

