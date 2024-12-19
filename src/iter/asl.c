
#include "num/ops.h"
#include "num/iovec.h"
#include "num/multind.h"

#include "linops/linop.h"
#include "linops/fmac.h"

#include "asl.h"

/**
 * This function extracts the size of the separated control and label image
 * from an input array of dimensions.
 * 
 * @param N        Number of dimensions.
 * @param asl_dim  Dimension along which to separate control and label images.
 * @param asl_dims Array of size N to store the separated dimensions.
 * @param in_dims  Array of size N specifying the size of each input dimension.
 */
void get_asl_dims(int N, int asl_dim, long asl_dims[N], const long in_dims[N])
{
	assert(in_dims[asl_dim] == 2);
	md_copy_dims(N, asl_dims, in_dims);
	asl_dims[asl_dim] = 1;
}

/**
 * This function creates a linear operator that separates control and label images along the ASL
 * dimension specified by `asl_dim` and computes the difference image.
 * The ASL dimension must have a size of 2 (control and label image).
 * The returned operator has a size of 1 along the ASL dimension.
 *
 * @param N        Number of dimensions.
 * @param img_dims Array of size N specifying the size of each dimension.
 * @param asl_dim  Dimension along which to separate control and label images.
 * @return         Pointer to the created linear operator structure.
 */
const struct linop_s* linop_asl_create(int N, const long img_dims[N], int asl_dim)
{
	assert(img_dims[asl_dim] == 2);
	
	long asl_img_dims[N];
	get_asl_dims(N, asl_dim, asl_img_dims, img_dims);

	long tdims[N];
	md_singleton_dims(N, tdims);
	tdims[asl_dim] = 2;

	complex float tensor[2] = { 1.0f, -1.0f };

	return linop_fmac_dims_create(N, asl_img_dims, img_dims, tdims, tensor);
}
