// for CUDA
#ifndef __device__
#define __device__
#endif


// This is an implementation of the Philox-4x32-10 PRNG, as described in:
// "Parallel Random Numbers: As Easy as 1, 2, 3", Salmon, Moraes, Dror & Shaw, SC11, Seattle, Washington, USA, 2011, ACM
// DOI: 10.1145/2063384.2063405

// 4x32-10 means: it uses 4 counters of 32 bits each, with 10 rounds.

// It has 2x32-bits of state, called key (collected in a 64-bit state for our convenience)
// Internally, it uses 4x32-bit counters, but we combine them into two 64-bit counters
// Each call, it also creates 4x32-bit output, but we combine those in two 64-bit outputs (passed as an array).


// Constants, taken from the reference above

static const uint32_t PHILOX_M0 = 0xD2511F53;
static const uint32_t PHILOX_M1 = 0xCD9E8D57;
// Truncated to 32 bits from the values in paper
static const uint32_t PHILOX_WEYL_32_0 = 0x9E3779B9;  // golden ratio
static const uint32_t PHILOX_WEYL_32_1 = 0xBB67AE85;  // sqrt(3)-1

struct prod_hilo {

	uint32_t lo;
	uint32_t hi;
};

__device__ static inline struct prod_hilo mulhilo(uint32_t M, uint32_t R)
{

	uint64_t product = ((uint64_t) M) * ((uint64_t) R);
	return (struct prod_hilo) {.lo = (uint32_t) product, .hi = (uint32_t) (product >> 32)};
}

__device__ static inline void _round(const uint32_t key[2], uint32_t ctrs[4])
{

	struct prod_hilo hilo0 = mulhilo(PHILOX_M0, ctrs[0]);
	struct prod_hilo hilo1 = mulhilo(PHILOX_M1, ctrs[2]);

	ctrs[0] = hilo1.hi ^ ctrs[1] ^ key[0];
	ctrs[1] = hilo1.lo;
	ctrs[2] = hilo0.hi ^ ctrs[3] ^ key[1];
	ctrs[3] = hilo0.lo;
}



__device__ static void philox_4x32(const uint64_t state, const uint64_t ctr1, const uint64_t ctr2, uint64_t out[2])
{

	uint32_t key[2] = {(uint32_t) state, (uint32_t) (state >> 32)};


	uint32_t ctrs[4] = {

		(uint32_t) ctr1,
		(uint32_t) (ctr1 >> 32),
		(uint32_t) ctr2,
		(uint32_t) (ctr2 >> 32),
	};

	const int nrounds = 10;
	for (int r = 0; r < nrounds; r++) {

		_round(key, ctrs);
		// key update, note that we are only changing a copy of the key
		key[0] += PHILOX_WEYL_32_0;
		key[1] += PHILOX_WEYL_32_1;
	}

	out[0] = ((uint64_t)ctrs[0]) << 32 | (uint64_t)ctrs[1];
	out[1] = ((uint64_t)ctrs[2]) << 32 | (uint64_t)ctrs[3];

}
