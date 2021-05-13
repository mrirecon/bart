/* Authors:
 * 2021 Tamás Hakkel <hakkelt@gmail.com>
 */

#include <stdlib.h>
#include "win/rand_r.h"
#include "misc/misc.h"

int rand_r(unsigned int* seed)
{
	UNUSED(seed);
	// Note: There is no rand_r implementation on Windows, but rand() is thread safe
	// TODO: Implement a thread safe rand_r that produces the same random sequence as the UNIX version
	// UNIX implementation: https://sourceware.org/git/?p=glibc.git;a=blob;f=stdlib/rand_r.c;h=30fafc0fa5f30592470a1796bc129f2b1613ecec;hb=refs/heads/master
	//    This implementation is deadlocked under Windows if it is called within a critical block of omp
	return rand();
}