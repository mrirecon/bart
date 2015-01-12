
/* Macro wrappers for functions to work around a limitation of
 * the C language standard: A pointer to an array cannot passed
 * as a pointer to a constant array without adding an explicit cast.
 * We hide this cast in the macro definitions. For GCC we can define
 * a type-safe version of the macro.
 *
 * A similar idea is used in Jens Gustedt's P99 preprocessor macros 
 * and functions package available at: http://p99.gforge.inria.fr/
 */
#ifndef AR2D_CAST
#ifndef __GNUC__
#define AR2D_CAST(t, n, m, x) (const t(*)[m])(x)
//#define AR2D_CAST(t, n, m, x) ((const t(*)[m])(0 ? (t(*)[m])0 : (x)))
//#define AR2D_CAST(t, n, m, x) ((const t(*)[m])(t(*)[m]){ &(x[0]) })
#else
#ifndef BUILD_BUG_ON
#define BUILD_BUG_ON(condition) ((void)sizeof(char[1 - 2*!!(condition)]))
#endif
#define AR2D_CAST(t, n, m, x) (BUILD_BUG_ON(!(__builtin_types_compatible_p(const t[m], __typeof__((x)[0])) \
				|| __builtin_types_compatible_p(t[m], __typeof__((x)[0])))), (const t(*)[m])(x))
#endif
#endif

