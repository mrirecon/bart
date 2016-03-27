/*
 *  Unit test based on the MinUnit sample code:
 *  http://www.jera.com/techinfo/jtns/jtn002.html
 */

#ifndef _MINUNIT_H
#define _MINUNIT_H



/* file: minunit.h */
#define MU_ASSERT(message, test) do { if (!(test)) return message; } while (0)
#define MU_RUN_TEST(test) do { char *message = test(); num_tests_run++; \
                                if (message) return message; } while (0)

#define TOL 1E-6

extern int num_tests_run;
int num_tests_run = 0;

#endif
