all: test_lrthresh test_sort bart_intel.a 

CC=icc
CXX=icpc
CFLAGS=-g -O3 -ipo -no-prec-div -fp-model fast=2 -xHOST -mkl -DMKL_Complex8="complex float" -DMKL_Complex16="complex double" -qopenmp -std=c99 -vec-threshold0 -static
CXXFLAGS=-g -O3 -ipo -no-prec-div -fp-model fast=2 -xHOST -mkl -DMKL_Complex8="complex float" -DMKL_Complex16="complex double" -qopenmp -vec-threshold0 -static
LDFLAGS=-mkl -qopenmp

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

.cc.o:
	$(CXX) -c $(CXXFLAGS) $< -o $@

test_lrthresh: lrthresh_intel.o test_lrthresh.o
	$(CC) test_lrthresh.o lrthresh_intel.o -o test_lrthresh $(LDFLAGS)

test_sort: sort_intel.o test_sort.o
	$(CC) test_sort.o sort_intel.o -o test_sort $(LDFLAGS)

bart_intel.a: sort_intel.o
	ar rcs bart_intel.a lrthresh_intel.o sort_intel.o

clean:
	rm -f test_lrthresh test_sort *.o *.a
