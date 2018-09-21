/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2017 Martin Uecker <uecker@martin.uecker@med.uni-goettingen.de>
 * 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#define _GNU_SOURCE

#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <complex.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <stdarg.h>
#include <limits.h>

#include <sys/mman.h>

#include "num/multind.h"

#include "misc/misc.h"
#include "misc/io.h"
#include "misc/debug.h"

#include "mmio.h"
#if defined(USE_MEM_CFL) || defined(MEMONLY_CFL)
#include "mmiocc.hh"
#endif

// for BSD compatibility
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#ifdef BART_WITH_PYTHON
#  include <Python.h>
#endif /* BART_WITH_PYTHON */



static void io_error(const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
#ifndef BART_WITH_PYTHON
#ifdef USE_LOG_BACKEND
	debug_vprintf_trace("error", __FILE__, __LINE__, DP_ERROR, fmt, ap);
	va_end(ap);
#else
	vfprintf(stderr, fmt, ap);	
	va_end(ap);
	fflush(stderr);
	perror(" ");
#endif
#else
	if (NULL == PyErr_Occurred()) {

		char err[1024] = { 0 };
	     vsnprintf(err, 1023, fmt, ap);
	     va_end(ap);
	     PyErr_SetString(PyExc_RuntimeError, err);
	}
	// No else required as the error indicator has already been set elsewhere

#endif /* !BART_WITH_PYTHON */

	error("");	// FIXME: we may leak open files descriptors...
}


#define err_assert(x)	({ if (!(x)) { debug_printf(DP_ERROR, "%s", #x); exit(EXIT_FAILURE); } })

static bool long_mul_overflow_p(long a, long b)
{
	bool of = false;

	of |= (a > 0) && (b > 0) && (a > LONG_MAX / b);
	of |= (a > 0) && (b < 0) && (b < LONG_MIN / a);
	of |= (a < 0) && (b > 0) && (a < LONG_MIN / b);
	of |= (a < 0) && (b < 0) && (b < LONG_MAX / a);

	return of;
}

static long io_calc_size(unsigned int D, const long dims[D], size_t size)
{
	if (0 == D)
		return size;

	long a = io_calc_size(D - 1, dims + 1, size);
	long b = dims[0];

	if ((a < 0) || (b < 0))
		return -1;

	if (long_mul_overflow_p(a, b))
		return -1;

	return a * b;
}



complex float* load_zra(const char* name, unsigned int D, long dims[D])
{
	int fd;
	if (-1 == (fd = open(name, O_RDONLY)))
		io_error("Loading ra file %s", name);

	if (-1 == read_ra(fd, D, dims))
		error("Loading ra file %s", name);

	long T;
	if (-1 == (T = io_calc_size(D, dims, sizeof(complex float))))
		error("Loading ra file %s", name);

	void* addr;
	struct stat st;

	if (-1 == fstat(fd, &st))
		io_error("Loading ra file %s", name);

	off_t header_size;

	if (-1 == (header_size = lseek(fd, 0, SEEK_CUR)))
		io_error("Loading ra file %s", name);

	// ra allows random stuff at the end
	if (T + header_size > st.st_size)
		error("Loading ra file %s", name);

	assert(header_size < 4096);

	if (MAP_FAILED == (addr = mmap(NULL, st.st_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0)))
		io_error("Loading ra file %s", name);

	if (-1 == close(fd))
		io_error("Loading ra file %s", name);

	return (complex float*)(addr + header_size);;
}


static void* create_data(int ofd, size_t header_size, size_t size)
{
	if (-1 == (ftruncate(ofd, size + header_size)))
		return NULL;

	size_t skip = header_size & ~4095UL;
	size_t off = header_size & 4095UL;
	void* addr;

	if (MAP_FAILED == (addr = mmap(NULL, size + off, PROT_READ|PROT_WRITE, MAP_SHARED, ofd, skip)))
		return NULL;

	return (char*)addr + off;
}

complex float* create_zra(const char* name, unsigned int D, const long dims[D])
{
	int ofd;
	if (-1 == (ofd = open(name, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR)))
		io_error("Creating ra file %s", name);

	if (-1 == write_ra(ofd, D, dims))
		error("Creating ra file %s", name);

	long T;
	if (-1 == (T = io_calc_size(D, dims, sizeof(complex float))))
		error("Creating ra file %s", name);

	off_t header_size;

	if (-1 == (header_size = lseek(ofd, 0, SEEK_CUR)))
		io_error("Creating ra file %s", name);

	void* data;

	if (NULL == (data = create_data(ofd, header_size, T)))
		error("Creating ra file %s", name);

	if (-1 == close(ofd))
		io_error("Creating ra file %s", name);

	return (complex float*)data;
}



float* create_coo(const char* name, unsigned int D, const long dims[D])
{	
	int ofd;

	if (-1 == (ofd = open(name, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR)))
		io_error("Creating coo file %s", name);

	if (-1 == write_coo(ofd, D, dims))
		error("Creating coo file %s", name);

	long T;

	if (-1 == (T = io_calc_size(D, dims, sizeof(float))))
		error("Creating coo file %s", name);

	void* addr;

	if (NULL == (addr = create_data(ofd, 4096, T)))
		error("Creating coo file %s", name);

	if (-1 == close(ofd))
		io_error("Creating coo file %s", name);

	return (float*)addr;
}






complex float* create_zcoo(const char* name, unsigned int D, const long dimensions[D])
{
	long dims[D + 1];
	dims[0] = 2; // complex
	memcpy(dims + 1, dimensions, D * sizeof(long));
	
	return (complex float*)create_coo(name, D + 1, dims);
}


complex float* create_cfl(const char* name, unsigned int D, const long dimensions[D])
{
	io_register_output(name);

#ifdef MEMONLY_CFL
	return create_mem_cfl(name, D, dimensions);
#else
	const char *p = strrchr(name, '.');

	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".ra")))
		return create_zra(name, D, dimensions);

	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".coo")))
		return create_zcoo(name, D, dimensions);

#ifdef USE_MEM_CFL
	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".mem")))
		return create_mem_cfl(name, D, dimensions);
#endif

 
	char name_bdy[1024];
	if (1024 <= snprintf(name_bdy, 1024, "%s.cfl", name))
		error("Creating cfl file %s", name);

	char name_hdr[1024];
	if (1024 <= snprintf(name_hdr, 1024, "%s.hdr", name))
		error("Creating cfl file %s", name);

	int ofd;
	if (-1 == (ofd = open(name_hdr, O_RDWR|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR)))
		io_error("Creating cfl file %s", name);

	if (-1 == write_cfl_header(ofd, D, dimensions))
		error("Creating cfl file %s", name);

	if (-1 == close(ofd))
		io_error("Creating cfl file %s", name);

	return shared_cfl(D, dimensions, name_bdy);
#endif /* MEMONLY_CFL */
}




float* load_coo(const char* name, unsigned int D, long dims[D])
{
	int fd;

	if (-1 == (fd = open(name, O_RDONLY)))
		io_error("Loading coo file %s", name);

	if (-1 == read_coo(fd, D, dims))
		error("Loading coo file %s", name);

	long T;

	if (-1 == (T = io_calc_size(D, dims, sizeof(float))))
		error("Loading coo file %s", name);

	void* addr;
	struct stat st;
        
	if (-1 == fstat(fd, &st))
		io_error("Loading coo file %s", name);

	if (T + 4096 != st.st_size)
		error("Loading coo file %s", name);

	if (MAP_FAILED == (addr = mmap(NULL, T, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 4096)))
		io_error("Loading coo file %s", name);

	if (-1 == close(fd))
		io_error("Loading coo file %s", name);

	return (float*)addr;
}


complex float* load_zcoo(const char* name, unsigned int D, long dimensions[D])
{
	long dims[D + 1];
	float* data = load_coo(name, D + 1, dims);

	if (2 != dims[0])
		error("Loading coo file %s", name);

	memcpy(dimensions, dims + 1, D * sizeof(long));

	return (complex float*)data;
}


static complex float* load_cfl_internal(const char* name, unsigned int D, long dimensions[D], bool priv)
{
	io_register_input(name);

#ifdef MEMONLY_CFL
	UNUSED(priv);

	complex float* ptr = load_mem_cfl(name, D, dimensions);

	if (NULL == ptr)
		io_error("Loading in-memory cfl file %s", name);

	return ptr;
#else
	const char *p = strrchr(name, '.');

	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".ra")))
		return load_zra(name, D, dimensions);

	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".coo")))
		return load_zcoo(name, D, dimensions);

#ifdef USE_MEM_CFL
	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".mem"))) {

	     complex float* ptr = load_mem_cfl(name, D, dimensions);

	     if (ptr == NULL) {
		     io_error("failed loading memory cfl file \"%s\"", name);
	     } else {
		  return ptr;
	     }
	}
#endif /* USE_MEM_CFL */


	char name_bdy[1024];
	if (1024 <= snprintf(name_bdy, 1024, "%s.cfl", name))
		error("Loading cfl file %s", name);

	char name_hdr[1024];
	if (1024 <= snprintf(name_hdr, 1024, "%s.hdr", name))
		error("Loading cfl file %s", name);

	int ofd;
	if (-1 == (ofd = open(name_hdr, O_RDONLY)))
		io_error("Loading cfl file %s", name);

	if (-1 == read_cfl_header(ofd, D, dimensions))
		error("Loading cfl file %s", name);

	if (-1 == close(ofd))
		io_error("Loading cfl file %s", name);

	return (priv ? private_cfl : shared_cfl)(D, dimensions, name_bdy);
#endif /* MEMONLY_CFL */
}


complex float* load_cfl(const char* name, unsigned int D, long dimensions[D])
{
	return load_cfl_internal(name, D, dimensions, true);
}


complex float* load_shared_cfl(const char* name, unsigned int D, long dimensions[D])
{
	return load_cfl_internal(name, D, dimensions, false);
}


#ifndef MEMONLY_CFL
complex float* shared_cfl(unsigned int D, const long dims[D], const char* name)
{
//	struct stat st;
	int fd;
	void* addr;
	long T;

	if (-1 == (T = io_calc_size(D, dims, sizeof(complex float))))
		error("shared cfl %s", name);

	err_assert(T > 0);

        if (-1 == (fd = open(name, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR)))
		io_error("shared cfl %s", name);

//	if (-1 == (fstat(fd, &st)))
//		error("abort");

//	if (!((0 == st.st_size) || (T == st.st_size)))
//		error("abort");

	if (NULL == (addr = create_data(fd, 0, T)))
		error("shared cfl %s", name);

	if (-1 == close(fd))
		io_error("shared cfl %s", name);

	return (complex float*)addr;
}
#endif /* !MEMONLY_CFL */


complex float* anon_cfl(const char* name, unsigned int D, const long dims[D])
{
	UNUSED(name);

#ifndef MEMONLY_CFL
	void* addr;
	long T;

	if (-1 == (T = io_calc_size(D, dims, sizeof(complex float))))
		error("anon cfl");

	if (MAP_FAILED == (addr = mmap(NULL, T, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0)))
		io_error("anon cfl");

	return (complex float*)addr;
#else
	return create_anon_mem_cfl(D, dims);
#endif
}



#if 0
void* private_raw(size_t* size, const char* name)
{
	int fd;
	void* addr;
	struct stat st;

	if (-1 == (fd = open(name, O_RDONLY)))
		error("abort");

	if (-1 == (fstat(fd, &st)))
		error("abort");

	*size = st.st_size;

	if (MAP_FAILED == (addr = mmap(NULL, *size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0)))
		error("abort");

	if (-1 == close(fd))
		error("abort");

	return addr;
}
#endif


#ifndef MEMONLY_CFL
complex float* private_cfl(unsigned int D, const long dims[D], const char* name)
{
	long T;

	if (-1 == (T = io_calc_size(D, dims, sizeof(complex float))))
		error("private cfl %s", name);

	int fd;
	void* addr;
	struct stat st;

	if (-1 == (fd = open(name, O_RDONLY)))
		io_error("private cfl %s", name);

	if (-1 == (fstat(fd, &st)))
		io_error("private cfl %s", name);

	if (T != st.st_size)
		error("private cfl %s", name);

	if (MAP_FAILED == (addr = mmap(NULL, T, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0)))
		io_error("private cfl %s", name);

	if (-1 == close(fd))
		io_error("private cfl %s", name);

	return (complex float*)addr;
}
#endif /* !MEMONLY_CFL */


void unmap_cfl(unsigned int D, const long dims[D], const complex float* x)
{
#ifdef MEMONLY_CFL
	UNUSED(D); UNUSED(dims);
	try_delete_mem_cfl(x);
#else

#ifdef USE_MEM_CFL
	if (is_mem_cfl(x)) {

		// only delete if the dirty flag has been set
		try_delete_mem_cfl(x);
		return;
	}
#endif

	long T;

	if (-1 == (T = io_calc_size(D, dims, sizeof(complex float))))
		error("unmap cfl");

	if (-1 == munmap((void*)((uintptr_t)x & ~4095UL), T))
		io_error("unmap cfl");
#endif
}

