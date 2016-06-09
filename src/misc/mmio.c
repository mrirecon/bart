/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012, 2016 Martin Uecker <uecker@martin.uecker@med.uni-goettingen.de>
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

#include <sys/mman.h>

#include "num/multind.h"

#include "misc/misc.h"
#include "misc/io.h"

#include "mmio.h"

// for BSD compatibility
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif



static void io_error(const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);	
	va_end(ap);
	fflush(stderr);
	perror(" ");
	exit(EXIT_FAILURE);

}


complex float* load_zra(const char* name, unsigned int D, long dims[D])
{
	int fd;
	if (-1 == (fd = open(name, O_RDONLY)))
		io_error("Loading ra file %s", name);

	if (-1 == read_ra(fd, D, dims))
		io_error("Loading ra file %s", name);

//	long T = md_calc_size(D, dims) * sizeof(complex float);

	void* addr;
	struct stat st;

	if (-1 == fstat(fd, &st))
		io_error("Loading ra file %s", name);

	off_t header_size;

	if (-1 == (header_size = lseek(fd, 0, SEEK_CUR)))
		io_error("Loading ra file %s", name);

	// ra allows random stuff at the end
//	if (T + header_size >= st.st_size)
//		io_error("Loading ra file %s", name);

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
		io_error("Creating ra file %s", name);

	long T = md_calc_size(D, dims) * sizeof(complex float);

	off_t header_size;

	if (-1 == (header_size = lseek(ofd, 0, SEEK_CUR)))
		io_error("Creating ra file %s", name);

	void* data;

	if (NULL == (data = create_data(ofd, header_size, T)))
		io_error("Creating ra file %s", name);

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
		io_error("Creating coo file %s", name);

	long T = md_calc_size(D, dims) * sizeof(float);

	void* addr;

	if (NULL == (addr = create_data(ofd, 4096, T)))
		io_error("Creating coo file %s", name);

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

	const char *p = strrchr(name, '.');

	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".ra")))
		return create_zra(name, D, dimensions);

	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".coo")))
		return create_zcoo(name, D, dimensions);


	char name_bdy[1024];
	if (1024 <= snprintf(name_bdy, 1024, "%s.cfl", name))
		io_error("Creating cfl file %s", name);

	char name_hdr[1024];
	if (1024 <= snprintf(name_hdr, 1024, "%s.hdr", name))
		io_error("Creating cfl file %s", name);

	int ofd;
	if (-1 == (ofd = open(name_hdr, O_RDWR|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR)))
		io_error("Creating cfl file %s", name);

	if (-1 == write_cfl_header(ofd, D, dimensions))
		io_error("Creating cfl file %s", name);

	if (-1 == close(ofd))
		io_error("Creating cfl file %s", name);

	return shared_cfl(D, dimensions, name_bdy);
}




float* load_coo(const char* name, unsigned int D, long dims[D])
{
	int fd;
	if (-1 == (fd = open(name, O_RDONLY)))
		io_error("Loading coo file %s", name);

	if (-1 == read_coo(fd, D, dims))
		io_error("Loading coo file %s", name);

	long T = md_calc_size(D, dims) * sizeof(float);

	void* addr;
	struct stat st;
        
	if (-1 == fstat(fd, &st))
		io_error("Loading coo file %s", name);

	if (T + 4096 != st.st_size)
		io_error("Loading coo file %s", name);

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
		io_error("Loading coo file %s", name);

	memcpy(dimensions, dims + 1, D * sizeof(long));

	return (complex float*)data;
}


static complex float* load_cfl_internal(const char* name, unsigned int D, long dimensions[D], bool priv)
{
	io_register_input(name);

	const char *p = strrchr(name, '.');

	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".ra")))
		return load_zra(name, D, dimensions);

	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".coo")))
		return load_zcoo(name, D, dimensions);


	char name_bdy[1024];
	if (1024 <= snprintf(name_bdy, 1024, "%s.cfl", name))
		io_error("Loading cfl file %s", name);

	char name_hdr[1024];
	if (1024 <= snprintf(name_hdr, 1024, "%s.hdr", name))
		io_error("Loading cfl file %s", name);

	int ofd;
	if (-1 == (ofd = open(name_hdr, O_RDONLY)))
		io_error("Loading cfl file %s", name);

	if (-1 == read_cfl_header(ofd, D, dimensions))
		io_error("Loading cfl file %s", name);

	if (-1 == close(ofd))
		io_error("Loading cfl file %s", name);

	return (priv ? private_cfl : shared_cfl)(D, dimensions, name_bdy);
}


complex float* load_cfl(const char* name, unsigned int D, long dimensions[D])
{
	return load_cfl_internal(name, D, dimensions, true);
}


complex float* load_shared_cfl(const char* name, unsigned int D, long dimensions[D])
{
	return load_cfl_internal(name, D, dimensions, false);
}



complex float* shared_cfl(unsigned int D, const long dims[D], const char* name)
{
//	struct stat st;
	int fd;
	void* addr;

	long T = md_calc_size(D, dims) * sizeof(complex float);

        if (-1 == (fd = open(name, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR)))
                abort();

//	if (-1 == (fstat(fd, &st)))
//		abort();

//	if (!((0 == st.st_size) || (T == st.st_size)))
//		abort();

	if (NULL == (addr = create_data(fd, 0, T)))
		abort();

	if (-1 == close(fd))
		abort();

	return (complex float*)addr;
}


complex float* anon_cfl(const char* name, unsigned int D, const long dims[D])
{
	UNUSED(name);
	void* addr;
	long T = md_calc_size(D, dims) * sizeof(complex float);

	if (MAP_FAILED == (addr = mmap(NULL, T, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0)))
		abort();

	return (complex float*)addr;
}



#if 0
void* private_raw(size_t* size, const char* name)
{
	int fd;
	void* addr;
	struct stat st;

	if (-1 == (fd = open(name, O_RDONLY)))
		abort();

	if (-1 == (fstat(fd, &st)))
		abort();

	*size = st.st_size;

	if (MAP_FAILED == (addr = mmap(NULL, *size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0)))
		abort();

	if (-1 == close(fd))
		abort();

	return addr;
}
#endif



complex float* private_cfl(unsigned int D, const long dims[D], const char* name)
{
	long T = md_calc_size(D, dims) * sizeof(complex float);

	int fd;
	void* addr;
	struct stat st;

	if (-1 == (fd = open(name, O_RDONLY)))
		abort();

	if (-1 == (fstat(fd, &st)))
		abort();

	if (!(T == st.st_size))
	{
		//printf("T = %ld, !=  st.st_size = %ld\n", T, st.st_size);
		//printf("not aborting...\n");
		abort();
	}

	if (MAP_FAILED == (addr = mmap(NULL, T, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0)))
		abort();

	if (-1 == close(fd))
		abort();

	return (complex float*)addr;
}


void unmap_cfl(unsigned int D, const long dims[D], const complex float* x)
{
	long T = md_calc_size(D, dims) * sizeof(complex float);

	if (-1 == munmap((void*)((uintptr_t)x & ~4095UL), T))
		abort();
}

