/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016-2021. Uecker Lab. University Center Göttingen.
 * Copyright 2023-2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
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

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include "win/mman.h"
#include "win/open_patch.h"
#else
#include <sys/mman.h>
#endif

#include "num/mpi_ops.h"
#include "num/multind.h"

#include "misc/misc.h"
#include "misc/list.h"
#include "misc/io.h"
#include "misc/debug.h"
#include "misc/memcfl.h"
#include "misc/stream.h"

#include "mmio.h"

// for BSD compatibility
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#ifdef BART_WITH_PYTHON
#  include <Python.h>
#endif /* BART_WITH_PYTHON */

#ifndef DIMS
#define DIMS 16
#endif


bool mmio_file_locking = true;

bool stream_create_binary_outputs = false;

#ifdef __EMSCRIPTEN__
// FIXME: This is a workaround for a bug in emscripten.
// https://github.com/emscripten-core/emscripten/issues/15140
// https://github.com/emscripten-core/emscripten/issues/17801

#define WASM_MAX_FDS 1024
int wasm_fds[WASM_MAX_FDS] = { };
int wasm_fd_offset = 0;

static void wasm_close_later(int fd)
{
	if (wasm_fd_offset >= WASM_MAX_FDS)
		error("WASM close_later: too many files.\n");

	wasm_fds[wasm_fd_offset++] = fd;
}

static void io_error(const char* fmt, ...);

void wasm_close_fds(void)
{
	while (wasm_fd_offset > 0) {

		int fd = wasm_fds[--wasm_fd_offset];

		if (-1 == close(fd))
			io_error("close_open_fds %d\n", fd);
	}
}
#endif


static void io_error(const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
#ifndef BART_WITH_PYTHON
#ifdef USE_LOG_BACKEND
	debug_vprintf_trace("error", __FILE__, __LINE__, DP_ERROR, fmt, ap);
	va_end(ap);
#else
	perror(" ");
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fflush(stderr);
#endif
#else
	if (NULL == PyErr_Occurred()) {

		char err[1024] = { };
		vsnprintf(err, 1023, fmt, ap);
		va_end(ap);
		PyErr_SetString(PyExc_RuntimeError, err);
	}
	// No else required as the error indicator has already been set elsewhere

#endif /* !BART_WITH_PYTHON */

	error("");	// FIXME: we may leak open files descriptors...
}


#define err_assert(x)	({ if (!(x)) { error("%s", #x); } })


bool mpi_shared_files = false;

unsigned long cfl_loop_rand_flags = ~0ul;
bool strided_cfl_loop = false;

struct cfl_loop_desc_s {

	int D;
	int omp_threads;
	unsigned long flags;
	long loop_dims[DIMS];
	long offs_dims[DIMS];
};

static struct cfl_loop_desc_s cfl_loop_desc = {

	.D = 0,
	.omp_threads = 1,
	.flags = 0UL,
	.loop_dims =  { [0 ... DIMS - 1] = 1 },
	.offs_dims =  { [0 ... DIMS - 1] = 0 },
};

#define MAX_WORKER 128
#define THREAD_BATCH_LVL 1

static long cfl_loop_index[MAX_WORKER] = { [ 0 ... MAX_WORKER - 1 ] = 0 };
static list_t unmap_addrs = NULL;


bool cfl_loop_omp(void)
{
#ifdef _OPENMP
	return (1 < cfl_loop_desc.omp_threads);
#else
	return false;
#endif
}

int cfl_loop_worker_id(void)
{
	if (!cfl_loop_desc_active())
		return 0;

	if (1 < mpi_get_num_procs()) {
		
		int procno = mpi_get_rank();

		if (MAX_WORKER <= procno)
			error("Maximum supported number of MPI workers (%d) exceeded!\n", MAX_WORKER);

		return procno;
	}

#ifdef _OPENMP
	if (1 < cfl_loop_desc.omp_threads) {

		int threadno = omp_get_ancestor_thread_num(THREAD_BATCH_LVL);

		if ((THREAD_BATCH_LVL < omp_get_level()) || (1 < omp_get_team_size(0)))
			debug_printf(DP_WARN, "File accessed in OMP region! Cannot guarantee thread safety!\n");

		if (MAX_WORKER <= threadno)
			error("Maximum supported number of MPI workers (%d) exceeded!\n", MAX_WORKER);

		return MAX(0, threadno);
	}
#endif

	return 0;
}

int cfl_loop_num_workers(void)
{
	if (!cfl_loop_desc_active())
		return 1;

	if (1 < mpi_get_num_procs()) {

		int nprocs = mpi_get_num_procs();

		if (MAX_WORKER < nprocs)
			error("Maximum supported number of MPI workers (%d) exceeded!\n", MAX_WORKER);

		return nprocs;
	}

	if (1 < cfl_loop_desc.omp_threads) {
		
		int omp_threads = cfl_loop_desc.omp_threads;
		
		if (MAX_WORKER < omp_threads)
			error("Maximum supported number of OMP workers exceeded!\n");

		return omp_threads;
	}

	return 1;
}



void init_cfl_loop_desc(int D, const long loop_dims[__VLA(D)], long start_dims[__VLA(D)], unsigned long flags, int omp_threads, int index)
{
	if (MAX_WORKER < omp_threads)
		error("Maximum supported number of OMP workers exceeded!\n");

	assert(DIMS >= D);

	cfl_loop_desc.omp_threads = omp_threads;
	cfl_loop_desc.flags = flags;

	cfl_loop_desc.D = D;
	md_copy_dims(D, cfl_loop_desc.loop_dims, loop_dims);
	md_copy_dims(D, cfl_loop_desc.offs_dims, start_dims);

	set_cfl_loop_index(index);

#pragma omp critical(unmap_addrs)
	if (NULL == unmap_addrs)
		unmap_addrs = list_create();
}


long cfl_loop_desc_total(void)
{
	return md_calc_size(cfl_loop_desc.D, cfl_loop_desc.loop_dims);
}


void set_cfl_loop_index(long index)
{
	if (!cfl_loop_desc_active())
		return;

	int worker_id = cfl_loop_worker_id();

	if (MAX_WORKER < worker_id)
		error("Worker id exceeds maximum supported workers!\n");

	cfl_loop_index[worker_id] = index;
}

long get_cfl_loop_index()
{
	int worker_id = cfl_loop_worker_id();

	if (MAX_WORKER < worker_id)
		error("Worker id exceeds maximum supported workers!\n");

	debug_printf(DP_DEBUG2, "loop index: %ld\n", cfl_loop_index[worker_id]);
	return cfl_loop_index[worker_id];
}


bool cfl_loop_desc_active(void)
{
	return cfl_loop_desc.flags > 0;
}

void cfl_loop_desc_set_inactive(void)
{
	cfl_loop_desc.flags = 0;
}

unsigned long cfl_loop_get_flags(void)
{
	return cfl_loop_desc.flags;
}

int cfl_loop_get_rank(void)
{
	return cfl_loop_desc.D;
}

void cfl_loop_get_dims(int D, long dims[D])
{
	assert(cfl_loop_desc.D == D);
	md_copy_dims(D, dims, cfl_loop_desc.loop_dims);
}

void cfl_loop_get_pos(int D, long pos[D])
{
	assert(cfl_loop_desc.D == D);
	md_set_dims(cfl_loop_desc.D , pos, 0);
	md_unravel_index(D, pos, cfl_loop_desc.flags, cfl_loop_desc.loop_dims, cfl_loop_index[cfl_loop_worker_id()]);

	for (int i = 0; i < cfl_loop_desc.D; i++)
		pos[i] += cfl_loop_desc.offs_dims[i];
}


struct cfl_file_desc_s {
	
	void* file_addr;
	void* data_addr;

	int D;
	long* file_dims; 
	long* data_dims;
	long* pos;

	bool writeback;
};

static bool cmp_addr(const void* _item, const void* _ref)
{
	const struct cfl_file_desc_s* item = _item;
	const complex float* ref = _ref;

	return (item->data_addr == ref);
}

static void work_buffer_get_pos(int D, const long dims[D], long pos[D], bool output, long index)
{
	md_set_dims(D, pos, 0);
	md_unravel_index(MIN(D, cfl_loop_desc.D), pos, cfl_loop_desc.flags, cfl_loop_desc.loop_dims, index);

	if (output)
		return;

	for (int i = 0; i < MIN(D, cfl_loop_desc.D); i++) {

		if (1 == dims[i])
			pos[i] = 0;
		else
			pos[i] += cfl_loop_desc.offs_dims[i];

		if (pos[i] >= dims[i])
			error("Position in cfl loop out of range!\n");
	}
}


/**
 * Creates working buffer containing slices of the original file
 * dims contain file_dims on entry and slice dims on return
**/
static void* create_worker_buffer(int D, long dims[D], void* addr, bool output)
{
	if (!cfl_loop_desc_active())
		return addr;

	if (output)
		assert(md_check_equal_dims(MIN(D, DIMS), dims, cfl_loop_desc.loop_dims, cfl_loop_desc.flags));

	long slc_dims[D];
	md_select_dims(D, ~cfl_loop_desc.flags, slc_dims, dims);

	long slc_strs[D];
	long tot_strs[D];

	md_calc_strides(D, tot_strs, dims, sizeof(complex float));
	md_calc_strides(D, slc_strs, slc_dims, sizeof(complex float));

	long pos[D];
	work_buffer_get_pos(D, dims, pos, output, cfl_loop_index[cfl_loop_worker_id()]);

	void* buf = NULL;

	if (md_check_equal_dims(D, tot_strs, slc_strs, ~cfl_loop_desc.flags)) {

		buf = addr + md_calc_offset(D, tot_strs, pos);

		if (output && (1 == mpi_get_num_procs()))
			output = false;

	} else {

		strided_cfl_loop = true;
		buf = md_alloc(D, slc_dims, sizeof(complex float));

		md_slice(D, cfl_loop_desc.flags, pos, dims, buf, addr, sizeof(complex float));
	}

	if (!mpi_shared_files && (1 < mpi_get_num_procs())) {

		for (int i = 1; i < mpi_get_num_procs(); i++) {

			complex float* src = NULL;
			
			if (mpi_is_main_proc()) {

				long tpos[D];
				work_buffer_get_pos(D, dims, tpos, output, cfl_loop_index[cfl_loop_worker_id()] + i);

				src = addr + md_calc_offset(D, tot_strs, tpos);
			}

			mpi_copy2(D, slc_dims, slc_strs, buf, tot_strs, src, sizeof(complex float), 0, i);
		}
	}

	PTR_ALLOC(struct cfl_file_desc_s, desc);

	desc->D = D;
	desc->file_dims = ARR_CLONE(long[D], dims);
	desc->data_dims = ARR_CLONE(long[D], slc_dims);
	desc->pos = ARR_CLONE(long[D], pos);

	desc->file_addr = addr;
	desc->data_addr = buf;

	desc->writeback = output;

#pragma omp critical(unmap_addrs)
	list_append(unmap_addrs, PTR_PASS(desc));

	md_copy_dims(D, dims, slc_dims);

	return buf;
}


/**
 * Check if addr contains a working buffer and return underlying pointer to file
 * dims contain slice dims on entry and file dims on return
**/
static const void* free_worker_buffer(int D, long dims[D], const void* addr)
{
	struct cfl_file_desc_s* desc = NULL;

	if (NULL == unmap_addrs)
		return addr;

#pragma omp critical(unmap_addrs)
	desc = list_get_first_item(unmap_addrs, addr, cmp_addr, true);

	if (NULL == desc)
		return addr;

	assert(D == desc->D);

	if (desc->writeback) {

		if (1 < mpi_get_num_procs()) {

			for (int i = 0; i < mpi_get_num_procs(); i++) {

				complex float* dst = NULL;

				long file_strs[D];
				md_calc_strides(D, file_strs, desc->file_dims, sizeof(complex float));

				long data_strs[D];
				md_calc_strides(D, data_strs, desc->data_dims, sizeof(complex float));


				if (mpi_is_main_proc()) {

					long tpos[D];
					work_buffer_get_pos(D, dims, tpos, true, cfl_loop_index[cfl_loop_worker_id()] + i);

					dst = desc->file_addr + md_calc_offset(D, file_strs, tpos);
				}

				mpi_copy2(D, desc->data_dims, file_strs, dst, data_strs, addr, sizeof(complex float), i, 0);
			}
		} else {

			md_copy_block(desc->D, desc->pos, desc->file_dims, desc->file_addr, desc->data_dims, desc->data_addr, sizeof(complex float));
		}
	}

	if (   (desc->file_addr > desc->data_addr)
	    || (desc->data_addr > desc->file_addr + ((long)sizeof(complex float) * md_calc_size(desc->D, desc->file_dims))))
		md_free(desc->data_addr);

	addr = desc->file_addr;
	md_copy_dims(D, dims, desc->file_dims);

	xfree(desc->file_dims);
	xfree(desc->data_dims);
	xfree(desc->pos);
	xfree(desc);

	return addr;
}


static complex float* load_zra_internal(int fd, const char* name, int D, long dims[D])
{
	if (-1 == read_ra(fd, D, dims))
		error("Loading ra file %s\n", name);

	long T;
	if (-1 == (T = io_calc_size(D, dims, sizeof(complex float))))
		error("Loading ra file %s\n", name);

	void* addr;
	struct stat st;

	if (-1 == fstat(fd, &st))
		io_error("Loading ra file %s\n", name);

	off_t header_size;

	if (-1 == (header_size = lseek(fd, 0, SEEK_CUR)))
		io_error("Loading ra file %s\n", name);

	// ra allows random stuff at the end
	if (T + header_size > st.st_size)
		error("Loading ra file %s\n", name);

	assert(header_size < 4096);

	if (MAP_FAILED == (addr = mmap(NULL, (size_t)st.st_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0)))
		io_error("Loading ra file %s\n", name);

#ifdef __EMSCRIPTEN__
	wasm_close_later(fd);
#else
	if (-1 == close(fd))
		io_error("Loading ra file %s\n", name);
#endif

	return addr + header_size;
}


complex float* load_zra(const char* name, int D, long dims[D])
{
	int fd;
	if (-1 == (fd = open(name, O_RDONLY)))
		io_error("Loading ra file %s\n", name);

	return load_zra_internal(fd, name, D, dims);
}

complex float* load_zshm(const char* name, int D, long dims[D])
{
	if ('/' != name[0])
		error("shm file name does not start with a slash.\n");

	int fd;
	if (-1 == (fd = shm_open(name, O_RDONLY, 0)))
		io_error("Loading shm file %s\n", name);

	return load_zra_internal(fd, name, D, dims);
}


static void* create_data(int ofd, size_t header_size, size_t size)
{
	if (-1 == ftruncate(ofd, (off_t)(size + header_size)))
		return NULL;

	size_t skip = header_size & ~4095UL;
	size_t off = header_size & 4095UL;
	void* addr;

	if (MAP_FAILED == (addr = mmap(NULL, size + off, PROT_READ|PROT_WRITE, MAP_SHARED, ofd, (off_t)skip)))
		return NULL;

	return (char*)addr + off;
}


static complex float* create_zra_internal(int ofd, const char* name, int D, const long dims[D])
{
	if (-1 == write_ra(ofd, D, dims))
		error("Creating ra file %s\n", name);

	long T;
	if (-1 == (T = io_calc_size(D, dims, sizeof(complex float))))
		error("Creating ra file %s\n", name);

	off_t header_size;

	if (-1 == (header_size = lseek(ofd, 0, SEEK_CUR)))
		io_error("Creating ra file %s\n", name);

	void* data;

	if (NULL == (data = create_data(ofd, (size_t)header_size, (size_t)T)))
		error("Creating ra file %s\n", name);

#ifdef __EMSCRIPTEN__
	wasm_close_later(ofd);
#else
	if (-1 == close(ofd))
		io_error("Creating ra file %s\n", name);
#endif

	return data;
}


complex float* create_zra(const char* name, int D, const long dims[D])
{
	int ofd;
	if (-1 == (ofd = open(name, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH)))
		io_error("Creating ra file %s\n", name);

	return create_zra_internal(ofd, name, D, dims);
}


complex float* create_zshm(const char* name, int D, const long dims[D])
{
	if ('/' != name[0])
		error("shm file name does not start with a slash.\n");

	int ofd;
	if (-1 == (ofd = shm_open(name, O_RDWR /* |O_CREAT */, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH)))
		io_error("Creating shm file %s\n", name);

	return create_zra_internal(ofd, name, D, dims);
}


float* create_coo(const char* name, int D, const long dims[D])
{
	int fd;

	if (-1 == (fd = open(name, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH)))
		io_error("Creating coo file %s\n", name);

	if (-1 == write_coo(fd, D, dims))
		error("Creating coo file %s\n", name);

	long T;

	if (-1 == (T = io_calc_size(D, dims, sizeof(float))))
		error("Creating coo file %s\n", name);

	void* addr;

	if (NULL == (addr = create_data(fd, 4096, (size_t)T)))
		error("Creating coo file %s\n", name);

#ifdef __EMSCRIPTEN__
	wasm_close_later(fd);
#else
	if (-1 == close(fd))
		io_error("Creating coo file %s\n", name);
#endif

	return (float*)addr;
}


complex float* create_zcoo(const char* name, int D, const long dimensions[D])
{
	long dims[D + 1];
	dims[0] = 2; // complex
	memcpy(dims + 1, dimensions, (size_t)(D * (long)sizeof(long)));

	return (complex float*)create_coo(name, D + 1, dims);
}


static complex float* stream_clone_if_exists(const char* name, stream_t* strm, bool in)
{
	*strm = stream_lookup_name(name, in);

	if (!(*strm))
		return NULL;

	stream_clone(*strm);
	return stream_get_data(*strm);
}

static complex float* create_binary_pipe(const char* name, int D, long dimensions[D], unsigned long stream_flags)
{
	complex float* ptr;
	stream_t strm;

	if (NULL != (ptr = stream_clone_if_exists(name, &strm, false)))
		return ptr;

	io_register_output(name);

	strm = stream_create_file(name, D, dimensions, stream_flags, NULL, false);

	if (NULL == strm)
		error("Creating stream");

	if (NULL == (ptr = anon_cfl(NULL, D, dimensions)))
		error("anon cfl\n");

	stream_attach(strm, ptr, true, true);

	if (cfl_loop_desc_active())
		stream_clone(strm);

	return ptr;
}

static complex float* create_pipe(const char* name, int D, long dimensions[D], unsigned long stream_flags)
{
	long T;
	stream_t strm;
	complex float* ptr;
	char filename[] = "bart-XXXXXX";
	int fd;
	char* dir;
	char* abs_filename;
	bool call_msync = false;

	if (NULL != (ptr = stream_clone_if_exists(name, &strm, false)))
		return ptr;

	if (stream_create_binary_outputs)
		return create_binary_pipe(name, D, dimensions, stream_flags);

	io_register_output(name);

	fd = mkstemp(filename);

	debug_printf(DP_DEBUG1, "Temp file for pipe: %s\n", filename);

	dir = xmalloc(BART_MAX_DIR_PATH_SIZE);

	if (!dir)
		error("Failed to allocate space for dir. name.\n");

	if (!getcwd(dir, BART_MAX_DIR_PATH_SIZE))
		error("Directory pathname too long.\n");

	abs_filename = ptr_printf("%s/%s", dir, filename);

#ifdef __EMSCRIPTEN__
	call_msync = true;
#endif
	strm = stream_create_file(name, D, dimensions, stream_flags, abs_filename, call_msync);

	xfree(dir);
	xfree(abs_filename);

	if (NULL == strm)
		error("Creating stream");

	if (-1 == (T = io_calc_size(D, dimensions, sizeof(complex float))))
		error("temp cfl %s\n", filename);

	err_assert(T > 0);

	if (NULL == (ptr = create_data(fd, 0, (size_t)T)))
		error("temp cfl %s\n", filename);

#ifdef __EMSCRIPTEN__
	wasm_close_later(fd);
#else
	if (-1 == close(fd))
		io_error("temp cfl %s\n", filename);
#endif

	stream_attach(strm, ptr, true, true);

	if (cfl_loop_desc_active())
		stream_clone(strm);

	return ptr;
}


static complex float* create_cfl_internal2(const char* name, int D, const long dims[D])
{
	char name_bdy[1024];
	if (1024 <= snprintf(name_bdy, 1024, "%s.cfl", name))
		error("Creating cfl file %s\n", name);

	char name_hdr[1024];
	if (1024 <= snprintf(name_hdr, 1024, "%s.hdr", name))
		error("Creating cfl file %s\n", name);

	int ofd;
	if (-1 == (ofd = open(name_hdr, O_RDWR|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH)))
		io_error("Creating cfl file %s\n", name);

	if (-1 == write_cfl_header(ofd, NULL, D, dims))
		error("Creating cfl file %s\n", name);

	if (-1 == close(ofd))
		io_error("Creating cfl file %s\n", name);

	return shared_cfl(D, dims, name_bdy);
}


static complex float* create_cfl_typed(enum file_types_e type, const char* name, int D, long dims[D], unsigned long stream_flags)
{
	complex float* addr = NULL;

	if (mpi_is_main_proc()) {

		switch (type) {

		case FILE_TYPE_PIPE:

			addr = create_pipe(name, D, dims, stream_flags);
			break;

		case FILE_TYPE_RA:
			addr = create_zra(name, D, dims);
			break;

		case FILE_TYPE_COO:
			addr = create_zcoo(name, D, dims);
			break;

		case FILE_TYPE_SHM:
			addr = create_zshm(name, D, dims);
			break;

		case FILE_TYPE_MEM:
			addr = memcfl_create(name, D, dims);
			break;

		case FILE_TYPE_CFL:

			addr = create_cfl_internal2(name, D, dims);
			break;

		default:
			error("Unknown filetype!\n");
		}

	} else {

		addr = anon_cfl(NULL, D, dims);
	}

	return addr;
}

static complex float* create_cfl_internal(const char* name, int D, const long dimensions[D], unsigned long stream_flags)
{
	long dims[D];
	md_copy_dims(D, dims, dimensions);

	if (cfl_loop_desc_active()) {

		if (!md_check_equal_dims(MIN(cfl_loop_desc.D, D), dimensions, MD_SINGLETON_DIMS(cfl_loop_desc.D), cfl_loop_desc.flags))
			io_error("Loop over altered dimensions!\n");

		for (int i = 0; i < MIN(D, cfl_loop_desc.D); ++i)
			dims[i] = MD_IS_SET(cfl_loop_desc.flags, i) ? cfl_loop_desc.loop_dims[i] : dimensions[i];

	} else {

		io_unlink_if_opened(name);
	}

	enum file_types_e type = file_type(name);

	complex float* addr = NULL;

	if (FILE_TYPE_PIPE != type)	// FIXME: Why?
		io_register_output(name);

	// FIXME:
	// bart_file_access mutex prevents multiple threads (when using bart -p)
	// from simultaneously creating the same file.
	// Required because we do not 'cache'.
	// Unfortunately, it also prevents multiple threads from simultaneously creating *different* files.
	//
	// When using bart streams with multiple FIFO outputs, this causes deadlock:
	// - open() on file A is called , blocks until a fifo has a reader (thread blocks on open).
	// - this prevents open() of file B (thread blocks on mutex).
	// - reading process tries to open B first, then A.
	// - Nothing ever happens.
	// Example:
	// bart phantom | bart tee 1.fifo 2.fifo > /dev/null & cat 2.fifo; cat 1.fifo
	// Possible solutions:
	// - per file mtx?
	// - improve looping framework, e.g.:
	//	- don't create/open same file twice in the first place. (address<->name lookup table)

	if (mmio_file_locking) {

#pragma 	omp critical (bart_file_access)
		addr = create_cfl_typed(type, name, D, dims, stream_flags);

	} else {

		assert(!cfl_loop_desc_active());
		addr = create_cfl_typed(type, name, D, dims, stream_flags);
	}

	return create_worker_buffer(D, dims, addr, true);
}


complex float* create_cfl(const char* name, int D, const long dimensions[D])
{
	return create_cfl_internal(name, D, dimensions, cfl_loop_desc.flags);
}


complex float* create_async_cfl(const char* name, const unsigned long flags, int D, const long dimensions[D])
{
	if (!cfl_loop_desc_active())
		return create_cfl_internal(name, D, dimensions, flags);

	if (0 != (md_nontriv_dims(D, dimensions) & flags))
		error("Creating stream %s: Cannot combine streaming and looping!\n", name);

	return create_cfl_internal(name, D, dimensions, 0UL);
}



float* load_coo(const char* name, int D, long dims[D])
{
	int fd;

	if (-1 == (fd = open(name, O_RDONLY)))
		io_error("Loading coo file %s\n", name);

	if (-1 == read_coo(fd, D, dims))
		error("Loading coo file %s\n", name);

	long T;

	if (-1 == (T = io_calc_size(D, dims, sizeof(float))))
		error("Loading coo file %s\n", name);

	void* addr;
	struct stat st;

	if (-1 == fstat(fd, &st))
		io_error("Loading coo file %s\n", name);

	if (T + 4096 != st.st_size)
		error("Loading coo file %s\n", name);

	if (MAP_FAILED == (addr = mmap(NULL, (size_t)T, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 4096)))
		io_error("Loading coo file %s\n", name);

#ifdef __EMSCRIPTEN__
	wasm_close_later(fd);
#else
	if (-1 == close(fd))
		io_error("Loading coo file %s\n", name);
#endif

	return (float*)addr;
}


complex float* load_zcoo(const char* name, int D, long dimensions[D])
{
	long dims[D + 1];
	float* data = load_coo(name, D + 1, dims);

	if (2 != dims[0])
		error("Loading coo file %s\n", name);

	memcpy(dimensions, dims + 1, (size_t)(D * (long)sizeof(long)));

	return (complex float*)data;
}


static complex float* load_cfl_internal(const char* name, int D, long dimensions[D], bool priv, bool stream)
{
	io_register_input(name);

	enum file_types_e type = file_type(name);

	complex float* addr = NULL;
	char* filename = NULL;
	stream_t strm;

#pragma omp critical (bart_file_access2)	// FIXME. this critical section is too big
	if (mpi_is_main_proc() || mpi_shared_files) {

		switch (type) {

		case FILE_TYPE_PIPE:

			// FIXME: should probably be moved into a file

			assert(1 == mpi_get_num_procs());

			strm = stream_lookup_name(name, true);

			if (NULL != strm) {

				addr = stream_get_data(strm);
				stream_get_dimensions(strm, D, dimensions);

				goto loadcfl_stream_end;
			}

			strm = stream_load_file(name, D, dimensions, &filename);

			if (NULL == strm)
				error("Creating stream\n");

			if (stream_is_binary(strm)) {

				assert(NULL == filename);
				addr = anon_cfl(NULL, D, dimensions);
				stream_attach(strm, addr, true, true);
			} else {

				addr = shared_cfl(D, dimensions, filename);
				//FIXME: MAP_PRIVATE states: It is unspecified whether changes made to the file
				//       after the mmap() call are visible in the mapped region.
				//	 Hence, we always load files with MAP_SHARED and protect files which are read only.
				//	 This will cause segfaults in tools like pics which uses the copy on write feature of MAP_PRIVATE!
				if (priv)
					mprotect(addr, (size_t)(io_calc_size(D, dimensions, sizeof(complex float))), PROT_READ);
				stream_attach(strm, addr, true, true);

				if (0 != unlink(filename))
					error("Error unlinking temporary file %s\n", filename);

				free(filename);
			}

		loadcfl_stream_end:

			if (cfl_loop_desc_active())
				stream_clone(strm);

			break;

		case FILE_TYPE_RA:
			addr = load_zra(name, D, dimensions);
			break;

		case FILE_TYPE_COO:
			addr = load_zcoo(name, D, dimensions);
			break;

		case FILE_TYPE_SHM:
			addr = load_zshm(name, D, dimensions);
			break;

		case FILE_TYPE_MEM:
			addr = memcfl_load(name, D, dimensions);
			break;

		case FILE_TYPE_CFL:

			char name_bdy[1024];

			if (1024 <= snprintf(name_bdy, 1024, "%s.cfl", name))
				error("Loading cfl file %s\n", name);

			char name_hdr[1024];

			if (1024 <= snprintf(name_hdr, 1024, "%s.hdr", name))
				error("Loading cfl file %s\n", name);

			int ofd;

			if (-1 == (ofd = open(name_hdr, O_RDONLY)))
				io_error("Loading cfl file %s\n", name);

			if (-1 == read_cfl_header(ofd, &filename, NULL, D, dimensions))
				error("Loading cfl file %s\n", name);

			if (-1 == close(ofd))
				io_error("Loading cfl file %s\n", name);

			addr = (priv ? private_cfl : shared_cfl)(D, dimensions, name_bdy);
		}
	}

	if (!stream || cfl_loop_desc_active()) {

		long pos[D];
		work_buffer_get_pos(D, dimensions, pos, false, cfl_loop_index[cfl_loop_worker_id()]);

		stream_t strm = stream_lookup(addr);

		stream_sync_slice(strm, D, dimensions, cfl_loop_desc.flags, pos);
	}

	if (1 < mpi_get_num_procs() && !mpi_shared_files) {

		mpi_sync_val(dimensions, D * (long)sizeof(long));

		if (!mpi_is_main_proc())
			addr = anon_cfl(NULL, D, dimensions);

		// sliced sync is performed in create_work_buffer
		if (!cfl_loop_desc_active())
			mpi_sync_val(addr, io_calc_size(D, dimensions, sizeof(complex float*)));
	}

	return create_worker_buffer(D, dimensions, addr, false);
}


complex float* load_cfl(const char* name, int D, long dimensions[D])
{
	return load_cfl_internal(name, D, dimensions, true, false);
}


complex float* load_shared_cfl(const char* name, int D, long dimensions[D])
{
	return load_cfl_internal(name, D, dimensions, false, false);
}


complex float* load_async_cfl(const char* name, int D, long dimensions[D])
{
	// we don't mix streaming via looping and explicit streaming
	if (cfl_loop_desc_active())
		return load_cfl_internal(name, D, dimensions, true, false);

	return load_cfl_internal(name, D, dimensions, true, true);
}


complex float* shared_cfl(int D, const long dims[D], const char* name)
{
//	struct stat st;
	int fd;
	void* addr;
	long T;

	if (-1 == (T = io_calc_size(D, dims, sizeof(complex float))))
		error("shared cfl %s\n", name);

	err_assert(T > 0);

        if (-1 == (fd = open(name, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH)))
		io_error("shared cfl %s\n", name);

//	if (-1 == (fstat(fd, &st)))
//		error("abort\n");

//	if (!((0 == st.st_size) || (T == st.st_size)))
//		error("abort\n");

	if (NULL == (addr = create_data(fd, 0, (size_t)T)))
		error("shared cfl %s\n", name);

#ifdef __EMSCRIPTEN__
	wasm_close_later(fd);
#else
	if (-1 == close(fd))
		io_error("shared cfl %s\n", name);
#endif

	return addr;
}


complex float* anon_cfl(const char* /*name*/, int D, const long dims[D])
{
	void* addr;
	long T;

	if (-1 == (T = io_calc_size(D, dims, sizeof(complex float))))
		error("anon cfl\n");

	if (MAP_FAILED == (addr = mmap(NULL, (size_t)T, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0)))
		io_error("anon cfl\n");

	return addr;
}



void unmap_raw(const void* data, size_t size)
{
	if (NULL == data)
		return;

	if (-1 == munmap((void*)data, size))
		io_error("unmap raw");
}


void* private_raw(size_t* size, const char* name)
{
	int fd;
	void* addr;
	struct stat st;

	if (-1 == (fd = open(name, O_RDONLY)))
		error("abort\n");

	if (-1 == (fstat(fd, &st)))
		error("abort\n");

	*size = (size_t)st.st_size;

	if (MAP_FAILED == (addr = mmap(NULL, *size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0)))
		error("abort\n");

#ifdef __EMSCRIPTEN__
	wasm_close_later(fd);
#else
	if (-1 == close(fd))
		error("abort\n");
#endif

	return addr;
}


complex float* private_cfl(int D, const long dims[D], const char* name)
{
	long T;

	if (-1 == (T = io_calc_size(D, dims, sizeof(complex float))))
		error("private cfl %s\n", name);

	int fd;
	void* addr;
	struct stat st;

	if (-1 == (fd = open(name, O_RDONLY)))
		io_error("private cfl %s\n", name);

	if (-1 == (fstat(fd, &st)))
		io_error("private cfl %s\n", name);

	if (T != st.st_size)
		error("private cfl %s\n", name);

	if (MAP_FAILED == (addr = mmap(NULL, (size_t)T, PROT_READ|PROT_WRITE, MAP_PRIVATE | MAP_NORESERVE, fd, 0)))
		io_error("private cfl %s\n", name);

#ifdef __EMSCRIPTEN__
	wasm_close_later(fd);
#else
	if (-1 == close(fd))
		io_error("private cfl %s\n", name);
#endif

	return addr;
}

static int munmap_rounded(const complex float* x, long sz)
{
	complex float* trunc_ptr = (complex float*)((uintptr_t)x & ~4095UL);

	ptrdiff_t pdiff = (const void*)x - (const void*)trunc_ptr;
	assert(0 <= pdiff);
	size_t offset = (size_t)pdiff;

	// we still need to provide the full size of the memory map to munmap
	// Therefore, we add the difference of the truncated pointer to the size here
	// Apparently, only emscripten checks for this
	return munmap(trunc_ptr, (size_t)sz + offset);
}


void unmap_cfl(int D, const long dims[D], const complex float* x)
{
	if (NULL == x)
		return;

	if (memcfl_unmap(x))
		return;

	long tdims[D?:1];
	md_copy_dims(D, tdims, dims);

	long pos[D?:1];
	md_set_dims(D, pos, 0);

	if (cfl_loop_desc_active())
		work_buffer_get_pos(D, NULL, pos, true, cfl_loop_index[cfl_loop_worker_id()]);

	x = free_worker_buffer(D, tdims, x);

	stream_t s = stream_lookup(x);

	if (NULL != s) {

		// sync remaining data
		stream_sync_slice(s, D, tdims, cfl_loop_desc.flags, pos);

		stream_free(s);

	} else {

		unmap_shared_cfl(D, tdims, x);
	}
}


void unmap_shared_cfl(int D, const long dims[D], const complex float* x)
{
	if (NULL == x)
		return;

	long T;

	if (-1 == (T = io_calc_size(D, dims, sizeof(complex float))))
		error("unmap cfl\n");

#ifdef _WIN32
	if (-1 == munmap((void*)x, T))
		io_error("unmap cfl\n");
#else
	if (-1 == munmap_rounded(x, T))
		io_error("unmap cfl\n");
#endif
}


/**
 * Create CFL file for multiple arrays
 *
 * @param name file name
 * @param N number of arrays to store in file
 * @param D[N] number of dimensions for each array
 * @param dimensions[N] pointer to dimensions of each array
 * @param args[N] pointer to the first element of each memory mapped array
 */
void create_multi_cfl(const char* name, int N, int D[N], const long* dimensions[N], complex float* args[N])
{
	io_register_output(name);

	if (cfl_loop_desc_active())
		error("multi cfl not supported for bart loop!\n");

#ifdef MEMONLY_CFL
	error("multi cfl not supported with MEMONLY_CFL\n");
#else
	const char *p = strrchr(name, '.');

	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".ra")))
		error("multi cfl does not not support .ra\n");

	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".coo")))
		error("multi cfl does not not support .coo\n");

#ifdef USE_MEM_CFL
	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".mem")))
		error("multi cfl does not not support .mem\n");
#endif


	char name_bdy[1024];
	if (1024 <= snprintf(name_bdy, 1024, "%s.cfl", name))
		error("Creating multi cfl file %s\n", name);

	char name_hdr[1024];
	if (1024 <= snprintf(name_hdr, 1024, "%s.hdr", name))
		error("Creating multi cfl file %s\n", name);

	int fd;
	if (-1 == (fd = open(name_hdr, O_RDWR|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH)))
		io_error("Creating multi cfl file %s\n", name);

	long num_ele = 0;
	for (int i = 0; i < N; i++)
		num_ele += md_calc_size(D[i], dimensions[i]);

	if (-1 == write_multi_cfl_header(fd, NULL, num_ele, N, D, dimensions))
		error("Creating multi cfl file %s\n", name);

#ifdef __EMSCRIPTEN__
	wasm_close_later(fd);
#else
	if (-1 == close(fd))
		io_error("Creating multi cfl file %s\n", name);
#endif

	args[0] = shared_cfl(1, &num_ele, name_bdy);

	for (int i = 1; i < N; i++)
		args[i] = args[i - 1] + md_calc_size(D[i - 1], dimensions[i - 1]);
#endif /* MEMONLY_CFL */
}


static int load_multi_cfl_internal(const char* name, int N_max, int D_max, int D[N_max], long dimensions[N_max][D_max], complex float* args[N_max], bool priv)
{
	io_register_input(name);

	//if (cfl_loop_desc_active())
	//	error("multi cfl not supported for bart loop!\n");

#ifdef MEMONLY_CFL
	error("multi cfl not supported with MEMONLY_CFL\n");
#else

	char* filename = NULL;

	const char *p = strrchr(name, '.');

	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".ra")))
		error("multi cfl does not not support .ra\n");

	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".coo")))
		error("multi cfl does not not support .coo\n");

#ifdef USE_MEM_CFL
	if ((NULL != p) && (p != name) && (0 == strcmp(p, ".mem")))
		error("multi cfl does not not support .mem\n");
#endif

	char name_bdy[1024];
	if (1024 <= snprintf(name_bdy, 1024, "%s.cfl", name))
		error("Loading multi cfl file %s\n", name);

	char name_hdr[1024];
	if (1024 <= snprintf(name_hdr, 1024, "%s.hdr", name))
		error("Loading multi cfl file %s\n", name);

	int fd;
	if (-1 == (fd = open(name_hdr, O_RDONLY)))
		io_error("Loading multi cfl file %s\n", name);

	if (-1 == read_multi_cfl_header(fd, &filename, N_max, D_max, D, dimensions))
		error("Loading multi cfl file %s\n", name);

#ifdef __EMSCRIPTEN__
	wasm_close_later(fd);
#else
	if (-1 == close(fd))
		io_error("Loading multi cfl file %s\n", name);
#endif


	long num_ele = 0;
	int N = 0;
	long off[N_max];

	for (int i = 0; i < N_max; i++) {

		off[i] = num_ele;

		if (0 == D[i])
			continue;

		N++;

		num_ele += md_calc_size(D[i], dimensions[i]);
	}

	long dims[1] = { num_ele };
	args[0] = (priv ? private_cfl : shared_cfl)(1, dims, name_bdy);

	for (int i = 1; i < N_max; i++) {

		args[i] = NULL;

		if (0 < D[i])
			args[i] = args[0] + off[i];
	}

	free(filename);

	return N;
#endif /* MEMONLY_CFL */
}


/**
 * Load CFL with multiple arrays
 *
 * @param name file name
 * @param N_max maximum number of arrays expected in file
 * @param D_max maximum number dimensions per array
 *
 * @param D[N_max] number of dimensions for each array read from header
 * @param dimensions[N_max][D_max] dimensions read from header
 * @param args[N] returned pointer to the first element of each memory mapped array
 */
int load_multi_cfl(const char* name, int N_max, int D_max, int D[N_max], long dimensions[N_max][D_max], complex float* args[N_max])
{
	return load_multi_cfl_internal(name, N_max, D_max, D, dimensions, args, true);
}

/**
 * Unmap CFL file for multiple arrays
 *
 * @param N number of arrays to store in file
 * @param D[N] number of dimensions for each array
 * @param dimensions[N] pointer to dimensions of each array
 * @param args[N] pointer to the first element of each memory mapped array
 */
void unmap_multi_cfl(int N, int D[N], const long* dimensions[N], complex float* args[N])
{
#ifdef MEMONLY_CFL
	error("multi cfl not supported with MEMONLY_CFL\n");
#else

#ifdef USE_MEM_CFL
	error("multi cfl not supported with USE_MEM_CFL\n");
#endif

	size_t T = 0;

	for (int i = 0; i < N; i++) {

		if (args[i] != args[0] + (T / (long)sizeof(complex float)))
			error("unmap multi cfl 1 %ld\n", T);

		long isize = io_calc_size(D[i], dimensions[i], sizeof(complex float));

		if (-1 == isize)
			error("unmap multi cfl 2\n");

		T += (size_t)isize;
	}

	if (-1 == munmap(args[0], T))
		io_error("unmap multi cfl 3\n");
#endif
}

