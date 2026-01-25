/* Copyright 2024-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Philip Schaten <philip.schaten@tugraz.at>
 * 2023 Moritz Blumenthal
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>

#ifdef _WIN32
#include "win/mman.h"
#include "win/open_patch.h"
#else
#include <sys/mman.h>
#endif

#include "num/multind.h"
#include "num/optimize.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/stream_protocol.h"
#include "misc/lock.h"
#include "misc/shrdptr.h"
#include "misc/list.h"

#include "stream.h"


struct pcfl {

	int D;
	long* dims;

	unsigned long stream_flags;

	long index;		// data continuously available including index
	long total; 		// total number of slices
	bool* synced;		// bitmask of synced slices

	long* index_map;		// for each event record synced slices
	long sync_counter;	// number of completed slices

	bart_lock_t* sync_lock;
};

struct stream {

	bool active;
	bool input;
	bool binary;

	const char* filename;
	const char* fifo_name;

	complex float* ptr;
	struct pcfl* pcfl;

	bool msync;

	int pipefd;

	bool busy;
	bart_lock_t *lock;
	bart_cond_t *cond;

	struct shared_obj_s sptr;
	bool unmap;

	list_t events;

	FILE* logfile;
	double* timestamps;
};

struct stream_settings {

	bool binary;

	unsigned long flags;
	long hdr_blocksize;
};



/* Creates a struct for a partially written complex float array.
 *
 * @param D: number of dimensions
 * @param dims: dimensions
 * @param flags: flags selecting which dims are 'incomplete'
 *
 * Returns: partial cfl struct
 *
 * A pcfl is an n-dimensional complex float array whose values are not yet known.
 * The flags define "slices" of the array.
 * Validity of the memory for every slice can be individually
 * checked/written using various pcfl_* functions.
 * Necessary state is kept in the struct returned from this function
 **/
static struct pcfl* pcfl_create(int D, const long dims[D], unsigned long flags)
{
	assert(bitcount(flags) <= D);

	PTR_ALLOC(struct pcfl, ret);

	ret->D = D;
	ret->stream_flags = flags;
	ret->index = -1;
	ret->dims = ARR_CLONE(long[D], dims);

	long stream_dims[D];
	md_select_dims(D, flags, stream_dims, dims);

	ret->synced = md_calloc(D, stream_dims, sizeof(bool));

	ret->total = md_calc_size(D, stream_dims);

	ret->index_map = md_calloc(D, stream_dims, sizeof(long));

	ret->sync_counter = 0;
	ret->sync_lock = bart_lock_create();

	return PTR_PASS(ret);
}

static void pcfl_free(struct pcfl* p)
{
	xfree(p->dims);

	md_free(p->synced);
	md_free(p->index_map);

	bart_lock_destroy(p->sync_lock);

	xfree(p);
}


/* Converts an N-dimensional position to the "slice index" for a partial CFL
 *
 * @param data: pcfl ptr
 * @param N: number of indices. Must at least include all selected dims for the pcfl!
 * @param pos: position
 *
 * Returns: slice index
 **/
static long pcfl_pos2index(struct pcfl* data, int N, const long pos[N])
{
	assert(N <= data->D);
	// make sure all stream flags are given!
	assert(0 == (data->stream_flags >> (N + 1)));	// FIXME

	long spos[data->D];
	md_set_dims(data->D, spos, 0);

	md_copy_dims(N, spos, pos);

	return md_ravel_index(data->D, spos, data->stream_flags, data->dims);
}


/* Gets the position up to which data is continuously available for a pcfl.
 *
 * @param p: pcfl ptr
 * @param N: number of indices
 * @param pos: output position
 **/
static void pcfl_get_latest_pos(struct pcfl* p, int N, long pos[N])
{
	md_set_dims(N, pos, 0);

	assert(N == p->D);

	md_unravel_index(N, pos, p->stream_flags, p->dims, MAX(0, p->index));
}


static void pcfl_get_dimensions(struct pcfl* p, int N, long dims[N])
{
	assert(N == p->D);

	md_copy_dims(N, dims, p->dims);
}




// Stream registry: Mapping between struct stream_s * <-> complex float *

static complex float* stream_ptr[MAXSTREAMS] = { };
static const char* stream_name[MAXSTREAMS] = { };
static stream_t stream_ptr_streams[MAXSTREAMS] = { };
static int stream_ptr_count = 0;


static void stream_deregister(const struct stream* s);

void stream_unmap_all(void)
{
	while (0 < stream_ptr_count) {

		stream_t stream = stream_ptr_streams[0];

		assert(stream->ptr && stream->unmap);

		int D = stream->pcfl->D;
		long dims[D];
		pcfl_get_dimensions(stream->pcfl, D, dims);

		unmap_cfl(D, dims, stream->ptr);

		stream_deregister(stream);
	}
}


static void stream_register(stream_t s)
{
	assert(s->ptr);

	int stream_ptr_pos = -1;

#pragma omp critical(stream_ptr_lock)
	{
		if (stream_ptr_count < MAXSTREAMS)
			stream_ptr_pos = stream_ptr_count++;
	}

	if (-1 == stream_ptr_pos)
		error("Maximum number of streams exceeded.\n");

	stream_ptr_streams[stream_ptr_pos] = s;
	stream_ptr[stream_ptr_pos] = s->ptr;
	stream_name[stream_ptr_pos] = s->filename;
}


static void stream_deregister(const struct stream* s)
{
#pragma omp critical(stream_ptr_lock)
	{
		for (int i = 0; i < stream_ptr_count; i++) {

			if (stream_ptr_streams[i] != s)
				continue;

			for (int j = i; j < stream_ptr_count - 1; j++) {

				stream_ptr[j] = stream_ptr[j + 1];
				stream_name[j] = stream_name[j + 1];
				stream_ptr_streams[j] = stream_ptr_streams[j + 1];
			}

			stream_ptr_count--;
		}
	}
}

stream_t stream_lookup(const complex float* ptr)
{
	stream_t s = NULL;

#pragma omp critical(stream_ptr_lock)
	{
		int i = 0;

		for (; i < stream_ptr_count; i++)
			if (stream_ptr[i] == ptr)
				break;

		if (i != stream_ptr_count)
			s = stream_ptr_streams[i];
	}

	return s;
}

static char* stream_mangle_name(const char* name, bool in);

stream_t stream_lookup_name(const char* filename, bool in)
{
	stream_t s = NULL;

	char* name = stream_mangle_name(filename, in);

#pragma omp critical(stream_ptr_lock)
	{
		int i = 0;

		for (; i < stream_ptr_count; i++)
			if (   (NULL != stream_name[i])
			    && (0 == strcmp(stream_name[i], name)))
				break;

		if (i != stream_ptr_count)
			s = stream_ptr_streams[i];
	}

	xfree(name);

	return s;
}

static char* stream_mangle_name(const char* name, bool in)
{
	assert(0 != strcmp(name, "in_-"));
	assert(0 != strcmp(name, "out_-"));

	const char* prefix = "";

	if (0 == strcmp("-", name))
		prefix = in ? "in_" : "out_";

	return ptr_printf("%s%s", prefix, name);
}



static void stream_del(const struct shared_obj_s* sptr);

static void stream_stop_log(const struct stream* s);
static void stream_init_log(stream_t s);
static void stream_log_index(stream_t s, long index, double t);

/* Creates a stream.
 *
 * @param N: Number of dimensions
 * @param dims: Dimensions of the memory area
 * @param data: Actual memory area
 * @param pipefd: File Descriptor used for synchronization
 * @param input: Whether this is an input stream (this process *reads* the data).
 * @param regist: Whether to register the stream
 * @param binary: Whether data is transferred serialized along with the metainformation.
 * @param name: Optional filename, for lookup.
 *
 * Returns: stream_t handle.
 *
 * Complex float memory shared between processes,
 * associated with a file descriptor used for synchronization and metainformation.
 **/
stream_t stream_create(int N, const long dims[N], int pipefd, bool input, bool binary, unsigned long flags, const char* name, bool msync)
{
	// msync only makes sense for output streams that are not binary.
	assert(!msync || !(input || binary));

	char* fifo_name = name ? (strcmp("-", name) ? strdup(name) : NULL) : NULL;


	PTR_ALLOC(struct stream, ret);

	*ret = (struct stream) {
		.lock = bart_lock_create(),
		.pipefd = pipefd,
		.input = input,
		.binary = binary,
		.msync = msync,
		.filename = (NULL == name) ? NULL : stream_mangle_name(name, input),
		.fifo_name = fifo_name,
		.cond = bart_cond_create(),
	};

	shared_obj_init(&ret->sptr, stream_del);

	ret->events = list_create();

	struct stream_settings settings = { .flags = flags, .binary = binary };

	if (-1 != pipefd) {

		if (input) {

			if (!stream_read_settings(pipefd, &settings))
				goto cleanup;

		} else {

			if (!stream_write_settings(pipefd, settings))
				goto cleanup;
		}

		if (binary != settings.binary)
			goto cleanup;

		ret->active = true;

		flags = settings.flags;
	}

	ret->pcfl = pcfl_create(N, dims, flags);

	stream_init_log(ret);

	return PTR_PASS(ret);

cleanup:
	stream_free(PTR_PASS(ret));

	return NULL;
}

void stream_free(stream_t s)
{
	shared_obj_destroy(&s->sptr);
}

static void stream_event_list_free(list_t events);

static void stream_del(const struct shared_obj_s* sptr)
{
	const struct stream* s = CONTAINER_OF_CONST(sptr, const struct stream, sptr);

	stream_deregister(s);

	stream_stop_log(s);

	if (s->pipefd > 1)
		close(s->pipefd);

	bart_lock_destroy(s->lock);
	bart_cond_destroy(s->cond);

	if (NULL != s->pcfl) {

		int D = s->pcfl->D;
		long dims[D];
		pcfl_get_dimensions(s->pcfl, D, dims);

		if (s->unmap)
			unmap_shared_cfl(D, dims, s->ptr);

		pcfl_free(s->pcfl);
	}

	xfree(s->filename);

	if (s->fifo_name && s->input)
		unlink(s->fifo_name);

	xfree(s->fifo_name);

	stream_event_list_free(s->events);

	xfree(s);
}

stream_t stream_clone(stream_t s)
{
	shared_obj_ref(&s->sptr);
	return s;
}

void stream_attach(stream_t s, complex float* x, bool unmap, bool regist)
{
	assert(NULL == s->ptr);

	s->ptr = x;
	s->unmap = unmap;

	if (regist) {

		assert(NULL == stream_lookup(x));

		stream_register(s);
	}
}

void stream_ensure_fifo(const char* name)
{
	struct stat statbuf = { };

	if (0 != stat(name, &statbuf))
#ifndef NO_FIFO
		mkfifo(name, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
#else
		error("BART compiled without fifo support, but requested an output stream that should be a fifo.\n");
#endif

	if (0 != stat(name, &statbuf))
		error("stat / mkfifo.\n");

	if (S_IFIFO != (statbuf.st_mode & S_IFMT))
		error(".fifo-file is not a FIFO!\n");
}

stream_t stream_load_file(const char* name, int D, long dims[D], char **datname)
{
	int fd = 0;
	bool is_stdin = (0 == strcmp(name, "-"));

	if (!is_stdin) {

		stream_ensure_fifo(name);
		fd = open(name, O_RDONLY);
	}

	if (-1 == fd)
		error("Opening FIFO %s\n", name);

	assert (!is_stdin || NULL == stdin_command_line);

	stream_t strm = stream_load_fd(fd, name, D, dims, datname, is_stdin ? &stdin_command_line : NULL);

	if (!strm)
		error("Reading input from %s\n", name);

	return strm;
}

stream_t stream_load_fd(int fd, const char* name, int D, long dims[D], char **datname, char** cmdline)
{
	char hdr[IO_MAX_HDR_SIZE] = { '\0' };
	bool binary = false;

	// read header from pipe
	int hdr_bytes = read_cfl_header2(ARRAY_SIZE(hdr) - 1, hdr, fd, name, datname, cmdline, D, dims);

	if (-1 == hdr_bytes)
		return NULL;

	if (NULL == *datname)
		binary = true;

	stream_t strm = stream_create(D, dims, fd, true, binary, 0, name, false);

	if (!strm)
		return NULL;

#ifdef __EMSCRIPTEN__
	if (!binary) {

		// mmap in emscripten is basically a read() of the whole file.
		// Thus, stream synchronization with shared files won't work.
		// Workaround: sync the whole file in the beginning, before the mmap in misc/mmio.c
		// happens.
		//
		// https://github.com/emscripten-core/emscripten/issues/17801
		// https://github.com/emscripten-core/emscripten/issues/21706

		debug_printf(DP_WARN, "WARNING: synchronous stream_load_file on EMSCRIPTEN.\n");
		stream_sync_all(strm);
	}
#endif

	return strm;
}


stream_t stream_create_file(const char* name, int D, long dims[D], unsigned long stream_flags, char* dataname, bool msync)
{
	int fd = 1;
	bool is_stdout = (0 == strcmp(name, "-"));
	bool binary = (NULL == dataname);

	// FIXME: Incompatible with LOOP!
	static bool once_w = false;

	if (is_stdout) {

		if (once_w)
			error("Writing two outputs to stdout is not possible.\n");

		once_w = true;
	}

	if (!is_stdout) {

		stream_ensure_fifo(name);
		fd = open(name, O_WRONLY);
	}

	if (-1 == fd)
		error("Opening FIFO %s\n", name);

	if (-1 == write_stream_header(fd, dataname, D, dims))
		error("Writing header of %s\n", name);

	return stream_create(D, dims, fd, false, binary, stream_flags, name, msync);
}





// Synchronization via file descriptors / message passing

void stream_get_raw(int pipefd, int N, long dims[N], long str[N], void* extptr, size_t elsize)
{
	long pos[N?:1];
	md_set_dims(N, pos, 0);

	do {
		char *ptr = extptr + md_calc_offset(N, str, pos);

		xread(pipefd, elsize, ptr);

	} while (md_next(N, dims, ~0UL, pos));
}

bool stream_get_msg(int pipefd, struct stream_msg* msg)
{
	char buffer[MSG_HDR_SIZE + 1] = { };

	xread(pipefd, MSG_HDR_SIZE, buffer);

	return stream_decode(msg, MSG_HDR_SIZE, buffer);
}


/* Send a message with additional data.
 *
 * @param pipefd: filedescriptor
 * @param msg: msg to be sent
 * @param n: number of dimensions of additional data
 * @param dims: dimensions of data
 * @param str: strides of data
 * @param extptr: ptr to store additional data.
 * @param elsize: element size of data
 */
bool stream_send_msg2(int pipefd, const struct stream_msg* msg,
		int N, const long dims[N], const long str[N], const void* extptr, size_t elsize)
{
	char buffer[MSG_HDR_SIZE] = { '\0' };

	if (!stream_encode(MSG_HDR_SIZE, buffer, msg))
		error("Stream_encode: Message failed to encode.\n");

	int w = xwrite(pipefd, MSG_HDR_SIZE, buffer);

	if (0 >= w)
		return false;

	if (NULL != extptr) {

		long pos[N?:1];
		md_set_dims(N, pos, 0);

		do {
			const void *ptr = extptr + md_calc_offset(N, str, pos);

			int w = xwrite(pipefd, elsize, ptr);

			if (0 >= w)
				return false;

		} while (md_next(N, dims, ~0UL, pos));
	}

	return true;
}



bool stream_send_msg(int pfd, const struct stream_msg* msg)
{
	return stream_send_msg2(pfd, msg, 1, (long[1]){ }, (long[1]){ }, NULL, 0UL);
}






// Calculate the memory layout on the 'wire-level' for binary streams
static long get_transport_layout(int D, const long dims[D], size_t size, int index, unsigned long flags,
		complex float* ptr, int *ND, long ndims[D], long nstr[D], complex float** nptr, size_t* nsize)
{
	long pos[D];
	md_set_dims(D, pos, 0);
	md_unravel_index(D, pos, flags, dims, MAX(0, index));

	md_calc_strides(D, nstr, dims, size);

	*nptr = &MD_ACCESS(D, nstr, pos, ptr);

	md_select_dims(D, ~flags, ndims, dims);
	md_select_strides(D, ~flags, nstr, nstr);

	*ND = simplify_dims(1, D, ndims, (long (*[])[D]){ (long (*)[])nstr });

	if (nstr[0] == (long)size) {

		*nsize = (size_t)((long)size * ndims[0]);
		ndims[0] = 1;

	} else {

		*nsize = size;
	}

	return md_calc_size(*ND, ndims) * (long)(*nsize);
}


static bool stream_add_event_intern(stream_t s, struct stream_event* event);
static struct stream_event* stream_event_create(long index, int type, const char* data, size_t size);
static struct list_s* stream_get_events_at_index(struct stream* s, long index);

// Stream Synchronization

static bool stream_receive_index_locked2(stream_t s)
{
	bool raw_received = false;
	struct stream_msg msg = { .type = STREAM_MSG_INVALID };

	if (!stream_get_msg(s->pipefd, &msg))
		return false;

	if (STREAM_MSG_INDEX != msg.type)
		return false;

	long index = msg.data.index;

	if (index < 0 || s->pcfl->total <= index)
		error("msg: invalid index");

	bool group_complete = false;

	while (!group_complete) {

		if (!stream_get_msg(s->pipefd, &msg))
			return false;

		switch (msg.type) {

		case STREAM_MSG_BREAK:

			group_complete = true;
			break;

		case STREAM_MSG_RAW:

			if ((!s->binary) || raw_received)
				return false;

		{
			assert(s->ptr);

			complex float* ptr = s->ptr;
			int ND = s->pcfl->D;
			long xdims[ND];
			long xstr[ND];
			size_t size = sizeof(complex float);

			long rx_size = get_transport_layout(ND, s->pcfl->dims, size,
							index, s->pcfl->stream_flags,
							ptr, &ND, xdims, xstr, &ptr, &size);

			if (rx_size != msg.data.extsize)
				return false;

			stream_get_raw(s->pipefd, ND, xdims, xstr, ptr, size);
		}

			raw_received = true;

			break;

		default:

			assert(msg.ext);
			long extsize = msg.data.extsize;

			if (0 == extsize)
				break;

			if (extsize < 0 || 4096 <= extsize)
				error("message with extended size too large\n");

			void* mem = xmalloc((size_t)extsize);

			xread(s->pipefd, extsize, mem);

			struct stream_event* event = stream_event_create(index, msg.type, mem, (size_t)extsize);

			xfree(mem);

			if (!stream_add_event_intern(s, event)) {

				xfree(event);
				return false;
			}
		}
	}

	if (s->binary && (!raw_received))
		return false;

	bart_lock(s->lock);

	struct pcfl* pcfl = s->pcfl;

	assert(pcfl->sync_counter < pcfl->total);
	assert(0 <= index && index < pcfl->total);

	pcfl->synced[index] = true;
	pcfl->index_map[pcfl->sync_counter++] = index;

	while (pcfl->index < pcfl->total - 1) {

		if (!pcfl->synced[pcfl->index + 1])
			break;

		pcfl->index++;
	}

	// if receiving, save timestamp after finished receiving!
	stream_log_index(s, index, timestamp());

	debug_printf(DP_DEBUG3, "data index rcvd: %ld\n", index);

	return true;
}

static bool stream_receive_index_locked(stream_t s)
{
	assert(!s->busy);
	s->busy = true;
	bart_unlock(s->lock);

	bool ret = stream_receive_index_locked2(s);

	if (!ret)
		bart_lock(s->lock);

	assert(s->busy);
	s->busy = false;

	return ret;
}

static bool stream_send_index_locked(stream_t s, long index)
{
	// if sending, save timestamp before starting sending of index.
	stream_log_index(s, index, timestamp());

	struct stream_msg msg = {

		.type = STREAM_MSG_INDEX,
		.data.index = index,
	};

	if (!stream_send_msg(s->pipefd, &msg))
		return false;

	list_t index_events = stream_get_events_at_index(s, index);

	if ((NULL != index_events) && (0 < list_count(index_events))) {

		struct stream_event* event;

		while (NULL != (event = list_pop(index_events))) {

			struct stream_msg block_msg = {

				.type = STREAM_MSG_BLOCK,
				.ext = true,
				.data.extsize = event->size,
			};

			if (!stream_send_msg2(s->pipefd, &block_msg,
				1, (long[1]){ 1 }, (long[1]){ 1 }, event->data, (size_t)event->size)) {

				xfree(event);

				stream_event_list_free(index_events);
				return false;
			}

			xfree(event);
		}
	}

	if (NULL != index_events)
		list_free(index_events);

	if (s->binary) {

		assert(s->ptr);

		complex float* ptr = s->ptr;
		struct pcfl* pcfl = s->pcfl;

		int ND = pcfl->D;
		long xdims[ND];
		long xstr[ND];
		size_t size = sizeof(complex float);

		long tx_size = get_transport_layout(ND, pcfl->dims, size, index, pcfl->stream_flags,
						    ptr, &ND, xdims, xstr, &ptr, &size);

		struct stream_msg msg = {

			.type = STREAM_MSG_RAW,
			.ext = true,
			.data.extsize = tx_size
		};

		if (!stream_send_msg2(s->pipefd, &msg, ND, xdims, xstr, ptr, size))
			return false;

	} else if (s->msync) {

		size_t size = (size_t)(md_calc_size(s->pcfl->D, s->pcfl->dims) * (long)sizeof(complex float));

		if (0 != msync(s->ptr, size, MS_SYNC))
			return false;
	}

	if (!stream_send_msg(s->pipefd, &(struct stream_msg){ .type = STREAM_MSG_BREAK }))
		return false;

	struct pcfl* pcfl = s->pcfl;

	pcfl->synced[index] = true;

	while (pcfl->index < pcfl->total - 1) {

		if (!pcfl->synced[pcfl->index + 1])
			break;

		pcfl->index++;
	}

	debug_printf(DP_DEBUG3, "data index sent: %ld\n", pcfl->index);

	return true;
}

static bool stream_sync_index(stream_t s, long index, bool allupto)
{
	if (allupto) {

		for (long i = s->pcfl->index + 1; i < index; i++)
			if (!stream_sync_index(s, i, false))
				return false;
	}

	bart_lock(s->lock);

	struct pcfl* pcfl = s->pcfl;

	if (s->input) {

		while (!pcfl->synced[index]) {

			// While other thread receiving and requested index missing, wait:
			while (s->busy && !pcfl->synced[index])
				bart_cond_wait(s->cond, s->lock);

			// Requested index received by other thread?
			if (pcfl->synced[index])
				break;

			assert(!s->busy && !pcfl->synced[index]);

			// (not busy) and (not synced) -> receive.
			if (!stream_receive_index_locked(s))
				break;

			bart_cond_notify_all(s->cond);
		}

	} else {

		if (!pcfl->synced[index])
			stream_send_index_locked(s, index);
	}

	bool synced = pcfl->synced[index];

	bart_unlock(s->lock);

	return synced;
}

bool stream_receive_serial(stream_t s, int N, long pos[N], long serial)
{
	assert(s->input);
	assert(N == s->pcfl->D);
	assert(serial <= s->pcfl->total);

	if (serial == s->pcfl->total)
		return false;

	bart_lock(s->lock);

	struct pcfl* pcfl = s->pcfl;

	while (serial >= pcfl->sync_counter) {

		while (serial >= pcfl->sync_counter && s->busy)
			bart_cond_wait(s->cond, s->lock);

		if (serial >= pcfl->sync_counter) {

			assert(!s->busy);

			if (!stream_receive_index_locked(s)) {

				bart_unlock(s->lock);
				return false;
			}

			bart_cond_notify_all(s->cond);
		}
	}

	assert(serial < pcfl->sync_counter);

	md_unravel_index(N, pos, pcfl->stream_flags, pcfl->dims, pcfl->index_map[serial]);

	bart_unlock(s->lock);

	return true;
}


/* Sync an arbitrary slice in a multidimensional array:
 * - flags & pos together define a set of fixed indices for an n-dimensional array.
 * - this syncs all stream slices (defined by the flags set in stream) that intersect the given slice.
 **/
bool stream_sync_slice_try(stream_t s, int N, const long dims[N], unsigned long flags, const long _pos[N])
{
	if (NULL == s)
		return true;

	long pos[N];
	md_copy_dims(N, pos, _pos);

	struct pcfl* pcfl = s->pcfl;

	for (int i = 0; i < MIN(N, s->pcfl->D); i++) {

		assert(MD_IS_SET(flags, i) || (0 == pos[i]));
		assert(!MD_IS_SET(flags, i) || (dims[i] == pcfl->dims[i]));
	}

	// loop over all stream dimensions which are not set in the given flags.
	unsigned long loop_flags = pcfl->stream_flags & ~flags;

	// for output streams, we'd rather need 'covered' slices instead of intersected slices.
	// Thus just fail if this is attempted.
	unsigned long lost_flags = flags & ~pcfl->stream_flags;

	assert(s->input || 0 == lost_flags);

	do {
		if (!stream_sync_index(s, pcfl_pos2index(pcfl, N, pos), false))
			return false;

	} while (md_next(pcfl->D, pcfl->dims, loop_flags, pos));

	return true;
}

void stream_sync_slice(stream_t s, int N, const long dims[N], unsigned long flags, const long pos[N])
{
	if (!stream_sync_slice_try(s, N, dims, flags, pos))
		error("Stream_sync_slice\n");
}

/* To allow disappearing in/outputs, use stream_sync_try AND catch SIGPIPE! see e.g. src/tee.c.
 * By default, a disappearing in-/ or output will end the program.
 */
bool stream_sync_try(stream_t s, int N, long pos[N])
{
	long index = pcfl_pos2index(s->pcfl, N, pos);

	return stream_sync_index(s, index, true);
}

void stream_sync(stream_t s, int N, long pos[N])
{
	if (!stream_sync_try(s, N, pos))
		error("Stream_sync\n");
}

void stream_sync_all(stream_t strm)
{
	struct pcfl* pcfl = strm->pcfl;
	int D = pcfl->D;

	long pos[D];
	md_set_dims(D, pos, 0);

	stream_sync_slice(strm, D, pcfl->dims, 0UL, pos);
}


bool stream_receive_next(stream_t s, int D, long pos[D])
{
	assert(s->input);
	assert(D == s->pcfl->D);

	bart_lock(s->lock);
	long serial = s->pcfl->sync_counter;
	bart_unlock(s->lock);

	return stream_receive_serial(s, D, pos, serial);
}


void stream_fetch(stream_t s)
{
	assert(s->input);

	int D = s->pcfl->D;
	long pos[D];
	(void)stream_receive_next(s, D, pos);
}


bool stream_read_settings(int pfd, struct stream_settings* settings)
{
	struct stream_msg msg;

	bool flags_rcvd = false;
	bool binary_rcvd = false;

	while (true) {

		if (!stream_get_msg(pfd, &msg))
			error("Reading stream settings\n");

		switch (msg.type) {

		case STREAM_MSG_FLAGS:

			if (flags_rcvd)
				return false;

			if (0 > msg.data.flags)
				return false;

			settings->flags = (unsigned long)msg.data.flags;
			flags_rcvd = true;
			break;

		case STREAM_MSG_BINARY:

			if (binary_rcvd)
				return false;

			settings->binary = true;
			binary_rcvd = true;
			break;

		case STREAM_MSG_BREAK:

			if (!flags_rcvd)
				return false;

			return true;

		default:
			return false;
		}
	}
}


bool stream_write_settings(int pfd, struct stream_settings settings)
{
	struct stream_msg msg;

	msg.type = STREAM_MSG_FLAGS;

	if (settings.flags > LONG_MAX)
		return false;

	msg.data.flags = (long)settings.flags;

	if (!stream_send_msg(pfd, &msg))
		return false;

	msg.type = STREAM_MSG_BINARY;

	if (settings.binary && !stream_send_msg(pfd, &msg))
		return false;

	msg.type = STREAM_MSG_BREAK;

	if (!stream_send_msg(pfd, &msg))
		return false;

	return true;
}


unsigned long stream_get_flags(stream_t s)
{
	return s->pcfl->stream_flags;
}

bool stream_is_binary(stream_t s)
{
	return s->binary;
}

complex float* stream_get_data(stream_t s)
{
	return s->ptr;
}

void stream_get_dimensions(stream_t s, int N, long dims[N])
{
	pcfl_get_dimensions(s->pcfl, N, dims);
}


void stream_get_latest_pos(stream_t s, int N, long pos[N])
{
	pcfl_get_latest_pos(s->pcfl, N, pos);
}

int stream_get_fd(stream_t s)
{
	return s->pipefd;
}

bool stream_is_synced(stream_t s, long index)
{
	assert(index < s->pcfl->total);

	return s->pcfl->synced[index];
}



static bool stream_event_id_eq(const void *item, const void* ref);

/* Add metadata to a stream.
 *
 * @param s: stream ptr
 * @param N: length of position array
 * @param pos: position at which the change occurs
 * @param size: size of metadata
 * @param data: metadata
 */
bool stream_add_event(stream_t s, int N, long pos[N], int type, const char* data, size_t size)
{
	bool ret = false;

	// only output streams can transmit events.
	// checking that here because stream_add_event_intern
	// should add events to an input stream.
	if (s->input)
		return ret;

	int index = pcfl_pos2index(s->pcfl, N, pos);

	bart_lock(s->pcfl->sync_lock);

	struct stream_event* item = stream_event_create(index, type, data, size);

	if (NULL == item)
		goto fail;

	if (!stream_add_event_intern(s, item)) {

		xfree(item);
		goto fail;
	}

	ret = true;
fail:
	bart_unlock(s->pcfl->sync_lock);

	return ret;
}

struct list_s* stream_get_events(struct stream* s, int N, long pos[N])
{
	return stream_get_events_at_index(s, pcfl_pos2index(s->pcfl, N, pos));
}

static bool stream_event_id_eq(const void *item, const void* ref)
{
	const struct stream_event* ev = item;
	long index = *((const long*)ref);

	return (ev->index == index);
}

static struct list_s* stream_get_events_at_index(struct stream* s, long index)
{
	return list_pop_sublist(s->events, &index, stream_event_id_eq);
}

static void stream_event_list_free(list_t events)
{
	if (NULL == events)
		return;

	struct stream_event* event;

	while (NULL != (event = list_pop(events)))
		xfree(event);

	list_free(events);
}

static bool stream_add_event_intern(stream_t s, struct stream_event* event)
{
	if ((event->index < 0) || (s->pcfl->total <= event->index))
		return false;

	if (s->pcfl->synced[event->index])
		return false;

	list_append(s->events, event);

	return true;
}

static struct stream_event* stream_event_create(long index, int type, const char* data, size_t size)
{
	size_t offset = sizeof(struct stream_event);

	void* mem = xmalloc(size + offset);

	struct stream_event* event = mem;
	event->index = index;
	event->type = type;
	event->size = (long)size;
	event->data = mem + offset;

	memcpy(event->data, data, size);

	return event;
}

static void stream_init_log(stream_t s)
{
	char* log_prefix = getenv("BART_STREAM_LOG");

	if (!log_prefix)
		return;

	const char* suffix2 = ".txt";
	unsigned int logfile_len = strlen(log_prefix) + strlen(s->filename) + strlen(suffix2) + 1;

	char* logfile_path = xmalloc(logfile_len);

	assert(logfile_path);
	snprintf(logfile_path, logfile_len, "%s%s%s", log_prefix, s->filename, suffix2);

	s->logfile = fopen(logfile_path, "a");

	if (!s->logfile)
		error("Failed to open stream logfile with 'a' flag: %s.\n", logfile_path);

	xfree(logfile_path);

	s->timestamps = xmalloc(sizeof(double) * (unsigned long)s->pcfl->total);

	fprintf(s->logfile, "# index, timestamp\n");
}

static void stream_stop_log(const struct stream* s)
{
	if (NULL == s->logfile || NULL == s->timestamps)
		return;

	for (long i = 0; i <= s->pcfl->index; i++)
		fprintf(s->logfile, "%ld, %f\n", i, s->timestamps[i]);

	if (NULL != s->logfile)
		fclose(s->logfile);

	xfree(s->timestamps);
}

static void stream_log_index(stream_t s, long index, double t)
{
	if (s->timestamps)
		s->timestamps[index] = t;
}

