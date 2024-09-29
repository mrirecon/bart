/* Copyright 2024. Institute of Biomedical Imaging. TU Graz.
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
#include <sys/mman.h>

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
	long index; //data continuously available including index.
	long tot;
	bool* synced;

	bart_lock_t* sync_lock;

	complex float* ptr;
};

struct stream {

	bool active;
	bool input;
	bool binary;

	const char* filename;

	int D;
	struct pcfl* data;
	bool call_msync;

	int pipefd;

	bool busy;
	bart_lock_t *lock;
	bart_cond_t *cond;

	struct shared_obj_s sptr;
	bool unmap;

	list_t events;
};

struct stream_settings {

	bool binary;

	unsigned long flags;
	long hdr_blocksize;
};



/** Creates a struct for a partially written complex float array.
 *
 * @param x: pointer to full cfl memory. Standard strides are assumed.
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
 */
static struct pcfl* pcfl_create(complex float* x, int D, const long dims[D], unsigned long flags)
{
	assert(bitcount(flags) <= D);

	PTR_ALLOC(struct pcfl, ret);

	ret->D = D;
	ret->ptr = x;
	ret->stream_flags = flags;
	ret->index = -1;
	ret->dims = ARR_CLONE(long[D], dims);

	long stream_dims[D];
	md_select_dims(D, flags, stream_dims, dims);

	ret->synced = md_calloc(D, stream_dims, sizeof(bool));

	ret->tot = md_calc_size(D, stream_dims);

	ret->sync_lock = bart_lock_create();

	return PTR_PASS(ret);
}

static void pcfl_free(struct pcfl* p)
{
	if (NULL != p->dims)
		xfree(p->dims);

	md_free(p->synced);

	bart_lock_destroy(p->sync_lock);

	xfree(p);
}


/** Converts an N-dimensional position to the "slice index" for a partial CFL
 *
 * @param data: pcfl ptr
 * @param N: number of indices. Must at least include all selected dims for the pcfl!
 * @param pos: position
 *
 * Returns: slice index
 */
static long pcfl_pos2offset(struct pcfl* data, int N, const long pos[N])
{
	assert(N <= data->D);
	// make sure all stream flags are given!
	assert(0 == (data->stream_flags >> (N + 1)));	// FIXME

	long spos[data->D];
	md_set_dims(data->D, spos, 0);

	md_copy_dims(N, spos, pos);

	return md_ravel_index(data->D, spos, data->stream_flags, data->dims);
}


/** Gets the position up to which data is continusly available for a pcfl.
 *
 * @param p: pcfl ptr
 * @param N: number of indices
 * @param pos: output position
 */
static
void pcfl_get_latest_pos(struct pcfl* p, int N, long pos[N])
{
	md_set_dims(N, pos, 0);

	assert(N == p->D);

	md_unravel_index(N, pos, p->stream_flags, p->dims, MAX(0, p->index));
}


static
complex float* pcfl_get_data(struct pcfl* p)
{
	return p->ptr;
}


static void pcfl_get_dimensions(struct pcfl* p, int N, long dims[N])
{
	assert(N == p->D);

	md_copy_dims(N, dims, p->dims);
}


static struct pcfl* stream_get_pcfl(stream_t s);


// Stream registry: Mapping between struct stream_s * <-> complex float *

static complex float* stream_ptr[MAXSTREAMS] = { };
static const char* stream_name[MAXSTREAMS] = { };
static stream_t stream_ptr_streams[MAXSTREAMS] = { };
static int stream_ptr_count = 0;


void stream_unmap_all(void)
{
	while (0 < stream_ptr_count) {

		stream_t stream = stream_ptr_streams[0];

		assert(!stream->binary);

		int D = stream->D;
		long dims[D];
		pcfl_get_dimensions(stream->data, D, dims);

		unmap_cfl(D, dims, pcfl_get_data(stream->data));
	}
}


static void stream_register(stream_t s)
{
	assert(!s->binary);

	int stream_ptr_pos = -1;

#pragma omp critical(stream_ptr_lock)
	{
		if (stream_ptr_count < MAXSTREAMS)
			stream_ptr_pos = stream_ptr_count++;
	}

	if (-1 == stream_ptr_pos)
		error("Maximum number of streams exceeded.\n");

	stream_ptr_streams[stream_ptr_pos] = s;
	stream_ptr[stream_ptr_pos] = pcfl_get_data(s->data);
	stream_name[stream_ptr_pos] = s->filename;
}


static void stream_deregister(const struct stream* s)
{
#pragma omp critical(stream_ptr_lock)
	{
		for (int i = 0; i < stream_ptr_count; i++) {

			if (stream_ptr_streams[i] == s) {

				for (int j = i; j < stream_ptr_count - 1; j++) {

					stream_ptr[j] = stream_ptr[j + 1];
					stream_name[j] = stream_name[j + 1];
					stream_ptr_streams[j] = stream_ptr_streams[j + 1];
				}

				stream_ptr_count--;
			}
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

stream_t stream_lookup_name(const char* name)
{
	stream_t s = NULL;

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

	return s;
}



static void stream_del(const struct shared_obj_s* sptr);

/**
 * Creates a stream.
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
 */
stream_t stream_create(int N, const long dims[N], complex float* data, int pipefd, bool input, bool regist, bool binary, unsigned long flags, const char* name, bool call_msync)
{
	PTR_ALLOC(struct stream, ret);

	ret->lock = bart_lock_create();
	ret->pipefd = pipefd;
	ret->input = input;
	ret->binary = binary;
	ret->call_msync = call_msync;

	// msync only makes sense for output streams that are not binary.
	assert(!call_msync || !(input || binary));

	ret->data = NULL;
	ret->unmap = false;
	ret->events =  NULL;

	ret->filename = (NULL == name) ? NULL : strdup(name);

	ret->busy = false;
	ret->cond = bart_cond_create();

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

	ret->D = N;
	ret->data = pcfl_create(data, N, dims, flags);


	assert(!(regist && (NULL == data)));

	if (regist)
		stream_register(ret);

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
	const struct stream* s = CONTAINER_OF(sptr, const struct stream, sptr);

	stream_deregister(s);

	if (s->pipefd > 1)
		close(s->pipefd);

	bart_lock_destroy(s->lock);
	bart_cond_destroy(s->cond);

	if (NULL != s->data) {

		int D = s->D;
		long dims[D];
		pcfl_get_dimensions(s->data, D, dims);

		if (s->unmap)
			unmap_shared_cfl(D, dims, pcfl_get_data(s->data));

		pcfl_free(s->data);
	}

	if (NULL != s->filename)
		xfree(s->filename);

	stream_event_list_free(s->events);

	xfree(s);
}

stream_t stream_clone(stream_t s)
{
	shared_obj_ref(&s->sptr);
	return s;
}

void stream_attach(stream_t s, complex float* x, bool unmap)
{
	assert(NULL == pcfl_get_data(s->data));

	s->data->ptr = x;
	s->unmap = unmap;

	if (!s->binary) {

		assert(NULL == stream_lookup(x));

		stream_register(s);
	}
}


stream_t stream_load_file(const char* name, int D, long dims[D], char **datname)
{
	int fd = 0;
	char hdr[IO_MAX_HDR_SIZE] = { '\0' };
	bool is_stdin = (0 == strcmp(name, "-"));

	bool binary = false;

	if (!is_stdin)
		fd = open(name, O_RDONLY);

	if (-1 == fd)
		error("Opening FIFO %s\n", name);

	// read header from pipe
	int hdr_bytes = read_cfl_header2(ARRAY_SIZE(hdr) - 1, hdr, fd, datname, D, dims);

	if (-1 == hdr_bytes)
		error("Reading input from %s\n", name);

	if (NULL == *datname)
		binary = true;

	const char* stream_name = ptr_printf("in_%s", name);

	stream_t strm = stream_create(D, dims, NULL, fd, true, false, binary, 0, stream_name, false);

	xfree(stream_name);

	if (NULL == strm)
		error("Could not create stream for %s\n", name);

	if (!is_stdin && (0 != unlink(name)))
		error("Unlinking temporary FIFO header %s\n", name);

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

		long pos[strm->data->D];
		md_set_dims(strm->data->D, pos, 0);
		stream_sync_slice(strm, strm->data->D, strm->data->dims, 0, pos);
	}
#endif

	return strm;
}


stream_t stream_create_file(const char* name, int D, long dims[D], unsigned long stream_flags, char* dataname, bool call_msync)
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

		struct stat statbuf;

		if (0 != stat(name, &statbuf))
			if (0 != mkfifo(name, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH))
				error("Creating fifo: %s!\n", name);

		fd = open(name, O_WRONLY);
	}

	if (-1 == fd)
		error("Opening FIFO %s\n", name);

	if (-1 == write_stream_header(fd, dataname, D, dims))
		error("Writing header of %s\n", name);

	const char* stream_name = ptr_printf("out_%s", name);

	stream_t strm = stream_create(D, dims, NULL, fd, false, false, binary, stream_flags, stream_name, call_msync);

	xfree(stream_name);

	return strm;
}






// Synchronization via file descriptors / message passing

void stream_get_raw(int pipefd, long n, long dims[n], long str[n], long el, void* ext)
{
	long pos[n?:1];
	md_set_dims(n, pos, 0);

	do {
		char *ptr = ext + md_calc_offset(n, str, pos);

		xread(pipefd, el, ptr);

	} while (md_next(n, dims, ~0UL, pos));
}

bool stream_get_msg(int pipefd, struct stream_msg* msg)
{
	char buffer[MSG_HDR_SIZE + 1] = { 0 };

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
 * @param el: element size of data
 * @param ext: ptr to store additional data.
 */
bool stream_send_msg2(int pipefd, const struct stream_msg* msg, long n, const long dims[n], const long str[n], long el, const void* ext)
{
	char buffer[MSG_HDR_SIZE] = { '\0' };

	if (!stream_encode(MSG_HDR_SIZE, buffer, msg))
		error("Stream_encode: Message failed to encode.\n");

	int w = xwrite(pipefd, MSG_HDR_SIZE, buffer);

	if (0 >= w)
		return false;

	if (NULL != ext) {

		long pos[n?:1];
		md_set_dims(n, pos, 0);

		do {
			const void *ptr = ext + md_calc_offset(n, str, pos);

			int w = xwrite(pipefd, el, ptr);

			if (0 >= w)
				return false;

		} while (md_next(n, dims, ~0UL, pos));
	}

	return true;
}



bool stream_send_msg(int pfd, const struct stream_msg* msg)
{
	return stream_send_msg2(pfd, msg, 1, (long[1]){ }, (long[1]){ }, 0, NULL);
}






// Calculate the memory layout on the 'wire-level' for binary streams
static long get_transport_layout(int D, const long dims[D], long size, int index, unsigned long flags, complex float* ptr, int *ND, long ndims[D], long nstr[D], long* nsize, complex float** nptr)
{
	long pos[D];
	md_set_dims(D, pos, 0);
	md_unravel_index(D, pos, flags, dims, MAX(0, index));

	md_calc_strides(D, nstr, dims, (size_t)size);

	*nptr = &MD_ACCESS(D, nstr, pos, ptr);

	md_select_dims(D, ~flags, ndims, dims);
	md_select_strides(D, ~flags, nstr, nstr);

	*ND = simplify_dims(1, D, ndims, (long (*[])[D]){ (long (*)[])nstr });

	if (nstr[0] == size) {

		*nsize = size * ndims[0];
		ndims[0] = 1;

	} else {

		*nsize = size;
	}

	return md_calc_size(*ND, ndims) * (*nsize);
}


static bool stream_add_event_intern(stream_t s, struct stream_event* event);
static struct stream_event* stream_event_create(int index, int type, long size, const char* data);
static struct list_s* stream_get_events_at_index(struct stream* s, long index);

// Stream Synchronization

static bool stream_receive_idx_locked(stream_t s)
{
	bool raw_received = false;
	long offset = -1;
	struct stream_msg msg = { .type = STREAM_MSG_INVALID };

	if (!stream_get_msg(s->pipefd, &msg))
		return false;

	if (STREAM_MSG_INDEX != msg.type)
		return false;

	offset = msg.data.offset;

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
			complex float* ptr = pcfl_get_data(s->data);
			int ND = s->data->D;
			long xdims[s->data->D];
			long xstr[s->data->D];
			long size = sizeof(complex float);

			long rx_size = get_transport_layout(ND, s->data->dims, size, offset, s->data->stream_flags, ptr, &ND, xdims, xstr, &size, &ptr);

			if (rx_size != msg.data.extsize)
				return false;

			stream_get_raw(s->pipefd, ND, xdims, xstr, size, ptr);
		}

			raw_received = true;

			break;

		default:

			assert(msg.ext);
			if (msg.data.extsize == 0)
				break;

			void* mem = xmalloc((size_t)msg.data.extsize);

			xread(s->pipefd, msg.data.extsize, mem);

			struct stream_event* event = stream_event_create(offset, msg.type, msg.data.extsize, mem);

			xfree(mem);

			if (!stream_add_event_intern(s, event)) {

				// analyzer doesn't understand xfree here! 2024-08 ~ps
				free(event);
				return false;
			}
		}
	}

	if (s->binary && (!raw_received))
		return false;


	s->data->synced[offset] = true;

	while ((s->data->index < s->data->tot - 1) && (s->data->synced[s->data->index + 1]))
		s->data->index++;

	bart_cond_notify_all(s->cond);

	debug_printf(DP_DEBUG3, "data offset rcvd: %ld\n", s->data->index);

	return true;
}

static bool stream_send_index_locked(stream_t s, long index)
{
	if (!stream_send_msg(s->pipefd, &(struct stream_msg){ .type = STREAM_MSG_INDEX, .data.offset = index }))
		return false;

	list_t index_events = stream_get_events_at_index(s, index);

	if ((NULL != index_events) && (0 < list_count(index_events))) {

		struct stream_event* event;

		while (NULL != (event = list_pop(index_events))) {

			struct stream_msg block_msg = { .type = STREAM_MSG_BLOCK, .ext = true, .data.extsize = event->size };
			if (!stream_send_msg2(s->pipefd, &block_msg, 1, (long[1]){1}, (long[1]){1}, event->size, event->data)) {

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

		complex float* ptr = pcfl_get_data(s->data);
		int ND = s->D;
		long xdims[ND];
		long xstr[ND];
		long size = sizeof(complex float);

		long tx_size = get_transport_layout(ND, s->data->dims, size, index, s->data->stream_flags,
						    ptr, &ND, xdims, xstr, &size, &ptr);

		struct stream_msg msg = {

			.type = STREAM_MSG_RAW,
			.ext = true,
			.data.extsize = tx_size
		};

		if (!stream_send_msg2(s->pipefd, &msg, ND, xdims, xstr, size, ptr))
			return false;

	} else if (s->call_msync) {

		if (0 != msync(s->data->ptr,
			       (size_t)(md_calc_size(s->data->D, s->data->dims) * (long)sizeof(complex float)), MS_SYNC))
			return false;
	}

	if (!stream_send_msg(s->pipefd, &(struct stream_msg){ .type = STREAM_MSG_BREAK }))
		return false;

	s->data->synced[index] = true;

	while ((s->data->index < s->data->tot - 1) && (s->data->synced[s->data->index + 1]))
		s->data->index++;

	debug_printf(DP_DEBUG3, "data offset sent: %ld\n", s->data->index);
	return true;
}

static void lock_stream(stream_t s)
{
	bart_lock(s->lock);

	while (s->busy)
		bart_cond_wait(s->cond, s->lock);

	s->busy = true;
	bart_unlock(s->lock);
}

static void unlock_stream(stream_t s)
{
	s->busy = false;
	bart_cond_notify_all(s->cond);
}

static bool stream_sync_index(stream_t s, long index, bool all)
{
	if (all) {

		for (long i = s->data->index + 1; i < index; i++)
			if (!stream_sync_index(s, i, false))
				return false;
	}

	lock_stream(s);

	if (s->input) {

		while (!s->data->synced[index])
			if (!stream_receive_idx_locked(s))
				break;
	} else {

		while (!s->data->synced[index])
			if (!stream_send_index_locked(s, index))
				break;
	}

	unlock_stream(s);

	return s->data->synced[index];
}

bool stream_sync_slice_try(stream_t s, int N, const long dims[N], unsigned long flags, const long _pos[N])
{
	if (NULL == s)
		return true;

	long pos[N];
	md_copy_dims(N, pos, _pos);

	for (int i = 0; i < MIN(N, s->data->D); i++) {

		assert(MD_IS_SET(flags, i) || (0 == pos[i]));
		assert(!MD_IS_SET(flags, i) || (dims[i] == s->data->dims[i]));
	}

	unsigned long loop_flags = s->data->stream_flags & ~flags;

	do {
		if (!stream_sync_index(s, pcfl_pos2offset(s->data, N, pos), false))
			return false;

	} while (md_next(s->data->D, s->data->dims, loop_flags, pos));

	return true;
}

void stream_sync_slice(stream_t s, int N, const long dims[N], unsigned long flags, const long _pos[N])
{
	if (!stream_sync_slice_try(s, N, dims, flags, _pos))
		error("Stream_sync_slice\n");
}

/* To allow disappearing in/outputs, use stream_sync_try AND catch SIGPIPE! see e.g. src/tee.c.
 * By default, a disappearing in-/ or output will end the program.
 */
bool stream_sync_try(stream_t s, int N, long pos[N])
{
	return stream_sync_index(s, pcfl_pos2offset(stream_get_pcfl(s), N, pos), true);
}

void stream_sync(stream_t s, int N, long pos[N])
{
	if (!stream_sync_try(s, N, pos))
		error("Stream_sync\n");
}


void stream_fetch(stream_t s)
{
	lock_stream(s);
	stream_receive_idx_locked(s);
	unlock_stream(s);
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
	return s->data->stream_flags;
}

extern bool stream_is_binary(stream_t s)
{
	return s->binary;
}

complex float* stream_get_data(stream_t s)
{
	return pcfl_get_data(s->data);
}

void stream_get_dimensions(stream_t s, int N, long dims[N])
{
	pcfl_get_dimensions(s->data, N, dims);
}


void stream_get_latest_pos(stream_t s, int N, long pos[N])
{
	pcfl_get_latest_pos(stream_get_pcfl(s), N, pos);
}

int stream_get_fd(stream_t s)
{
	return s->pipefd;
}

static
struct pcfl* stream_get_pcfl(stream_t s)
{
	return s->data;
}

bool* stream_get_synced(stream_t s)
{
	return s->data->synced;
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
bool stream_add_event(stream_t s, int N, long pos[N], int type, long size, const char* data)
{
	bool ret = false;

	// only output streams can transmit events.
	// checking that here because stream_add_event_intern
	// should add events to an input stream.
	if (s->input)
		return ret;

	int index = pcfl_pos2offset(stream_get_pcfl(s), N, pos);

	bart_lock(s->data->sync_lock);

	struct stream_event* item = stream_event_create(index, type, size, data);

	if (NULL == item)
		goto fail;

	if (!stream_add_event_intern(s, item)) {

		xfree(item);
		goto fail;
	}

	ret = true;
fail:
	bart_unlock(s->data->sync_lock);

	return ret;
}

struct list_s* stream_get_events(struct stream* s, int N, long pos[N])
{
	return stream_get_events_at_index(s, pcfl_pos2offset(stream_get_pcfl(s), N, pos));
}

static bool stream_event_id_eq(const void *item, const void* ref)
{
	const struct stream_event* ev = item;
	long index = *((long*)ref);

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
	// check if index is in range
	if ((event->index >= s->data->tot) || (0 > event->index))
		return false;

	// check if this has already been synced
	if (s->data->synced[event->index])
		return false;

	list_append(s->events, event);

	return true;
}

static struct stream_event* stream_event_create(int index, int type, long size, const char* data)
{
	size_t offset = sizeof(struct stream_event);

	void* mem = xmalloc((size_t)size + offset);

	struct stream_event* event = mem;
	event->index = index;
	event->type = type;
	event->size = size;
	event->data = mem + offset;

	memcpy(event->data, data, (size_t)size);

	return event;
}

