
#ifndef _STREAM_H
#define _STREAM_H 1

#include <stddef.h>

#include "misc/cppwrap.h"

#define MAXSTREAMS 10

struct stream_settings;
struct stream_msg;
struct stream;

struct stream_event {

	long index;
	int type;
	long size;
	char* data;
};


struct list_s;

typedef struct stream* stream_t;

extern stream_t stream_lookup(const _Complex float* ptr);
extern stream_t stream_lookup_name(const char* name, _Bool in);

extern _Bool stream_is_synced(stream_t s, long index);


extern stream_t stream_create(int N, const long dims[__VLA(N)], int pipefd, _Bool input, _Bool binary, unsigned long flags, const char* name, _Bool call_msync);
extern stream_t stream_clone(stream_t s);

extern void stream_free(stream_t s);

extern void stream_unmap_all(void);

void stream_attach(stream_t s, _Complex float* x, _Bool unmap, _Bool regist);
extern void stream_ensure_fifo(const char* name);
extern stream_t stream_load_file(const char* name, int D, long dims[__VLA(D)], char **datname);
extern stream_t stream_load_fd(int fd, const char* name, int D, long dims[D], char **datname, char** cmdline);
extern stream_t stream_create_file(const char* name, int D, long dims[__VLA(D)], unsigned long stream_flags, char* dataname, _Bool call_msync);

extern void stream_sync(stream_t s, int N, long pos[__VLA(N)]);
extern _Bool stream_sync_try(stream_t s, int N, long pos[__VLA(N)]);

extern void stream_sync_slice(stream_t s, int N, const long dims[__VLA(N)], unsigned long flags, const long pos[__VLA(N)]);
extern _Bool stream_sync_slice_try(stream_t s, int N, const long dims[__VLA(N)], unsigned long flags, const long pos[__VLA(N)]);

extern void stream_sync_all(stream_t strm);

extern void stream_fetch(stream_t s);

extern _Bool stream_receive_next(stream_t s, int N, long pos[__VLA(N)]);
extern _Bool stream_receive_serial(stream_t s, int N, long pos[__VLA(N)], long serial);

extern _Complex float* stream_get_data(stream_t s);
extern void stream_get_dimensions(stream_t s, int D, long dims[__VLA(D)]);

extern unsigned long stream_get_flags(stream_t s);
extern _Bool stream_is_binary(stream_t s);

extern void stream_get_latest_pos(stream_t s, int N, long pos[__VLA(N)]);
extern int stream_get_fd(stream_t s);

extern _Bool stream_get_msg(int pipefd, struct stream_msg* msg);
extern _Bool stream_send_msg(int pipefd, const struct stream_msg* msg);
extern void stream_get_raw(int pipefd, int N, long dims[__VLA(N)], long str[__VLA(N)], void* ext, size_t elsize);
extern _Bool stream_send_msg2(int pipefd, const struct stream_msg* msg, int N, const long dims[__VLA(N)], const long str[__VLA(N)], const void* extptr, size_t elsize);

extern _Bool stream_read_settings(int pipefd, struct stream_settings* settings);
extern _Bool stream_write_settings(int pipefd, struct stream_settings settings);

extern _Bool stream_add_event(stream_t s, int N, long pos[__VLA(N)], int type, const char* data, size_t size);
extern struct list_s* stream_get_events(struct stream* s, int N, long pos[__VLA(N)]);

#include "misc/cppwrap.h"

#endif	// _STREAM_H

