
#ifndef _STREAM_PROTOCOL_H
#define _STREAM_PROTOCOL_H 1

#include "misc/cppwrap.h"

#define MSG_HDR_SIZE 24
#define MSG_PADDING ' '

enum stream_msg_type {

	STREAM_MSG_INVALID,

	// grouping
	STREAM_MSG_BREAK,

	// setup
	STREAM_MSG_FLAGS, STREAM_MSG_BINARY,

	// stream progress
	STREAM_MSG_INDEX,

	// data
	STREAM_MSG_RAW, STREAM_MSG_BLOCK
};


struct stream_msg {

	enum stream_msg_type type;

	union {

		long offset;
		long extsize;
		long flags;
		long data_long;
	} data;

	_Bool ext;
};

extern _Bool stream_encode(int l, char buf[__VLA(l)], const struct stream_msg* msg);
extern _Bool stream_decode(struct stream_msg* msg, int l, const char buf[__VLA(l)]);

#include "misc/cppwrap.h"

#endif	// _STREAM_PROTOCOL_H

