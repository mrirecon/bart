
#ifndef __STREAM_PROTOCOL_H
#define __STREAM_PROTOCOL_H 1

#include "misc/cppwrap.h"

#define MSG_HDR_SIZE 16

struct comm;

enum stream_msg_type {

	STREAM_MSG_INVALID,

	// grouping
	STREAM_MSG_BEGINGROUP, STREAM_MSG_ENDGROUP,

	// setup
	STREAM_MSG_FLAGS, STREAM_MSG_SERIAL, STREAM_MSG_BINARY,

	// stream progress
	STREAM_MSG_INDEX, STREAM_MSG_RAW,

	STREAM_MSG_BLOCK
};


struct stream_msg {

	enum stream_msg_type type;

	union {

		long offset;
		long extsize;
		long data_long;

		unsigned long flags;
		unsigned long data_unsigned_long;
	} data;

	_Bool ext;
};

extern int stream_encode(int l, char buf[__VLA(l)], const struct stream_msg* msg);
extern int stream_decode(struct stream_msg* msg, int l, const char buf[__VLA(l)]);

#include "misc/cppwrap.h"

#endif	// __STREAM_PROTOCOL_H

