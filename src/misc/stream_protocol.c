/* Copyright 2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2023-2024 Philip Schaten <philip.schaten@tugraz.at>
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "misc/misc.h"

#include "stream_protocol.h"


static char* keywords[] = {
	[STREAM_MSG_INVALID] = NULL,
	[STREAM_MSG_BEGINGROUP] = "# {",
	[STREAM_MSG_ENDGROUP] = "# }",
	[STREAM_MSG_FLAGS] = "# flags",
	[STREAM_MSG_SERIAL] = "# serial",
	[STREAM_MSG_BINARY] = "# binary",
	[STREAM_MSG_INDEX] = "# index",
	[STREAM_MSG_RAW] = "# raw",
	[STREAM_MSG_BLOCK] = "# block",
};


int stream_encode(int l, char buf[l], const struct stream_msg* msg)
{
	if (STREAM_MSG_INVALID == msg->type)
		return -1;

	int written = snprintf(buf, (size_t)l, "%s ", keywords[msg->type]);

	switch (msg->type) {

	// stream syntax msg, long parameter
	case STREAM_MSG_RAW:
	case STREAM_MSG_INDEX:
	case STREAM_MSG_BLOCK:

		written += snprintf(buf + written, (size_t)(l - written), "%ld", msg->data.data_long);
		break;

	// stream syntax msg, unsigned long parameter
	case STREAM_MSG_FLAGS:

		written += snprintf(buf + written, (size_t)(l - written), "%lu", msg->data.data_unsigned_long);
		break;

	// stream syntax msg, no parameters
	case STREAM_MSG_BINARY:
	case STREAM_MSG_BEGINGROUP:
	case STREAM_MSG_ENDGROUP:
	case STREAM_MSG_SERIAL:
		break;

	default:
		return -1;
	}

	written += snprintf(buf + written, (size_t)(l - written), "\n");

	return written;
}


int stream_decode(struct stream_msg* msg, int l, const char buf[l])
{
	msg->type = STREAM_MSG_INVALID;
	msg->ext = false;
	msg->data.extsize = 0;

	char* startptr = memchr(buf, '#', (size_t)l);

	int len = -1;

	if (NULL == startptr)
		goto error;

	char* endptr = memchr(startptr, '\n', (size_t)(l - (startptr - buf)));

	if (NULL == endptr)
		goto error;

	len = (endptr - buf) + 1;

	int match = STREAM_MSG_INVALID + 1;
	int keylen = 0;

	for (; match < (int)ARRAY_SIZE(keywords); match++) {

		keylen = strlen(keywords[match]);

		if (keylen > endptr - startptr)
			continue;

		// use strncmp to just compare the initial keylen bytes of
		// startptr; string behind startptr may be longer!

		if (   (0 == strncmp(startptr, keywords[match], (size_t)keylen))
		    && (   (' ' == startptr[keylen])
			|| ('\n' == startptr[keylen])))
			break;
	}

	if (match == ARRAY_SIZE(keywords))
		goto error;

	msg->type = match;

	// parameter parsing
	char* valptr = startptr + keylen + 1;
	char* endptr2 = NULL;

	switch (match) {

	// stream syntax msg, one parameter
	case STREAM_MSG_RAW:
	case STREAM_MSG_BLOCK:

		msg->ext = true;
		msg->data.extsize = strtol(valptr, &endptr2, 10);
		/* FALLTHRU */

	case STREAM_MSG_INDEX:

		msg->data.data_long = strtol(valptr, &endptr2, 10);

		if (endptr2 != endptr)
			msg->type = STREAM_MSG_INVALID;

		break;

	case STREAM_MSG_FLAGS:

		msg->data.data_unsigned_long = strtoul(valptr, &endptr2, 10);

		if (endptr2 != endptr)
			msg->type = STREAM_MSG_INVALID;

		break;

	// stream syntax msg, no parameters
	case STREAM_MSG_BINARY:
	case STREAM_MSG_BEGINGROUP:
	case STREAM_MSG_ENDGROUP:
	case STREAM_MSG_SERIAL:

		if (valptr != endptr)
			msg->type = STREAM_MSG_INVALID;

		break;

	default:
	}

error:
	return len;
}
