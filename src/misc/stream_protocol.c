/* Copyright 2024-2026. Institute of Biomedical Imaging. TU Graz.
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

enum stream_param { NO_PARAMS = 0, LONG_PARAM };

struct typeinfo {

	const char* keyword;
	unsigned int keylen;
	bool ext;
	enum stream_param param;
};

#define TOKEN "\n# "
const int token_len = strlen(TOKEN);
#define KW_PADDED(x) x "\n"
#define KW(x) .keyword = KW_PADDED(x), .keylen = (unsigned int)strlen(KW_PADDED(x))

// Max keyword len: MSG_HDR_SIZE - 1 - token_len
static const struct typeinfo types[] = {

	[STREAM_MSG_INVALID] = { .keyword = NULL, .keylen = 0 },
	[STREAM_MSG_BREAK] = { KW("---") },
	[STREAM_MSG_FLAGS] = { KW("flags"), .param = LONG_PARAM },
	[STREAM_MSG_BINARY] = { KW("binary") },
	[STREAM_MSG_INDEX] = { KW("index"), .param = LONG_PARAM },
	[STREAM_MSG_RAW] = { KW("raw"), .ext = true, .param = LONG_PARAM },
	[STREAM_MSG_BLOCK] = { KW("block"), .ext = true, .param = LONG_PARAM },
};

bool stream_encode(int l, char buf[l], const struct stream_msg* msg)
{
	if (STREAM_MSG_INVALID == msg->type)
		return false;

	if (l < MSG_HDR_SIZE)
		return false;

	memset(buf, MSG_PADDING, MSG_HDR_SIZE);

	int w = 0;
	if (types[msg->type].param)
		w = snprintf(buf, MSG_HDR_SIZE, "%s%s%ld",
				TOKEN, types[msg->type].keyword, msg->data.data_long);
	else
		w = snprintf(buf, MSG_HDR_SIZE, "%s%s", TOKEN, types[msg->type].keyword);

	if (w >= MSG_HDR_SIZE)
		return false;

	// replace final null byte & make binary data start on a new line.
	buf[MSG_HDR_SIZE - 1] = types[msg->type].ext ? '\n' : MSG_PADDING;

	return true;
}

bool stream_decode(struct stream_msg* msg, int l, const char buf[l])
{
	// strncmp to check for equality in the token_len bytes.
	if (MSG_HDR_SIZE > l || 0 != strncmp(buf, TOKEN, token_len))
		return false;

	// copy to null-terminated buffer to secure strtol call.
	char str[MSG_HDR_SIZE] = { '\0' };
	memcpy(str, buf + token_len, MSG_HDR_SIZE - token_len - 1);

	for (msg->type = ARRAY_SIZE(types) - 1; msg->type > STREAM_MSG_INVALID; msg->type--)
		if (0 == strncmp(types[msg->type].keyword, str, types[msg->type].keylen))
			break;

	if (STREAM_MSG_INVALID == msg->type)
		return false;

	msg->ext = types[msg->type].ext;

	if (types[msg->type].param)
		msg->data.data_long = strtol(str + types[msg->type].keylen, NULL, 10);

	return true;
}

