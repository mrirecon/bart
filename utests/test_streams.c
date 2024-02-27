/* Copyright 2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <signal.h>
#include <unistd.h>
#include <complex.h>
#include <stdio.h>

#include "misc/stream_protocol.h"

#include "utest.h"

#define BUFLEN 100

// FIXME: for some reason we want to abort
#define UTEST_ERR	abort()

static bool generic_test_stream_transcode(struct stream_msg* out, const struct stream_msg msg_ref)
{
	char buf[BUFLEN] = { '\0' };

	int x = stream_encode(BUFLEN, buf, &msg_ref);

	if (x >= BUFLEN)
		UTEST_ERR;

	int y = stream_decode(out, x, buf);

	if (y != x)
		UTEST_ERR;

	return true;
}

static bool test_stream_transcode(void)
{
	struct stream_msg msg_recv;
	struct stream_msg msg_ref;
	struct stream_msg msg_default = { .type = STREAM_MSG_INVALID };

	msg_recv = msg_default;
	msg_ref = (struct stream_msg){ .type = STREAM_MSG_INDEX, .data.offset = 10 };

	if (!generic_test_stream_transcode(&msg_recv, msg_ref))
		UTEST_ERR;

	if ((msg_recv.data.offset != msg_ref.data.offset) || (msg_recv.type != msg_ref.type))
		UTEST_ERR;

	msg_recv = msg_default;
	msg_ref = (struct stream_msg){ .type = STREAM_MSG_FLAGS, .data.flags = 1024 };

	if (!generic_test_stream_transcode(&msg_recv, msg_ref))
		UTEST_ERR;

	if ((msg_recv.data.flags != msg_ref.data.flags) || (msg_recv.type != msg_ref.type))
		UTEST_ERR;

	msg_recv = msg_default;
	msg_ref = (struct stream_msg){ .type = STREAM_MSG_SERIAL };

	if (!generic_test_stream_transcode(&msg_recv, msg_ref))
		UTEST_ERR;

	if (msg_recv.type != msg_ref.type)
		UTEST_ERR;

	return true;
}


UT_REGISTER_TEST(test_stream_transcode);

