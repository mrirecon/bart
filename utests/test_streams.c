/* Copyright 2024-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <signal.h>
#include <unistd.h>
#include <complex.h>
#include <stdio.h>
#include <string.h>

#include "misc/list.h"

#include "misc/stream.h"
#include "misc/stream_protocol.h"

#include "utest.h"


#define BUFLEN 100

// FIXME: for some reason we want to abort
#define UTEST_ERR	abort()

static bool generic_test_stream_transcode(struct stream_msg* out, const struct stream_msg msg_ref)
{
	char buf[BUFLEN] = { '\0' };

	if (!stream_encode(BUFLEN, buf, &msg_ref))
		UTEST_ERR;

	if (!stream_decode(out, BUFLEN, buf))
		UTEST_ERR;

	return true;
}

static bool test_stream_transcode(void)
{
	struct stream_msg msg_recv;
	struct stream_msg msg_ref;
	struct stream_msg msg_default = { .type = STREAM_MSG_INVALID };

	msg_recv = msg_default;
	msg_ref = (struct stream_msg){ .type = STREAM_MSG_INDEX, .data.index = 10 };

	if (!generic_test_stream_transcode(&msg_recv, msg_ref))
		UTEST_ERR;

	if ((msg_recv.data.index != msg_ref.data.index) || (msg_recv.type != msg_ref.type))
		UTEST_ERR;

	msg_recv = msg_default;
	msg_ref = (struct stream_msg){ .type = STREAM_MSG_FLAGS, .data.flags = 1024 };

	if (!generic_test_stream_transcode(&msg_recv, msg_ref))
		UTEST_ERR;

	if ((msg_recv.data.flags != msg_ref.data.flags) || (msg_recv.type != msg_ref.type))
		UTEST_ERR;

	msg_recv = msg_default;
	msg_ref = (struct stream_msg){ .type = STREAM_MSG_BREAK };

	if (!generic_test_stream_transcode(&msg_recv, msg_ref))
		UTEST_ERR;

	if (msg_recv.type != msg_ref.type)
		UTEST_ERR;

	return true;
}

static bool generic_test_stream_transceive(int pipefds[2], struct stream_msg* out, const struct stream_msg msg_ref)
{
	if (!stream_send_msg(pipefds[1], &msg_ref))
		UTEST_ERR;

	if (!stream_get_msg(pipefds[0], out))
		UTEST_ERR;

	return true;
}

static bool test_stream_transceive(void)
{
	struct stream_msg msg_recv;
	struct stream_msg msg_ref;
	struct stream_msg msg_default = { .type = STREAM_MSG_INVALID };

	int pipefds[2];
	if (0 != pipe(pipefds))
		UTEST_ERR;

	msg_ref = (struct stream_msg){ .type = STREAM_MSG_INDEX, .data.index = 2 };
	msg_recv = msg_default;

	if (!generic_test_stream_transceive(pipefds, &msg_recv,  msg_ref))
		UTEST_ERR;

	if (msg_recv.data.index != msg_ref.data.index || msg_recv.type != msg_ref.type)
		UTEST_ERR;

	msg_ref = (struct stream_msg){ .type = STREAM_MSG_BINARY };
	msg_recv = msg_default;

	if (!generic_test_stream_transceive(pipefds, &msg_recv,  msg_ref))
		UTEST_ERR;

	if (msg_recv.type != msg_ref.type)
		UTEST_ERR;

	close(pipefds[1]);
	close(pipefds[0]);

	return true;
}

static bool test_comm_msg2(void)
{
	int pipefds[2];

	if (0 != pipe(pipefds))
		UTEST_ERR;

	complex float a[3] = { 1, 2, 3 };
	complex float b[3] = { 0, 0, 0 };

	long dims[1] = { 3 };
	long str[1] = { sizeof(complex float) };
	long size = 3 * sizeof(complex float);

	struct stream_msg msg = { .type = STREAM_MSG_RAW, .ext = true, .data.extsize = size };

	if (!stream_send_msg2(pipefds[1], &msg, 1, dims, str, a, sizeof(complex float)))
		UTEST_ERR;

	struct stream_msg recv;

	if (!stream_get_msg(pipefds[0], &recv))
		UTEST_ERR;

	if (STREAM_MSG_RAW != recv.type)
		UTEST_ERR;

	stream_get_raw(pipefds[0], 1 , dims, str, b, sizeof(complex float));

	for (unsigned int i = 0 ; i < ARRAY_SIZE(a); i++)
		if (a[i] != b[i])
			UTEST_ERR;

	close(pipefds[1]);
	close(pipefds[0]);

	return true;
}

static bool test_comm_followup(void)
{
	int pipefds[2];

	if (0 != pipe(pipefds))
		UTEST_ERR;

	struct stream_msg msg_ref = { .type = STREAM_MSG_INDEX, .data.index = 2 };
	struct stream_msg msg_recv;

	for (int i = 0; i < 2; i++)
		if (!stream_send_msg(pipefds[1], &msg_ref))
			UTEST_ERR;

	if (!stream_get_msg(pipefds[0], &msg_recv))
		UTEST_ERR;

	if (!stream_get_msg(pipefds[0], &msg_recv))
		UTEST_ERR;

	if (msg_recv.data.index != msg_ref.data.index || msg_recv.type != msg_ref.type)
		UTEST_ERR;

	close(pipefds[1]);
	close(pipefds[0]);

	return true;
}


static bool test_stream_registry(void)
{
	long dims[1] = { 1 };
	const int N = 1;

	complex float a[6];
	stream_t s[6];

	for (int i = 0; i < 5; i++) {

		s[i] = stream_create(N, dims, -1, true, false, 1, NULL, false);
		stream_attach(s[i], a + i, false, true);
	}

	s[5] = stream_create(N, dims, -1, true, false, 1, NULL, false);

	for (int i = 0; i < 5; i++)
		if (s[i] != stream_lookup(a + i))
			UTEST_ERR;

	if (NULL != stream_lookup(a + 5))
		UTEST_ERR;

	for (int i = 0; i < 6; i++)
		stream_free(s[i]);

	for (int i = 0; i < 6; i++)
		if (NULL != stream_lookup(a + i))
			UTEST_ERR;

	return true;
}

static bool test_stream_sync(void)
{
	int pipefds[2];

	if (0 != pipe(pipefds))
		UTEST_ERR;

	stream_t strm_in, strm_out;

	if (!(strm_out = stream_create(1, (long[1]){ 1 }, pipefds[1], false, false, 1, NULL, false)))
		UTEST_ERR;

	if (!(strm_in = stream_create(1, (long[1]){ 1 }, pipefds[0], true, false, 1, NULL, false)))
		UTEST_ERR;

	if (stream_is_synced(strm_in, 0) || stream_is_synced(strm_out, 0))
		UTEST_ERR;

	stream_sync(strm_out, 1, (long[1]){ 0 });

	if (!stream_is_synced(strm_out, 0))
		UTEST_ERR;

	stream_sync(strm_in, 1, (long[1]){ 0 });

	if (!stream_is_synced(strm_in, 0) || !stream_is_synced(strm_out, 0))
		UTEST_ERR;

	stream_free(strm_in);
	stream_free(strm_out);

	close(pipefds[1]);
	close(pipefds[0]);

	return true;
}

static bool test_binary_stream(void)
{
	int pipefds[2];

	if (0 != pipe(pipefds))
		UTEST_ERR;

	long dims[2] = { 1, 3 };
	complex float out[3] = { 1, 2, 3 };
	complex float  in[3] = { 0, 0, 0 };

	stream_t strm_in, strm_out;
	//write end of the pipe
	if (!(strm_out = stream_create(2, dims, pipefds[1], false, true, 2, NULL, false)))
		UTEST_ERR;
	// read end
	if (!(strm_in = stream_create(2, dims, pipefds[0], true, true, 2, NULL, false)))
		UTEST_ERR;

	stream_attach(strm_in, in, false, false);
	stream_attach(strm_out, out, false, false);

	if (stream_is_synced(strm_in, 0) || stream_is_synced(strm_out, 0))
		UTEST_ERR;

	stream_sync_slice(strm_out, 2, dims, 2, (long[2]){ 0, 0 });
	stream_sync_slice(strm_out, 2, dims, 2, (long[2]){ 0, 2 });

	stream_sync_slice(strm_in, 2, dims, 2, (long[2]){ 0, 0 });
	stream_sync_slice(strm_in, 2, dims, 2, (long[2]){ 0, 2 });

	if (out[0] != in[0])
		UTEST_ERR;

	if (0 != in[1])
		UTEST_ERR;

	if (out[2] != in[2])
		UTEST_ERR;

	stream_free(strm_in);
	stream_free(strm_out);

	close(pipefds[1]);
	close(pipefds[0]);

	return true;
}

static bool test_stream_events(void)
{
	int pipefds[2];

	if (0 != pipe(pipefds))
		UTEST_ERR;

	stream_t strm_in, strm_out;

	if (!(strm_out = stream_create(1, (long[1]){ 2 }, pipefds[1], false, false, 1, NULL, false)))
		UTEST_ERR;

	if (!(strm_in = stream_create(1, (long[1]){ 2 }, pipefds[0], true, false, 1, NULL, false)))
		UTEST_ERR;

	enum { LEN = 4 };
	char teststr[][LEN] = { "HFS", "HFP" };

	// add 2 events
	for (int i = 0; i < 2; i++)
		if(!stream_add_event(strm_out, 1, (long[1]){ i }, 0, teststr[i], LEN))
			UTEST_ERR;

	// verify that we can't add to an input stream
	if (stream_add_event(strm_in, 1, (long[1]){ 0 }, 0, teststr[0], LEN))
		UTEST_ERR;

	stream_sync(strm_out, 1, (long[1]){ 0 });


	// check that we can't add to already synced position
	if (stream_add_event(strm_out, 1, (long[1]){ 0 }, 0, teststr[0], LEN))
		UTEST_ERR;


	stream_sync(strm_out, 1, (long[1]){ 1 });

	stream_sync(strm_in, 1, (long[1]){ 0 });

	stream_sync(strm_in, 1, (long[1]){ 1 });


	list_t rx_event_lists[] = {
		stream_get_events(strm_in, 1, (long[1]){ 0 }),
		stream_get_events(strm_in, 1, (long[1]){ 1 })
	};

	for (int i = 0; i < 2; i++) {

		if (1 != list_count(rx_event_lists[i]))
			UTEST_ERR;

		struct stream_event* e = list_pop(rx_event_lists[i]);

		xfree(rx_event_lists[i]);

		if (e->index != i)
			UTEST_ERR;

		if (e->size != (long)(strlen(teststr[i]) + 1))
			UTEST_ERR;

		if (0 != strcmp(teststr[i], e->data))
			UTEST_ERR;

		xfree(e);
	}

	stream_free(strm_in);
	stream_free(strm_out);

	close(pipefds[1]);
	close(pipefds[0]);
	return true;
}


UT_REGISTER_TEST(test_stream_transcode);
UT_REGISTER_TEST(test_stream_transceive);
UT_REGISTER_TEST(test_comm_msg2);
UT_REGISTER_TEST(test_comm_followup);
UT_REGISTER_TEST(test_stream_registry);
UT_REGISTER_TEST(test_stream_sync);
UT_REGISTER_TEST(test_binary_stream);
UT_REGISTER_TEST(test_stream_events);

