/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2015 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

/* NOTE: This code packs pixel data into very simple dicom images
 * with only image related tags. Other mandatory DICOM tags are
 * missing. We only support 16 bit little endian gray scale images.
 *
 * FOR RESEARCH USE ONLY - NOT FOR DIAGNOSTIC USE
 */

#define _GNU_SOURCE

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "dicom.h"



// US unsigned short
// LS unsigned long
// IS integer string
// LT long text
// CS code string
// OW other word string

#define DGRP_IMAGE		0x0028
#define DTAG_IMAGE_SAMPLES_PER_PIXEL	0x0002
#define DTAG_IMAGE_PHOTOM_INTER		0x0004
#define DTAG_IMAGE_ROWS			0x0010
#define DTAG_IMAGE_COLS			0x0011
#define DTAG_IMAGE_BITS_ALLOC		0x0100
#define DTAG_IMAGE_BITS_STORED		0x0101
#define DTAG_IMAGE_PIXEL_HIGH_BIT	0x0102
#define DTAG_IMAGE_PIXEL_REP		0x0103	// 0 unsigned 2 two's complement

#define MONOCHROME2			"MONOCHROME2"

#define DGRP_PIXEL			0x7FE0
#define DTAG_PIXEL_DATA			0x0010

#define DGRP_FILE		0x0002
#define DTAG_META_SIZE			0x0000
#define DTAG_TRANSFER_SYNTAX		0x0010
#define LITTLE_ENDIAN_EXPLICIT		"1.2.840.10008.1.2.1"

#define DGRP_IMAGE2		0x0020
#define DTAG_IMAGE_INSTANCE_NUM		0x0013
#define DTAG_COMMENT			0x4000



// order matters...
enum eoffset {
	EOFF_BEGIN,
	ITAG_META_SIZE = EOFF_BEGIN,

	ITAG_TRANSFER_SYNTAX, 
	ITAG_IMAGE_INSTANCE_NUM, 
	ITAG_COMMENT, 
	ITAG_IMAGE_SAMPLES_PER_PIXEL, 
	ITAG_IMAGE_PHOTOM_INTER, 
	ITAG_IMAGE_ROWS, 
	ITAG_IMAGE_COLS, 
	ITAG_IMAGE_BITS_ALLOC, 
	ITAG_IMAGE_BITS_STORED, 
	ITAG_IMAGE_PIXEL_HIGH_BIT, 
	ITAG_IMAGE_PIXEL_REP, 
	ITAG_PIXEL_DATA, 

	EOFF_END,
};


struct element {

	enum eoffset eoff; // sanity check

	uint16_t group;
	uint16_t element;
	char vr[2];

	unsigned int len;
	const void* data;
};



struct element dicom_elements_default[EOFF_END] = {

	{ ITAG_META_SIZE, DGRP_FILE, DTAG_META_SIZE, "UL", 4, &(uint32_t){ 28 } },
	{ ITAG_TRANSFER_SYNTAX, DGRP_FILE, DTAG_TRANSFER_SYNTAX, "UI", sizeof(LITTLE_ENDIAN_EXPLICIT), LITTLE_ENDIAN_EXPLICIT },
	{ ITAG_IMAGE_INSTANCE_NUM, DGRP_IMAGE2, DTAG_IMAGE_INSTANCE_NUM, "IS", 0, NULL },
	{ ITAG_COMMENT, DGRP_IMAGE2, DTAG_COMMENT, "LT", 22, "NOT FOR DIAGNOSTIC USE\0\0" },
	{ ITAG_IMAGE_SAMPLES_PER_PIXEL, DGRP_IMAGE, DTAG_IMAGE_SAMPLES_PER_PIXEL, "US", 2, &(uint16_t){ 1 } }, 		// gray scale
	{ ITAG_IMAGE_PHOTOM_INTER, DGRP_IMAGE, DTAG_IMAGE_PHOTOM_INTER, "CS", sizeof(MONOCHROME2), MONOCHROME2 },	// 0 is black
	{ ITAG_IMAGE_ROWS, DGRP_IMAGE, DTAG_IMAGE_ROWS, "US", 2, &(uint16_t){ 0 } },
	{ ITAG_IMAGE_COLS, DGRP_IMAGE, DTAG_IMAGE_COLS, "US", 2, &(uint16_t){ 0 } },
	{ ITAG_IMAGE_BITS_ALLOC, DGRP_IMAGE, DTAG_IMAGE_BITS_ALLOC, "US", 2, &(uint16_t){ 16 } },			//
	{ ITAG_IMAGE_BITS_STORED, DGRP_IMAGE, DTAG_IMAGE_BITS_STORED, "US", 2, &(uint16_t){ 16 } },			// 12 for CT
	{ ITAG_IMAGE_PIXEL_HIGH_BIT, DGRP_IMAGE, DTAG_IMAGE_PIXEL_HIGH_BIT, "US", 2, &(uint16_t){ 15 } },
	{ ITAG_IMAGE_PIXEL_REP, DGRP_IMAGE, DTAG_IMAGE_PIXEL_REP, "US", 2, &(uint16_t){ 0 } },			// unsigned
	{ ITAG_PIXEL_DATA, DGRP_PIXEL, DTAG_PIXEL_DATA, "OW", 0, NULL },
};




static bool vr_oneof(const char a[2], unsigned int N, const char b[N][2])
{
	for (unsigned int i = 0; i < N; i++)
		if ((a[0] == b[i][0]) && (a[1] == b[i][1]))
			return true;

	return false;
}

static int dicom_write_element(unsigned int len, char buf[static 8 + len], struct element e)
{
	assert((((union { uint16_t s; uint8_t b; }){ 1 }).b));	// little endian

	assert(len == e.len);
	assert(0 == len % 2);

	int o = 0;

	buf[o++] = ((e.group >> 0) & 0xFF);
	buf[o++] = ((e.group >> 8) & 0xFF);

	buf[o++] = ((e.element >> 0) & 0xFF);
	buf[o++] = ((e.element >> 8) & 0xFF);

 	buf[o++] = e.vr[0];
	buf[o++] = e.vr[1];

	if (!vr_oneof(e.vr, 5, (const char[5][2]){ "OB", "OW", "SQ", "UN", "UT" })) {

		buf[o++] = ((len >> 0) & 0xFF);
		buf[o++] = ((len >> 8) & 0xFF);
	
	} else {
	
		buf[o++] = 0; // reserved
		buf[o++] = 0; // reserved
		buf[o++] = ((len >>  0) & 0xFF);
		buf[o++] = ((len >>  8) & 0xFF);
		buf[o++] = ((len >> 16) & 0xFF);
		buf[o++] = ((len >> 24) & 0xFF);
	}

	memcpy(buf + o, e.data, len);
	return len + o;
}




int dicom_write(const char* name, unsigned int cols, unsigned int rows, long inum, const unsigned char* img)
{
	int fd;
	void* addr;
	struct stat st;
	int ret = -1;

	// allocate before any goto calls
	int entries = EOFF_END;

	struct element dicom_elements[entries];

	for (int i = 0; i < entries; i++) {
		memcpy(&dicom_elements[i], &dicom_elements_default[i], sizeof(struct element));
		assert(dicom_elements[i].eoff == (enum eoffset)i);
	}


	if (-1 == (fd = open(name, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR)))
		goto cleanup;

	if (-1 == fstat(fd, &st))
		goto cleanup;

	size_t size = 128 + 4;


	dicom_elements[ITAG_IMAGE_ROWS].data = &(uint16_t){ rows };
	dicom_elements[ITAG_IMAGE_COLS].data = &(uint16_t){ cols };

	char inst_num[12]; // max number of bytes for InstanceNumber tag
	sprintf(inst_num, "+%04ld", inum);

	dicom_elements[ITAG_IMAGE_INSTANCE_NUM].data = inst_num;
	dicom_elements[ITAG_IMAGE_INSTANCE_NUM].len = sizeof(inst_num);

	dicom_elements[ITAG_PIXEL_DATA].data = img;
	dicom_elements[ITAG_PIXEL_DATA].len = 2 * rows * cols;


	size += 4;	// the pixel data element is larger

	for (int i = 0; i < entries; i++)
		size += 8 + dicom_elements[i].len;


	if (-1 == ftruncate(fd, size))
		goto cleanup;

	if (MAP_FAILED == (addr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0)))
		goto cleanup;


	// write header

	memset(addr, 0, 128);
	memcpy(addr + 128, "DICM", 4);

	size_t off = 128 + 4;
	
	uint16_t last_group = 0;
	uint16_t last_element = 0;

	// make sure tags are in ascending order
	for (int i = 0; i < entries; i++) {

		assert(((last_group == dicom_elements[i].group) && (last_element < dicom_elements[i].element))
			|| (last_group < dicom_elements[i].group));

		last_group = dicom_elements[i].group;
		last_element = dicom_elements[i].element;

		off += dicom_write_element(dicom_elements[i].len, addr + off, dicom_elements[i]);
	}

	assert(0 == size - off);

	ret = 0;

	if (-1 == munmap((void*)addr, size))
		abort();

cleanup:
	if (-1 == close(fd))
		abort();

	return ret;
}


