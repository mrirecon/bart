/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <sys/types.h>
#include <stdint.h>
#include <complex.h>
#include <fcntl.h>
#include <unistd.h>

#include "num/multind.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/debug.h"
#include "misc/opts.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

#ifdef _WIN32
#include "win/open_patch.h"
#endif


/* Information about twix files can be found here:
 * (Matlab code by Philipp Ehses and others, Yarra by Tobias Block)
 * https://github.com/cjohnevans/Gannet2.0/blob/master/mapVBVD.m
 * https://bitbucket.org/yarra-dev/yarramodules-setdcmtags/src/
 */ 
struct hdr_s {

	uint32_t offset;
	uint32_t nscans;
};

struct entry_s {

	uint32_t measid;
	uint32_t fileid;
	uint64_t offset;
	uint64_t length;
        char patient[64];
        char protocol[64];
};

static void xread(int fd, void* buf, size_t size)
{
	size_t rsize = (size_t)read(fd, buf, size);
	if (size != rsize)
		error("Error reading %zu bytes, read returned %zu\n", size, rsize);
}

static void xseek(int fd, off_t pos)
{
        if (-1 == lseek(fd, pos, SEEK_SET))
		error("seeking");
}

static bool siemens_meas_setup(int fd, struct hdr_s* hdr)
{
	off_t start = 0;

	xseek(fd, start);
	xread(fd, hdr, sizeof(struct hdr_s));

	// check for VD version
	bool vd = ((0 == hdr->offset) && (hdr->nscans < 64));

	if (vd) {

		assert((0 < hdr->nscans) && (hdr->nscans < 30));

		struct entry_s entries[hdr->nscans];
		xread(fd, &entries, sizeof(entries));

		int n = hdr->nscans - 1;

		debug_printf(DP_INFO, "VD/VE Header. MeasID: %d FileID: %d Scans: %d\n",
				entries[n].measid, entries[n].fileid, hdr->nscans);

		debug_printf(DP_INFO, "Patient: %.64s\nProtocol: %.64s\n", entries[n].patient, entries[n].protocol);


		start = entries[n].offset;

		// reread offset
		xseek(fd, start);
		xread(fd, &hdr->offset, sizeof(hdr->offset));

	} else {

		debug_printf(DP_INFO, "VB Header.\n");
		hdr->nscans = 1;
	}

	start += hdr->offset;

	xseek(fd, start);

	return vd;
}

enum adc_flags {
	ACQEND = 0,
	SYNCDATA = 5,
};

enum adc_return {
	ADC_ERROR = -1,
	ADC_OK = 0,
	ADC_SKIP = 1,
	ADC_END = 2,
};

struct mdh1 {

	uint32_t flags_dmalength;
	int32_t measUID;
	uint32_t scounter;
	uint32_t timestamp;
	uint32_t pmutime;
};

struct mdh2 {	// second part of mdh

	uint32_t evalinfo[2];
	uint16_t samples;
	uint16_t channels;
	uint16_t sLC[14];
	uint16_t dummy1[2];
	uint16_t clmnctr;
	uint16_t dummy2[5];
	uint16_t linectr;
	uint16_t partctr;
};


static enum adc_return skip_to_next(const char* hdr, int fd, size_t offset)
{
	struct mdh1 mdh1;
	memcpy(&mdh1, hdr, sizeof(mdh1));

	size_t dma_length = mdh1.flags_dmalength & 0x01FFFFFFL;

	if (dma_length < offset)
		error("dma_length < offset.\n");

	if (-1 == lseek(fd, dma_length - offset, SEEK_CUR))
		error("seeking");

	return ADC_SKIP;
}


static enum adc_return siemens_bounds(bool vd, int fd, long min[DIMS], long max[DIMS])
{
	char scan_hdr[vd ? 192 : 0];
	size_t size = sizeof(scan_hdr);

	if (size != (size_t)read(fd, scan_hdr, size))
		return ADC_ERROR;

	long pos[DIMS] = { 0 };

	for (pos[COIL_DIM] = 0; pos[COIL_DIM] < max[COIL_DIM]; pos[COIL_DIM]++) {

		char chan_hdr[vd ? 32 : 128];
		size_t size = sizeof(chan_hdr);

		if (size != (size_t)read(fd, chan_hdr, size))
			return ADC_ERROR;

		struct mdh2 mdh;
		memcpy(&mdh, vd ? (scan_hdr + 40) : (chan_hdr + 20), sizeof(mdh));

		if (MD_IS_SET(mdh.evalinfo[0], ACQEND))
			return ADC_END;
		if (MD_IS_SET(mdh.evalinfo[0], SYNCDATA)) {

			size_t offset = sizeof(scan_hdr) + sizeof(chan_hdr);
			return skip_to_next(vd ? scan_hdr : chan_hdr, fd, offset);

		}

		if (0 == max[READ_DIM]) {

			max[READ_DIM] = mdh.samples;
			max[COIL_DIM] = mdh.channels;
		}


		if (max[READ_DIM] != mdh.samples)
			return ADC_ERROR;

		if (max[COIL_DIM] != mdh.channels)
			return ADC_ERROR;

		pos[PHS1_DIM]	= mdh.sLC[0];
		pos[AVG_DIM]	= mdh.sLC[1];
		pos[SLICE_DIM]	= mdh.sLC[2];
		pos[PHS2_DIM]	= mdh.sLC[3];
		pos[TE_DIM]	= mdh.sLC[4];
		pos[COEFF_DIM]	= mdh.sLC[5];
		pos[TIME_DIM]	= mdh.sLC[6];
		pos[TIME2_DIM]	= mdh.sLC[7];


		for (int i = 0; i < DIMS; i++) {

			max[i] = MAX(max[i], pos[i] + 1);
			min[i] = MIN(min[i], pos[i] + 0);
		}

		size = mdh.samples * CFL_SIZE;
		char buf[size];

		if (size != (size_t)read(fd, buf, size))
			return ADC_ERROR;
	}

	return ADC_OK;
}


static enum adc_return siemens_adc_read(bool vd, int fd, bool linectr, bool partctr, bool radial, const long dims[DIMS], long pos[DIMS], complex float* buf)
{
	char scan_hdr[vd ? 192 : 0];
	xread(fd, scan_hdr, sizeof(scan_hdr));

	for (pos[COIL_DIM] = 0; pos[COIL_DIM] < dims[COIL_DIM]; pos[COIL_DIM]++) {

		char chan_hdr[vd ? 32 : 128];
		xread(fd, chan_hdr, sizeof(chan_hdr));

		struct mdh2 mdh;
		memcpy(&mdh, vd ? (scan_hdr + 40) : (chan_hdr + 20), sizeof(mdh));

		if (MD_IS_SET(mdh.evalinfo[0], ACQEND))
			return ADC_END;

		if (MD_IS_SET(mdh.evalinfo[0], SYNCDATA)
			|| (dims[READ_DIM] != mdh.samples)) {

			size_t offset = sizeof(scan_hdr) + sizeof(chan_hdr);
			return skip_to_next(vd ? scan_hdr : chan_hdr, fd, offset);
		}


			if (0 == pos[COIL_DIM]) {

				// TODO: rethink this
				pos[PHS1_DIM]	= mdh.sLC[0] + (linectr ? mdh.linectr : 0);
				pos[AVG_DIM]	= mdh.sLC[1];
				if (radial) { // reorder for radial

					pos[SLICE_DIM]	= mdh.sLC[3];
				} else {

					pos[SLICE_DIM]	= mdh.sLC[2];
					pos[PHS2_DIM]	= mdh.sLC[3] + (partctr ? mdh.partctr : 0);
				}
				pos[TE_DIM]	= mdh.sLC[4];
				pos[COEFF_DIM]	= mdh.sLC[5];
				pos[TIME_DIM]	= mdh.sLC[6];
				pos[TIME2_DIM]	= mdh.sLC[7];
			}

			debug_print_dims(DP_DEBUG3, DIMS, pos);

			if (dims[READ_DIM] != mdh.samples) {

				debug_printf(DP_WARN, "Wrong number of samples: %d != %d.\n", dims[READ_DIM], mdh.samples);
				return ADC_ERROR;
			}

			if ((0 != mdh.channels) && (dims[COIL_DIM] != mdh.channels)) {

				debug_printf(DP_WARN, "Wrong number of channels: %d != %d.\n", dims[COIL_DIM], mdh.channels);
				return ADC_ERROR;
			}

			xread(fd, buf + pos[COIL_DIM] * dims[READ_DIM], dims[READ_DIM] * CFL_SIZE);
	}

	pos[COIL_DIM] = 0;
	return ADC_OK;
}




static const char help_str[] = "Read data from Siemens twix (.dat) files.";


int main_twixread(int argc, char* argv[argc])
{
	const char* dat_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &dat_file, "dat file"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	long adcs = 0;
	long radial_lines = -1;

	bool autoc = false;
	bool linectr = false;
	bool partctr = false;
	bool mpi = false;

	long dims[DIMS];
	md_singleton_dims(DIMS, dims);

	struct opt_s opts[] = {

		OPT_LONG('x', &(dims[READ_DIM]), "X", "number of samples (read-out)"),
		OPT_LONG('r', &radial_lines, "R", "radial lines"),
		OPT_LONG('y', &(dims[PHS1_DIM]), "Y", "phase encoding steps"),
		OPT_LONG('z', &(dims[PHS2_DIM]), "Z", "partition encoding steps"),
		OPT_LONG('s', &(dims[SLICE_DIM]), "S", "number of slices"),
		OPT_LONG('v', &(dims[AVG_DIM]), "V", "number of averages"),
		OPT_LONG('c', &(dims[COIL_DIM]), "C", "number of channels"),
		OPT_LONG('n', &(dims[TIME_DIM]), "N", "number of repetitions"),
		OPT_LONG('a', &adcs, "A", "total number of ADCs"),
		OPT_SET('A', &autoc, "automatic [guess dimensions]"),
		OPT_SET('L', &linectr, "use linectr offset"),
		OPT_SET('P', &partctr, "use partctr offset"),
		OPT_SET('M', &mpi, "MPI mode"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	bool radial = false;
	if (-1 != radial_lines) {

		dims[PHS1_DIM] = radial_lines;
		radial = true;
	}

	if (0 == adcs)
		adcs = dims[PHS1_DIM] * dims[PHS2_DIM] * dims[SLICE_DIM] * dims[TIME_DIM];

	debug_print_dims(DP_DEBUG1, DIMS, dims);

        int ifd;
        if (-1 == (ifd = open(dat_file, O_RDONLY)))
                error("error opening file.");

	struct hdr_s hdr;
	bool vd = siemens_meas_setup(ifd, &hdr);

	enum adc_return sar = ADC_OK;
	long off[DIMS] = { 0 };

	if (autoc) {

		long max[DIMS] = { [COIL_DIM] = 1000 };
		long min[DIMS] = { 0 }; // min is always 0

		adcs = 0;

		while (ADC_END != sar) {


			sar = siemens_bounds(vd, ifd, min, max);

			if (ADC_SKIP == sar) {

				continue;
			} else if (ADC_ERROR == sar) {

				error("Could not automatically determine dimensions, adc read error!\n");
			} else if (ADC_OK == sar) {

				debug_print_dims(DP_DEBUG3, DIMS, max);
				adcs++;
			}
		}

		debug_printf(DP_DEBUG2, "found %d adcs\n", adcs);

		for (int i = 0; i < DIMS; i++) {

			off[i] = -min[i];
			dims[i] = max[i] + off[i];
		}

		debug_printf(DP_DEBUG2, "Dimensions: ");
		debug_print_dims(DP_DEBUG2, DIMS, dims);
		debug_printf(DP_DEBUG2, "Offset: ");
		debug_print_dims(DP_DEBUG2, DIMS, off);

		siemens_meas_setup(ifd, &hdr); // reset
	}

	long odims[DIMS];
	md_copy_dims(DIMS, odims, dims);

	if (-1 != radial_lines) {

		// change output dims (must have identical layout!)
		odims[0] = 1;
		odims[1] = dims[0];
		odims[2] = dims[1];
		assert(1 == dims[2]);
	}

	complex float* out = create_cfl(out_file, DIMS, odims);
	md_clear(DIMS, odims, out, CFL_SIZE);

	debug_printf(DP_DEBUG1, "___ reading measured data (%d adcs).\n", adcs);

	long adc_dims[DIMS];
	md_select_dims(DIMS, READ_FLAG|COIL_FLAG, adc_dims, dims);

	void* buf = md_alloc(DIMS, adc_dims, CFL_SIZE);

	long mpi_slice = -1;

	sar = ADC_OK;

	while (ADC_END != sar) {

		if (mpi && (0 == adcs)) //with MPI, we cannot rely on ADC_END
			break;

		long pos[DIMS] = { [0 ... DIMS - 1] = 0 };

		sar = siemens_adc_read(vd, ifd, linectr, partctr, radial, dims, pos, buf);

		if (ADC_ERROR == sar) {

			debug_printf(DP_WARN, "ADC read error, stopping\n");
			break;
		} else if (ADC_SKIP == sar) {

			debug_printf(DP_DEBUG2, "Skipping.\n");
			continue;
		} else if (ADC_OK == sar) {

			adcs--; // count ADC

			for (int i = 0; i < DIMS; i++)
				pos[i] += off[i];

			if (mpi) {

				pos[SLICE_DIM] = mpi_slice;

				if ((0 == pos[TIME_DIM]) && (0 == pos[PHS1_DIM]))
					mpi_slice++;

				if (0 > pos[SLICE_DIM]) { //skip first

					// FIXME: Why do we not count this ADC?
					continue;
				}
			}

			debug_printf(DP_DEBUG2, "pos: ");
			debug_print_dims(DP_DEBUG2, DIMS, pos);

			if (!md_is_index(DIMS, pos, dims)) {

				// FIXME: This should be an error or fixed earlier
				debug_printf(DP_WARN, "Index out of bounds.\n");
				debug_printf(DP_WARN, " dims: ");
				debug_print_dims(DP_WARN, DIMS, dims);
				debug_printf(DP_WARN, "  pos: ");
				debug_print_dims(DP_WARN, DIMS, pos);
				continue;
			}

			md_copy_block(DIMS, pos, dims, out, adc_dims, buf, CFL_SIZE);
		}

	}

	if (0 != adcs)
		error("Incorrect number of ADCs read! ADC count difference: %d != 0!\n", adcs);
	md_free(buf);

	unmap_cfl(DIMS, dims, out);

	return 0;
}

