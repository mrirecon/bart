/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015-2021. Uecker Lab. University Center Göttingen.
 * Copyright 2017-2018. Damien Nguyen.
 * Copyright 2022-2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2021 Martin Uecker
 * 2017-2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
 */

#define _GNU_SOURCE
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include "win/mman.h"
#include "win/vdprintf.h"
#else
#include <sys/mman.h>
#endif
#include <complex.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include "num/multind.h"

#include "misc/version.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "io.h"



static char* node_id = NULL;

static void toolgraph_add_input(const char* node, const char* file);

static void toolgraph_save_iofiles(void);


static int xdprintf(int fd, const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	int ret = vdprintf(fd, fmt, ap);
	va_end(ap);

	if (ret < 0)
		error("Error writing.\n");

	return ret;
}


int xwrite(int fd, int N, const char buf[N])
{
	int w = 0;

	while (w < N) {

		int ww = write(fd, buf + w, (size_t)(N - w));

		if (0 >= ww)
			return -1;

		w += ww;
	}

	return w;
}

int xread(int fd, int N, char buf[N])
{
	int r = 0;

	while (r < N) {

		int rr = read(fd, buf + r, (size_t)(N - r));

		if (0 >= rr)
			return -1;

		r += rr;
	}

	return r;
}

static int xread0(int fd, int N, char buf[N])
{
	int r = 0;

	while (r < N) {

		int rr = read(fd, buf + r, (size_t)(N - r));

		if (0 > rr)
			error("io read error\n");

		if (0 == rr)
			break;

		r += rr;
	}

	return r;
}



enum file_types_e file_type(const char* name)
{
	if ((0 == strcmp("-", name)))
		return FILE_TYPE_PIPE;

	const char *p = strrchr(name, '.');

	if ((NULL != p) && (p != name)) {

		if (0 == strcmp(p, ".ra"))
			return FILE_TYPE_RA;

		if (0 == strcmp(p, ".coo"))
			return FILE_TYPE_COO;

		if (0 == strcmp(p, ".shm"))
			return FILE_TYPE_SHM;

		if (0 == strcmp(p, ".mem"))
			return FILE_TYPE_MEM;

		if (0 == strcmp(p, ".fifo"))
			return FILE_TYPE_PIPE;
	}

	return FILE_TYPE_CFL;
}


struct iofile_s {

	const char* name;
	bool output;
	bool input;
	bool open;
	struct iofile_s* prev;
};

static _Thread_local struct iofile_s* iofiles = NULL;



static void io_register(const char* name, bool output, bool input, bool open)
{
	struct iofile_s* iop = iofiles;
	bool new = true;

	while (NULL != iop) {

		if (0 == strcmp(name, iop->name)) {

			new = false;

			if (iop->open) {

				if ((output || iop->output) && (0 != strcmp(name, "-")))
					debug_printf(DP_WARN, "Overwriting file: %s\n", name);

			} else {

				if (open) {

					if (output && !iop->output)
						error("%s: Input opened for writing!\n", name);

					if (input &&  !iop->input)
						error("%s: Output opened for reading!\n", name);
				}

				iop->open = open;
				iop->output = output || iop->output;
				iop->input = input || iop->input;
			}
		}

		iop = iop->prev;
	}

	if (new) {

		if (open)
			debug_printf(DP_WARN, "%s: Opening file which was not previously reserved for in- or output!\n", name);

		PTR_ALLOC(struct iofile_s, ion);

		ion->name = strdup(name);
		ion->output = output;
		ion->input = input;
		ion->open = open;
		ion->prev = iofiles;

		iofiles = PTR_PASS(ion);
	}
}

void io_register_input(const char* name)
{
	io_register(name, false, true, true);
}

void io_register_output(const char* name)
{
	io_register(name, true, false, true);
}

void io_reserve_input(const char* name)
{
	io_register(name, false, true, false);
}

void io_reserve_output(const char* name)
{
	io_register(name, true, false, false);
}

void io_reserve_inout(const char* name)
{
	io_register(name, true, true, false);
}

void io_unregister(const char* name)
{
	struct iofile_s** iop = &iofiles;
	struct iofile_s* io;

	while (NULL != (io = *iop)) {

		if (0 == strcmp(name, io->name)) {

			*iop = io->prev;

			xfree(io->name);
			xfree(io);

			return;
		}

		iop = &io->prev;
	}
}

void io_close(const char* name)
{
	struct iofile_s* iop = iofiles;


	while (NULL != iop) {

		if (0 == strcmp(name, iop->name))
			iop->open = false;

		iop = iop->prev;
	}
}


void io_memory_cleanup(void)
{
	toolgraph_save_iofiles();
	while (NULL != iofiles)
		io_unregister(iofiles->name);
}



void io_unlink_if_opened(const char* name)
{
	const struct iofile_s* iop = iofiles;

	while (NULL != iop) {

		if ((0 == strcmp(name, iop->name)) && iop->open) {

			debug_printf(DP_DEBUG1, "Unlinking file %s\n", name);

			enum file_types_e type = file_type(name);

			switch (type) {

			case FILE_TYPE_RA:
			case FILE_TYPE_COO:

				if (0 != unlink(name))
					error("Failed to unlink file %s\n", name);

				break;

			case FILE_TYPE_CFL:

				;

				char name_bdy[1024];

				if (1024 <= snprintf(name_bdy, 1024, "%s.cfl", name))
					error("Failed to unlink cfl file %s\n", name);

				if (0 != unlink(name_bdy))
					error("Failed to unlink file %s\n", name);

				char name_hdr[1024];

				if (1024 <= snprintf(name_hdr, 1024, "%s.hdr", name))
					error("Failed to unlink cfl file %s\n", name);

				if (0 != unlink(name_hdr))
					error("Failed to unlink file %s\n", name);

				break;

			case FILE_TYPE_SHM:

				if (0 != shm_unlink(name))
					error("Failed to unlink shared memory segment %s\n", name);

				break;

			case FILE_TYPE_PIPE:
				break;

			case FILE_TYPE_MEM:
				break;
			}

			io_close(name);

			break;
		}

		iop = iop->prev;
	}
}


int write_cfl_header(int fd, const char* filename, int n, const long dimensions[n])
{
	int written = 0;

	written += xdprintf(fd, "# Dimensions\n");

	for (int i = 0; i < n; i++)
		written += xdprintf(fd, "%ld ", dimensions[i]);

	written += xdprintf(fd, "\n");

	if (NULL != filename) {

		written += xdprintf(fd, "# Data\n");
		written += xdprintf(fd, "%s\n", filename);
	}

	if (NULL != command_line) {

		written += xdprintf(fd, "# Command\n");
		if (NULL != stdin_command_line)
			written += xdprintf(fd, "%s| ", stdin_command_line);
		written += xdprintf(fd, "%s\n", command_line);
	}

	if (NULL != iofiles) {

		struct iofile_s* in = iofiles;

		written += xdprintf(fd, "# Files\n");

		while (in) {

			written += xdprintf(fd, " %s%s%s", in->input ? "<" : "", in->output ? ">" : "", in->name);
			in = in->prev;
		}

		written += xdprintf(fd, "\n");
	}

	if (node_id)
		written += xdprintf(fd, "# Node-ID\n%s\n", node_id);

	written += xdprintf(fd, "# Creator\nBART %s\n", bart_version);

	return written;
}


int write_stream_header(int fd, const char* dataname, int D, const long dims[D])
{
	// determine header length first by writing it to /dev/null
#ifdef _WIN32
	const char* null_file = "nul";
#else
	const char* null_file = "/dev/null";
#endif
	int null_fd = open(null_file, O_WRONLY);

	int written = write_cfl_header(null_fd, dataname, D, dims);

	close(null_fd);


	int MM = IO_MIN_HDR_SIZE;

	int padding = 0;

	if (written + 9 + 4 < MM)
		padding = MM - (written + 9 + 4);

	int l = xdprintf(fd, "# Header\n%d\n", written + 9 + 4 + padding);

	// This works for a header size of 100 to 999
	assert(l == 9 + 4);

	write_cfl_header(fd, dataname, D, dims);

	// avoid vla with 0 length
	if (0 == padding)
		return 0;

	char pad[padding];
	memset(pad, '.', (size_t)padding);
	pad[padding - 1] = '\n';
	xwrite(fd, padding, pad);

	return 0;
}


static int parse_cfl_header_len(long N, const char header[N + 1])
{
	int pos = 0;
	int delta = 0;
	int header_len = 0;

	char keyword[32] = { };

	// first line is not a keyword
	if (1 != sscanf(header + pos, "# %31s\n%n", keyword, &delta))
		return 0;

	pos += delta;

	if (0 == strcmp(keyword, "Header")) {

		if (1 != sscanf(header + pos, "%d\n%n", &header_len, &delta))
			error("Malformatted # Header\n");
	}

	return header_len;
}


int read_cfl_header2(int N, char header[N + 1], int fd, const char* hdrname, char** file, char** cmd, int n, long dimensions[n])
{
	*file = NULL;
	memset(header, 0, (size_t)(N + 1));

	int r = 0;
	int header_len = 0;
	int M = IO_MIN_HDR_SIZE;

	r = xread0(fd, M, header);

	header_len = parse_cfl_header_len(N, header);

	if (header_len > N)
		error("Header too large");

	if (header_len > 0)
		M = header_len;
	else
		M = N;

	if (r < M)
		r += xread0(fd, M - r, header + r);

	assert(r <= M);

	char* node = NULL;
	if (0 > parse_cfl_header(r, header, file, cmd, &node, n, dimensions))
		return -1;

	if (node) {

		toolgraph_add_input(node, hdrname);
		xfree(node);
	}

	return r;
}


int read_cfl_header(int fd, const char* hdrname, char** file, char** cmd, int n, long dimensions[n])
{
	char header[IO_MAX_HDR_SIZE + 1];

	return read_cfl_header2(IO_MAX_HDR_SIZE, header, fd, hdrname, file, cmd, n, dimensions);
}



int parse_cfl_header(long N, const char header[N + 1], char** file, char** cmd, char** node, int n, long dimensions[n])
{
	*file = NULL;

	int pos = 0;
	int delta = 0;
	bool ok = false;

	while (pos < N) {

		char keyword[32];

		if (1 != sscanf(header + pos, "# %31s\n%n", keyword, &delta))
			return -1;

		pos += delta;

		if (0 == strcmp(keyword, "Dimensions")) {

			if (ok)
				return -1;

			for (int i = 0; i < n; i++)
				dimensions[i] = 1;

			long val;
			int i = 0;

			while (1 == sscanf(header + pos, "%ld%n", &val, &delta)) {

				pos += delta;

				if (i < n)
					dimensions[i] = val;
				else if (1 != val)
					return -1;

				i++;
			}

			if (0 != sscanf(header + pos, "\n%n", &delta))
				return -1;

			pos += delta;

			ok = true;

		} else if (0 == strcmp(keyword, "Data")) {

			char filename[256];

			if (1 != sscanf(header + pos, "%255s\n%n", filename, &delta))
				return -1;

			pos += delta;

			*file = strdup(filename);

		} else if (NULL != cmd && 0 == strcmp(keyword, "Command")) {

			char* last_char = memchr(header + pos, '\n', (size_t)(N - pos));
			assert(NULL != last_char);

			delta = 1 + last_char - (header + pos);

			*cmd = xmalloc((size_t)delta);

			memcpy(*cmd, header + pos, (size_t)(delta - 1));

			(*cmd)[delta - 1] = '\0';

			pos += delta;

		} else if (node && 0 == strcmp(keyword, "Node-ID")) {

			char* last_char = memchr(header + pos, '\n', (size_t)(N - pos));
			assert(NULL != last_char);

			delta = 1 + last_char - (header + pos);

			*node = xmalloc((size_t)delta);

			memcpy(*node, header + pos, (size_t)(delta - 1));

			(*node)[delta - 1] = '\0';

			pos += delta;
		}



		// skip lines not starting with '#'

		while ('#' != header[pos]) {

			if ('\0' == header[pos])
				goto out;

			if (0 != sscanf(header + pos, "%*[^\n]\n%n", &delta))
				return -1;

			if (0 == delta)
				goto out;

			pos += delta;
		}
	}

out:
	return ok ? 0 : -1;
}


/**
 * Writes a header for multiple md_arrays in one cfl file
 *
 * @param fd file to write in
 * @param num_ele total number of elements in all arrays, used for backwards compatibility to load_cfl
 * @param D number of arrays written in file
 * @param n[D] number of dimensions per array
 * @param dimensions[D] pointer to dimensions of each array
 */
int write_multi_cfl_header(int fd, const char* filename, long num_ele, int D, int n[D], const long* dimensions[D])
{
	xdprintf(fd, "# Dimensions\n%ld \n", num_ele);

	xdprintf(fd, "# SizesDimensions\n");

	for (int i = 0; i < D; i++)
		xdprintf(fd, "%ld ", n[i]);

	xdprintf(fd, "\n");

	xdprintf(fd, "# MultiDimensions\n");

	for (int i = 0; i < D; i++) {

		for (int j = 0; j < n[i]; j++)
			xdprintf(fd, "%ld ", dimensions[i][j]);

		xdprintf(fd, "\n");
	}

	if (NULL != filename) {

		xdprintf(fd, "# Data\n");
		xdprintf(fd, "%s\n", filename);
	}

	if (NULL != command_line) {

		xdprintf(fd, "# Command\n");
		xdprintf(fd, "%s\n", command_line);
	}

	if (NULL != iofiles) {

		struct iofile_s* in = iofiles;

		xdprintf(fd, "# Files\n");

		while (in) {

			xdprintf(fd, " %s%s%s", in->input ? "<" : "", in->output ? ">" : "", in->name);
			in = in->prev;
		}

		xdprintf(fd, "\n");
	}

	xdprintf(fd, "# Creator\nBART %s\n", bart_version);

	return 0;
}




/**
 * Reads a header for multiple md_arrays in one cfl file
 *
 * @param fd file to read from
 * @param D_max maximal number of arrays expected in file
 * @param n_max maximal number of dimension expected per array
 * @param D number of arrays written in file
 * @param n[D_max] number of dimensions per array read from header
 * @param dimensions[D_max][n_max] dimensions read from header
 *
 * @return number of arrays in file
 */
int read_multi_cfl_header(int fd, char** file, int D_max, int n_max, int n[D_max], long dimensions[D_max][n_max])
{
	*file = NULL;
	char header[IO_MAX_HDR_SIZE + 1] = { };

	long dims[1];
	int max = read_cfl_header2(IO_MAX_HDR_SIZE, header, fd, NULL, file, NULL, 1, dims);

	if (-1 == max)
		return -1;

	int pos = 0;
	int delta = 0;
	bool ok = false;
	bool multi_cfl = false;

	int D = 0;
	long num_ele = dims[0];
	long num_ele_dims = 0;

	while (true) {

		char keyword[32];

		if (1 != sscanf(header + pos, "# %31s\n%n", keyword, &delta))
			return -1;

		pos += delta;

		if (0 == strcmp(keyword, "SizesDimensions")) {

			for (int i = 0; i < D_max; i++)
				n[i] = 0;

			long val;

			while (1 == sscanf(header + pos, "%ld%n", &val, &delta)) {

				pos += delta;

				if (D < D_max)
					n[D] = val;
				else
					return -1;

				D++;
			}

			if (0 != sscanf(header + pos, "\n%n", &delta))
				return -1;

			pos += delta;

			if (multi_cfl)
				return -1;

			multi_cfl = true;

		} else if (0 == strcmp(keyword, "MultiDimensions")) {

			if (ok)
				return -1;

			for (int i = 0; i < D; i++)
				for (int j = 0; j < n_max; j++)
					dimensions[i][j] = 1;

			long val;
			int i = 0;
			int j = 0;
			long size_tensor = 1;

			while (1 == sscanf(header + pos, "%ld%n", &val, &delta)) {

				pos += delta;

				if (j == n[i]) {

					num_ele_dims += size_tensor;

					size_tensor = 1;
					j = 0;
					i++;
				}

				dimensions[i][j] = val;
				j++;
				size_tensor *= val;
			}

			if (0 != sscanf(header + pos, "\n%n", &delta))
				return -1;

			pos += delta;

			ok = true;
		}

		// skip lines not starting with '#'

		while ('#' != header[pos]) {

			if ('\0' == header[pos])
				goto out;

			if (0 != sscanf(header + pos, "%*[^\n]\n%n", &delta))
				return -1;

			if (0 == delta)
				goto out;

			pos += delta;
		}
	}

	ok &= (num_ele == num_ele_dims);

out:
	return ok ? 0 : -1;
}




int write_coo(int fd, int n, const long dimensions[n])
{
	char header[4096];
	long len = (long)ARRAY_SIZE(header);
	memset(header, 0, 4096);

	int pos = 0;
	int ret;

	ret = snprintf(header + pos, (size_t)len, "Type: float\nDimensions: %d\n", n);

	if ((ret < 0) || (ret >= (int)len))
		return -1;

	pos += ret;
	len -= ret;

	long start = 0;
	long stride = 1;

	for (int i = 0; i < n; i++) {

		long size = dimensions[i];

		ret = snprintf(header + pos, (size_t)len, "[%ld\t%ld\t%ld\t%ld]\n", start, stride * size, size, stride);

		if ((ret < 0) || (ret >= len))
			return -1;

		pos += ret;
		len -= ret;

		stride *= size;
	}

	if (4096 != write(fd, header, 4096))
		return -1;

	return 0;
}


int read_coo(int fd, int n, long dimensions[n])
{
	char header[4096];

	if (4096 != xread(fd, 4096, header))
		return -1;

	int pos = 0;
	int delta = 0;

	if (0 != sscanf(header + pos, "Type: float\n%n", &delta))
		return -1;

	if (0 == delta)
		return -1;

	pos += delta;

	int dim;

	if (1 != sscanf(header + pos, "Dimensions: %d\n%n", &dim, &delta))
		return -1;

	pos += delta;

//	if (n != dim)
//		return -1;

	for (int i = 0; i < n; i++)
		dimensions[i] = 1;

	for (int i = 0; i < dim; i++) {

		long val;

		if (1 != sscanf(header + pos, "[%*d %*d %ld %*d]\n%n", &val, &delta))
			return -1;

		pos += delta;

		if (i < n)
			dimensions[i] = val;
		else
		if (1 != val)	// fail if we have to many dimensions not equal 1
			return -1;
	}

	return 0;
}





struct ra_hdr_s {

	uint64_t magic;
	uint64_t flags;
	uint64_t eltype;
	uint64_t elbyte;
	uint64_t size;
	uint64_t ndims;
};

#define RA_MAGIC_NUMBER		0x7961727261776172ULL
#define RA_FLAG_BIG_ENDIAN	(1ULL << 0)

enum ra_types {

	RA_TYPE_USER = 0,
	RA_TYPE_INT,
	RA_TYPE_UINT,
	RA_TYPE_FLOAT,
	RA_TYPE_COMPLEX,
};


#define err_assert(x)	({ if (!(x)) { debug_printf(DP_ERROR, "%s", #x); return -1; } })


int read_ra(int fd, int n, long dimensions[n])
{
	struct ra_hdr_s header;

	if (sizeof(header) != xread(fd, sizeof(header), (void*)&header))
		return -1;

	err_assert(RA_MAGIC_NUMBER == header.magic);
	err_assert(!(header.flags & RA_FLAG_BIG_ENDIAN));
	err_assert(RA_TYPE_COMPLEX == header.eltype);
	err_assert(sizeof(complex float) == header.elbyte);
	err_assert(header.ndims <= 100);

	uint64_t dims[header.ndims];

	if ((int)sizeof(dims) != xread(fd, sizeof(dims), (void*)&dims))
		return -1;

	md_singleton_dims(n, dimensions);

	for (int i = 0; i < (int)header.ndims; i++) {

		if (i < n)
			dimensions[i] = (long)dims[i];
		else
			err_assert(1 == dims[i]);
	}

	// this can overflow, but we check in mmio
	err_assert((long)header.size == md_calc_size(n, dimensions) * (long)sizeof(complex float));

	return 0;
}



int write_ra(int fd, int n, const long dimensions[n])
{
	struct ra_hdr_s header = {

		.magic = RA_MAGIC_NUMBER,
		.flags = 0ULL,
		.eltype = RA_TYPE_COMPLEX,
		.elbyte = sizeof(complex float),
		.size = (size_t)(md_calc_size(n, dimensions) * (long)sizeof(complex float)),
		.ndims = (size_t)n,
	};

	if (sizeof(header) != write(fd, &header, sizeof(header)))
		return -1;

	uint64_t dims[n];

	for (int i = 0; i < n; i++)
		dims[i] = (uint64_t)dimensions[i];

	if ((int)sizeof(dims) != write(fd, &dims, sizeof(dims)))
		return -1;

	return 0;
}



#define MAX_INPUT_NODES 100
char* input_nodes [MAX_INPUT_NODES] = { NULL };

static int toolgraph_fd = -1;

static const char* toolgraph_iofiles = NULL;

static void toolgraph_save_iofiles(void)
{
	if (!iofiles)
		return;

	bool set = false;
#pragma omp critical(toolgraph_iofiles)
	if (toolgraph_iofiles)
		set = true;
	else
		toolgraph_iofiles = "X";

	if (set)
		return;

	char* tmp1 = NULL;

	struct iofile_s* in = iofiles;

	tmp1 = ptr_printf("# Files\n");

	while (in) {

		char*  tmp2 = ptr_printf("%s %s%s%s", tmp1, in->input ? "<" : "", in->output ? ">" : "", in->name);
		xfree(tmp1);
		tmp1 = tmp2;
		in = in->prev;
	}

	toolgraph_iofiles = tmp1;
}

static void toolgraph_add_input(const char* node, const char* file)
{
	if (-1 == toolgraph_fd)
		return;

	int i = 0;
	bool found = false;
#pragma omp critical(toolgraph_inputs)
	for (; i < MAX_INPUT_NODES && input_nodes[i] && !found; i++)
		if (0 == strcmp(node, input_nodes[i]))
			found = true;
	if (found)
		return;

	if (MAX_INPUT_NODES == i)
		error("BART tool graph: Too many input files\n");

	input_nodes[i] = strdup(node);

#pragma omp critical(toolgraph_fd)
	xdprintf(toolgraph_fd, "%s:%s ", node, file);
}

static int toolgraph_create_node_fd(const char* tool_name, int dirfd, char** node)
{
	int l = tool_name ? strlen(tool_name) : 0;
	l += 1 + 20 + 1 + 8 + 1; // delimiter + timestamp + pid (up to 8 digits ) + zero-byte
	char name[l];

	if (snprintf(name, (unsigned)l, "%s_%d_%d", tool_name ? : "", (int)timestamp(), getpid()) >= l)
		error("graph_generate_name.\n");

	int fd = openat(dirfd, name, O_RDWR|O_CREAT|O_EXCL, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);

	if (-1 == fd)
		error("graph_generate_name.\n");

	*node = strdup(name);

	return fd;
}

void toolgraph_create(const char* tool_name, int argc, char* argv[static argc])
{
	assert(-1 == toolgraph_fd);

	char* str = getenv("BART_TOOL_GRAPH");

	if (!str || 0 == strlen(str))
		return;

	struct stat sb = {};

	if (0 != stat(str, &sb) || (S_IFDIR != (sb.st_mode & S_IFMT)))
		error("Environment variable BART_TOOL_GRAPH is not a directory.\n");

	int dirfd = open(str, O_RDONLY);

	toolgraph_fd = toolgraph_create_node_fd(tool_name, dirfd, &node_id);

	close(dirfd);

	char dir[BART_MAX_DIR_PATH_SIZE];

	if (NULL == getcwd(dir, BART_MAX_DIR_PATH_SIZE))
		error("Current working directory path too long.\n");

	xdprintf(toolgraph_fd, "# Directory\n%s\n", dir);

	char* cmd = serialize_command_line(argc, argv);

	if (cmd) {

		xdprintf(toolgraph_fd, "# Command\n%s\n", cmd);
		xfree(cmd);
	}

	xdprintf(toolgraph_fd, "# Input Nodes\n");
}

void toolgraph_close(void)
{
	if (-1 == toolgraph_fd)
		return;

	for (int i = 0; i < MAX_INPUT_NODES && input_nodes[i]; i++)
		xfree(input_nodes[i]);

	xdprintf(toolgraph_fd, "\n");

	if (toolgraph_iofiles) {

		xdprintf(toolgraph_fd, "%s\n", toolgraph_iofiles);
		xfree(toolgraph_iofiles);
	}

	if (node_id)
		XFREE(node_id);

	close(toolgraph_fd);
	toolgraph_fd = -1;
}
