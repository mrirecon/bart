
#include <stdbool.h>
#include <stdlib.h>

#include "misc/misc.h"
#include "misc/shrdptr.h"

#include "num/multind.h"
#include "num/iovec.h"

#include "na.h"


struct buf_s {

	void* data;
	bool owner;

	struct shared_obj_s sptr;
};


struct na_s {

	struct iovec_s iov;
	struct buf_s* buf;
	long offset;
};

static void buf_del(const struct shared_obj_s* sptr)
{
	struct buf_s* p = CONTAINER_OF(sptr, struct buf_s, sptr);

	if (p->owner)
		md_free(p->data);

	xfree(p);
}


na na_wrap(unsigned int N, const long (*dims)[N], const long (*strs)[N], void* data, size_t size)
{
	PTR_ALLOC(struct na_s, n);

	iovec_init2(&n->iov, N, *dims, *strs, size);

	n->buf = TYPE_ALLOC(struct buf_s);
	n->buf->data = data;
	n->buf->owner = false;

	n->offset = 0;

	shared_obj_init(&n->buf->sptr, buf_del);

	return PTR_PASS(n);
}
	
na na_new(unsigned int N, const long (*dims)[N], size_t size)
{
	void* data = md_alloc(N, *dims, size);
	
	long strs[N];
	md_calc_strides(N, strs, *dims, size);
	
	na n = na_wrap(N, dims, &strs, data, size);
	n->buf->owner = true;
	return n;
}

void na_free(na x)
{
	iovec_destroy(&x->iov);
	shared_obj_destroy(&x->buf->sptr);
	xfree(x);
}

na na_slice(na x, unsigned int flags, unsigned int N, const long (*pos)[N])
{
	PTR_ALLOC(struct na_s, n);

	assert(na_rank(x) == N);
	long dims[N];
	md_select_dims(N, flags, dims, x->iov.dims);

	long strs[N];
	md_copy_strides(N, strs, x->iov.strs);
	
	for (unsigned int i = 0; i < N; i++)
		if (!MD_IS_SET(flags, i))
			strs[i] = 0;

	iovec_init2(&n->iov, N, dims, strs, x->iov.size); 
	
	shared_obj_ref(&x->buf->sptr);
	n->buf = x->buf;

	n->offset = x->offset + md_calc_offset(N, strs, *pos); 

	return PTR_PASS(n);
}

unsigned int na_rank(na x)
{
	return x->iov.N;
}

size_t na_element_size(na x)
{
	return x->iov.size;
}

struct long_array_s na_get_dimensions(na x)
{
	return (struct long_array_s){ na_rank(x), x->iov.dims };
}

struct long_array_s na_get_strides(na x)
{
	return (struct long_array_s){ na_rank(x), x->iov.strs };
}

na na_view(na x)
{
	unsigned int N = na_rank(x);

	long pos[N];
	return na_slice(x, ~0, N, &pos);
}

na na_clone(na x)
{
	return na_new(na_rank(x), NA_DIMS(x), na_element_size(x));
}


void* na_ptr(na x)
{
	return x->buf->data + x->offset;
}

void na_copy(na dst, na src)
{
	assert(na_rank(dst) == na_rank(src));
	assert(na_element_size(dst) == na_element_size(src));
	// check compat

	md_copy2(na_rank(dst), *NA_DIMS(dst), *NA_STRS(dst), na_ptr(dst),
			*NA_STRS(src), na_ptr(src), na_element_size(dst));
}

void na_clear(na dst)
{
	md_clear2(na_rank(dst), *NA_DIMS(dst), *NA_STRS(dst), na_ptr(dst), na_element_size(dst));
}




