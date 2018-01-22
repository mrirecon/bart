/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include "misc/types.h"

#include "utest.h"


typedef struct superclass_s { TYPEID* TYPEID; } superclass;

struct class1 {

	INTERFACE(superclass);
};

DEF_TYPEID(class1);


static bool test_cast_down_pos(void)
{
	bool ok = true;
	PTR_ALLOC(struct class1, cp);
	SET_TYPEID(class1, cp);

	struct superclass_s* sp = CAST_UP(cp);

	if (cp != CAST_MAYBE(class1, sp))
		ok = false;

	PTR_FREE(cp);

	return ok;
}


UT_REGISTER_TEST(test_cast_down_pos);


struct class2 {

	INTERFACE(superclass);
};

DEF_TYPEID(class2);


static bool test_cast_down_neg(void)
{
	bool ok = true;
	PTR_ALLOC(struct class2, cp);
	SET_TYPEID(class2, cp);

	struct superclass_s* sp = CAST_UP(cp);

	if (NULL != CAST_MAYBE(class1, sp))
		ok = false;

	PTR_FREE(cp);

	return ok;
}


UT_REGISTER_TEST(test_cast_down_neg);


