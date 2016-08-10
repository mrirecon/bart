/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


// some classic C preprocessor hackery

/* Ideas from:
 *
 * https://github.com/swansontec/map-macro
 * https://github.com/pfultz2/Cloak/wiki/Is-the-C-preprocessor-Turing-complete%3F
 */

#define EMPTY()
#define DEFER1(...) __VA_ARGS__ EMPTY()
#define DEFER2(...) __VA_ARGS__ DEFER1(EMPTY)()
#define DEFER3(...) __VA_ARGS__ DEFER2(EMPTY)()

#define EXPAND6(...) __VA_ARGS__
#define EXPAND5(...) EXPAND6(EXPAND6(__VA_ARGS__))
#define EXPAND4(...) EXPAND5(EXPAND5(__VA_ARGS__))
#define EXPAND3(...) EXPAND4(EXPAND4(__VA_ARGS__))
#define EXPAND2(...) EXPAND3(EXPAND3(__VA_ARGS__))
#define EXPAND1(...) EXPAND2(EXPAND2(__VA_ARGS__))
#define EXPAND0(...) EXPAND1(EXPAND1(__VA_ARGS__))
#define EXPAND(...) EXPAND0(EXPAND0(__VA_ARGS__))

#define CAT0(x, y) x ## y
#define CAT(x, y) CAT0(x, y)
#define NIL_TEST() DUMMY, TRUE,
#define RET2ND0(a, b, ...) b
#define RET2ND(...) RET2ND0(__VA_ARGS__)
#define NIL_P(x) RET2ND(NIL_TEST x, FALSE)
#define IF_TRUE(a, b) a
#define IF_FALSE(a, b) b
#define IF(x, a, b) CAT(IF_, x)(a, b)
#define MAP1() MAP0
#define MAP0(f, a, b, ...) f(a) IF(NIL_P(b), , DEFER3(MAP1)()(f, b, __VA_ARGS__))
#define MAP(f, ...) EXPAND(MAP0(f, __VA_ARGS__, ()))

