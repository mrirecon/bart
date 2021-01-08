/* Copyright 2021. Uecker Lab, University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __VERSION_H
#define __VERSION_H



extern const char* bart_version;

extern _Bool version_parse(unsigned int v[5], const char* version);

extern int version_compare(const unsigned int va[5], const unsigned int vb[5]);

extern _Bool use_compat_to_version(const char* compat_version);

#endif // __VERSION_H
