
#ifndef _VERSION_H
#define _VERSION_H


extern const char* bart_version;

extern _Bool version_parse(unsigned int v[5], const char* version);
extern int version_compare(const unsigned int va[5], const unsigned int vb[5]);

extern _Bool use_compat_to_version(const char* compat_version);

#endif // __VERSION_H
