
#include "version.h"

#define STRINGIFY(x) # x 
#define VERSION(x) STRINGIFY(x)
const char* bart_version = 
#include "version.inc"
;

