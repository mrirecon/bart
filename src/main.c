#include "misc/cppwrap.h"

extern int main_real(int argc, char** argv);

int main(int argc, char** argv)
{
	return main_real(argc, argv);
}

#include "misc/cppwrap.h"
