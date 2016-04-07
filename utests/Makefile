
OMP?=1
CC = gcc

here  = $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
utestdir = $(here)
root := $(utestdir)/..

srcdir = $(root)/src
libdir = $(root)/lib
bindir = $(root)/bin

OPT = -O3 -ffast-math -g
CPPFLAGS ?= -Wall -Wextra
CFLAGS ?= $(OPT) -Wmissing-prototypes

BART_H := -I$(srcdir)
BART_L := -L$(libdir)

ifeq ($(OMP),1)
CFLAGS += -fopenmp
else
CFLAGS += -Wno-unknown-pragmas
endif

CFLAGS += -std=c99 $(BART_H) $(BART_L)

BART_MODULES = num misc num misc

UTARGETS = test_md_zwavg


# define space to faciliate running executables
define \n


endef



BART_LIBS = $(patsubst %, $(libdir)/lib%.a, $(BART_MODULES))
BART_LLIBS = $(patsubst %, -l%, $(BART_MODULES))


$(UTARGETS): % : %.o $(BART_LIBS)
	$(CC) $(CFLAGS) -o $@ $< $(BART_LLIBS) -lm 

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<


utests: $(UTARGETS)

run: utests
	$(patsubst %,$(\n)./%,$(UTARGETS))


all: run utests

default: all

clean:
	rm -f `find $(utestdir) -name "*.o"`

allclean: clean
	rm -f $(patsubst %, %, $(UTARGETS))


