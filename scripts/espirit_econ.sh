#!/bin/bash
# Copyright 2018. Martin Uecker.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Authors:
# 2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
#
# Memory-saving ESPIRiT
#
set -e

LOGFILE=/dev/stdout

title=$(cat <<- EOF
	ESPIRiT-ECON
EOF
)

helpstr=$(cat <<- EOF
-l logfile
-h help
EOF
)

usage="Usage: $0 [-h] <kspace> <output>"

echo "$title"
echo

while getopts "hl:" opt; do
        case $opt in
	h)
		echo "$usage"
		echo
		echo "$helpstr"
		exit 0
	;;
	l)
		LOGFILE=$(readlink -f "$OPTARG")
	;;
        \?)
        	echo "$usage" >&2
		exit 1
        ;;
        esac
done

shift $((OPTIND - 1))


if [ $# -lt 2 ] ; then

        echo "$usage" >&2
        exit 1
fi


export PATH=$TOOLBOX_PATH:$PATH

input=$(readlink -f "$1")
output=$(readlink -f "$2")

if [ ! -e $input.cfl ] ; then
	echo "Input file does not exist." >&2
	echo "$usage" >&2
	exit 1
fi

if [ ! -e $TOOLBOX_PATH/bart ] ; then
        echo "\$TOOLBOX_PATH is not set correctly!" >&2
	exit 1
fi


#WORKDIR=$(mktemp -d)
# Mac: http://unix.stackexchange.com/questions/30091/fix-or-alternative-for-mktemp-in-os-x
WORKDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`
trap 'rm -rf "$WORKDIR"' EXIT
cd $WORKDIR

# start group for redirection of output to the logfile
{

XX=$(bart show -d0 $input)
YY=$(bart show -d1 $input)
ZZ=$(bart show -d2 $input)

DIM=2
# To decouple along another dimension:
# 1. change DIM
# 2. replace ZZ below
# 3. change the ecaltwo command

bart ecalib -1 $input eon

# zero-pad
bart fft $(bart bitmask ${DIM}) eon eon_fft
bart resize -c ${DIM} ${ZZ} eon_fft eon_fft2
bart fft -i $(bart bitmask ${DIM}) eon_fft2 eon

for i in `seq -w 0 $(($ZZ - 1))` ; do
	
	bart slice ${DIM} $i eon sl
	bart ecaltwo ${XX} ${YY} 1 sl sens-$i.coo
done

# # join slices back together
bart join ${DIM} sens-*.coo $output

} > $LOGFILE

exit 0
