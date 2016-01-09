#!/bin/bash
# Copyright 2015. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Authors:
# 2015 Martin Uecker <uecker@eecs.berkeley.edu>
#
# Compressed sensing parallel imaging reconstruction with temporal
# total-variation regularization for Siemens radial VIBE sequence
# with golden-angle sampling (GRASP).
#
set -e

# default settings
export SPOKES=21
export SKIP=0
export CALIB=400
export ITER=30
export REG=0.05
SCALE=0.6
LOGFILE=/dev/stdout
MAXPROC=4
MAXTHREADS=4

title=$(cat <<- EOF
	(BART-)GRASP v0.3 (Berkeley Advanced Reconstruction Toolbox)
	      --- EXPERIMENTAL --- FOR RESEARCH USE ONLY ---
EOF
)

helpstr=$(cat <<- EOF
Compressed sensing parallel imaging reconstruction with temporal
total-variation regularization for Siemens radial VIBE sequence
with golden-angle sampling (GRASP).
This script requires the Berkeley Advanced Reconstruction Toolbox
version 0.2.09. (later versions may also work).

-s spokes	number of spokes per frame
-r lambda	regularization parameter
-p maxproc	max. number of slices processed in parallel
-t maxthreads	max. number of threads per slice
-l logfile
-h help
EOF
)

usage="Usage: $0 [-h] [-s spokes] [-r lambda] <meas.dat> <output>"

echo "$title"
echo

while getopts "hl:s:p:t:r:" opt; do
        case $opt in
        s)
                SPOKES=$OPTARG
        ;;
	r)
		REG=$OPTARG
	;;
	h)
		echo "$usage"
		echo
		echo "$helpstr"
		exit 0
	;;
	l)
		LOGFILE=$(readlink -f "$OPTARG")
	;;
	p)
		MAXPROC=$OPTARG
	;;
	t)
		MAXTHREADS=$OPTARG
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

if [ ! -e $input ] ; then
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

# read TWIX file
bart twixread -A $input grasp

export READ=$(bart show -d0 grasp)
export COILS=$(bart show -d3 grasp)
export PHASES=$(($(bart show -d1 grasp) / $SPOKES))

export OMP_NUM_THREADS=$((MAXPROC * $MAXTHREADS))

# zero-pad
#flip $(bitmask 2) grasp grasp2
#resize 2 64 grasp2 grasp
#circshift 2 10 grasp grasp2
#fft -u $(bitmask 2) grasp2 grasp_hybrid
#rm grasp.* grasp2.*

# inverse FFT along 3rd dimension
bart fft -i -u $(bart bitmask 2) grasp grasp_hybrid
rm grasp.cfl grasp.hdr

SLICES=$(bart show -d2 grasp_hybrid)


# create trajectory with 400 spokes and 2x oversampling
bart traj -G -x$READ -y$CALIB r
bart scale $SCALE r rcalib

# create trajectory with 2064 spokes and 2x oversampling
bart traj -G -x$READ -y$(($SPOKES * $PHASES)) r
bart scale $SCALE r r2

# split off time dimension into index 10
bart reshape $(bart bitmask 2 10) $SPOKES $PHASES r2 rfull

# number of threads per slice
export OMP_NUM_THREADS=$MAXTHREADS

calib_slice()
{
	# extract slice
	bart slice 2 $1 grasp_hybrid grasp1-$1

	# extract first $CALIB spokes
	bart extract 1 $(($SKIP + 0)) $(($SKIP + $CALIB - 1)) grasp1-$1 grasp2-$1

	# reshape dimensions
	bart reshape $(bart bitmask 0 1 2 3) 1 $READ $CALIB $COILS grasp2-$1 grasp3-$1

	# apply inverse nufft to first $CALIB spokes
	bart nufft -i -t rcalib grasp3-$1 img-$1.coo
}

recon_slice()
{
	# extract sensitivities for slice
	bart slice 2 $1 sens sens-$1

	# extract spokes and split-off time dim
	bart extract 1 $(($SKIP + 0)) $(($SKIP + $SPOKES * $PHASES - 1)) grasp1-$1 grasp2-$1
	bart reshape $(bart bitmask 1 2) $SPOKES $PHASES grasp2-$1 grasp1-$1

	# move time dimensions to dim 10 and reshape
	bart transpose 2 10 grasp1-$1 grasp2-$1
	bart reshape $(bart bitmask 0 1 2) 1 $READ $SPOKES grasp2-$1 grasp1-$1
	rm grasp2-$1.cfl grasp2-$1.hdr

	# reconstruction with tv penality along dimension 10
	# old (v0.2.08):
	# pics -S -d5 -lv -u10. -r$REG -R$(bitmask 10) -i$ITER -t rfull grasp1-$1 sens-$1 i-$1.coo
	# new (v0.2.09):
	bart pics -S -d5 -u10. -RT:$(bart bitmask 10):0:$REG -i$ITER -t rfull grasp1-$1 sens-$1 i-$1.coo

	# clean up temp files
	rm *-$1.cfl *-$1.hdr
}

export -f calib_slice
export -f recon_slice

# loop over slices
seq -w 0 $(($SLICES - 1)) | xargs -I {} -P $MAXPROC bash -c "calib_slice {}"

# transform back to k-space and compute sensitivities
bart join 2 img-*.coo img
bart fft -u $(bart bitmask 0 1 2) img ksp

#ecalib -S -c0.8 -m1 -r20 ksp sens

# transpose because we already support off-center calibration region
# in dim 0 but here we might have it in 2
bart transpose 0 2 ksp ksp2
bart ecalib -S -c0.8 -m1 -r20 ksp2 sens2
bart transpose 0 2 sens2 sens

# loop over slices
seq -w 0 $(($SLICES - 1)) | xargs -I {} -P $MAXPROC bash -c "recon_slice {}"
#echo 20 | xargs -i --max-procs=$MAXPROC bash -c "recon_slice {}"

# join slices back together
bart join 2 i-*.coo $output

# generate dicoms
#for s in $(seq -w 0 $(($SLICES - 1))) ; do
#	for p in $(seq -w 0 $(($PHASES - 1))) ; do
#		bart slice 10 $p i-$s.coo i-$p-$s.coo
#		bart toimg i-$p-$s.coo $output.series$p.slice$s.dcm
#	done
#done

} > $LOGFILE

exit 0
