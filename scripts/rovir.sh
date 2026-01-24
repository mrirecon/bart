#!/bin/bash
# Copyright 2023-2026. TU Graz. Institute of Biomedical Imaging.
# Author: Moritz Blumenthal
#
# Kim, D, Cauley, SF, Nayak, KS, Leahy, RM, Haldar, JP.
# Region-optimized virtual (ROVir) coils: Localization and/or 
# suppression of spatial regions using sensor-domain beamforming. 
# Magn Reson Med. 2021; 86: 197–212.
#

set -eu

helpstr=$(cat <<- EOF
Compute coil compression following the ROVir method.
<kspace>	Signal to be compressed
<pos/neg roi>	Mask (1/0) for region of interest to be optimized for.
		Defines also low resolution image.
<cc/coeffs>	Compressed signal or coefficient matrix

-p N	compress to N virtual channels 
-t file trajectory
-B file subspace basis
-M	output coefficients
-g 	use GPU
-h	help
EOF
)


usage="Usage: $0 [-h] [-g] [-t <trajectory>][-B <basis>] <kspace> <pos roi> <neg roi> <cc/coeffs>"

CC=""
GPU=""
TRAJ=""
BASIS=""
COEFFS=0

while getopts "hgB:t:Mp:" opt; do
        case $opt in
        g)
		GPU=" -g"
        ;;
        t)
		TRAJ=$(readlink -f "$OPTARG")
        ;;
        p)
		CC=" -p $OPTARG"
        ;;
        M)
		COEFFS=1
        ;;
        B)
		BASIS="-B $(readlink -f "$OPTARG")"
        ;;
	h)
		echo "$usage"
		echo
		echo "$helpstr"
		exit 0
	;;
	\?)
		echo "$usage" >&2
		exit 1
	;;
        esac
done

shift $((OPTIND - 1))


if [ $# -ne 4 ] ; then

        echo "$usage" >&2
        exit 1
fi

if [ ! -e "$BART_TOOLBOX_PATH"/bart ] ; then
	echo "\$BART_TOOLBOX_PATH is not set correctly!" >&2
	exit 1
fi
export PATH="$BART_TOOLBOX_PATH:$PATH"

sig=$(readlink -f "$1")
pos=$(readlink -f "$2")
neg=$(readlink -f "$3")
out=$(readlink -f "$4")

WORKDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`
trap 'rm -rf "$WORKDIR"' EXIT
cd $WORKDIR

if [ -z "$TRAJ" ] ; then

	DIMS="0 $(bart show -d 0 $pos) 1 $(bart show -d 1 $pos) 2 $(bart show -d 2 $pos)"
	bart resize -c $DIMS $sig res
	bart fft -i 7 res img
	bart fmac img $pos pos
	bart fmac img $neg neg
else

	DIMS="$(bart show -d 0 $pos):$(bart show -d 1 $pos):$(bart show -d 2 $pos)"
	bart nufftbase $DIMS $TRAJ pat
	bart nufft $GPU $BASIS -p pat -i -x$DIMS $TRAJ $sig cim

	bart fmac cim $pos ipos
	bart fmac cim $neg ineg

	bart nufft $GPU $BASIS -ppat $TRAJ ipos pos
	bart nufft $GPU $BASIS -ppat $TRAJ ineg neg
fi

bart rovir pos neg compress

if [[ "$COEFFS" -eq 1 ]]; then
	bart copy compress $out
else
	bart ccapply $CC $sig compress $out
fi

