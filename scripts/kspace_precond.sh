#!/bin/bash
# Copyright 2022. TU Graz. Institute of Biomedical Imaging.
# Author: Moritz Blumenthal
#
# F. Ong, M. Uecker and M. Lustig, Accelerating Non-Cartesian 
# MRI Reconstruction Convergence Using k-Space Preconditioning
# IEEE TMI, 2020 39:1646-1654
#


helpstr=$(cat <<- EOF
Compute k-space preconditioner P such that ||P^2 AA^H - 1|| is minimal
Note the square in the definition. The preconditioner can be used directly as wights in PICS.

<ones>	contains ones with image dimensions

-B file subspace basis
-g 	use GPU
-h	help
EOF
)


usage="Usage: $0 [-h] [-g] [-B <basis>] <ones> <trajectory> <output>"

GPU=""
BASIS=""

while getopts "hgB:" opt; do
        case $opt in
        g)
		GPU="-g"
        ;;
        B)
		BASIS=$(readlink -f "$OPTARG")
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


if [ $# -lt 3 ] ; then

        echo "$usage" >&2
        exit 1
fi

ones=$(readlink -f "$1")
traj=$(readlink -f "$2")
prec=$(readlink -f "$3")


WORKDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`
trap 'rm -rf "$WORKDIR"' EXIT
cd $WORKDIR

X=$(bart show -d0 $ones)
Y=$(bart show -d1 $ones)
Z=$(bart show -d2 $ones)

s1=$((X*Y*Z))

if [[ 1 != $X ]] ; then X=$((2*X)); fi
if [[ 1 != $Y ]] ; then Y=$((2*Y)); fi
if [[ 1 != $Z ]] ; then Z=$((2*Z)); fi

s2=$((X*Y*Z))
s3=$(echo "$s1*e(-1.5*l($s2))"|bc -l)

bart fmac -C -s7 $ones $ones mps_norm2
bart scale $s3 mps_norm2 scale


ksp_dims="1"
for i in $(seq 15); do
	ksp_dims+=" $(bart show -d$i $traj)"
done

if [[ "$BASIS" != "" ]] ; then

	bart fmac -C -s$(bart bitmask 6) $BASIS $BASIS bas_scale
	bart fmac scale bas_scale scale2
	bart copy scale2 scale

	COE=$(bart show -d6 $BASIS)

	bart transpose 6 7 $BASIS basis_r 
	bart fmac -C $BASIS basis_r basis_2
	bart reshape $(bart bitmask 6 7) $((COE*COE)) 1 basis_2 basis
	BASIS="-B basis"
fi

bart ones 16 $ksp_dims ksp
bart scale 2 $traj traj2

bart nufft $BASIS -P --lowmem --no-precomp -a $GPU -x$X:$Y:$Z traj2 ksp psf

bart resize -c 0 $X 1 $Y 2 $Z $ones ones_os
bart fft -u 7 ones_os ones_ksp1
bart fmac -C ones_ksp1 ones_ksp1 ones_ksp
bart fft -u -i 7 ones_ksp ones_img

bart fmac psf ones_img psf_mul

bart nufft $BASIS -P --lowmem --no-precomp $GPU traj2 psf_mul pre_inv

bart creal pre_inv pre_inv_real
bart invert pre_inv_real pre_real

bart fmac pre_real scale pre_sqr

bart spow -- 0.5 pre_sqr $prec

