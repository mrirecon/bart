#!/bin/bash
# Copyright 2024. TU Graz. Institute of Biomedical Imaging.
# Author: Moritz Blumenthal
#

IFS='' read -r -d '' helpstr << 'EOF'
Transform k-space from a moved image (IM) to a reference image (IR) by an affine transform.
The affine coordinate transform is defined such that:
IR(xr) = IM(xm) = IM(Axr) = IM(Rxr + a) where A is a 4x4 matrix with

    (\ | / | | )
A = (- R - | a )
    (/ | \ | | )
    (0 0 0 | 1 )

The transform needs to be defined following the usual convention:
1.) Shifts are measured in units of FOV as the trajectory measures k-space coordinates in units 1/FOV
2.) The image origin (x=0) is at grid position N//2 (integer division for odd numbers)

The transformed k-space is given by
F[IR] (R^T k) = exp(i2pi ak) F[IM](k)

Warning: A factor 1/|det(R)| is missing for non-rigid transforms!

<ksp_raw>	moved k-space:			F[IM]
<trj_raw>	moved trajectory:		k
<affine>	affine transformation matrix:	A
<ksp_ref>	reference k-space:		F[IR]
<trj_ref>	reference trajectory:		R^T k

-h		help
EOF



usage="Usage: $0 [-h] <ksp_raw> <trj_raw> <affine> <ksp_trans> <trj_trans>"

GPU=""
BASIS=""

while getopts "h" opt; do
        case $opt in
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


if [ $# -lt 5 ] ; then

        echo "$usage" >&2
        exit 1
fi

if [ ! -e "$BART_TOOLBOX_PATH"/bart ] ; then
	if [ -e "$TOOLBOX_PATH"/bart ] ; then
		BART_TOOLBOX_PATH="$TOOLBOX_PATH"
	else
		echo "\$BART_TOOLBOX_PATH is not set correctly!" >&2
		exit 1
	fi
fi
export PATH="$BART_TOOLBOX_PATH/:$PATH"

KSP_IN=$(readlink -f "$1")
TRJ_IN=$(readlink -f "$2")
AFFINE=$(readlink -f "$3")
KSP_OUT=$(readlink -f "$4")
TRJ_OUT=$(readlink -f "$5")

WORKDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`
trap 'rm -rf "$WORKDIR"' EXIT
cd $WORKDIR

bart extract 0 0 3  1 0 3 $AFFINE rot		# rotation of affine transform
bart extract 0 0 3  1 3 4 $AFFINE shift		# shift of affine transform

bart transpose 0 1 rot rott
bart transpose 1 3 rott rot

READ=$(bart show -d1 $TRJ_IN)
PHS1=$(bart show -d2 $TRJ_IN)

bart transpose 0 3 $TRJ_IN trjt
bart fmac -s8 rot trjt $TRJ_OUT

#FIXME: TEST and add scaling by determinant.
#bart determinant rott det
#bart invert det idet
#bart fmac idet $KSP_IN ksp

bart fovshift -Sshift -t$TRJ_IN $KSP_IN $KSP_OUT

