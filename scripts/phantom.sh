#!/bin/bash
# Copyright 2022. TU Graz. Institute of Biomedical Imaging.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Author:
# 2022 Nick Scholand <scholand@tugraz.at>
#
# Creation of digital reference object.

set -e

LOGFILE=/dev/stdout
KSPACE=false
SENS=1
ROT_ANGLE=0
ROT_STEPS=1
GEOM=NIST

title=$(cat <<- EOF
	Digital Reference Object
EOF
)

helpstr=$(cat <<- EOF
-S \t\t Diagnostic Sonar geometry (NIST phantom is default)
-k \t\t simulate in k-space
-a d \t\t angle of rotation
-r d \t\t number of rotation steps
-s d \t\t number of simulated coils
-t <traj> \t define custom trajectory file
-l \t\t logfile
-h \t\t help

Please adjust simulation parameters inside the script.
EOF
)

usage="Usage: $0 [-h] [-k] [-r d] [-s d] [-t <traj>] <output>"

echo "$title"
echo

while getopts "hSka:r:s:t:l:" opt; do
        case $opt in
	h)
		echo "$usage"
		echo
		echo -e "$helpstr"
		exit 0
	;;
        S)
		GEOM=SONAR
	;;
        k)
		KSPACE=true
	;;
        a)
		ROT_ANGLE=$OPTARG
	;;
        r)
		ROT_STEPS=$OPTARG
	;;
        s)
		SENS=$OPTARG
	;;
        t)
		TRAJ=$(readlink -f "$OPTARG")
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

if [ $# != 1 ] ; then

        echo "$usage" >&2
        exit 1
fi


export PATH=$TOOLBOX_PATH:$PATH

if [ ! -e $TOOLBOX_PATH/bart ] ; then
        echo "\$TOOLBOX_PATH is not set correctly!" >&2
	exit 1
fi


output=$(readlink -f "$1")


# Tests for usefull input

if [ ! -z "${TRAJ}" ] && [ "$KSPACE" = false ]; then

        echo "Trajectory only works in k-space domain. Please add [-k]!" >&2
	exit 1
fi


#WORKDIR=$(mktemp -d)
# Mac: http://unix.stackexchange.com/questions/30091/fix-or-alternative-for-mktemp-in-os-x
WORKDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`
trap 'rm -rf "$WORKDIR"' EXIT
cd $WORKDIR


# start group for redirection of output to the logfile
{

case $GEOM in

NIST)
        echo "NIST Phantom Geometry"
        echo "T2 Sphere of Model 130"
        echo "Relaxation Paramters for 3 T"
        echo ""

        ## Relaxation parameters for T2 Sphere of NIST phantom at 3 T (Model 130)
        ##      Stupic, KF, Ainslie, M, Boss, MA, et al.
        ##      A standard system phantom for magnetic resonance imaging.
        ##      Magn Reson Med. 2021; 86: 1194– 1211. https://doi.org/10.1002/mrm.28779
        T1=(3 2.48 2.173 1.907 1.604 1.332 1.044 0.802 0.609 0.458 0.337 0.244 0.177 0.127 0.091)
        T2=(1 0.581 0.404 0.278 0.191 0.133 0.097 0.064 0.046 0.032 0.023 0.016 0.011 0.008 0.006)
        ;;

SONAR)
        echo "Diagnostic Sonar Phantom Geometry"
        echo "Eurospin II"
        echo "Gels: 3, 4, 7, 10, 14, and 16"
        echo ""

        ## Relaxation parameters for Diagnostic Sonar phantom
        ## Eurospin II, gel nos 3, 4, 7, 10, 14, and 16)
        ## T1 from reference measurements in
        ##      Wang, X., Roeloffs, V., Klosowski, J., Tan, Z., Voit, D., Uecker, M. and Frahm, J. (2018),
        ##      Model-based T1 mapping with sparsity constraints using single-shot inversion-recovery radial FLASH.
        ##      Magn. Reson. Med, 79: 730-740. https://doi.org/10.1002/mrm.26726
        ## T2 from
        ##      T. J. Sumpf, A. Petrovic, M. Uecker, F. Knoll and J. Frahm,
        ##      Fast T2 Mapping With Improved Accuracy Using Undersampled Spin-Echo MRI and Model-Based Reconstructions With a Generating Function
        ##      IEEE Transactions on Medical Imaging, vol. 33, no. 12, pp. 2213-2222, Dec. 2014, doi: 10.1109/TMI.2014.2333370.
        T1=(3 0.311 0.458 0.633 0.805 1.1158 1.441 3)
        T2=(1 0.046 0.081 0.101 0.132 0.138 0.166 1)
        ;;
*)
    echo -n "Unknown geometry!\n"
    exit 1
    ;;
esac


# Simulation Parameters
#       Run `bart sim --seq h` for more details
SEQ=IR-FLASH    # Sequence Type
TR=0.0034       # Repetition Time [s]
TE=0.0021       # Echo Time [s]
REP=600         # Number of repetitions
IPL=0.01        # Inversion Pulse Length [s]
ISP=0.005       # Inversion Spoiler Gradient Length [s]
PPL=0           # Preparation Pulse Length [s]
TRF=0.001       # Pulse Duration [s]
FA=6            # Flip Angle [degree]
BWTP=4          # Bandwidth-Time-Product
OFF=0           # Off-Resonance [rad/s]
SLGRAD=0        # Slice Selection Gradient Strength [T/m]
SLTHICK=0       # Thickness of Simulated Slice [m]
NSPINS=1        # Number of Simulated Spins

# Run Simulation
for i in `seq 0 $((${#T1[@]}-1))`; do

        echo -e "Tube $i\t T1: ${T1[$i]} s,\tT2[$i]: ${T2[$i]} s"

        bart sim        --ODE \
                        --seq $SEQ,TR=$TR,TE=$TE,Nrep=$REP,ipl=$IPL,isp=$ISP,ppl=$PPL,Trf=$TRF,FA=$FA,BWTP=$BWTP,off=$OFF,sl-grad=$SLGRAD,slice-thickness=$SLTHICK,Nspins=$NSPINS \
                        -1 ${T1[$i]}:${T1[$i]}:1 -2 ${T2[$i]}:${T2[$i]}:1 \
                        _simu$(printf "%02d" $i)
done


# Join individual simulations
bart join 7 $(ls _simu*.cfl | sed -e 's/\.cfl//') simu

# Join simulations in a single dimension (-> 6)
bart reshape $(bart bitmask 6 7) ${#T1[@]} 1 simu simu2


# Create Geometry
if [ -z "${TRAJ}" ]; then

        if $KSPACE; then

                # Create default trajectory
                DIM=192
                SPOKES=$((DIM-1))

                bart traj -x $DIM -y $SPOKES traj

                bart phantom --${GEOM} -b -s $SENS --rotation-steps $ROT_STEPS --rotation-angle $ROT_ANGLE  -t traj geom
        else

                bart phantom --${GEOM} -b -s $SENS --rotation-steps $ROT_STEPS --rotation-angle $ROT_ANGLE geom
        fi
else
        if $KSPACE; then

                bart phantom --${GEOM} -b -s $SENS --rotation-steps $ROT_STEPS --rotation-angle $ROT_ANGLE -k -t ${TRAJ} geom
        else

                bart phantom --${GEOM} -b -s $SENS --rotation-steps $ROT_STEPS --rotation-angle $ROT_ANGLE geom
        fi
fi

# Combine simulated signal and geometry

bart fmac -s $(bart bitmask 6) geom simu2 $output

} > $LOGFILE

[ -d $WORKDIR ] && rm -rf $WORKDIR

exit 0

