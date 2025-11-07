#!/bin/bash
# Copyright 2024. Institute of Biomedical Imaging. TU Graz.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Authors:
# 2024 Moritz Blumenthal <blumenthal@tugraz.at>
# 2024 Philip Schaten <philip.schaten@tugraz.at>

set -e
set -o pipefail

LOGFILE=/dev/stderr

title=$(cat <<- EOF
	Real-Time Reconstruction
EOF
)

helpstr=$(cat <<- EOF
-h help
-l logfile
-L timestamp logfile
-t #turns / tiny golden angle parameter
-f median filter
-R ROVIR
-G Real-time geometric decomposition coil compression
-S Static Coil Compression matrix estimated from first frame.
-T Output timestamp on stderr
-s spokes per frame, activates RAGA reconstruction.
EOF
)

usage="Usage: rtreco.sh [-h,l,f,n,T,A,d] [(-R|-G|-S|-N)] [-p <# virt. chan.>] [-l <logfile>] [-L <timestamp logfile>] [-s <spokes per frame>] [-t <TURNS=5 / Tiny GA=1>] <kspace> <output> [<coils>]"

TURNS=
ROVIR=false
GEOM=false
FILTER=false
NLMEANS=false
STATIC_COILS=false
OVERGRIDDING=1.5
DELAY=2
CHANNELS=8
RAGA=false
TINY=1
SPOKES_PER_FRAME=0
TOPTS="-o2 -r -D -O"
SLW=false
CC_NONE=false
PHASE_POLE=
FAST=" --fast"
GD_CORR=true

: "${NLMEANS_OPTS:=-p3 -d3 -H0.00005 3}"
: "${NLINV_OPTS:=--cgiter=10 -S --real-time --sens-os=1.25 -i6}"
: "${ESTDELAY_OPTS:=-R -p20 -r2}"

export TMPDIR=/dev/shm/
export TMP_TEMPLATE=tmp.rtreco.XXXXXXXXXX

export OMP_NUM_THREADS=1

while getopts "hl:t:fnRTp:SGL:s:ANPd" opt; do
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
	L)
		TIMELOG=$(readlink -f "$OPTARG")
	;;
	t)
		TURNS=$OPTARG
	;;
	f)
		FILTER=true
	;;
	n)
		NLMEANS=true
	;;
	T)
		TIME=-t
	;;
	R)
		ROVIR=true
	;;
	S)
		STATIC_COILS=true
	;;
	p)
		CHANNELS="$OPTARG"
	;;
	P)
		PHASE_POLE=" --phase-pole=6"
		FAST=""
	;;
	G)
		GEOM=true
	;;
	s)
		SPOKES_PER_FRAME="$OPTARG"
	;;
	A)
		SLW=true
	;;
	N)
		CC_NONE=true
	;;
	d)
		GD_CORR=false
	;;
        \?)
		echo "$usage" >&2
		exit 1
        ;;
        esac
done

shift $((OPTIND - 1))


if [ 0 -lt $SPOKES_PER_FRAME ]; then

	: "${TURNS:=1}"

	RAGA=true
	TINY=$TURNS
	TURNS=1
else
	: "${TURNS:=5}"
fi

export GD_CORR
export ESTDELAY_OPTS
export ROVIR
export TURNS
export DELAY
export CHANNELS
export STATIC_COILS
export GEOM
export TOPTS
export RAGA
export TINY
export SLW
export CC_NONE
export PHASE_POLE
export FAST

if $SLW; then
	OVERGRIDDING=1
fi

export BART_DEBUG_STREAM=1

if [ $# -lt 2 ] ; then

	echo "$usage" >&2
	exit 1
fi

if [ $# -gt 3 ] ; then

	echo "$usage" >&2
	exit 1
fi

echo "$title"	>>$LOGFILE
echo		>>$LOGFILE



if [ ! -e "$BART_TOOLBOX_PATH"/bart ] ; then
	if [ -e "$TOOLBOX_PATH"/bart ] ; then
		BART_TOOLBOX_PATH="$TOOLBOX_PATH"
	else
		echo "\$BART_TOOLBOX_PATH is not set correctly!" >&2
		exit 1
	fi
fi
export PATH="$BART_TOOLBOX_PATH:$PATH"


GPU=

if [ -n "$(bart version -V | grep CUDA=1)" ]; then
	GPU="-g"
fi

export GPU


get_file() {
	[ "-" = "$1" ] && echo - || echo $(readlink -f "$1")
}

delay () (

	#delays input by prepending the first frame START times and cropps the last END frames

	DIM=$1
	START=$2
	END=$3

	SRC=$(get_file $4)
	DST=$(get_file $5)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit


	bart tee -i $SRC --out0 meta.fifo -n first.fifo end1.fifo		&
	TOT=$(bart show -d $DIM meta.fifo)

	END=$((TOT-END))

	bart -l$(bart bitmask $DIM) -e$END	copy end1.fifo end2.fifo	&

	bart -l$(bart bitmask $DIM) -e$START	copy -- first.fifo -		| \
	bart					join -s -- $DIM - end2.fifo $DST
)


# Reshape incoming k-space data
reshape_radial_ksp() (

	ksp=$(get_file $1)
	output=$(get_file $2)

        WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
        trap 'rm -rf "$WORKDIR"' EXIT
        cd "$WORKDIR" || exit

	bart tee -i $ksp -n --out0 meta.fifo ksp.fifo &

        dims=$(bart show -m meta.fifo | tail -n1 | cut -f2-)

        samples=$(echo $dims | cut -f1 -d' ')
        spokes=$(echo $dims | cut -f2 -d' ')
	frames=$(echo $dims | cut -f11 -d' ')

	if [ 1 -eq $samples ]; then

		bart -r ksp.fifo copy ksp.fifo $output
	else
		bart reshape -s $(bart bitmask 2 10) $(bart bitmask 0 1 2 10) 1 $samples $spokes $frames ksp.fifo $output
	fi
)

# Arguments:
# $1 = Input File (stream) to be reordered.
# $2 = Rebinned and reordered output file.
#
# Environment parameters:
# SPOKES_PER_FRAME = Spokes per frame after raga rebinning.
# TINY =
rebin_raga() (

        [ "-" = "$1" ] && SRC=- || SRC=$(readlink -f "$1")
        [ "-" = "$2" ] && DST=- || DST=$(readlink -f "$2")

        WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
        trap 'rm -rf "$WORKDIR"' EXIT
        cd "$WORKDIR" || exit

	bart tee -i $SRC -n --out0 meta.fifo ksp.fifo &

        dims=$(bart show -m meta.fifo | tail -n1 | cut -f2-)

        # Full frame
        spokes=$(echo $dims | cut -f3 -d' ')
        frames=$(echo $dims | cut -f11 -d' ')
        spokes2=$SPOKES_PER_FRAME

        all_spokes=$((spokes * frames))
        frames2=$((1 + all_spokes / spokes2))
        all_spokes2=$((frames2 * spokes2))

        # Calculate raga indices
        bart raga -s $TINY $spokes ind
        bart index 0 $all_spokes - |\
        bart reshape $(bart bitmask 0 1 2 10) 1 1 $spokes $frames - range

        bart bin -o ind range order
        bart reshape 1028 $all_spokes 1 order order_flat

        # Pad final frame with zeros
        pad=$(($all_spokes2-$all_spokes))
        bart zeros $(echo $dims | wc -w) $dims - | bart resize 2 $pad 10 1 - pad

        # Rebin
        bart reshape -s1028 1028 $all_spokes 1 ksp.fifo - |\
        bart bin --stream -o order_flat - - |\
        bart join -s 2 - pad - |
        bart reshape -s1028 1028 $spokes2 $frames2 - $DST
)


filter () (

	#temporal median filter with filter size WIN

	WIN=$1
	SRC=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	FIFOS=""

	for i in $(seq $WIN) ; do
		SLIC+=" fil_0_$i.fifo"
		JOIN+=" fil_1_$i.fifo"
	done


	delay 10 $((WIN-1)) 0 $SRC delay.fifo &

	bart copy --stream 1024 -- delay.fifo - | bart tee --out0 meta.fifo | bart tee bart $SLIC > /dev/null &
	TOT=$(bart show -d10 meta.fifo)

	for i in $(seq $WIN) ; do
		bart -l1024 -s $((i-1)) -e $((TOT-WIN+i)) flip 0 fil_0_$i.fifo fil_1_$i.fifo &
	done

	bart -r fil_1_1.fifo		join -- 11 $JOIN -					| \
	bart -r -			filter -m11 -l5 -- - $DST
)

sliding_window() (

	INPUT=$(get_file $1)
	WINDOW_SIZE=$2
	DELAYS=$((WINDOW_SIZE - 1))
	DST=$(get_file $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	bart copy --stream 1024 $INPUT -							|\
	bart tee -n --out0 meta.fifo tmp.fifo $(seq -s' ' -f "input_%g.fifo" $DELAYS)		&

	dims=$(bart show -m meta.fifo | tail -n1 | cut -f2-)

        spokes=$(echo $dims | cut -f3 -d' ')
        frames=$(echo $dims | cut -f11 -d' ')

	for i in $(seq $DELAYS); do
		delay 10 $i $i input_$i.fifo delay_$i.fifo					&
	done

	# tricky: need looping framework, WONT work with builtin stream support of these tools.
	bart -l1024 -r tmp.fifo join 9 tmp.fifo $(seq -s' ' -f "delay_%g.fifo" $DELAYS) -	|\
	bart -r - reshape $(bart bitmask 2 9) $((spokes * WINDOW_SIZE)) 1 - $DST
)

rl_filter_ksp () (

	KSP=$(get_file $1)
	TRJ=$(get_file $2)
	OUT=$(get_file $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	bart -r $TRJ rss 1 $TRJ filter.fifo							&
	bart -r $KSP fmac filter.fifo $KSP $OUT
)


trajectory () (

	# generate trajectory and correct gradient delays with ring
	# gradient delays are taken from previous turns shifted by DELAY

	KSP=$(readlink -f $1)
	DST=$(readlink -f $2)

	GRAD_DELAYS=
	if [ $# -eq 3 ]; then
		GRAD_DELAYS=$(readlink -f $3)
	fi

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	bart tee -i $KSP --out0 meta.fifo	| \
	bart copy --stream 1024 - ksp_tmp.fifo	&

	dims=$(bart show -m meta.fifo | tail -n1 | cut -f2-)
	READ=$(($(echo $dims | cut -f2 -d' ')/2))
	PHS1=$(echo $dims | cut -f3 -d' ')
	TOT=$(echo $dims | cut -f11 -d' ')

	DST1=$DST
	trj_gd=
	gd_delay_dim=10
	if $GD_CORR; then
		DST1=trj.fifo
		trj_gd=trj_gd.fifo
	fi

	if $RAGA; then
		TOPTS="$TOPTS -x$READ -y$SPOKE_FF"

		bart traj $TOPTS - 					|\
		bart repmat 10 $FULL_FRAMES - trj_full
		rebin_raga trj_full trj_rebinned.fifo			&
		bart tee -n -i trj_rebinned.fifo $trj_gd $DST1	 	&

		if $GD_CORR; then
			bart -r ksp_tmp.fifo copy ksp_tmp.fifo ksp_gd.fifo &
		fi
	else
		TOPTS="$TOPTS -x$READ -y$PHS1 -t$TURNS"

		bart traj $TOPTS trj_tmp
		if $SLW && ! $GD_CORR; then
			bart repmat 11 $((TOT / TURNS)) trj_tmp -		|\
			bart reshape $(bart bitmask 10 11) $TOT 1 - $DST1 	&
		else
			bart copy trj_tmp $DST1					&
		fi

		if $GD_CORR; then
			gd_delay_dim=11
			bart reshape $(bart bitmask 2 10)			\
				$((PHS1 * TURNS)) 1 trj_tmp $trj_gd		&

			bart 	reshape -s 2048 $(bart bitmask 2 10 11) 	\
				$((PHS1 * TURNS)) 1 $((TOT / TURNS)) 		\
				ksp_tmp.fifo ksp_gd.fifo 			&
		fi
	fi

	if $GD_CORR; then
		echo "estdelay options: $ESTDELAY_OPTS"				>&2

		bart -t4 -r ksp_gd.fifo estdelay $ESTDELAY_OPTS			\
			$trj_gd ksp_gd.fifo - 					|\
		bart tee -n $GRAD_DELAYS predelay.fifo 				&

		delay $gd_delay_dim $DELAY $DELAY predelay.fifo delays.fifo 	&

		bart -t4 -r delays.fifo						\
			trajcor -V delays.fifo $DST1 trj_cor.fifo		&

		if $RAGA; then
			bart -r trj_cor.fifo copy trj_cor.fifo $DST		&
		else
			bart reshape -s 1024 $(bart bitmask 2 10 11) 		\
				$PHS1 $TOT 1 trj_cor.fifo $DST
		fi
	else
		bart show -m ksp_tmp.fifo > /dev/null &
	fi

	# prevent early deletion of static files:
	wait
)


coilcompression_svd () (

	# SVD based coil compression
	# TRJ is void but provided for easy replacement with rovir

	KSP=$(readlink -f $1)
	TRJ=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	bart tee -i $KSP --out0 meta.fifo 					| \
	bart copy --stream 1024 -- - ksp_tmp.fifo				&

	dims=$(bart show -m meta.fifo | tail -n1 | cut -f2-)
	PHS=$(echo $dims | cut -f3 -d' ')
	TOT=$(echo $dims | cut -f11 -d' ')

	mkfifo $TRJ || true
	cat $TRJ > /dev/null &

	bart		tee -i ksp_tmp.fifo tmp.fifo							| \
	bart 		reshape -s1024 -- $(bart bitmask 2 10) $((PHS*TURNS)) $((TOT/TURNS)) - -	| \
	bart -r -	cc -M -- - predelay.fifo							&

	delay 10 $DELAY $DELAY predelay.fifo cc.fifo							&

	bart -r cc.fifo	repmat -- 9 $TURNS cc.fifo -							| \
	bart		reshape -s1024 -- $(bart bitmask 9 10) 1 $TOT - - 				| \
	bart -r -	ccapply -p$CHANNELS -- tmp.fifo - $DST
)

coilcompression_svd_first () (

	# SVD based coil compression
	# TRJ is void but provided for easy replacement with rovir

	KSP=$(readlink -f $1)
	TRJ=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	bart tee -i $KSP --out0 meta.fifo 					| \
	bart copy --stream 1024 -- - ksp_tmp.fifo				&

	dims=$(bart show -m meta.fifo | tail -n1 | cut -f2-)
	PHS=$(echo $dims | cut -f3 -d' ')
	TOT=$(echo $dims | cut -f11 -d' ')

	mkfifo $TRJ || true;
	cat $TRJ > /dev/null &

	bart		tee -i ksp_tmp.fifo tmp.fifo							| \
	bart 		reshape -s1024 -- $(bart bitmask 2 10) $((PHS*TURNS)) $((TOT/TURNS)) - -	| \
	bart -r -	cc -M -- - - | bart tee -n cc.fifo						&

	bart -l 1024	copy -- cc.fifo cc2

	bart -r tmp.fifo	ccapply -p$CHANNELS -- tmp.fifo cc2 $DST
)


coilcompression_rovir () (

	# ROVir based coil compression

	KSP=$(readlink -f $1)
	TRJ=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	bart tee -i $KSP --out0 meta.fifo	| \
	bart copy --stream 1024 - ksp_tmp.fifo	&

	dims=$(bart show -m meta.fifo | tail -n1 | cut -f2-)
	READ=$(($(echo $dims | cut -f2 -d' ')/2))
	PHS=$(echo $dims | cut -f3 -d' ')
	TOT=$(echo $dims | cut -f11 -d' ')

	bart ones 2 20 20 o
	bart resize -c 0 40 1 40 o pos

	bart ones 2 25 25 o
	bart resize -c 0 40 1 40 o t
	bart ones 2 40 40 o
	bart saxpy -- -1 t o neg

	topts=(-o2 -r -D -l -x"$READ" -y"$PHS" -t"$TURNS" -O)

	bart traj "${topts[@]}" -- - | bart reshape -- $(bart bitmask 2 10) $((TURNS*PHS)) 1 - trj
	bart scale 2 trj trjos
	DIMS=40:40:1
	bart nufftbase $DIMS trjos pat


	bart		tee -i ksp_tmp.fifo tmp.fifo								| \
	bart 		reshape -s1024 -- $(bart bitmask 2 10) $((PHS*TURNS)) $((TOT/TURNS)) - ksp_rovir.fifo	&

	bart -r $TRJ copy $TRJ -										| \
	bart -t4 -r -	scale -- 2 - - 										| \
	bart 		reshape -s1024 -- $(bart bitmask 2 10) $((PHS*TURNS)) $((TOT/TURNS)) - -		| \
	bart 		tee trj_rovir1.fifo trj_rovir2.fifo							| \
	bart -r - 	nufft $GPU -p pat -i -x$DIMS -- - ksp_rovir.fifo -			 		| \
	bart 		tee cim1.fifo > cim2.fifo								&

	bart -r cim1.fifo	fmac -- cim1.fifo pos ipos.fifo							&
	bart -r cim2.fifo	fmac -- cim2.fifo neg ineg.fifo							&

	bart -t4 -r trj_rovir1.fifo	nufft -p pat -- trj_rovir1.fifo ipos.fifo pos.fifo			&
	bart -t4 -r trj_rovir2.fifo	nufft -p pat -- trj_rovir2.fifo ineg.fifo neg.fifo			&

	bart -t4 -r pos.fifo		rovir -- pos.fifo neg.fifo predelay.fifo				&

	delay 10 $DELAY $DELAY predelay.fifo cc.fifo 								&

	bart -r cc.fifo			repmat -- 9 $TURNS cc.fifo -						| \
	bart				reshape -s1024 -- $(bart bitmask 9 10) 1 $TOT - - 			| \
	bart -r -			ccapply -p$CHANNELS -- tmp.fifo - $DST
)

coilcompression_geom () (

	# SVD Coil Compression with alignment along time dim.
	# TRJ is void but provided for easy replacement with rovir

	KSP=$(readlink -f $1)
	TRJ=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	bart tee -i $KSP --out0 meta.fifo 					| \
	bart copy --stream 1024 -- - ksp_tmp.fifo				&

	dims=$(bart show -m meta.fifo | tail -n1 | cut -f2-)
	PHS=$(echo $dims | cut -f3 -d' ')
	TOT=$(echo $dims | cut -f11 -d' ')

	mkfifo $TRJ || true
	cat $TRJ > /dev/null &

	bart -r ksp_tmp.fifo copy ksp_tmp.fifo -					| \
	bart		tee tmp.fifo									| \
	bart 		reshape -s1024 -- $(bart bitmask 2 10) $((PHS*TURNS)) $((TOT/TURNS)) - -	| \
	bart -r -	cc -M -- - predelay.fifo							&

	delay 10 $DELAY $DELAY predelay.fifo cc.fifo							&

	bart -r cc.fifo	repmat -- 9 $TURNS cc.fifo -							| \
	bart		reshape -s1024 -- $(bart bitmask 9 10) 1 $TOT - - 				| \
	bart		ccapply -A10 -p$CHANNELS -- tmp.fifo - $DST
)


coilcompression_none () (

	# SVD based coil compression
	# TRJ is void but provided for easy replacement with rovir

	KSP=$(readlink -f $1)
	TRJ=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	bart -r $KSP copy $KSP $DST;
)



build_pipeline ()
{
	KSP=$(get_file $1)
	REC=$(get_file $2)
	if [ $# -eq 3 ]; then
		COILS=$(get_file $3)
		COILS_TMP=coils.fifo
	else
		COILS=
		COILS_TMP=
	fi


	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"; kill $(jobs -p) || true' EXIT
	cd "$WORKDIR" || exit

	echo "WORKING_DIR:    $WORKDIR" >> $LOGFILE
	echo "k-Space:        $KSP" 	>> $LOGFILE
	echo "Reconstruction: $REC" 	>> $LOGFILE

	reshape_radial_ksp $KSP - |\
	BART_STREAM_LOG=$TIMELOG'_incoming' bart tee -n --out0 meta.fifo ksp0.fifo &

	dims=$(bart show -m meta.fifo | tail -n1 | cut -f2-)

	# cut uses one based indexing
	READ=$(echo $dims | cut -f2 -d' ')
	SPOKE_FF=$(echo $dims | cut -f3 -d' ')
	FULL_FRAMES=$(echo $dims | cut -f11 -d' ')
	export SPOKE_FF
	export FULL_FRAMES

	DIM0=$(echo $dims | cut -f1 -d' ')
	if [ 1 -ne $DIM0 ]; then
		echo "Radial k-Space needs dim[0] == 1. Exiting.."
	fi


	if $RAGA; then
		rebin_raga ksp0.fifo ksp.fifo &
	else
		bart -r ksp0.fifo copy ksp0.fifo ksp.fifo &
	fi


	RDIMS=$((READ/2))
	GDIMS=$(echo "scale=0;($RDIMS*$OVERGRIDDING+0.5)/1" | bc -l)

	: ${IMG_SIZE:=$GDIMS:$GDIMS:1}

	trajectory ksp_gd.fifo trj.fifo $GRAD_DELAYS &


	if $ROVIR ; then
		coilcompression_rovir		ksp_cc.fifo trj_cc.fifo ksp_reco.fifo &
	elif $STATIC_COILS ; then
		coilcompression_svd_first	ksp_cc.fifo trj_cc.fifo ksp_reco.fifo &
	elif $GEOM; then
		coilcompression_geom		ksp_cc.fifo trj_cc.fifo ksp_reco.fifo &
	elif $CC_NONE; then
		coilcompression_none		ksp_cc.fifo trj_cc.fifo ksp_reco.fifo &
	else
		coilcompression_svd		ksp_cc.fifo trj_cc.fifo ksp_reco.fifo &
	fi

	bart tee -i trj.fifo trj_cc.fifo | bart -r - scale -- $OVERGRIDDING - trj_reco.fifo &
	bart tee -i ksp.fifo -n ksp_gd.fifo ksp_cc.fifo &

	if $FILTER; then
		OUT=reco.fifo
	else
		OUT=$REC
	fi

	if $SLW; then

		window_size=$TURNS

		sliding_window trj_reco.fifo $window_size -		|\
			bart tee -n trj_sw1.fifo trj_sw2.fifo		&

		sliding_window ksp_reco.fifo $window_size -		|\
			rl_filter_ksp - trj_sw1.fifo tmp.fifo		&

		OMP_NUM_THREADS=4 BART_STREAM_LOG=$TIMELOG bart -r tmp.fifo		\
			nufft -x$RDIMS:$RDIMS:1 -a trj_sw2.fifo tmp.fifo tmp2.fifo	&

		bart -r tmp2.fifo rss 8 tmp2.fifo - |\
		bart -r - flip 3 - $OUT &
	else

		BART_STREAM_LOG=$TIMELOG bart nlinv			\
			$NLINV_OPTS					\
			$FAST $PHASE_POLE $GPU -x$IMG_SIZE 		\
			-t trj_reco.fifo ksp_reco.fifo - $COILS_TMP |\
		bart -r - flip 3 - -				| \
		bart -r - resize -c 0 $RDIMS 1 $RDIMS - $OUT	&

		if [ ! -z $COILS_TMP ]; then
			bart flip -- 3 $COILS_TMP - |\
			bart resize -c -- 0 $RDIMS 1 $RDIMS - $COILS &
		fi
	fi


	if $FILTER ; then
		filter 5 reco.fifo reco_fil.fifo &
		if $NLMEANS; then

			OMP_NUM_THREADS=16 BART_STREAM_LOG=$TIMELOG${TIMELOG:+_filter} bart -r reco_fil.fifo \
				nlmeans $NLMEANS_OPTS reco_fil.fifo $REC &
		else
			BART_STREAM_LOG=$TIMELOG${TIMELOG:+_filter} bart -r reco_fil.fifo copy reco_fil.fifo $REC &

		fi
	fi

	# before removing tmpdir:
	wait
}

KSP=$1
REC=$2
COILS=
if [ $# -eq 3 ]; then
	COILS=$(readlink -f "$3")
fi

if [ "-" = "$1" ]; then
	stdin_tmpdir=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$stdin_tmpdir"' EXIT
	KSP=$stdin_tmpdir/stdin.fifo
	mkfifo $KSP
fi

build_pipeline $KSP $REC $COILS 2>>$LOGFILE &

if [ "-" = "$1" ]; then
	# 'catch' stdin and block until it's closed:
	cat > $KSP
fi

wait
