#!/bin/bash
# Copyright 2024. Institute of Biomedical Imaging. TU Graz.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Authors:
# 2024 Moritz Blumenthal <blumenthal@tugraz.at>
# 2024 Philip Schaten <philip.schaten@tugraz.at>
 
set -e

LOGFILE=/dev/stderr

title=$(cat <<- EOF
	Real-Time Reconstruction
EOF
)

helpstr=$(cat <<- EOF
-h help
-l logfile
-t #turns
-f median filter
-R ROVIR
-G Real-time geometric decomposition coil compression
-S Static Coil Compression matrix estimated from first frame.
EOF
)

usage="Usage: $0 [-h,l,t,f] [(-R|-G)]  <kspace> <output> [<coils>]"

TURNS=5
ROVIR=false
GEOM=false
FILTER=false
STATIC_COILS=false
OVERGRIDDING=1.5
DELAY=2
CHANNELS=8

export TMPDIR=/dev/shm/

while getopts "hl:t:fRTp:SG" opt; do
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
	t)
		TURNS=$OPTARG
	;;
	f)
		FILTER=true
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
	G)
		GEOM=true
	;;
        \?)
        	echo "$usage" >&2
		exit 1
        ;;
        esac
done

shift $((OPTIND - 1))

export ROVIR
export TURNS
export DELAY
export CHANNELS
export STATIC_COILS
export GEOM

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

if [ "-" = "$1" ]; then
	KSP=-
else
	KSP=$(readlink -f "$1")
fi

if [ "-" = "$2" ]; then
	REC=-
else
	REC=$(readlink -f "$2")
fi

if [ $# -eq 3 ]; then
	COILS=$(readlink -f "$3")
fi

delay () (

	#delays input by prepending the first frame START times and cropps the last END frames

	DIM=$1
	START=$2
	END=$3

	SRC=$(readlink -f $4)
	DST=$(readlink -f $5)

	WORKDIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	mkfifo first.fifo
	mkfifo end1.fifo
	mkfifo end2.fifo
	mkfifo meta.fifo

	cat $SRC | bart tee --out0 meta.fifo -n first.fifo end1.fifo			&
	TOT=$(bart show -d $DIM meta.fifo)

	END=$((TOT-END))

	bart -l$(bart bitmask $DIM) -e$END	copy end1.fifo end2.fifo	&

	bart -l$(bart bitmask $DIM) -e$START	copy -- first.fifo -		| \
	bart					join -s -- $DIM - end2.fifo $DST
)

filter () (

	#temporal median filter with filter size WIN

	WIN=$1
	SRC=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	FIFOS=""

	for i in $(seq $WIN) ; do

		mkfifo fil_0_$i.fifo
		mkfifo fil_1_$i.fifo

		SLIC+=" fil_0_$i.fifo"
		JOIN+=" fil_1_$i.fifo"
	done

	mkfifo meta.fifo
	mkfifo delay.fifo

	delay 10 $((WIN-1)) 0 $SRC delay.fifo &

	bart copy --stream 1024 -- delay.fifo - | bart tee --out0 meta.fifo | bart tee bart $SLIC > /dev/null &
	TOT=$(bart show -d10 meta.fifo)

	for i in $(seq $WIN) ; do

		bart -l1024 -s $((i-1)) -e $((TOT-WIN+i)) flip 0 fil_0_$i.fifo fil_1_$i.fifo &
	done

	bart -R fil_1_1.fifo		join -- 11 $JOIN -					| \
	bart -R -			filter -m11 -l5 -- - $DST
)


trajectory () (

	# generate trajectory and correct gradient delays with ring
	# gradient delays are taken from previous turns shifted by DELAY

	KSP=$(readlink -f $1)
	DST=$(readlink -f $2)

	WORKDIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	mkfifo ksp_tmp.fifo
	mkfifo meta0.fifo
	mkfifo meta1.fifo
	mkfifo meta2.fifo

	cat $KSP 					| \
	bart tee --out0 meta0.fifo 			| \
	bart tee --out0 meta1.fifo 			| \
	bart tee --out0 meta2.fifo 			| \
	bart copy --stream 1024 -- - ksp_tmp.fifo	&

	#FIXME DEADLOCK:
	#> ksp_tmp.fifo	&
	
	READ=$(($(bart show -d 1 meta0.fifo)/2))
	PHS1=$(bart show -d 2 meta1.fifo)
	TOT=$(bart show -d 10 meta2.fifo)
	
	topts=(-o2 -r -D -l -x"$READ" -y"$PHS1" -t"$TURNS" -O)

	bart traj "${topts[@]}" trj_tmp
	bart reshape -- $(bart bitmask 2 10) $((PHS1*TURNS)) 1 trj_tmp trj_gd

	bart zeros 1 3 init

	mkfifo predelay.fifo
	mkfifo postdelay.fifo

	bart 		reshape -s 2048 -- $(bart bitmask 2 10 11) $((PHS1*TURNS)) 1 $((TOT/TURNS)) ksp_tmp.fifo -				| \
	bart -t4 -R - 	estdelay -p10 -R -r2 -- trj_gd - predelay.fifo										| \

	delay 11 $DELAY $DELAY predelay.fifo postdelay.fifo &
	
	bart -t4 -R postdelay.fifo 	traj "${topts[@]}" -V postdelay.fifo -- -								| \
	bart 				reshape -s 1024 -- $(bart bitmask 2 10 11) $PHS1 $TOT 1 - $DST
)


coilcompression_svd () (

	# SVD based coil compression
	# TRJ is void but provided for easy replacement with rovir

	KSP=$(readlink -f $1)
	TRJ=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	mkfifo ksp_tmp.fifo
	mkfifo meta0.fifo
	mkfifo meta1.fifo
	mkfifo predelay.fifo
	mkfifo cc.fifo
	mkfifo tmp.fifo

	cat $KSP 											| \
	bart tee --out0 meta0.fifo 									| \
	bart tee --out0 meta1.fifo 									| \
	bart copy --stream 1024 -- - ksp_tmp.fifo							&

	PHS=$(bart show -d 2 meta0.fifo)
	TOT=$(bart show -d 10 meta1.fifo)

	cat $TRJ > /dev/null &

	cat ksp_tmp.fifo										| \
	bart		tee tmp.fifo									| \
	bart 		reshape -s1024 -- $(bart bitmask 2 10) $((PHS*TURNS)) $((TOT/TURNS)) - -	| \
	bart -R -	cc -M -- - predelay.fifo							&
	
	delay 10 $DELAY $DELAY predelay.fifo cc.fifo							&
	
	bart -R cc.fifo	repmat -- 9 $TURNS cc.fifo -							| \
	bart		reshape -s1024 -- $(bart bitmask 9 10) 1 $TOT - - 				| \
	bart -R -	ccapply -p$CHANNELS -- tmp.fifo - $DST
)

coilcompression_svd_first () (

	# SVD based coil compression
	# TRJ is void but provided for easy replacement with rovir

	KSP=$(readlink -f $1)
	TRJ=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	mkfifo ksp_tmp.fifo
	mkfifo meta0.fifo
	mkfifo meta1.fifo
	mkfifo predelay.fifo
	mkfifo cc.fifo
	mkfifo cc2.fifo
	mkfifo tmp.fifo
	mkfifo ccmat.fifo

	cat $KSP 											| \
	bart tee --out0 meta0.fifo 									| \
	bart tee --out0 meta1.fifo 									| \
	bart copy --stream 1024 -- - ksp_tmp.fifo							&

	PHS=$(bart show -d 2 meta0.fifo)
	TOT=$(bart show -d 10 meta1.fifo)

	cat $TRJ > /dev/null &

	cat ksp_tmp.fifo										| \
	bart		tee tmp.fifo									| \
	bart 		reshape -s1024 -- $(bart bitmask 2 10) $((PHS*TURNS)) $((TOT/TURNS)) - -	| \
	bart -R -	cc -M -- - - | bart tee -n cc.fifo						&

	bart -l 1024	copy -- cc.fifo cc2	

	bart -R tmp.fifo	ccapply -p$CHANNELS -- tmp.fifo cc2 $DST
)


coilcompression_rovir () (

	# ROVir based coil compression

	KSP=$(readlink -f $1)
	TRJ=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	mkfifo ksp_tmp.fifo
	mkfifo meta0.fifo
	mkfifo meta1.fifo
	mkfifo meta2.fifo

	cat $KSP 					| \
	bart tee --out0 meta0.fifo 			| \
	bart tee --out0 meta1.fifo 			| \
	bart tee --out0 meta2.fifo 			| \
	bart copy --stream 1024 -- - ksp_tmp.fifo	&

	READ=$(($(bart show -d 1 meta0.fifo)/2))
	PHS=$(bart show -d 2 meta1.fifo)
	TOT=$(bart show -d 10 meta2.fifo)

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

	mkfifo ksp_rovir.fifo
	mkfifo trj_rovir1.fifo
	mkfifo trj_rovir2.fifo
	mkfifo cim1.fifo
	mkfifo cim2.fifo
	mkfifo ipos.fifo
	mkfifo ineg.fifo
	mkfifo pos.fifo
	mkfifo neg.fifo
	mkfifo ksp.fifo
	mkfifo cc.fifo
	mkfifo cc_init.fifo
	mkfifo ksp_cc.fifo
	mkfifo predelay.fifo

	cat ksp_tmp.fifo											| \
	bart		tee tmp.fifo										| \
	bart 		reshape -s1024 -- $(bart bitmask 2 10) $((PHS*TURNS)) $((TOT/TURNS)) - ksp_rovir.fifo	&
	
	cat $TRJ												| \
	bart -t4 -R -	scale -- 2 - - 										| \
	bart 		reshape -s1024 -- $(bart bitmask 2 10) $((PHS*TURNS)) $((TOT/TURNS)) - -		| \
	bart 		tee trj_rovir1.fifo trj_rovir2.fifo							| \
	bart -R - 	nufft -g -p pat -i -x$DIMS -- - ksp_rovir.fifo -			 		| \
	bart 		tee cim1.fifo > cim2.fifo								&

	bart -R cim1.fifo	fmac -- cim1.fifo pos ipos.fifo							&
	bart -R cim2.fifo	fmac -- cim2.fifo neg ineg.fifo							&

	bart -t4 -R trj_rovir1.fifo	nufft -p pat -- trj_rovir1.fifo ipos.fifo pos.fifo			&
	bart -t4 -R trj_rovir2.fifo	nufft -p pat -- trj_rovir2.fifo ineg.fifo neg.fifo			&

	bart -t4 -R pos.fifo		rovir -- pos.fifo neg.fifo predelay.fifo				&
	
	delay 10 $DELAY $DELAY predelay.fifo cc.fifo 								&
	
	bart -R cc.fifo			repmat -- 9 $TURNS cc.fifo -						| \
	bart				reshape -s1024 -- $(bart bitmask 9 10) 1 $TOT - - 			| \
	bart -R -			ccapply -p$CHANNELS -- tmp.fifo - $DST
)

coilcompression_geom () (

	# SVD Coil Compression with alignment along time dim.
	# TRJ is void but provided for easy replacement with rovir

	KSP=$(readlink -f $1)
	TRJ=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	mkfifo tmp.fifo
	mkfifo ksp_tmp.fifo
	mkfifo meta0.fifo
	mkfifo meta1.fifo
	mkfifo predelay.fifo
	mkfifo cc.fifo

	cat $KSP 											| \
	bart tee --out0 meta0.fifo 									| \
	bart tee --out0 meta1.fifo 									| \
	bart copy --stream 1024 -- - ksp_tmp.fifo							&

	PHS=$(bart show -d 2 meta0.fifo)
	TOT=$(bart show -d 10 meta1.fifo)

	cat $TRJ > /dev/null &

	cat ksp_tmp.fifo										| \
	bart		tee tmp.fifo									| \
	bart 		reshape -s1024 -- $(bart bitmask 2 10) $((PHS*TURNS)) $((TOT/TURNS)) - -	| \
	bart -R -	cc -M -- - predelay.fifo							&

	delay 10 $DELAY $DELAY predelay.fifo cc.fifo							&

	bart -R cc.fifo	repmat -- 9 $TURNS cc.fifo -							| \
	bart		reshape -s1024 -- $(bart bitmask 9 10) 1 $TOT - - 				| \
	bart		ccapply -A10 -p$CHANNELS -- tmp.fifo - $DST
)

WORKDIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')
trap 'rm -rf "$WORKDIR"; kill $(jobs -p) || true' EXIT
cd "$WORKDIR" || exit

{

mkfifo ksp.fifo
mkfifo meta0.fifo
mkfifo meta1.fifo

bart -R $KSP copy -- $KSP - | bart tee --out0 meta0.fifo | bart tee --out0 meta1.fifo -n ksp.fifo &

echo "WORKING_DIR:    $WORKDIR" >> $LOGFILE
echo "k-Space:        $KSP" 	>> $LOGFILE
echo "Reconstruction: $REC" 	>> $LOGFILE

READ=$(bart show -d 1 meta0.fifo)
PHS1=$(bart show -d 2 meta1.fifo)

RDIMS=$((READ/2))
GDIMS=$(echo "scale=0;($RDIMS*$OVERGRIDDING+0.5)/1" | bc -l)

bart ones 3 1 $READ $PHS1 pat

mkfifo reco.fifo

mkfifo ksp_reco.fifo
mkfifo trj_reco.fifo

mkfifo ksp_gd.fifo
mkfifo trj.fifo
trajectory ksp_gd.fifo trj.fifo &


mkfifo ksp_cc.fifo
mkfifo trj_cc.fifo

if $ROVIR ; then
	coilcompression_rovir		ksp_cc.fifo trj_cc.fifo ksp_reco.fifo &
elif $STATIC_COILS ; then
	coilcompression_svd_first	ksp_cc.fifo trj_cc.fifo ksp_reco.fifo &
elif $GEOM; then
	coilcompression_geom	ksp_cc.fifo trj_cc.fifo ksp_reco.fifo &
else
	coilcompression_svd		ksp_cc.fifo trj_cc.fifo ksp_reco.fifo &
fi

cat trj.fifo | bart tee trj_cc.fifo | bart -R - scale -- $OVERGRIDDING - trj_reco.fifo &
cat ksp.fifo | bart tee -n ksp_gd.fifo ksp_cc.fifo &

bart 		nlinv --cgiter=10 -S --real-time --fast -g --sens-os=1.25 -i6 -x$GDIMS:$GDIMS:1 -ppat -t trj_reco.fifo -- ksp_reco.fifo - $COILS	| \
bart		tee $TIME																| \
bart -R - 	flip -- 3 - -																| \
bart -R - 	resize -c -- 0 $RDIMS 1 $RDIMS - reco.fifo												&

if $FILTER ; then 

	mkfifo reco_fil.fifo

	filter 5 reco.fifo reco_fil.fifo &
	bart -R reco_fil.fifo copy -- reco_fil.fifo $REC &
else
	bart -R reco.fifo copy -- reco.fifo $REC &
fi

} 2>>$LOGFILE

wait

if [ -f "$COILS.hdr" ]; then
	bart flip -- 3 $COILS tmp_coils
	bart resize -c -- 0 $RDIMS 1 $RDIMS tmp_coils $COILS;
fi
