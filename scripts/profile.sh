#!/bin/sh
set -e

usage="Usage: $0 <input.log> <output.log>"

helpstr=$(cat <<- EOF
Postprocess debugging output from BART to extract profiling
information and to translate pointer values to symbol names.

-h help
EOF
)


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


if [ $# -lt 2 ] ; then

	echo "$usage" >&2
	exit 1
fi

in=$(readlink -f "$1")
out=$(readlink -f "$2")

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


nm --defined-only $TOOLBOX_PATH/bart | cut -c11-16,19- | sort > bart.syms


cat $in	| grep "^TRACE" \
	| grep " 0x" \
	| cut -c7-23,25-31,34- \
	| sort -k3 \
	| join -11 -23 bart.syms - \
	| cut -c8- \
	| sort -k2 > $out

