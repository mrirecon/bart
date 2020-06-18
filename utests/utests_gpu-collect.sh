#!/bin/sh

UTESTS_GPU=$(grep UT_GPU_REGISTER $1 | cut -f2 -d'(' | cut -f1 -d')')

for i in $UTESTS_GPU; do
	echo "call_$i,"
done
