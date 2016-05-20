#!/bin/sh

UTESTS=$(grep UT_REGISTER $1 | cut -f2 -d'(' | cut -f1 -d')')

for i in $UTESTS; do
	echo "call_$i,"
done
