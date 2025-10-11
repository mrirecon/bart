#!/bin/sh

UTEST_TYPE="$1"
shift
UTEST_RUN="$1"
shift

fail=0
for test in "$@"
do
#	echo "${UTEST_RUN}" ./"${test}"
	${UTEST_RUN} ./"${test}"
	EX=$?
	if [ 0 -ne "${EX}" ]
	then
		fail=$((fail+1))
	fi
done

if [ "${fail}" -ne 0 ]
then
	printf "%2d %s UNIT TEST(S) FAILED!\n" "${fail}" "${UTEST_TYPE}" >&2
fi

exit $(( fail != 0 ))

