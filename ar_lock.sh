#!/bin/bash
set -e
if command -v flock > /dev/null ; then
	flock `dirname $2`/.`basename $2`.lock -c "ar $*"
	exit 0
fi

if command -v shlock > /dev/null ; then
	LOCK=/tmp/`basename $2`.lock
	trap 'rm -f ${LOCK} ; exit 1' 1 2 3 15

	while true ; do
		if shlock -p $$ -f ${LOCK} ; then
        		ar $*
			rm -rf ${LOCK}
			exit 0
		else
        		sleep 1
		fi
	done
fi

echo "Error: no flock/shlock command!"
exit 1


