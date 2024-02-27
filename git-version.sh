#!/bin/bash

if test -d ${GIT_DIR:-.git} -o -f .git
then
    GDOUT=`git describe --abbrev=7 --match "v*" --dirty 2>&1`
    if [[ $? -eq 0 ]]; then
	echo ${GDOUT}
	git describe --abbrev=7 --match "v*" | cut -f1 -d'-' > version.txt
    else
	if git diff --quiet --exit-code
	then
	    cat version.txt
	else
	    var=`cat version.txt`
	    echo ${var}-dirty
	fi
    fi
else
    cat version.txt
fi

