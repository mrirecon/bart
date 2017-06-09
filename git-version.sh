#!/bin/bash

if test -d ${GIT_DIR:-.git} -o -f .git
then
    git describe --match "v*" --dirty
    if [[ $? -eq 0 ]]; then
	git describe --match "v*" | cut -f1 -d'-' > version.txt
    else
	var=`cat version.txt`
	echo ${var}-dirty
    fi
else
    cat version.txt
fi

