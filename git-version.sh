#!/bin/bash

if test -d ${GIT_DIR:-.git} -o -f .git
then
	git describe --match "v*" --dirty
	git describe --match "v*" | cut -f1 -d'-' > version.txt
else
	cat version.txt
fi

