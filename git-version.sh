#!/bin/bash

if test -d ${GIT_DIR:-.git} -o -f .git
then
	git describe --dirty
	git describe | cut -f1 -d'-' > version.txt
else
	cat version.txt
fi

