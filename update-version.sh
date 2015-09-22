#!/bin/bash

echo 'VERSION('`./git-version.sh`')' > version.new
cmp -s version.new src/misc/version.inc || mv version.new src/misc/version.inc
rm -f version.new

