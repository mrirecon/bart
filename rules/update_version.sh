#!/bin/bash

echo 'VERSION('`./git-version.sh`')' > version.new.$$
./rules/update_if_changed.sh version.new.$$ src/misc/version.inc
