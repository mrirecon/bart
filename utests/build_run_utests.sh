#!/bin/bash

set -eu
set -o pipefail

export DEBUG_LEVEL=3

pushd utests
make allclean
make all
popd
