#!/bin/bash

set -eu
set -o pipefail

pushd utests
make allclean
make all
popd
