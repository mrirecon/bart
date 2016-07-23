#!/bin/bash

EXEC=$1
OUT=$2

nm --defined-only ${EXEC} | cut -c11-16,19- | sort > ${OUT}
