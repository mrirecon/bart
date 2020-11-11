#!/bin/bash
{ /usr/bin/time -v $* ; } 2>&1 | grep "Maximum resident" | awk '{print $6}'
