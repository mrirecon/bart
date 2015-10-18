#!/bin/bash
cmp -s $1 $2 || mv $1 $2
rm -f $1
