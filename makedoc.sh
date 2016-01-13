#!/bin/bash

( cat doxyconfig ; echo "PROJECT_NUMBER=$(cat version.txt)" ) | doxygen -
