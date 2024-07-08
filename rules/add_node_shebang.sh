#!/bin/bash

for exe in "$@"
do
	if [ ! -f "$exe" ]; then
		continue
	fi
	if head -n 1 "$exe" | grep -q "/usr/bin/env"; then
		continue
	fi
	chmod +x "$exe"
	sed -i '1s|^|#!/usr/bin/env node\n|' "$exe"
done
