
tests/test-bitmask: bitmask
	set -e					 						;\
	[ 0 -eq `$(TOOLDIR)/bitmask` ] 								&&\
	[ 1 -eq `$(TOOLDIR)/bitmask 0` ] 							&&\
	[ 1587 -eq `$(TOOLDIR)/bitmask 0 1 4 5 9 10` ] 						&&\
	[ 65535 -eq `$(TOOLDIR)/bitmask 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15` ] 		&&\
	true
	touch $@


tests/test-bitmask-reverse: bitmask
	set -e					 							;\
	[ -z `$(TOOLDIR)/bitmask -b 0 | tr -d '\n'` ]	 						&&\
	[ "0 " = "`$(TOOLDIR)/bitmask -b 1 | tr -d '\n'`" ]						&&\
	[ "0 1 4 5 9 10 " == "`$(TOOLDIR)/bitmask -b 1587 | tr -d '\n'`" ] 				&&\
	[ "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 " == "`$(TOOLDIR)/bitmask -b 65535 | tr -d '\n'`" ]	&&\
	true
	touch $@



TESTS += tests/test-bitmask tests/test-bitmask-reverse

