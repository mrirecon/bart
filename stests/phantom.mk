
stests/phantom: sim join reshape traj phantom fmac
	set -e; mkdir $(STESTS_TMP) ; cd $(STESTS_TMP)	               	;\
	$(TOOLDIR)/traj t.ra 		                  		;\
	$(SCRIPTDIR)/phantom.sh -S -k -a 90 -r 3 -s 3 -l log.txt -t t.ra test.ra	;\
	exit 0 					;\
	rm *.ra *.txt ; cd .. ; rmdir $(STESTS_TMP)
	touch $@

STESTS += stests/phantom