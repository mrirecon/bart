

# check if bart can parse its own version
tests/test-version: version
	$(TOOLDIR)/version -t `cat $(ROOTDIR)/version.txt`
	touch $@

# check if BART_DEBUG_LEVEL is usable together with BART_COMPAT_VERSION
tests/test-version-estdelay-ring: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)									;\
	$(TOOLDIR)/traj -D -r -q0.3:0.1:0.2 -c -y5 t.ra									;\
	$(TOOLDIR)/traj -D -c -r -y5 n.ra										;\
	$(TOOLDIR)/phantom -s8 -k -t t.ra k.ra										;\
	$(TOOLDIR)/traj -D -r -q`BART_COMPAT_VERSION=v0.4.00 BART_DEBUG_LEVEL=0 $(TOOLDIR)/estdelay -R n.ra k.ra` -y5 -c t2.ra	;\
	$(TOOLDIR)/nrmse -t 0.0045 t.ra t2.ra										;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-version tests/test-version-estdelay-ring
