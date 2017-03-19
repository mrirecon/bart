
tests/test-estdelay: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -q1.5:1:-0.5 -r -y8 t.ra						;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -q`$(TOOLDIR)/estdelay t.ra k.ra` -r -y8 t2.ra			;\
	$(TOOLDIR)/nrmse -t 0.00001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-estdelay

