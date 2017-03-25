
tests/test-estdelay: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -q1.5:1:-0.5 -r -y8 t.ra						;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -q`$(TOOLDIR)/estdelay t.ra k.ra` -r -y8 t2.ra			;\
	$(TOOLDIR)/nrmse -t 0.00001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-transverse: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -O -q1.5:1:-0.5 -r -y8 t.ra						;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -O -q`$(TOOLDIR)/estdelay t.ra k.ra` -r -y8 t2.ra			;\
	$(TOOLDIR)/nrmse -t 0.001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-coils: estdelay scale traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -G -q0.3:-0.1:0.2 -y5 t.ra						;\
	$(TOOLDIR)/traj -G -r -y5 n.ra								;\
	$(TOOLDIR)/scale 0.5 n.ra ns.ra								;\
	$(TOOLDIR)/scale 0.5 t.ra ts.ra								;\
	$(TOOLDIR)/phantom -s8 -k -t ts.ra k.ra							;\
	$(TOOLDIR)/traj -G -q`$(TOOLDIR)/estdelay ns.ra k.ra` -y5 t2.ra				;\
	$(TOOLDIR)/nrmse -t 0.004 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-ring: estdelay scale traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -G -q0.3:-0.1:0.2 -c -y5 t.ra						;\
	$(TOOLDIR)/traj -G -c -r -y5 n.ra							;\
	$(TOOLDIR)/scale 0.5 n.ra ns.ra								;\
	$(TOOLDIR)/scale 0.5 t.ra ts.ra								;\
	$(TOOLDIR)/phantom -k -t ts.ra k.ra							;\
	$(TOOLDIR)/traj -G -q`$(TOOLDIR)/estdelay -R ns.ra k.ra` -c -y5 t2.ra			;\
	$(TOOLDIR)/nrmse -t 0.003 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-scale: estdelay scale traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -q1.5:1:-0.5 -r -y8 t.ra						;\
	$(TOOLDIR)/scale 0.5 t.ra ts.ra								;\
	$(TOOLDIR)/phantom -k -t ts.ra k.ra							;\
	$(TOOLDIR)/traj -D -q`$(TOOLDIR)/estdelay ts.ra k.ra` -r -y8 t2.ra			;\
	$(TOOLDIR)/scale 0.5 t2.ra t2s.ra							;\
	$(TOOLDIR)/nrmse -t 0.0001 ts.ra t2s.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-estdelay tests/test-estdelay-transverse
TESTS += tests/test-estdelay-ring tests/test-estdelay-coils tests/test-estdelay-scale

