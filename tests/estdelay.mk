
tests/test-estdelay: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -q1.5:1:-0.5 -r -y8 t.ra						;\
	$(TOOLDIR)/traj -D -r -y8 t0.ra								;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -q`$(TOOLDIR)/estdelay t0.ra k.ra` -r -y8 t2.ra			;\
	$(TOOLDIR)/nrmse -t 0.00001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-dccen: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -q1.5:1.:-0.5 -r -y8 -c t.ra						;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -r -y8 -c t0.ra							;\
	$(TOOLDIR)/traj -D -q`$(TOOLDIR)/estdelay t0.ra k.ra` -r -y8 -c t2.ra			;\
	$(TOOLDIR)/nrmse -t 0.0001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-dccen-scale: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -q1.5:1:-0.5 -r -y8 -o2. -c t.ra					;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -r -y8 -o2. -c t0.ra							;\
	$(TOOLDIR)/traj -D -q`$(TOOLDIR)/estdelay t0.ra k.ra` -r -y8 -o2. -c t2.ra		;\
	$(TOOLDIR)/nrmse -t 0.0001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-transverse: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -O -q1.5:1:-0.5 -r -y8 t.ra						;\
	$(TOOLDIR)/traj -D -r -y8 t0.ra								;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -O -q`$(TOOLDIR)/estdelay t0.ra k.ra` -r -y8 t2.ra			;\
	$(TOOLDIR)/nrmse -t 0.0011 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-coils: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -G -q0.3:-0.1:0.2 -y5 -o2. -c t.ra					;\
	$(TOOLDIR)/traj -G -r -y5 -o2. n.ra							;\
	$(TOOLDIR)/phantom -s8 -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -G -q`$(TOOLDIR)/estdelay n.ra k.ra` -y5 -o2. t2.ra			;\
	$(TOOLDIR)/nrmse -t 0.004 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-ring-scale: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -r -q0.3:0.1:0.2 -O -c -y5 -o2. t.ra					;\
	$(TOOLDIR)/traj -D -c -r -y5 -o2. n.ra							;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -r -q`$(TOOLDIR)/estdelay -R n.ra k.ra` -O -y5 -o2. -c t2.ra		;\
	$(TOOLDIR)/nrmse -t 0.0001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-ring-uncen: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -r -q0.3:0.1:0.2 -y5 -o2. t.ra					;\
	$(TOOLDIR)/traj -D -r -y5 -o2. n.ra							;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -r -q`$(TOOLDIR)/estdelay -R n.ra k.ra` -y5 -o2. t2.ra		;\
	$(TOOLDIR)/nrmse -t 0.002 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-ring: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -r -q0.3:0.1:0.2 -c -y5 t.ra						;\
	$(TOOLDIR)/traj -D -c -r -y5 n.ra							;\
	$(TOOLDIR)/phantom -s8 -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -r -q`$(TOOLDIR)/estdelay -R n.ra k.ra` -y5 -c t2.ra			;\
	$(TOOLDIR)/nrmse -t 0.0035 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-estdelay-scale: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -q1.5:1:-0.5 -r -y8 -o2. t.ra					;\
	$(TOOLDIR)/traj -D -r -y8 -o2. t0.ra							;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -q`$(TOOLDIR)/estdelay t0.ra k.ra` -r -y8 -o2. t2.ra			;\
	$(TOOLDIR)/nrmse -t 0.0001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-asym: estdelay traj phantom nrmse extract
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -q1.5:1.:-0.5 -r -y8 -x128 t.ra					;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -r -y8 -x128 t0.ra							;\
	$(TOOLDIR)/extract 1 1 128 k.ra kb.ra 							;\
	$(TOOLDIR)/extract 1 1 128 t0.ra t0b.ra 						;\
	$(TOOLDIR)/traj -D -q`$(TOOLDIR)/estdelay t0b.ra kb.ra` -r -y8 -x128 t2.ra		;\
	$(TOOLDIR)/nrmse -t 0.0001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-estdelay-dccen-asym: estdelay traj phantom nrmse extract
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -q1.5:1.:-0.5 -r -y8 -c -x128 t.ra					;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -r -y8 -c -x128 t0.ra						;\
	$(TOOLDIR)/extract 1 1 128 k.ra kb.ra 							;\
	$(TOOLDIR)/extract 1 1 128 t0.ra t0b.ra 						;\
	$(TOOLDIR)/traj -D -q`$(TOOLDIR)/estdelay t0b.ra kb.ra` -r -y8 -c -x128 t2.ra		;\
	$(TOOLDIR)/nrmse -t 0.0001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-estdelay tests/test-estdelay-dccen tests/test-estdelay-transverse
TESTS += tests/test-estdelay-ring tests/test-estdelay-coils tests/test-estdelay-scale
TESTS += tests/test-estdelay-dccen-scale tests/test-estdelay-ring-scale
TESTS += tests/test-estdelay-ring-uncen tests/test-estdelay-asym tests/test-estdelay-dccen-asym

