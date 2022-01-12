


tests/test-roistat-std: zeros noise ones resize roistat std nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 3 100 100 3 z.ra						;\
	$(TOOLDIR)/noise -s1 -n1. z.ra n.ra						;\
	$(TOOLDIR)/ones 3 50 50 1 oy.ra							;\
	$(TOOLDIR)/resize -c 0 100 1 100 oy.ra oy2.ra					;\
	$(TOOLDIR)/roistat -b -D oy2.ra n.ra dy.ra					;\
	$(TOOLDIR)/resize -c 0 50 1 50 n.ra ny2.ra					;\
	$(TOOLDIR)/std 3 ny2.ra dy2.ra							;\
	$(TOOLDIR)/nrmse -t 0. dy2.ra dy.ra 						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-roistat-var: zeros noise ones resize roistat var nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 3 100 100 3 z.ra						;\
	$(TOOLDIR)/noise -s1 -n1. z.ra n.ra						;\
	$(TOOLDIR)/ones 3 50 50 1 oy.ra							;\
	$(TOOLDIR)/resize -c 0 100 1 100 oy.ra oy2.ra					;\
	$(TOOLDIR)/roistat -b -V oy2.ra n.ra dy.ra					;\
	$(TOOLDIR)/resize -c 0 50 1 50 n.ra ny2.ra					;\
	$(TOOLDIR)/var 3 ny2.ra dy2.ra							;\
	$(TOOLDIR)/nrmse -t 0.0000001 dy2.ra dy.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-roistat-mean: ones index fmac noise resize roistat nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 3 100 100 1 o.ra						;\
	$(TOOLDIR)/index 2 3 i.ra							;\
	$(TOOLDIR)/fmac o.ra i.ra oi.ra							;\
	$(TOOLDIR)/noise -s1 -n0.1 oi.ra n.ra						;\
	$(TOOLDIR)/ones 3 50 50 1 oy.ra							;\
	$(TOOLDIR)/resize -c 0 100 1 100 oy.ra oy2.ra					;\
	$(TOOLDIR)/roistat -M oy2.ra n.ra dy.ra						;\
	$(TOOLDIR)/nrmse -t 0.006 i.ra dy.ra 						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-roistat-mult: zeros noise ones resize roistat join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 3 100 100 3 z.ra						;\
	$(TOOLDIR)/noise -s1 -n1. z.ra n.ra						;\
	$(TOOLDIR)/ones 3 50 50 1 oy.ra							;\
	$(TOOLDIR)/resize -c 0 100 1 100 oy.ra oy2.ra					;\
	$(TOOLDIR)/ones 3 70 70 1 ox.ra							;\
	$(TOOLDIR)/resize -c 0 100 1 100 ox.ra ox2.ra					;\
	$(TOOLDIR)/roistat -b -D oy2.ra n.ra dy.ra					;\
	$(TOOLDIR)/roistat -b -D ox2.ra n.ra dx.ra					;\
	$(TOOLDIR)/join 4 dy.ra dx.ra d2.ra 						;\
	$(TOOLDIR)/join 4 oy2.ra ox2.ra o2.ra 						;\
	$(TOOLDIR)/roistat -b -D o2.ra n.ra d.ra					;\
	$(TOOLDIR)/nrmse -t 0. d2.ra d.ra 						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-roistat-std tests/test-roistat-var tests/test-roistat-mult
TESTS += tests/test-roistat-mean

