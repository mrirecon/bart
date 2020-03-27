
tests/test-filter-median: index filter extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/index 2 10 a.ra								;\
	$(TOOLDIR)/filter -m2 -l5 a.ra b.ra							;\
	$(TOOLDIR)/extract 2 2 8 a.ra c.ra							;\
	$(TOOLDIR)/nrmse -t 0. c.ra b.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-filter-median2: index filter extract nrmse repmat
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/index 1 10 a.ra								;\
	$(TOOLDIR)/repmat 0 13 a.ra a0.ra							;\
	$(TOOLDIR)/repmat 2 11 a0.ra a2.ra							;\
	$(TOOLDIR)/filter -m1 -l5 a2.ra b.ra							;\
	$(TOOLDIR)/extract 1 2 8 a2.ra c.ra							;\
	$(TOOLDIR)/nrmse -t 0. c.ra b.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-filter-movingavg: ones zeros join scale filter nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 3 4 2 1 O.ra	;\
	$(TOOLDIR)/zeros 3 4 2 1 Z.ra	;\
	$(TOOLDIR)/join 0 O.ra Z.ra O.ra Z.ra j.ra	;\
	$(TOOLDIR)/ones 3 1 2 1 o.ra	;\
	$(TOOLDIR)/scale 0.75 o.ra o75.ra	;\
	$(TOOLDIR)/scale 0.5 o.ra o5.ra	;\
	$(TOOLDIR)/scale 0.25 o.ra o25.ra	;\
	$(TOOLDIR)/scale 0 o.ra o0.ra	;\
	$(TOOLDIR)/join 0 o.ra o75.ra o5.ra o25.ra o0.ra o25.ra o5.ra o75.ra o.ra o75.ra o5.ra o25.ra o0.ra j_syn.ra	;\
	$(TOOLDIR)/filter -a 0 -l4 j.ra j_filt.ra	;\
	$(TOOLDIR)/nrmse -t 0. j_syn.ra j_filt.ra	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-filter-median tests/test-filter-median2 tests/test-filter-movingavg

