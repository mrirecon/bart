
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


TESTS += tests/test-filter-median tests/test-filter-median2

