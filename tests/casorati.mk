


tests/test-casorati: ones noise casorati reshape avg slice squeeze extract nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 2 6 6 o1.ra							;\
	$(TOOLDIR)/noise o1.ra o2.ra							;\
	$(TOOLDIR)/casorati 1 2 o2.ra o3.ra						;\
	$(TOOLDIR)/reshape 7 5 6 2 o3.ra o4.ra						;\
	$(TOOLDIR)/avg 4 o4.ra o5.ra							;\
	$(TOOLDIR)/slice 0 1 o5.ra o6.ra						;\
	$(TOOLDIR)/squeeze o6.ra o7.ra							;\
	$(TOOLDIR)/extract 1 1 2 o2.ra i2.ra						;\
	$(TOOLDIR)/avg 2 i2.ra i3.ra							;\
	$(TOOLDIR)/nrmse -t 0.001 i3.ra o7.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-casorati
