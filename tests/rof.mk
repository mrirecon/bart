


tests/test-rof: phantom slice noise rof nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s2 x.ra							;\
	$(TOOLDIR)/slice 3 0 x.ra x0.ra							;\
	$(TOOLDIR)/noise -n 1000000. x0.ra x0n.ra					;\
	$(TOOLDIR)/rof 700. 3 x0n.ra xr.ra						;\
	$(TOOLDIR)/nrmse -t 0.028 x0.ra xr.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-rof

