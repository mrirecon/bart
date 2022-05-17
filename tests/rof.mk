


tests/test-rof: phantom slice noise rof nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s2 x.ra							;\
	$(TOOLDIR)/slice 3 0 x.ra x0.ra							;\
	$(TOOLDIR)/noise -n 1000000. x0.ra x0n.ra					;\
	$(TOOLDIR)/rof 700. 3 x0n.ra xd.ra						;\
	$(TOOLDIR)/slice 15 0 xd.ra xd0.ra						;\
	$(TOOLDIR)/nrmse -t 0.028 x0.ra xd0.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-tgv: phantom slice noise tgv nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s2 x.ra							;\
	$(TOOLDIR)/slice 3 0 x.ra x0.ra							;\
	$(TOOLDIR)/noise -n 1000000. x0.ra x0n.ra					;\
	$(TOOLDIR)/tgv 650. 3 x0n.ra xd.ra						;\
	$(TOOLDIR)/slice 15 0 xd.ra xd0.ra						;\
	$(TOOLDIR)/nrmse -t 0.024 x0.ra xd0.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-rof tests/test-tgv

