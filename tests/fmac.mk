
tests/test-fmac-sum: ones fmac noise nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 3 2 1 3 a0.ra							;\
	$(TOOLDIR)/noise a0.ra a1.ra							;\
	$(TOOLDIR)/fmac -s 4 a1.ra a2.ra						;\
	$(TOOLDIR)/ones 1 1 o.ra							;\
	$(TOOLDIR)/fmac -s 4 a1.ra o.ra a3.ra						;\
	$(TOOLDIR)/nrmse -t 0. a2.ra a3.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-fmac-sum

