tests/test-poly: ones poly nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP) ;\
	$(TOOLDIR)/ones 1 128 ones.ra                ;\
	$(TOOLDIR)/poly 128 0 1 poly.ra              ;\
	$(TOOLDIR)/nrmse -t 0. ones.ra poly.ra       ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-poly
