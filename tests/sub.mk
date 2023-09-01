tests/test-sub: ones scale sub nrmse bart # bart is called by sub
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP) 	;\
	$(TOOLDIR)/ones 1 128 o.ra                	;\
	$(TOOLDIR)/scale 2. o.ra t.ra                	;\
	$(TOOLDIR)/sub o.ra t.ra d.ra			;\
	$(TOOLDIR)/nrmse -t 0. o.ra d.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-sub
