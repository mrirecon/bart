
# This tests if identical input and output is supported
# The hope is that, if it works in these tools, it will also work in all others
tests/test-io: ones slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 50 50 a.ra								;\
	$(TOOLDIR)/slice 0 12 a.ra b.ra								;\
	$(TOOLDIR)/slice 0 12 a.ra a.ra								;\
	$(TOOLDIR)/nrmse -t 0. a.ra b.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-io2: ones std noise nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 50 50 a.ra								;\
	$(TOOLDIR)/noise a.ra b.ra								;\
	$(TOOLDIR)/std 3 b.ra c.ra								;\
	$(TOOLDIR)/std 3 b.ra b.ra								;\
	$(TOOLDIR)/nrmse -t 0. b.ra c.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-io tests/test-io2

