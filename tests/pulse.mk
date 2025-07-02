


tests/test-pulse-sms: pulse nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/pulse --sinc s.ra								;\
	$(TOOLDIR)/pulse --sms --mb 1 m.ra							;\
	$(TOOLDIR)/nrmse -t 0. s.ra m.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-pulse-sms

