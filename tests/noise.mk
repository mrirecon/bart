


tests/test-noise: zeros noise std ones nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 2 100 100 z.ra							;\
	$(TOOLDIR)/noise -s1 -n1. z.ra n.ra						;\
	$(TOOLDIR)/std 3 n.ra d.ra							;\
	$(TOOLDIR)/ones 2 1 1 o.ra							;\
	$(TOOLDIR)/nrmse -t 0.02 o.ra d.ra 						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-noise-real: zeros noise std ones nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 2 100 100 z.ra							;\
	$(TOOLDIR)/noise -s1 -n1. -r z.ra n.ra						;\
	$(TOOLDIR)/std 3 n.ra d.ra							;\
	$(TOOLDIR)/ones 2 1 1 o.ra							;\
	$(TOOLDIR)/nrmse -t 0.02 o.ra d.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

# In order to use a small tolerance, a rather large array of 100k values is needed.
# Test idea is simply: add spike noise to an array of zeros, multiply with its own
# inverse and count the number of ones. Then check that it is close to the expected number.
tests/test-noise-spike: zeros noise invert fmac vec nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 2 100 1000 z.ra						;\
	$(TOOLDIR)/noise -s1 -n1. -S 0.67 z.ra n.ra				;\
	$(TOOLDIR)/invert n.ra o.ra							;\
	$(TOOLDIR)/fmac -s3 o.ra n.ra nz.ra						;\
	$(TOOLDIR)/vec 67000 r.ra							;\
	$(TOOLDIR)/nrmse -t 0.006 r.ra nz.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-noise tests/test-noise-real tests/test-noise-spike

