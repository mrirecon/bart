


tests/test-zexp: ones zeros zexp nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/zeros 2 5 5 a.ra								;\
	$(TOOLDIR)/zexp a.ra b.ra								;\
	$(TOOLDIR)/ones 2 5 5 c.ra								;\
	$(TOOLDIR)/nrmse -t 0. c.ra b.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-zexp2: zeros noise saxpy fmac zexp nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/zeros 2 5 5 a.ra								;\
	$(TOOLDIR)/noise -s1 a.ra a1.ra								;\
	$(TOOLDIR)/noise -s2 a.ra a2.ra								;\
	$(TOOLDIR)/saxpy 1. a1.ra a2.ra as.ra							;\
	$(TOOLDIR)/zexp a1.ra b1.ra								;\
	$(TOOLDIR)/zexp a2.ra b2.ra								;\
	$(TOOLDIR)/zexp as.ra bs.ra								;\
	$(TOOLDIR)/fmac b1.ra b2.ra bs2.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 bs.ra bs2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-zexp-imag: zeros noise scale zexp nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/zeros 2 5 5 a.ra								;\
	$(TOOLDIR)/noise -s1 a.ra a1.ra								;\
	$(TOOLDIR)/scale -- 1.i a1.ra a2.ra							;\
	$(TOOLDIR)/zexp a2.ra b1.ra								;\
	$(TOOLDIR)/zexp -i a1.ra b2.ra								;\
	$(TOOLDIR)/nrmse -t 0.000001 b1.ra b2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@




TESTS += tests/test-zexp tests/test-zexp2 tests/test-zexp-imag

