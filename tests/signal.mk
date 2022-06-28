tests/test-signal-spoke-averaging-2: signal slice join avg nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/signal -I -F -r0.0041 -e0.00258 -f8 -n1000 -1 3:3:1 -2 1:1:1 ref.ra	;\
	$(TOOLDIR)/signal -I -F -r0.0041 -e0.00258 -f8 -n500 --av-spokes 2 -1 3:3:1 -2 1:1:1 signal.ra	;\
	$(TOOLDIR)/slice 5 0 ref.ra ref1.ra ;\
	$(TOOLDIR)/slice 5 1 ref.ra ref2.ra ;\
	$(TOOLDIR)/join 5 ref1.ra ref2.ra comb.ra ;\
	$(TOOLDIR)/avg 32 comb.ra avg.ra ;\
	$(TOOLDIR)/slice 5 0 signal.ra s.ra;\
	$(TOOLDIR)/nrmse -t 0.00001 avg.ra s.ra			    			;\
	$(TOOLDIR)/slice 5 500 ref.ra ref1.ra ;\
	$(TOOLDIR)/slice 5 501 ref.ra ref2.ra ;\
	$(TOOLDIR)/join 5 ref1.ra ref2.ra comb.ra ;\
	$(TOOLDIR)/avg 32 comb.ra avg.ra ;\
	$(TOOLDIR)/slice 5 250 signal.ra s.ra;\
	$(TOOLDIR)/nrmse -t 0.00001 avg.ra s.ra			    			;\
	$(TOOLDIR)/slice 5 998 ref.ra ref1.ra ;\
	$(TOOLDIR)/slice 5 999 ref.ra ref2.ra ;\
	$(TOOLDIR)/join 5 ref1.ra ref2.ra comb.ra ;\
	$(TOOLDIR)/avg 32 comb.ra avg.ra ;\
	$(TOOLDIR)/slice 5 499 signal.ra s.ra;\
	$(TOOLDIR)/nrmse -t 0.00001 avg.ra s.ra			    			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-signal-spoke-averaging-3: signal slice join avg nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/signal -I -F -r0.0041 -e0.00258 -f8 -n999 -1 3:3:1 -2 1:1:1 ref.ra	;\
	$(TOOLDIR)/signal -I -F -r0.0041 -e0.00258 -f8 -n333 --av-spokes 3 -1 3:3:1 -2 1:1:1 signal.ra	;\
	$(TOOLDIR)/slice 5 0 ref.ra ref1.ra ;\
	$(TOOLDIR)/slice 5 1 ref.ra ref2.ra ;\
	$(TOOLDIR)/slice 5 2 ref.ra ref3.ra ;\
	$(TOOLDIR)/join 5 ref1.ra ref2.ra ref3.ra comb.ra ;\
	$(TOOLDIR)/avg 32 comb.ra avg.ra ;\
	$(TOOLDIR)/slice 5 0 signal.ra s.ra;\
	$(TOOLDIR)/nrmse -t 0.00001 avg.ra s.ra			    			;\
	$(TOOLDIR)/slice 5 501 ref.ra ref1.ra ;\
	$(TOOLDIR)/slice 5 502 ref.ra ref2.ra ;\
	$(TOOLDIR)/slice 5 503 ref.ra ref3.ra ;\
	$(TOOLDIR)/join 5 ref1.ra ref2.ra ref3.ra comb.ra ;\
	$(TOOLDIR)/avg 32 comb.ra avg.ra ;\
	$(TOOLDIR)/slice 5 167 signal.ra s.ra;\
	$(TOOLDIR)/nrmse -t 0.00001 avg.ra s.ra			    			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-signal-spoke-averaging-2 tests/test-signal-spoke-averaging-3
