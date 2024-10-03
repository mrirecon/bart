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

tests/test-signal-se: signal ones scale slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/signal -S -i 0.5 -n 3 -e 0.01 -r 100 -1 3:3:1 -2 0.01:0.01:1 sig.ra	;\
	$(TOOLDIR)/ones 1 1 one.ra				;\
	$(TOOLDIR)/slice 5 0 sig.ra t0.ra ;\
	$(TOOLDIR)/nrmse -t 0.00001 t0.ra one.ra			    	;\
	$(TOOLDIR)/scale -- 0.367879 one.ra ref1.ra			    	;\
	$(TOOLDIR)/slice 5 1 sig.ra t1.ra ;\
	$(TOOLDIR)/nrmse -t 0.00001 t1.ra ref1.ra			    	;\
	$(TOOLDIR)/scale -- 0.135335 one.ra ref2.ra			    	;\
	$(TOOLDIR)/slice 5 2 sig.ra t2.ra ;\
	$(TOOLDIR)/nrmse -t 0.00001 t2.ra ref2.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-signal-fse: signal nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/signal -S -n 8 -e 0.01 -r 100 -1 1:3:10 -2 0.01:0.3:10 sig.ra	;\
	$(TOOLDIR)/signal -Z -n 8 -e 0.01 -r 100 -f 180 -1 1:3:10 -2 0.01:0.3:10 sig2.ra	;\
	$(TOOLDIR)/nrmse -t 0.00001 sig.ra sig2.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-signal-fse-epg: signal epg extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/signal -Z -n 8 -e 0.01 -r 12 -f 120 -1 1:1:1 -2 0.1:0.1:1 sig.ra	;\
	$(TOOLDIR)/epg -C -n 7 -e 0.01 -r 12 -f 120 -1 1 -2 0.1 sig2.ra	;\
	$(TOOLDIR)/extract 5 1 8 sig.ra sig3.ra					;\
	$(TOOLDIR)/nrmse -t 0.00001 sig2.ra sig3.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-signal-ir-se: signal ones scale slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/signal -S -I -i 0.5 -n 3 -e 0.0001 -r 100 -1 3:3:1 -2 1000:1000:1 sig.ra	;\
	$(TOOLDIR)/ones 1 1 one.ra				;\
	$(TOOLDIR)/scale -- -1 one.ra ref0.ra			    	;\
	$(TOOLDIR)/slice 5 0 sig.ra t0.ra ;\
	$(TOOLDIR)/nrmse -t 0.00001 t0.ra ref0.ra			    	;\
	$(TOOLDIR)/scale -- -0.692963 one.ra ref1.ra			    	;\
	$(TOOLDIR)/slice 5 1 sig.ra t1.ra ;\
	$(TOOLDIR)/nrmse -t 0.00001 t1.ra ref1.ra			    	;\
	$(TOOLDIR)/scale -- -0.4330626 one.ra ref2.ra			    	;\
	$(TOOLDIR)/slice 5 2 sig.ra t2.ra ;\
	$(TOOLDIR)/nrmse -t 0.00001 t2.ra ref2.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-signal-se-single: signal slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/signal -S -i 0.5 -n 3 -e 0.01 -r 100 -1 3:3:1 -2 0.01:0.01:1 sig.ra	;\
	$(TOOLDIR)/signal -S -i 0.5 -n 1 -e 0.01 -r 100 -1 3:3:1 -2 0.01:0.01:1 sig2.ra	;\
	$(TOOLDIR)/signal -S -i 0.5 -n 1 -e 0.02 -r 100 -1 3:3:1 -2 0.01:0.01:1 sig3.ra	;\
	$(TOOLDIR)/slice 5 1 sig.ra t1.ra ;\
	$(TOOLDIR)/nrmse -t 0.00001 t1.ra sig2.ra			    	;\
	$(TOOLDIR)/slice 5 2 sig.ra t2.ra ;\
	$(TOOLDIR)/nrmse -t 0.00001 t2.ra sig3.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-signal-ir-se-single: signal slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/signal -S -I -i 0.5 -n 3 -e 0.0001 -r 100 -1 3:3:1 -2 1000:1000:1 sig.ra	;\
	$(TOOLDIR)/signal -S -I -i 0.5 -n 1 -e 0.0001 -r 100 -1 3:3:1 -2 1000:1000:1 sig2.ra	;\
	$(TOOLDIR)/signal -S -I -i 1 -n 1 -e 0.0001 -r 100 -1 3:3:1 -2 1000:1000:1 sig3.ra	;\
	$(TOOLDIR)/slice 5 1 sig.ra t1.ra ;\
	$(TOOLDIR)/nrmse -t 0.00001 t1.ra sig2.ra			    	;\
	$(TOOLDIR)/slice 5 2 sig.ra t2.ra ;\
	$(TOOLDIR)/nrmse -t 0.00001 t2.ra sig3.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-signal-LL-short-TR-approx: signal slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/signal -I -F -r0.0001 -f8 -n1000 -1 3:3:1 -2 1:1:1 signal.ra	;\
	$(TOOLDIR)/signal -I -F -r0.0001 -f8 -n1000 -1 3:3:1 -2 1:1:1 --short-TR-LL-approx signal2.ra	;\
	$(TOOLDIR)/nrmse -t 0.0001 signal.ra signal2.ra			    			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-signal-spoke-averaging-2 tests/test-signal-spoke-averaging-3
TESTS += tests/test-signal-se tests/test-signal-ir-se
TESTS += tests/test-signal-fse tests/test-signal-fse-epg
TESTS += tests/test-signal-se-single tests/test-signal-ir-se-single
TESTS += tests/test-signal-LL-short-TR-approx