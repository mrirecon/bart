
tests/test-calc-zsqrt: ones scale calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- -45.+28i o.ra o2.ra							;\
	$(TOOLDIR)/calc zsqrt o2.ra o3.ra							;\
	$(TOOLDIR)/scale -- 2.+7i o.ra r.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 o3.ra r.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-calc-zconj: ones scale calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- -45.+28i o.ra o2.ra							;\
	$(TOOLDIR)/calc zconj o2.ra o3.ra							;\
	$(TOOLDIR)/scale -- -45.-28i o.ra r.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 o3.ra r.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-calc-zreal: ones scale calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- -45.+28i o.ra o2.ra							;\
	$(TOOLDIR)/calc zreal o2.ra o3.ra							;\
	$(TOOLDIR)/scale -- -45. o.ra r.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 o3.ra r.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-calc-zimag: ones scale calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- -45.+28i o.ra o2.ra							;\
	$(TOOLDIR)/calc zimag o2.ra o3.ra							;\
	$(TOOLDIR)/scale -- +28i o.ra r.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 o3.ra r.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-calc-zarg: ones scale calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- 1.+1i o.ra o2.ra							;\
	$(TOOLDIR)/calc zarg o2.ra o3.ra							;\
	$(TOOLDIR)/scale -- 0.785398 o.ra r.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 o3.ra r.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-calc-zabs: ones scale calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- 1.+1i o.ra o2.ra							;\
	$(TOOLDIR)/calc zabs o2.ra o3.ra							;\
	$(TOOLDIR)/scale -- 1.414213 o.ra r.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 o3.ra r.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-calc-zphsr: ones scale calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- 1.+1i o.ra o2.ra							;\
	$(TOOLDIR)/calc zphsr o2.ra o3.ra							;\
	$(TOOLDIR)/scale -- 0.707107+0.707107i o.ra r.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 o3.ra r.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-calc-zlog: ones scale calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- 1.+1i o.ra o2.ra							;\
	$(TOOLDIR)/calc zlog o2.ra o3.ra							;\
	$(TOOLDIR)/scale -- 0.346573+0.785398i o.ra r.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 o3.ra r.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-calc-zexp: ones scale calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- 1.+1i o.ra o2.ra							;\
	$(TOOLDIR)/calc zexp o2.ra o3.ra 							;\
	$(TOOLDIR)/calc zlog o3.ra o4.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 o4.ra o2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-calc-zsin: ones scale calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- 1.+1i o.ra o2.ra							;\
	$(TOOLDIR)/calc zsin o2.ra o3.ra							;\
	$(TOOLDIR)/scale -- 1.298457+0.634963i o.ra r.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 o3.ra r.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-calc-zcos: ones scale calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- 1.+1i o.ra o2.ra							;\
	$(TOOLDIR)/calc zcos o2.ra o3.ra							;\
	$(TOOLDIR)/scale -- 0.833730-0.988897i o.ra r.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 o3.ra r.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-calc-zsinh: ones scale calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- 1.+1i o.ra o2.ra							;\
	$(TOOLDIR)/calc zsinh o2.ra o3.ra							;\
	$(TOOLDIR)/scale -- 0.634963+1.298457i o.ra r.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 o3.ra r.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-calc-zcosh: ones scale calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- 1.+1i o.ra o2.ra							;\
	$(TOOLDIR)/calc zcosh o2.ra o3.ra							;\
	$(TOOLDIR)/scale -- 0.833730+0.988897i o.ra r.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 o3.ra r.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-calc-zatanr: ones scale calc spow fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- 0.5 o.ra o2.ra							;\
	$(TOOLDIR)/calc zcos o2.ra o3.ra							;\
	$(TOOLDIR)/spow -- -1 o3.ra o4.ra							;\
	$(TOOLDIR)/calc zsin o2.ra o5.ra							;\
	$(TOOLDIR)/fmac o4.ra o5.ra o6.ra						;\
	$(TOOLDIR)/calc zatanr o6.ra o7.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 o2.ra o7.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-calc-zacosr: ones scale calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/scale -- 0.5 o.ra o2.ra							;\
	$(TOOLDIR)/calc zcos o2.ra o3.ra							;\
	$(TOOLDIR)/calc zacosr o3.ra o4.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 o2.ra o4.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-calc-zsqrt tests/test-calc-zconj
TESTS += tests/test-calc-zreal tests/test-calc-zimag tests/test-calc-zarg tests/test-calc-zabs
TESTS += tests/test-calc-zphsr
TESTS += tests/test-calc-zlog tests/test-calc-zexp
TESTS += tests/test-calc-zsin tests/test-calc-zcos
TESTS += tests/test-calc-zsinh tests/test-calc-zcosh
TESTS += tests/test-calc-zatanr tests/test-calc-zacosr