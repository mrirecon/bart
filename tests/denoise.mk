

tests/test-ictv-denoise: phantom noise denoise nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/phantom -s2 x.ra									;\
	$(TOOLDIR)/noise -n 1000000. x.ra xn.ra								;\
	$(TOOLDIR)/denoise -w1. -i30 -C5 -u0.1 --tvscales 2:2:5 --tvscales2 1:1:0.5 -S -RC:67:0:750. xn.ra xg.ra		;\
	$(TOOLDIR)/nrmse -t 0.034 xg.ra x.ra								;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-ictgv-denoise: phantom noise denoise nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/phantom -s2 x.ra									;\
	$(TOOLDIR)/noise -n 1000000. x.ra xn.ra								;\
	$(TOOLDIR)/denoise -w1. -i30 -C5 -u0.1 --tvscales 2:2:5 --tvscales2 1:1:0.5 -S -RV:67:0:750. xn.ra xg.ra		;\
	$(TOOLDIR)/nrmse -t 0.040 xg.ra x.ra								;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ictv-denoise2: phantom slice vec transpose fmac repmat noise saxpy denoise nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/phantom -N2 -b tb.ra									;\
	$(TOOLDIR)/slice 6 0 tb.ra t1.ra								;\
	$(TOOLDIR)/slice 6 1 tb.ra t2.ra								;\
	$(TOOLDIR)/vec 5 15 30 60 20 v.ra								;\
	$(TOOLDIR)/transpose 0 6 v.ra v.ra								;\
	$(TOOLDIR)/fmac t2.ra v.ra r.ra									;\
	$(TOOLDIR)/repmat 2 5 r.ra r.ra									;\
	$(TOOLDIR)/noise -n 5 r.ra rn.ra								;\
	$(TOOLDIR)/transpose 6 2 v.ra v.ra								;\
	$(TOOLDIR)/fmac t1.ra v.ra t1.ra								;\
	$(TOOLDIR)/repmat 6 5 t1.ra	t1.ra								;\
	$(TOOLDIR)/saxpy -- 10 t1.ra rn.ra on.ra							;\
	$(TOOLDIR)/saxpy -- 10 t1.ra r.ra o.ra								;\
	$(TOOLDIR)/denoise -w1. -i20 -C5 -u0.1 -S  --tvscales 1:1:0.5:2 --tvscales2 1:1:2:0.5 -RC:71:0:2 on.ra o2.ra	;\
	$(TOOLDIR)/nrmse -t 0.07 o.ra o2.ra								;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-ictgv-denoise2: phantom slice vec transpose fmac noise denoise repmat saxpy nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/phantom -N2 -b tb.ra									;\
	$(TOOLDIR)/slice 6 0 tb.ra t1.ra								;\
	$(TOOLDIR)/slice 6 1 tb.ra t2.ra								;\
	$(TOOLDIR)/vec 5 15 30 60 20 v.ra								;\
	$(TOOLDIR)/transpose 0 6 v.ra v.ra								;\
	$(TOOLDIR)/fmac t2.ra v.ra r.ra									;\
	$(TOOLDIR)/repmat 2 5 r.ra r.ra									;\
	$(TOOLDIR)/noise -n 5 r.ra rn.ra								;\
	$(TOOLDIR)/transpose 6 2 v.ra v.ra								;\
	$(TOOLDIR)/fmac t1.ra v.ra t1.ra								;\
	$(TOOLDIR)/repmat 6 5 t1.ra	t1.ra								;\
	$(TOOLDIR)/saxpy -- 10 t1.ra rn.ra on.ra							;\
	$(TOOLDIR)/saxpy -- 10 t1.ra r.ra o.ra								;\
	$(TOOLDIR)/denoise -w1. -i20 -C5 -u0.1 -S  --tvscales 1:1:0.5:2 --tvscales2 1:1:2:0.5 -RV:71:0:2 on.ra o2.ra	;\
	$(TOOLDIR)/nrmse -t 0.06 o.ra o2.ra								;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-ictv-denoise tests/test-ictgv-denoise tests/test-ictv-denoise2

TESTS_SLOW += tests/test-ictgv-denoise2


