

$(TESTS_OUT)/exponentials.ra: index fmac scale zexp 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/index 0 128 i.ra							;\
	$(TOOLDIR)/index 6 32 i2.ra							;\
	$(TOOLDIR)/fmac i.ra i2.ra i3.ra						;\
	$(TOOLDIR)/scale -- -0.001 i3.ra i4.ra						;\
	$(TOOLDIR)/zexp i4.ra $@							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)


$(TESTS_OUT)/phantom-exp.ra: phantom fmac $(TESTS_OUT)/exponentials.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom x.ra								;\
	$(TOOLDIR)/fmac x.ra $(TESTS_OUT)/exponentials.ra $@				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

	
$(TESTS_OUT)/expon-basis.ra: squeeze svd transpose extract reshape $(TESTS_OUT)/exponentials.ra 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/squeeze $(TESTS_OUT)/exponentials.ra es.ra				;\
	$(TOOLDIR)/svd es.ra u.ra s.ra v.ra						;\
	$(TOOLDIR)/transpose 0 1 v.ra vT.ra 						;\
	$(TOOLDIR)/extract 1 0 4 vT.ra v2.ra						;\
	$(TOOLDIR)/reshape 99 1 1 32 4 v2.ra $@						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)


tests/test-pics-linsub: fft reshape ones delta repmat reshape fmac pics transpose nrmse $(TESTS_OUT)/phantom-exp.ra $(TESTS_OUT)/expon-basis.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/fft -u 7 $(TESTS_OUT)/phantom-exp.ra xek.ra 				;\
	$(TOOLDIR)/reshape 96 32 1 xek.ra xek2.ra					;\
	$(TOOLDIR)/ones 2 128 128 o.ra							;\
	$(TOOLDIR)/delta 8 33 8 p.ra							;\
	$(TOOLDIR)/repmat 1 16 p.ra p2.ra						;\
	$(TOOLDIR)/repmat 6 4 p2.ra p3.ra						;\
	$(TOOLDIR)/reshape 3 128 1 p3.ra p4.ra						;\
	$(TOOLDIR)/reshape 96 32 1 p4.ra p5.ra						;\
	$(TOOLDIR)/fmac xek2.ra p5.ra xek3.ra						;\
	$(TOOLDIR)/repmat 1 128 p5.ra p6.ra						;\
	$(TOOLDIR)/pics -w1. -B$(TESTS_OUT)/expon-basis.ra -pp6.ra xek3.ra o.ra x2.ra	;\
	$(TOOLDIR)/fmac -s 64 x2.ra $(TESTS_OUT)/expon-basis.ra xx.ra			;\
	$(TOOLDIR)/transpose 5 6 xx.ra xxT.ra						;\
	$(TOOLDIR)/nrmse -t 0.001 $(TESTS_OUT)/phantom-exp.ra xxT.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pics-linsub-noncart: traj scale nufft reshape ones delta repmat reshape fmac pics transpose nrmse $(TESTS_OUT)/phantom-exp.ra $(TESTS_OUT)/expon-basis.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/traj -r -x256 -y128 t.ra								;\
	$(TOOLDIR)/scale 0.5 t.ra t2.ra									;\
	$(TOOLDIR)/nufft t2.ra $(TESTS_OUT)/phantom-exp.ra xek.ra					;\
	$(TOOLDIR)/reshape 96 32 1 xek.ra xek2.ra							;\
	$(TOOLDIR)/ones 2 128 128 o.ra									;\
	$(TOOLDIR)/delta 8 33 8 p.ra									;\
	$(TOOLDIR)/repmat 1 16 p.ra p2.ra								;\
	$(TOOLDIR)/repmat 6 4 p2.ra p3.ra								;\
	$(TOOLDIR)/reshape 7 1 1 128 p3.ra p4.ra							;\
	$(TOOLDIR)/reshape 96 32 1 p4.ra p5.ra								;\
	$(TOOLDIR)/fmac xek2.ra p5.ra xek3.ra								;\
	$(TOOLDIR)/repmat 1 256 p5.ra p6.ra								;\
	$(TOOLDIR)/scale 2 $(TESTS_OUT)/expon-basis.ra basis.ra						;\
	$(TOOLDIR)/pics -S -RT:7:0:0.02 -i100 -e -w1. -Bbasis.ra -pp6.ra -t t2.ra xek3.ra o.ra x2.ra	;\
	$(TOOLDIR)/fmac -s 64 x2.ra $(TESTS_OUT)/expon-basis.ra xx.ra					;\
	$(TOOLDIR)/transpose 5 6 xx.ra xxT.ra								;\
	$(TOOLDIR)/scale 2. xxT.ra xxS.ra								;\
	$(TOOLDIR)/nrmse -t 0.05 $(TESTS_OUT)/phantom-exp.ra xxS.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-pics-linsub tests/test-pics-linsub-noncart

