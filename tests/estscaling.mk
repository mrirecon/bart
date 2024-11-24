
tests/test-estscaling: phantom estscaling pics ones fft fmac nrmse resize
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 64 64 col.ra								;\
	$(TOOLDIR)/phantom -x64 -k ksp.ra							;\
	$(TOOLDIR)/resize -c 0 48 1 48 ksp.ra cal.ra						;\
	$(TOOLDIR)/estscaling -i -x64:64:1 cal.ra scl.ra					;\
	$(TOOLDIR)/fft -i -u 3 ksp.ra fft.ra							;\
	$(TOOLDIR)/fmac fft.ra scl.ra fftscl.ra							;\
	$(TOOLDIR)/pics ksp.ra col.ra recscl.ra							;\
	$(TOOLDIR)/nrmse -t 1.e-6 recscl.ra fftscl.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

# Estimate scaling from rss image estimated with ncalib
tests/test-estscaling-ncalib-cart: phantom traj estscaling pics mip ncalib nrmse fft ones fmac rss
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x64 -k -s4 ksp.ra							;\
	$(TOOLDIR)/ncalib -i5 ksp.ra col.ra img.ra						;\
	$(TOOLDIR)/fft -u 7 img.ra ksp_scl.ra 							;\
	$(TOOLDIR)/estscaling -x64:64:1 -i ksp_scl.ra scl1.ra					;\
	$(TOOLDIR)/estscaling -x64:64:1 -i ksp.ra     scl2.ra					;\
	$(TOOLDIR)/nrmse -t 0.01 scl1.ra scl2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

#FIXME: Scaling is based on p90 so difference to mip is fine!
tests/test-estscaling-ncalib-cart2: phantom traj estscaling pics mip ncalib nrmse fft ones fmac rss
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x64 -k -s4 ksp.ra							;\
	$(TOOLDIR)/ncalib -i10 ksp.ra col.ra img.ra						;\
	$(TOOLDIR)/fft -u 7 img.ra ksp_scl.ra 							;\
	$(TOOLDIR)/estscaling -x64:64:1 -i ksp_scl.ra scl.ra					;\
	$(TOOLDIR)/fmac ksp.ra scl.ra ksp_nrm.ra						;\
	$(TOOLDIR)/pics -w1. ksp_nrm.ra col.ra rec.ra						;\
	$(TOOLDIR)/fmac col.ra rec.ra cim.ra							;\
	$(TOOLDIR)/rss 8 cim.ra rss.ra								;\
	$(TOOLDIR)/mip 7 rec.ra max.ra								;\
	$(TOOLDIR)/mip 7 rss.ra max_rss.ra							;\
	$(TOOLDIR)/ones 1 1 one.ra								;\
	$(TOOLDIR)/nrmse -t 0.3 one.ra max_rss.ra						;\
	$(TOOLDIR)/nrmse -t 0.4 one.ra max.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

#FIXME: Scaling is based on p90 so difference to mip is fine!
tests/test-estscaling-ncalib-noncart: phantom traj estscaling pics mip ncalib nrmse fft ones fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -x64 -y 64 -r trj.ra							;\
	$(TOOLDIR)/phantom -k -s4 -t trj.ra ksp.ra						;\
	$(TOOLDIR)/ncalib -i12 -t trj.ra ksp.ra col.ra img.ra					;\
	$(TOOLDIR)/fft -u 7 img.ra ksp_scl.ra 							;\
	$(TOOLDIR)/estscaling -x64:64:1 -i ksp_scl.ra scl.ra					;\
	$(TOOLDIR)/fmac ksp.ra scl.ra ksp_nrm.ra						;\
	$(TOOLDIR)/pics -w1. -t trj.ra ksp_nrm.ra col.ra rec.ra					;\
	$(TOOLDIR)/fmac col.ra rec.ra cim.ra							;\
	$(TOOLDIR)/rss 8 cim.ra rss.ra								;\
	$(TOOLDIR)/mip 7 rec.ra max.ra								;\
	$(TOOLDIR)/mip 7 rss.ra max_rss.ra							;\
	$(TOOLDIR)/ones 1 1 one.ra								;\
	$(TOOLDIR)/nrmse -t 0.3 one.ra max_rss.ra						;\
	$(TOOLDIR)/nrmse -t 0.45 one.ra max.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-estscaling
TESTS += tests/test-estscaling-ncalib-cart tests/test-estscaling-ncalib-cart2 tests/test-estscaling-ncalib-noncart
