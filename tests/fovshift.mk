


tests/test-fovshift: phantom fft fovshift circshift nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom x.ra									;\
	$(TOOLDIR)/fft -u 7 x.ra k.ra								;\
	$(TOOLDIR)/fovshift -s0:0.5:0. k.ra ks.ra						;\
	$(TOOLDIR)/fft -u -i 7 ks.ra xs.ra							;\
	$(TOOLDIR)/circshift 1 64 x.ra xc.ra							;\
	$(TOOLDIR)/nrmse -t 0.00001 xc.ra xs.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-fovshift-nc: traj scale phantom fovshift nufft fft nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -x256 -r -y201 t1.ra							;\
	$(TOOLDIR)/scale 0.5 t1.ra t.ra								;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/fovshift -s0:0.1:0. -t t.ra k.ra ks.ra					;\
	$(TOOLDIR)/nufft -i -t t.ra ks.ra xs.ra							;\
	$(TOOLDIR)/phantom -k k2.ra								;\
	$(TOOLDIR)/fovshift -s0:0.1:0. k2.ra k2s.ra						;\
	$(TOOLDIR)/fft -u -i 7 k2s.ra x2s.ra							;\
	$(TOOLDIR)/nrmse -t 0.06 x2s.ra xs.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-fovshift tests/test-fovshift-nc

