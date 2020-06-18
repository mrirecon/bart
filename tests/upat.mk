


tests/test-upat-traj: traj phantom nufft upat squeeze fmac fft nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -y16 -x16 -a2 t.ra						;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra						;\
	$(TOOLDIR)/nufft -a -t t.ra k.ra x.ra 						;\
	$(TOOLDIR)/upat -Y16 -Z16 -z2 u.ra 						;\
	$(TOOLDIR)/squeeze u.ra u2.ra 							;\
	$(TOOLDIR)/phantom -x16 -k k2.ra						;\
	$(TOOLDIR)/fmac k2.ra u2.ra ku.ra 						;\
	$(TOOLDIR)/fft -u -i 7 ku.ra x2.ra 						;\
	$(TOOLDIR)/nrmse -t 0.06 x.ra x2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
	

TESTS += tests/test-upat-traj

