
tests/test-psf: traj psf nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -r trj.ra						;\
	$(TOOLDIR)/psf --oversampled trj.ra psf1.ra				;\
	$(TOOLDIR)/psf --oversampled-decomposed trj.ra psf2.ra			;\
	$(TOOLDIR)/nrmse -t 1.5e-5 psf1.ra psf2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-psf-odd: traj psf nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x127 trj.ra						;\
	$(TOOLDIR)/psf --oversampled trj.ra psf1.ra				;\
	$(TOOLDIR)/psf --oversampled-decomposed trj.ra psf2.ra			;\
	$(TOOLDIR)/nrmse -t 1.e-3 psf1.ra psf2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-psf
TESTS += tests/test-psf-odd