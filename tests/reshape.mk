
tests/test-reshape: phantom reshape repmat slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom x.ra								;\
	$(TOOLDIR)/reshape 7 256 1 64 x.ra x2.ra					;\
	$(TOOLDIR)/repmat 1 5 x2.ra x3.ra						;\
	$(TOOLDIR)/reshape 5 128 128 x3.ra x4.ra					;\
	$(TOOLDIR)/slice 1 2 x4.ra x5.ra						;\
	$(TOOLDIR)/reshape 7 128 128 1 x5.ra x6.ra					;\
	$(TOOLDIR)/nrmse -t 0. x.ra x6.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reshape-mpi: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(ROOTDIR)/bart phantom -s 6 x							;\
	$(ROOTDIR)/bart noise x x_noise							;\
	$(ROOTDIR)/bart reshape 7 256 1 64 x x2						;\
	$(ROOTDIR)/bart reshape 7 256 1 64 x_noise x_noise2				;\
	$(ROOTDIR)/bart join 15 x_noise2 x_noise2 x2 x2_ref				;\
	$(ROOTDIR)/bart join 15 x_noise x_noise x x_p					;\
	mpirun -n 4 $(ROOTDIR)/bart -l 32768 -e 3 reshape 7 256 1 64 x_p x_p2		;\
	$(ROOTDIR)/bart nrmse -t 0. x2_ref x_p2						;\
	rm *.cfl ; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-reshape

TESTS_MPI += tests/test-reshape-mpi


