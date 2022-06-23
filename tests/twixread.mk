TWIX_SingleRaid=$(AGUE_TWIX_REF)/TWIX_SingleRaid.dat
TWIX_MultiRaid=$(AGUE_TWIX_REF)/TWIX_MultiRaid.dat
TWIX_MPI=$(AGUE_TWIX_REF)/TWIX_MPI.dat
TWIX_VE=$(AGUE_TWIX_REF)/TWIX_VE.dat

tests/test-twixread: twixread ${TIWX_SingleRaid}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -A ${TWIX_SingleRaid} ksp.ra				;\
	echo "c304bf3bb41c7571408e776ec1280870 *ksp.ra" | md5sum -c			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-twixread-multiraid: twixread ${TWIX_MultiRaid}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -A ${TWIX_MultiRaid} ksp.ra					;\
	echo "c304bf3bb41c7571408e776ec1280870 *ksp.ra" | md5sum -c			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-twixread-mpi: twixread ${TWIX_MPI}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -M -x320 -r11 -s3 -n217 -c34 ${TWIX_MPI} ksp.ra		;\
	echo "9ee480b68ed70a7388851277e209b38d *ksp.ra" | md5sum -c			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-twixread-ve: twixread ${TWIX_VE}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -A ${TWIX_VE} ksp.ra					;\
	echo "ca5478fa82f1fc1052358c376c41d5e2  ksp.ra" | md5sum -c 			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS_AGUE += tests/test-twixread tests/test-twixread-multiraid tests/test-twixread-mpi tests/test-twixread-ve

