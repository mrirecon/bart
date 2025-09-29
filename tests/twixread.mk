TWIX_SingleRaid=$(AGUE_REF)/twix/TWIX_SingleRaid.dat
TWIX_MultiRaid=$(AGUE_REF)/twix/TWIX_MultiRaid.dat
TWIX_MPI=$(AGUE_REF)/twix/TWIX_MPI.dat
TWIX_VE=$(AGUE_REF)/twix/TWIX_VE.dat
TWIX_PMU=$(AGUE_REF)/twix/TWIX_PMU.dat

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

tests/test-twixread-pmu: twixread ${TWIX_PMU}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/twixread -A ${TWIX_PMU} ksp.ra pmu.ra			;\
	echo "3ce40a3777cf9c9924073696b2598bb8  ksp.ra" | md5sum -c		;\
	echo "f7628c4adaef57f6615f7ff7d85dab05  pmu.ra" | md5sum -c		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-twixread-pmu-chrono: twixread transpose nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/twixread -A    ${TWIX_PMU} ksp_ref.ra pmu_ref.ra		;\
	$(TOOLDIR)/twixread -A -C ${TWIX_PMU} ksp_chr.ra pmu_chr.ra		;\
	$(TOOLDIR)/transpose 1 10 ksp_chr.ra ksp_chr_re.ra			;\
	$(TOOLDIR)/nrmse -t 0 ksp_chr_re.ra ksp_ref.ra				;\
	$(TOOLDIR)/transpose 1 10 pmu_chr.ra pmu_chr_re.ra			;\
	$(TOOLDIR)/nrmse -t 0 pmu_chr_re.ra pmu_ref.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS_AGUE += tests/test-twixread tests/test-twixread-multiraid tests/test-twixread-mpi tests/test-twixread-ve tests/test-twixread-pmu tests/test-twixread-pmu-chrono

