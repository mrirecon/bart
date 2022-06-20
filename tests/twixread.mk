
TWIXFILE_SINGLE=/home/ague/archive/pha/2019/2019-03-25_Other-MU_Export/single_raid/meas_MID00311_FID65562_t1_fl2d.dat
TWIXFILE_MPI=/home/ague/data/T1825/T1825.dat
TWIXFILE_MULTI=/home/ague/archive/pha/2019/2019-03-25_Other-MU_Export/meas_MID00311_FID65562_t1_fl2d.dat
TWIXFILE_VE=/home/ague/data/SCBI_2022/2022-05-13_SCBI/MultiRaid/meas_MID00052_FID03520_localizer.dat

tests/test-twixread: twixread ${TWIXFILE_SINGLE}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -A ${TWIXFILE_SINGLE} ksp.ra				;\
	echo "c304bf3bb41c7571408e776ec1280870 ksp.ra" | md5sum -c			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-twixread-multiraid: twixread ${TWIXFILE_MULTI}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -A ${TWIXFILE_MULTI} ksp.ra					;\
	echo "c304bf3bb41c7571408e776ec1280870 ksp.ra" | md5sum -c			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-twixread-mpi: twixread ${TWIXFILE_MPI}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -M -x320 -r11 -s3 -n217 -c34 ${TWIXFILE_MPI} ksp.ra		;\
	echo "9ee480b68ed70a7388851277e209b38d ksp.ra" | md5sum -c			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-twixread-ve: twixread ${TWIXFILE_VE}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -A ${TWIXFILE_VE} ksp.ra		;\
	echo "d6553079149dd6d0689828546681a4da  ksp.ra" | md5sum -c			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS_AGUE += tests/test-twixread tests/test-twixread-multiraid tests/test-twixread-mpi tests/test-twixread-ve
