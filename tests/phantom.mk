
$(TESTS_OUT)/shepplogan.ra: phantom
	$(TOOLDIR)/phantom $@

$(TESTS_OUT)/shepplogan_ksp.ra: phantom
	$(TOOLDIR)/phantom -k $@

$(TESTS_OUT)/shepplogan_coil.ra: phantom
	$(TOOLDIR)/phantom -s8 $@

$(TESTS_OUT)/shepplogan_coil_ksp.ra: phantom
	$(TOOLDIR)/phantom -s8 -k $@

$(TESTS_OUT)/coils.ra: phantom
	$(TOOLDIR)/phantom -S8 $@


tests/test-phantom-ksp: fft nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/fft -i 7 $(TESTS_OUT)/shepplogan_ksp.ra shepplogan_img.ra			;\
	$(TOOLDIR)/nrmse -t 0.22 $(TESTS_OUT)/shepplogan.ra shepplogan_img.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-phantom-noncart: traj phantom reshape nrmse $(TESTS_OUT)/shepplogan_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj traj.ra									;\
	$(TOOLDIR)/phantom -k -t traj.ra shepplogan_ksp2.ra					;\
	$(TOOLDIR)/reshape 7 128 128 1 shepplogan_ksp2.ra shepplogan_ksp3.ra			;\
	$(TOOLDIR)/nrmse -t 0. $(TESTS_OUT)/shepplogan_ksp.ra shepplogan_ksp3.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP) 
	touch $@


tests/test-phantom-coil: fmac nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_coil.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/fmac $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/coils.ra sl_coil2.ra		;\
	$(TOOLDIR)/nrmse -t 0. $(TESTS_OUT)/shepplogan_coil.ra sl_coil2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-phantom-ksp-coil: fft nrmse $(TESTS_OUT)/shepplogan_coil.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/fft -i 7 $(TESTS_OUT)/shepplogan_coil_ksp.ra shepplogan_cimg.ra		;\
	$(TOOLDIR)/nrmse -t 0.22 $(TESTS_OUT)/shepplogan_coil.ra shepplogan_cimg.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-phantom-bart: fft nrmse phantom
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -B -k k.ra								;\
	$(TOOLDIR)/fft -i 3 k.ra x.ra								;\
	$(TOOLDIR)/phantom -B r.ra								;\
	$(TOOLDIR)/nrmse -t 0.21 r.ra x.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-phantom-bart-basis: nrmse phantom fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -B -k k0.ra								;\
	$(TOOLDIR)/phantom -B -b -k k1.ra							;\
	$(TOOLDIR)/fmac -s 64 k1.ra k2.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 k0.ra k2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-phantom-basis: nrmse phantom fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -T -k k0.ra								;\
	$(TOOLDIR)/phantom -T -b -k k1.ra							;\
	$(TOOLDIR)/fmac -s 64 k1.ra k2.ra							;\
	$(TOOLDIR)/nrmse -t 0. k0.ra k2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-phantom-random-tubes: nrmse phantom fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -N 5 -k k0.ra							;\
	$(TOOLDIR)/phantom -N 5 -b -k k1.ra							;\
	$(TOOLDIR)/fmac -s 64 k1.ra k2.ra							;\
	$(TOOLDIR)/nrmse -t 0. k0.ra k2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-NIST: fft nrmse phantom
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom --NIST -k k.ra								;\
	$(TOOLDIR)/fft -i 3 k.ra x.ra								;\
	$(TOOLDIR)/phantom --NIST r.ra								;\
	$(TOOLDIR)/nrmse -t 0.14 r.ra x.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-NIST-basis: nrmse phantom fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom --NIST -k k0.ra							;\
	$(TOOLDIR)/phantom --NIST -b 11 -k k1.ra						;\
	$(TOOLDIR)/fmac -s 64 k1.ra k2.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 k0.ra k2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-rotation-tubes: phantom flip nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 21 -T o.ra					;\
	$(TOOLDIR)/flip 3 o.ra of.ra								;\
	$(TOOLDIR)/phantom -x 21 -T --rotation-angle 180 r.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 of.ra r.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-rotation-tubes-kspace: phantom flip nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 21 -k -T o.ra					;\
	$(TOOLDIR)/flip 3 o.ra of.ra								;\
	$(TOOLDIR)/phantom -x 21 -k -T --rotation-angle 180 r.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 of.ra r.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-rotation-tubes-basis: phantom flip nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 21 -b -T o.ra					;\
	$(TOOLDIR)/flip 3 o.ra of.ra								;\
	$(TOOLDIR)/phantom -x 21 -b -T --rotation-angle 180 r.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 of.ra r.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-rotation-NIST: phantom flip nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 21 --NIST o.ra					;\
	$(TOOLDIR)/flip 3 o.ra of.ra								;\
	$(TOOLDIR)/phantom -x 21 --NIST --rotation-angle 180 r.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 of.ra r.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-rotation-NIST-basis: phantom flip nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 21 --NIST -b o.ra					;\
	$(TOOLDIR)/flip 3 o.ra of.ra								;\
	$(TOOLDIR)/phantom -x 21 --NIST -b --rotation-angle 180 r.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 of.ra r.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-rotation-NIST-kspace: phantom flip nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 21 --NIST -k o.ra					;\
	$(TOOLDIR)/flip 3 o.ra of.ra								;\
	$(TOOLDIR)/phantom -x 21 --NIST -k --rotation-angle 180 r.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 of.ra r.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-rotation-tubes-multistep: phantom slice flip nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 21 -T --rotation-steps 4 --rotation-angle 90 o.ra					;\
	$(TOOLDIR)/slice 10 0 o.ra o2.ra							;\
	$(TOOLDIR)/flip 3 o2.ra o2f.ra								;\
	$(TOOLDIR)/slice 10 2 o.ra o3.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 o2f.ra o3.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-rotation-tubes-kspace-multistep: phantom slice flip nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 21 -k -T --rotation-steps 4 --rotation-angle 90 o.ra					;\
	$(TOOLDIR)/slice 10 0 o.ra o2.ra							;\
	$(TOOLDIR)/flip 3 o2.ra o2f.ra								;\
	$(TOOLDIR)/slice 10 2 o.ra o3.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 o2f.ra o3.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-rotation-tubes-basis-multistep: phantom fmac slice flip nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 21 -k -T k.ra					;\
	$(TOOLDIR)/phantom -x 21 -k -b -T --rotation-steps 4 --rotation-angle 90 k2.ra					;\
	$(TOOLDIR)/fmac -s 64 k2.ra o.ra							;\
	$(TOOLDIR)/slice 10 1 o.ra o2.ra							;\
	$(TOOLDIR)/flip 3 k.ra kf.ra								;\
	$(TOOLDIR)/nrmse -t 0.000001 kf.ra o2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-rotation-NIST-multistep: phantom slice flip nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 21 --NIST --rotation-steps 4 --rotation-angle 90 o.ra					;\
	$(TOOLDIR)/slice 10 0 o.ra o2.ra							;\
	$(TOOLDIR)/flip 3 o2.ra o2f.ra								;\
	$(TOOLDIR)/slice 10 2 o.ra o3.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 o2f.ra o3.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-rotation-NIST-kspace-multistep: phantom slice flip nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 21 -k --NIST --rotation-steps 4 --rotation-angle 90 o.ra					;\
	$(TOOLDIR)/slice 10 0 o.ra o2.ra							;\
	$(TOOLDIR)/flip 3 o2.ra o2f.ra								;\
	$(TOOLDIR)/slice 10 2 o.ra o3.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 o2f.ra o3.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-rotation-NIST-basis-multistep: phantom fmac slice flip nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 21 -k --NIST k.ra					;\
	$(TOOLDIR)/phantom -x 21 -k -b --NIST --rotation-steps 4 --rotation-angle 90 k2.ra					;\
	$(TOOLDIR)/fmac -s 64 k2.ra o.ra							;\
	$(TOOLDIR)/slice 10 1 o.ra o2.ra							;\
	$(TOOLDIR)/flip 3 k.ra kf.ra								;\
	$(TOOLDIR)/nrmse -t 0.000001 kf.ra o2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-SONAR: fft nrmse phantom
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom --SONAR -k k.ra								;\
	$(TOOLDIR)/fft -i 3 k.ra x.ra								;\
	$(TOOLDIR)/phantom --SONAR r.ra								;\
	$(TOOLDIR)/nrmse -t 0.14 r.ra x.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-SONAR-basis: nrmse phantom fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom --SONAR -k k0.ra							;\
	$(TOOLDIR)/phantom --SONAR -b 11 -k k1.ra						;\
	$(TOOLDIR)/fmac -s 64 k1.ra k2.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 k0.ra k2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-rotation-SONAR: phantom flip nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 21 --SONAR o.ra					;\
	$(TOOLDIR)/flip 3 o.ra of.ra								;\
	$(TOOLDIR)/phantom -x 21 --SONAR --rotation-angle 180 r.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 of.ra r.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-phantom-rotation-SONAR-multistep: phantom slice flip nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 21 --SONAR --rotation-steps 4 --rotation-angle 90 o.ra					;\
	$(TOOLDIR)/slice 10 0 o.ra o2.ra							;\
	$(TOOLDIR)/flip 3 o2.ra o2f.ra								;\
	$(TOOLDIR)/slice 10 2 o.ra o3.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 o2f.ra o3.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-phantom-ksp tests/test-phantom-noncart tests/test-phantom-coil tests/test-phantom-ksp-coil
TESTS += tests/test-phantom-bart tests/test-phantom-bart-basis
TESTS += tests/test-phantom-basis tests/test-phantom-random-tubes
TESTS += tests/test-phantom-NIST tests/test-phantom-NIST-basis
TESTS += tests/test-phantom-rotation-tubes tests/test-phantom-rotation-tubes-kspace tests/test-phantom-rotation-tubes-basis
TESTS += tests/test-phantom-rotation-NIST tests/test-phantom-rotation-NIST-kspace tests/test-phantom-rotation-NIST-basis
TESTS += tests/test-phantom-rotation-tubes-multistep tests/test-phantom-rotation-tubes-kspace-multistep tests/test-phantom-rotation-tubes-basis-multistep
TESTS += tests/test-phantom-rotation-NIST-multistep tests/test-phantom-rotation-NIST-kspace-multistep tests/test-phantom-rotation-NIST-basis-multistep
TESTS += tests/test-phantom-SONAR tests/test-phantom-SONAR-basis tests/test-phantom-rotation-SONAR tests/test-phantom-rotation-SONAR-multistep
