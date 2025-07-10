
tests/test-affine-rigid: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP) ; export BART_TOOLBOX_DIR=$(ROOTDIR)	;\
	$(ROOTDIR)/bart traj -x64 -y64 t1a							;\
	$(ROOTDIR)/bart scale -- 0.8 t1a t1							;\
	$(ROOTDIR)/bart traj -x64 -y64 -R5 t2a							;\
	$(ROOTDIR)/bart scale -- 0.8 t2a t2							;\
	$(ROOTDIR)/bart phantom -k -t t1 k1							;\
	$(ROOTDIR)/bart phantom -k -t t2 kp							;\
	$(ROOTDIR)/bart traj -x64 -y64 t1							;\
	$(ROOTDIR)/bart fovshift -s 0.05:-0.02:0 -t t1 kp k2					;\
	$(ROOTDIR)/bart nufft -a t1 k1 ph1							;\
	$(ROOTDIR)/bart nufft -a t1 k2 ph2							;\
	$(ROOTDIR)/bart affinereg -R ph1 ph2 aff						;\
	$(ROOTDIR)/scripts/affine_kspace.sh k2 t1 aff k3 t3					;\
	$(ROOTDIR)/bart nufft -a -x64:64:1 t3 k3 ph3						;\
	$(ROOTDIR)/bart interpolate -A 7 ph2 aff ph4						;\
	$(ROOTDIR)/bart nrmse -t 0.1 ph3 ph1							;\
	$(ROOTDIR)/bart nrmse -t 0.3 ph4 ph1							;\
	rm *.{cfl,hdr} ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-affine-affine: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP) ; export BART_TOOLBOX_DIR=$(ROOTDIR)	;\
	$(ROOTDIR)/bart traj -x64 -y64 t1a							;\
	$(ROOTDIR)/bart scale -- 0.8 t1a t1							;\
	$(ROOTDIR)/bart traj -x64 -y64 -R5 t2a							;\
	$(ROOTDIR)/bart scale -- 0.9 t2a t2							;\
	$(ROOTDIR)/bart phantom -k -t t1 k1							;\
	$(ROOTDIR)/bart phantom -k -t t2 kp							;\
	$(ROOTDIR)/bart traj -x64 -y64 t1							;\
	$(ROOTDIR)/bart fovshift -s 0.05:-0.02:0 -t t1 kp k2					;\
	$(ROOTDIR)/bart nufft -a t1 k1 ph1							;\
	$(ROOTDIR)/bart nufft -a t1 k2 ph2							;\
	$(ROOTDIR)/bart affinereg -A ph1 ph2 aff						;\
	$(ROOTDIR)/scripts/affine_kspace.sh k2 t1 aff k3 t3					;\
	$(ROOTDIR)/bart nufft -a -x64:64:1 t3 k3 ph3						;\
	$(ROOTDIR)/bart interpolate -A 7 ph2 aff ph4						;\
	$(ROOTDIR)/bart nrmse -s -t 0.1 ph3 ph1							;\
	$(ROOTDIR)/bart nrmse -s -t 0.3 ph4 ph1							;\
	rm *.{cfl,hdr} ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estmotion: traj phantom fovshift nufft estmotion interpolate nrmse scale ones pics
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -x64 -y64 t1a							;\
	$(TOOLDIR)/scale -- 0.8 t1a t1							;\
	$(TOOLDIR)/traj -x64 -y64 -R5 t2a						;\
	$(TOOLDIR)/scale -- 0.8 t2a t2							;\
	$(TOOLDIR)/phantom -k -t t1 k1							;\
	$(TOOLDIR)/phantom -k -t t2 kp							;\
	$(TOOLDIR)/traj -x64 -y64 t1							;\
	$(TOOLDIR)/fovshift -s 0.05:-0.02:0 -t t1 kp k2					;\
	$(TOOLDIR)/nufft -a t1 k1 ph1							;\
	$(TOOLDIR)/nufft -a t1 k2 ph2							;\
	$(TOOLDIR)/estmotion 3 ph1 ph2 disp idisp					;\
	$(TOOLDIR)/interpolate -D -C 3 ph2 disp ph3 					;\
	$(TOOLDIR)/nrmse -t 0.25 ph3 ph1						;\
	$(TOOLDIR)/ones 2 64 64 o		 					;\
	$(TOOLDIR)/pics --motion-field idisp -RT:3:0:0.001 -t t1 k2 o ph4		;\
	$(TOOLDIR)/nrmse -s -t 0.25 ph4 ph1						;\
	rm *.{cfl,hdr} ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estmotion-optical-flow-ksp: traj phantom fovshift nufft estmotion interpolate nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -x64 -y64 t2							;\
	$(TOOLDIR)/phantom -k -t t2 kp							;\
	$(TOOLDIR)/fovshift -s 0.02:-0.08:0 -t t2 kp k2					;\
	$(TOOLDIR)/nufft -a t2 kp ph1							;\
	$(TOOLDIR)/nufft -a t2 k2 ph2							;\
	$(TOOLDIR)/estmotion -r0.3 --optical-flow 3 ph1 ph2 disp			;\
	$(TOOLDIR)/interpolate -D -N 3 ph2 disp ph3					;\
	$(TOOLDIR)/nrmse -t 0.25 ph1 ph3						;\
	rm *.{cfl,hdr} ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estmotion-optical-flow-img: traj phantom circshift nufft estmotion interpolate nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -x64 -y64 t2							;\
	$(TOOLDIR)/phantom -k -t t2 kp							;\
	$(TOOLDIR)/nufft -a t2 kp ph1							;\
	$(TOOLDIR)/circshift 1 5 ph1 ph2						;\
	$(TOOLDIR)/estmotion -r0.3 --optical-flow 3 ph1 ph2 disp			;\
	$(TOOLDIR)/interpolate -D -N 3 ph2 disp ph3					;\
	$(TOOLDIR)/nrmse -t 0.03 ph1 ph3						;\
	rm *.{cfl,hdr} ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-affine-rigid
TESTS += tests/test-estmotion-optical-flow-ksp tests/test-estmotion
TESTS += tests/test-estmotion-optical-flow-img

TESTS_SLOW += tests/test-affine-affine

