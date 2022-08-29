tests/test-sim-to-signal-irflash: sim cabs mip spow fmac scale signal nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq ir-flash,tr=0.0041,te=0.0025,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=8,bwtp=4 -1 3:3:1 -2 1:1:1 _simu.ra ;\
	$(TOOLDIR)/cabs _simu.ra _simu_abs.ra								;\
	$(TOOLDIR)/mip 32 _simu_abs.ra max.ra								;\
	$(TOOLDIR)/spow -- -1 max.ra scale.ra								;\
	$(TOOLDIR)/fmac _simu.ra scale.ra _simu2.ra 							;\
	$(TOOLDIR)/scale -- -1i _simu2.ra simu.ra							;\
	$(TOOLDIR)/signal -I -F -r0.0041 -e0.00258 -f8 -n1000 -1 3:3:1 -2 1:1:1 signal.ra		;\
	$(TOOLDIR)/nrmse -t 0.003 simu.ra signal.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-to-signal-flash: sim cabs mip spow fmac scale signal nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq flash,tr=0.0041,te=0.0025,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=8,bwtp=4 -1 3:3:1 -2 1:1:1 _simu.ra ;\
	$(TOOLDIR)/cabs _simu.ra _simu_abs.ra								;\
	$(TOOLDIR)/mip 32 _simu_abs.ra max.ra								;\
	$(TOOLDIR)/spow -- -1 max.ra scale.ra								;\
	$(TOOLDIR)/fmac _simu.ra scale.ra _simu2.ra 							;\
	$(TOOLDIR)/scale -- -1i _simu2.ra simu.ra							;\
	$(TOOLDIR)/signal -F -r0.0041 -e0.00258 -f8 -n1000 -1 3:3:1 -2 1:1:1 signal.ra		;\
	$(TOOLDIR)/nrmse -t 0.003 simu.ra signal.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-to-signal-irbSSFP: sim scale signal nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 _sim.ra ;\
	$(TOOLDIR)/scale -- -1i _sim.ra sim.ra							;\
	$(TOOLDIR)/signal -I -B -r0.0045 -e0.00225 -f45 -n1000 -1 3:3:1 -2 1:1:1 signal.ra		;\
	$(TOOLDIR)/nrmse -t 0.003 sim.ra signal.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-spoke-averaging-3: sim slice join avg nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 ref.ra	;\
	$(TOOLDIR)/sim --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=45,bwtp=4,av-spokes=3 -1 3:3:1 -2 1:1:1 signal.ra	;\
	$(TOOLDIR)/slice 5 0 ref.ra ref1.ra ;\
	$(TOOLDIR)/slice 5 1 ref.ra ref2.ra ;\
	$(TOOLDIR)/slice 5 2 ref.ra ref3.ra ;\
	$(TOOLDIR)/join 5 ref1.ra ref2.ra ref3.ra comb.ra ;\
	$(TOOLDIR)/avg 32 comb.ra avg.ra ;\
	$(TOOLDIR)/slice 5 0 signal.ra s.ra;\
	$(TOOLDIR)/nrmse -t 0.00001 avg.ra s.ra			    			;\
	$(TOOLDIR)/slice 5 501 ref.ra ref1.ra ;\
	$(TOOLDIR)/slice 5 502 ref.ra ref2.ra ;\
	$(TOOLDIR)/slice 5 503 ref.ra ref3.ra ;\
	$(TOOLDIR)/join 5 ref1.ra ref2.ra ref3.ra comb.ra ;\
	$(TOOLDIR)/avg 32 comb.ra avg.ra ;\
	$(TOOLDIR)/slice 5 167 signal.ra s.ra;\
	$(TOOLDIR)/nrmse -t 0.00001 avg.ra s.ra			    			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-to-signal-irbSSFP-averaged-spokes: sim scale signal nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=45,bwtp=4,av-spokes=3 -1 3:3:1 -2 1:1:1 _sim.ra ;\
	$(TOOLDIR)/scale -- -1i _sim.ra sim.ra							;\
	$(TOOLDIR)/signal -I -B -r0.0045 -e0.00225 -f45 -n1000 -1 3:3:1 -2 1:1:1 --av-spokes 3 signal.ra		;\
	$(TOOLDIR)/nrmse -t 0.001 sim.ra signal.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-to-signal-slice-profile: sim signal nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0,ppl=0.00225,trf=0.001,fa=45,bwtp=4,slice-profile-spins=40 -1 3:3:1 -2 1:1:1 sim.ra ;\
	$(TOOLDIR)/sim --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0,ppl=0.00225,trf=0.001,fa=45,bwtp=4,slice-profile-spins=42 -1 3:3:1 -2 1:1:1 sim2.ra ;\
	$(TOOLDIR)/nrmse -t 0.0015 sim.ra sim2.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-hp-irflash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq ir-flash,tr=0.0041,te=0.0025,nrep=1000,pinv,ipl=0,ppl=0,trf=0.0001,fa=8,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --seq ir-flash,tr=0.0041,te=0.0025,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=8,bwtp=4 -1 3:3:1 -2 1:1:1 simu_hp.ra ;\
	$(TOOLDIR)/nrmse -t 0.0006 simu_hp.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-hp-flash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq flash,tr=0.0041,te=0.0025,nrep=1000,pinv,ipl=0,ppl=0,trf=0.0001,fa=8,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --seq flash,tr=0.0041,te=0.0025,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=8,bwtp=4 -1 3:3:1 -2 1:1:1 simu_hp.ra ;\
	$(TOOLDIR)/nrmse -t 0.0006 simu_hp.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-hp-irbssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0.0001,ppl=0,trf=0,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_hp.ra ;\
	$(TOOLDIR)/nrmse -t 0.00015 simu_hp.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-hp-bssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0.0001,ppl=0,trf=0,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --seq bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_hp.ra ;\
	$(TOOLDIR)/nrmse -t 0.00015 simu_hp.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-sim-multi-relaxation: sim slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq bssfp,tr=0.0045,te=0.00225,nrep=1000,ipl=0.0001,ppl=0,trf=0,fa=45,bwtp=4 -1 3:3:4 -2 1:1:5 simu.ra ;\
	$(TOOLDIR)/slice 6 0 7 0 simu.ra slice1.ra				;\
	$(TOOLDIR)/slice 6 3 7 4 simu.ra slice2.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 slice1.ra slice2.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-bssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0.01,ppl=0.00225,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --STM --seq bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0.01,ppl=0.00225,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_stm.ra ;\
	$(TOOLDIR)/nrmse -t 0.003 simu_stm.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-irbssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0.01,ppl=0.00225,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --STM --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0.01,ppl=0.00225,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_stm.ra ;\
	$(TOOLDIR)/nrmse -t 0.003 simu_stm.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-flash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq flash,tr=0.003,te=0.0017,nrep=1000,pinv,ipl=0.01,ppl=0,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --STM --seq flash,tr=0.003,te=0.0017,nrep=1000,pinv,ipl=0.01,ppl=0,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_stm.ra ;\
	$(TOOLDIR)/nrmse -t 0.001 simu_stm.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-irflash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq ir-flash,tr=0.003,te=0.0017,nrep=1000,pinv,ipl=0.01,ppl=0,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --STM --seq ir-flash,tr=0.003,te=0.0017,nrep=1000,pinv,ipl=0.01,ppl=0,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_stm.ra ;\
	$(TOOLDIR)/nrmse -t 0.001 simu_stm.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-rot-bssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq bssfp,tr=0.0045,te=0.00225,nrep=50,pinv,ipl=0.01,ppl=0.00225,trf=0.0001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --ROT --seq bssfp,tr=0.0045,te=0.00225,nrep=50,pinv,ipl=0.01,ppl=0.00225,trf=0.0001,fa=45,bwtp=4 --other sampling-rate=10E5 -1 3:3:1 -2 1:1:1 simu_rot.ra ;\
	$(TOOLDIR)/nrmse -t 0.003 simu_rot.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-rot-irbssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=50,pinv,ipl=0.01,ppl=0.00225,trf=0.0001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --ROT --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=50,pinv,ipl=0.01,ppl=0.00225,trf=0.0001,fa=45,bwtp=4 --other sampling-rate=10E5 -1 3:3:1 -2 1:1:1 simu_rot.ra ;\
	$(TOOLDIR)/nrmse -t 0.003 simu_rot.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-rot-flash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq flash,tr=0.003,te=0.0017,nrep=1000,pinv,ipl=0.01,ppl=0,trf=0.001,fa=8,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --ROT --seq flash,tr=0.003,te=0.0017,nrep=1000,pinv,ipl=0.01,ppl=0,trf=0.001,fa=8,bwtp=4 --other sampling-rate=10E5 -1 3:3:1 -2 1:1:1 simu_rot.ra ;\
	$(TOOLDIR)/nrmse -t 0.006 simu_rot.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-rot-irflash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq ir-flash,tr=0.003,te=0.0017,nrep=1000,pinv,ipl=0.01,ppl=0,trf=0.001,fa=8,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --ROT --seq ir-flash,tr=0.003,te=0.0017,nrep=1000,pinv,ipl=0.01,ppl=0,trf=0.001,fa=8,bwtp=4 --other sampling-rate=10E5 -1 3:3:1 -2 1:1:1 simu_rot.ra ;\
	$(TOOLDIR)/nrmse -t 0.01 simu_rot.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-split-dim-mag: sim slice saxpy nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=10,pinv,ipl=0,ppl=0.00225,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 mxy.ra ;\
	$(TOOLDIR)/sim --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=10,pinv,ipl=0,ppl=0.00225,trf=0.001,fa=45,bwtp=4 --split-dim -1 3:3:1 -2 1:1:1 sim.ra ;\
	$(TOOLDIR)/slice 0 0 sim.ra x.ra ;\
	$(TOOLDIR)/slice 0 1 sim.ra y.ra ;\
	$(TOOLDIR)/saxpy -- 1i y.ra x.ra mxy2.ra ;\
	$(TOOLDIR)/nrmse -t 0.002 mxy.ra mxy2.ra ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-split-dim-deriv: sim slice saxpy nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=10,pinv,ipl=0,ppl=0.00225,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 mxy.ra dxy.ra;\
	$(TOOLDIR)/sim --seq ir-bssfp,tr=0.0045,te=0.00225,nrep=10,pinv,ipl=0,ppl=0.00225,trf=0.001,fa=45,bwtp=4 --split-dim -1 3:3:1 -2 1:1:1 sim.ra deriv.ra;\
	$(TOOLDIR)/slice 0 0 deriv.ra x.ra ;\
	$(TOOLDIR)/slice 0 1 deriv.ra y.ra ;\
	$(TOOLDIR)/saxpy -- 1i y.ra x.ra d.ra ;\
	$(TOOLDIR)/nrmse -t 0.002 dxy.ra d.ra ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-flash-te-eq-trf-eq-tr: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq flash,tr=0.001,te=0.001,nrep=100,pinv,ipl=0,ppl=0,trf=0.001,fa=180,bwtp=1,isp=0,mom-sl=16050 --split-dim -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --STM --seq flash,tr=0.001,te=0.001,nrep=100,pinv,ipl=0,ppl=0,trf=0.001,fa=180,bwtp=1,isp=0,mom-sl=16050 --split-dim -1 3:3:1 -2 1:1:1 simu_stm.ra ;\
	$(TOOLDIR)/nrmse -t 0.001 simu_stm.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-sim-to-signal-irflash tests/test-sim-to-signal-flash
TESTS += tests/test-sim-to-signal-irbSSFP
TESTS += tests/test-sim-spoke-averaging-3 tests/test-sim-to-signal-irbSSFP-averaged-spokes
TESTS += tests/test-sim-to-signal-slice-profile
TESTS += tests/test-sim-ode-hp-irflash tests/test-sim-ode-hp-flash
TESTS += tests/test-sim-ode-hp-irbssfp tests/test-sim-ode-hp-bssfp
TESTS += tests/test-sim-multi-relaxation
TESTS += tests/test-sim-ode-stm-bssfp tests/test-sim-ode-stm-irbssfp
TESTS += tests/test-sim-ode-stm-flash tests/test-sim-ode-stm-irflash
TESTS += tests/test-sim-ode-rot-bssfp tests/test-sim-ode-rot-irbssfp
TESTS += tests/test-sim-ode-rot-flash tests/test-sim-ode-rot-irflash
TESTS += tests/test-sim-split-dim-mag tests/test-sim-split-dim-deriv
TESTS += tests/test-sim-ode-stm-flash-te-eq-trf-eq-tr

