tests/test-sim-to-signal-irflash: sim cabs mip spow fmac scale signal nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq IR-FLASH,TR=0.0041,TE=0.0025,Nrep=1000,pinv,ipl=0,ppl=0,Trf=0,FA=8,BWTP=4 -1 3:3:1 -2 1:1:1 _simu.ra ;\
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
	$(TOOLDIR)/sim --seq FLASH,TR=0.0041,TE=0.0025,Nrep=1000,pinv,ipl=0,ppl=0,Trf=0,FA=8,BWTP=4 -1 3:3:1 -2 1:1:1 _simu.ra ;\
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
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0,ppl=0,Trf=0,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 _sim.ra ;\
	$(TOOLDIR)/scale -- -1i _sim.ra sim.ra							;\
	$(TOOLDIR)/signal -I -B -r0.0045 -e0.00225 -f45 -n1000 -1 3:3:1 -2 1:1:1 signal.ra		;\
	$(TOOLDIR)/nrmse -t 0.003 sim.ra signal.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-spoke-averaging-3: sim slice join avg nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0,ppl=0,Trf=0,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 ref.ra	;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0,ppl=0,Trf=0,FA=45,BWTP=4,av-spokes=3 -1 3:3:1 -2 1:1:1 signal.ra	;\
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
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0,ppl=0,Trf=0,FA=45,BWTP=4,av-spokes=3 -1 3:3:1 -2 1:1:1 _sim.ra ;\
	$(TOOLDIR)/scale -- -1i _sim.ra sim.ra							;\
	$(TOOLDIR)/signal -I -B -r0.0045 -e0.00225 -f45 -n1000 -1 3:3:1 -2 1:1:1 --av-spokes 3 signal.ra		;\
	$(TOOLDIR)/nrmse -t 0.001 sim.ra signal.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-slice-profile-spins: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0,ppl=0.00225,Trf=0.001,FA=45,BWTP=4,slice-thickness=0.020,sl-grad=0.01,Nspins=41 -1 3:3:1 -2 1:1:1 sim.ra ;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0,ppl=0.00225,Trf=0.001,FA=45,BWTP=4,slice-thickness=0.020,sl-grad=0.01,Nspins=61 -1 3:3:1 -2 1:1:1 sim2.ra ;\
	$(TOOLDIR)/nrmse -t 0.01 sim.ra sim2.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-slice-profile-slicethickness: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0,ppl=0.00225,Trf=0.001,FA=45,BWTP=4,slice-thickness=0.040,sl-grad=0.01,Nspins=41 -1 3:3:1 -2 1:1:1 sim.ra ;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0,ppl=0.00225,Trf=0.001,FA=45,BWTP=4,slice-thickness=0.020,sl-grad=0.01,Nspins=41 -1 3:3:1 -2 1:1:1 sim2.ra ;\
	$(TOOLDIR)/nrmse -t 0.02 sim.ra sim2.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-slice-profile-density: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0,ppl=0.00225,Trf=0.001,FA=45,BWTP=4,slice-thickness=0.040,sl-grad=0.01,Nspins=41 -1 3:3:1 -2 1:1:1 sim.ra ;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0,ppl=0.00225,Trf=0.001,FA=45,BWTP=4,slice-thickness=0.020,sl-grad=0.01,Nspins=31 -1 3:3:1 -2 1:1:1 sim2.ra ;\
	$(TOOLDIR)/nrmse -t 0.02 sim.ra sim2.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-slice-profile-density2: sim scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0,ppl=0.00225,Trf=0.001,FA=45,BWTP=4,slice-thickness=0.041,sl-grad=0,Nspins=41 -1 3:3:1 -2 1:1:1 sim.ra ;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0,ppl=0.00225,Trf=0.001,FA=45,BWTP=4,sl-grad=0 -1 3:3:1 -2 1:1:1 sim2.ra ;\
	$(TOOLDIR)/scale -- 41 sim2.ra sim3.ra			;\
	$(TOOLDIR)/nrmse -t 0.02 sim.ra sim3.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-hp-irflash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq IR-FLASH,TR=0.0041,TE=0.0025,Nrep=1000,pinv,ipl=0,ppl=0,Trf=0.0001,FA=8,BWTP=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --seq IR-FLASH,TR=0.0041,TE=0.0025,Nrep=1000,pinv,ipl=0,ppl=0,Trf=0,FA=8,BWTP=4 -1 3:3:1 -2 1:1:1 simu_hp.ra ;\
	$(TOOLDIR)/nrmse -t 0.0006 simu_hp.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-hp-flash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq FLASH,TR=0.0041,TE=0.0025,Nrep=1000,pinv,ipl=0,ppl=0,Trf=0.0001,FA=8,BWTP=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --seq FLASH,TR=0.0041,TE=0.0025,Nrep=1000,pinv,ipl=0,ppl=0,Trf=0,FA=8,BWTP=4 -1 3:3:1 -2 1:1:1 simu_hp.ra ;\
	$(TOOLDIR)/nrmse -t 0.0006 simu_hp.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-hp-irbssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0.0001,ppl=0,Trf=0,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0,ppl=0,Trf=0,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 simu_hp.ra ;\
	$(TOOLDIR)/nrmse -t 0.00015 simu_hp.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-hp-bssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0.0001,ppl=0,Trf=0,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --seq BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0,ppl=0,Trf=0,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 simu_hp.ra ;\
	$(TOOLDIR)/nrmse -t 0.00015 simu_hp.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-sim-multi-relaxation: sim slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,ipl=0.0001,ppl=0,Trf=0,FA=45,BWTP=4 -1 3:3:4 -2 1:1:5 simu.ra ;\
	$(TOOLDIR)/slice 6 0 7 0 simu.ra slice1.ra				;\
	$(TOOLDIR)/slice 6 3 7 4 simu.ra slice2.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 slice1.ra slice2.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-bssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0.01,ppl=0.00225,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --STM --seq BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0.01,ppl=0.00225,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 simu_stm.ra ;\
	$(TOOLDIR)/nrmse -t 0.003 simu_stm.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-irbssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0.01,ppl=0.00225,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --STM --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=1000,pinv,ipl=0.01,ppl=0.00225,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 simu_stm.ra ;\
	$(TOOLDIR)/nrmse -t 0.003 simu_stm.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-flash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq FLASH,TR=0.003,TE=0.0017,Nrep=1000,pinv,ipl=0.01,ppl=0,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --STM --seq FLASH,TR=0.003,TE=0.0017,Nrep=1000,pinv,ipl=0.01,ppl=0,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 simu_stm.ra ;\
	$(TOOLDIR)/nrmse -t 0.001 simu_stm.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-irflash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq IR-FLASH,TR=0.003,TE=0.0017,Nrep=1000,pinv,ipl=0.01,ppl=0,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --STM --seq IR-FLASH,TR=0.003,TE=0.0017,Nrep=1000,pinv,ipl=0.01,ppl=0,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 simu_stm.ra ;\
	$(TOOLDIR)/nrmse -t 0.001 simu_stm.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-rot-bssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq BSSFP,TR=0.0045,TE=0.00225,Nrep=50,pinv,ipl=0.01,ppl=0.00225,Trf=0.0001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --ROT --seq BSSFP,TR=0.0045,TE=0.00225,Nrep=50,pinv,ipl=0.01,ppl=0.00225,Trf=0.0001,FA=45,BWTP=4 --other sampling-rate=10E5 -1 3:3:1 -2 1:1:1 simu_rot.ra ;\
	$(TOOLDIR)/nrmse -t 0.003 simu_rot.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-rot-irbssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=50,pinv,ipl=0.01,ppl=0.00225,Trf=0.0001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --ROT --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=50,pinv,ipl=0.01,ppl=0.00225,Trf=0.0001,FA=45,BWTP=4 --other sampling-rate=10E5 -1 3:3:1 -2 1:1:1 simu_rot.ra ;\
	$(TOOLDIR)/nrmse -t 0.003 simu_rot.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-rot-flash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq FLASH,TR=0.003,TE=0.0017,Nrep=1000,pinv,ipl=0.01,ppl=0,Trf=0.001,FA=8,BWTP=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --ROT --seq FLASH,TR=0.003,TE=0.0017,Nrep=1000,pinv,ipl=0.01,ppl=0,Trf=0.001,FA=8,BWTP=4 --other sampling-rate=10E5 -1 3:3:1 -2 1:1:1 simu_rot.ra ;\
	$(TOOLDIR)/nrmse -t 0.006 simu_rot.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-rot-irflash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq IR-FLASH,TR=0.003,TE=0.0017,Nrep=1000,pinv,ipl=0.01,ppl=0,Trf=0.001,FA=8,BWTP=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --ROT --seq IR-FLASH,TR=0.003,TE=0.0017,Nrep=1000,pinv,ipl=0.01,ppl=0,Trf=0.001,FA=8,BWTP=4 --other sampling-rate=10E5 -1 3:3:1 -2 1:1:1 simu_rot.ra ;\
	$(TOOLDIR)/nrmse -t 0.01 simu_rot.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-split-dim-mag: sim slice saxpy nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=10,pinv,ipl=0,ppl=0.00225,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 mxy.ra ;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=10,pinv,ipl=0,ppl=0.00225,Trf=0.001,FA=45,BWTP=4 --split-dim -1 3:3:1 -2 1:1:1 sim.ra ;\
	$(TOOLDIR)/slice 0 0 sim.ra x.ra ;\
	$(TOOLDIR)/slice 0 1 sim.ra y.ra ;\
	$(TOOLDIR)/saxpy -- 1i y.ra x.ra mxy2.ra ;\
	$(TOOLDIR)/nrmse -t 0.002 mxy.ra mxy2.ra ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-split-dim-deriv: sim slice saxpy nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=10,pinv,ipl=0,ppl=0.00225,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 mxy.ra dxy.ra;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=10,pinv,ipl=0,ppl=0.00225,Trf=0.001,FA=45,BWTP=4 --split-dim -1 3:3:1 -2 1:1:1 sim.ra deriv.ra;\
	$(TOOLDIR)/slice 0 0 deriv.ra x.ra ;\
	$(TOOLDIR)/slice 0 1 deriv.ra y.ra ;\
	$(TOOLDIR)/saxpy -- 1i y.ra x.ra d.ra ;\
	$(TOOLDIR)/nrmse -t 0.002 dxy.ra d.ra ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-flash-te-eq-trf-eq-tr: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --ODE --seq FLASH,TR=0.001,TE=0.001,Nrep=100,pinv,ipl=0,ppl=0,Trf=0.001,FA=180,BWTP=1,isp=0,slice-thickness=0.040,sl-grad=0.001 --split-dim -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --STM --seq FLASH,TR=0.001,TE=0.001,Nrep=100,pinv,ipl=0,ppl=0,Trf=0.001,FA=180,BWTP=1,isp=0,slice-thickness=0.040,sl-grad=0.001 --split-dim -1 3:3:1 -2 1:1:1 simu_stm.ra ;\
	$(TOOLDIR)/nrmse -t 0.001 simu_stm.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-deriv-r1: sim slice saxpy scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/sim --ODE --seq IR-BSSFP,TR=0.004,TE=0.002,Nrep=1000,pinv,ipl=0.01,ppl=0.002,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 --other ode-tol=1E-6 s.ra d.ra;\
	$(TOOLDIR)/slice 4 0 d.ra d_r1.ra ;\
	$(TOOLDIR)/sim --ODE --seq IR-BSSFP,TR=0.004,TE=0.002,Nrep=1000,pinv,ipl=0.01,ppl=0.002,Trf=0.001,FA=45,BWTP=4 -1 3.003:3.003:1 -2 1:1:1 --other ode-tol=1E-6 s2.ra ;\
	$(TOOLDIR)/saxpy -- -1 s.ra s2.ra diff.ra ;\
	$(TOOLDIR)/scale -- 333 diff.ra g.ra ;\
	$(TOOLDIR)/scale -- -9 g.ra g2.ra ;\
	$(TOOLDIR)/nrmse -t 0.013 d_r1.ra g2.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-deriv-r2: sim slice saxpy scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/sim --ODE --seq IR-BSSFP,TR=0.004,TE=0.002,Nrep=1000,pinv,ipl=0.01,ppl=0.002,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 --other ode-tol=1E-6 s.ra d.ra;\
	$(TOOLDIR)/slice 4 2 d.ra d_r2.ra ;\
	$(TOOLDIR)/sim --ODE --seq IR-BSSFP,TR=0.004,TE=0.002,Nrep=1000,pinv,ipl=0.01,ppl=0.002,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1.01:1.01:1 --other ode-tol=1E-6 s2.ra ;\
	$(TOOLDIR)/saxpy -- -1 s.ra s2.ra diff.ra ;\
	$(TOOLDIR)/scale -- 100 diff.ra g.ra ;\
	$(TOOLDIR)/scale -- -1 g.ra g2.ra ;\
	$(TOOLDIR)/nrmse -t 0.013 d_r2.ra g2.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-deriv-b1: sim slice saxpy scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/sim --ODE --seq IR-BSSFP,TR=0.004,TE=0.002,Nrep=1000,pinv,ipl=0.01,ppl=0.002,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 --other ode-tol=1E-6 s.ra d.ra;\
	$(TOOLDIR)/slice 4 3 d.ra d_b1.ra ;\
	$(TOOLDIR)/sim --ODE --seq IR-BSSFP,TR=0.004,TE=0.002,Nrep=1000,pinv,ipl=0.01,ppl=0.002,Trf=0.001,FA=45.1,BWTP=4 -1 3:3:1 -2 1:1:1 --other ode-tol=1E-6 s2.ra ;\
	$(TOOLDIR)/saxpy -- -1 s.ra s2.ra diff.ra ;\
	$(TOOLDIR)/scale -- 10 diff.ra g.ra ;\
	$(TOOLDIR)/scale -- 45 g.ra g2.ra ;\
	$(TOOLDIR)/nrmse -t 0.01 d_b1.ra g2.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-deriv: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/sim --ODE --seq IR-BSSFP,TR=0.004,TE=0.002,Nrep=1000,pinv,ipl=0.01,ppl=0.002,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1:1:1 --other ode-tol=1E-6 s.ra d.ra;\
	$(TOOLDIR)/sim --STM --seq IR-BSSFP,TR=0.004,TE=0.002,Nrep=1000,pinv,ipl=0.01,ppl=0.002,Trf=0.001,FA=45,BWTP=4 -1 3:3:1 -2 1.01:1.01:1 --other ode-tol=1E-6 s2.ra d2.ra ;\
	$(TOOLDIR)/nrmse -t 0.005 d.ra d2.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-sim-to-signal-irflash tests/test-sim-to-signal-flash
TESTS += tests/test-sim-to-signal-irbSSFP
TESTS += tests/test-sim-spoke-averaging-3 tests/test-sim-to-signal-irbSSFP-averaged-spokes
TESTS += tests/test-sim-slice-profile-spins tests/test-sim-slice-profile-slicethickness tests/test-sim-slice-profile-density tests/test-sim-slice-profile-density2
TESTS += tests/test-sim-ode-hp-irflash tests/test-sim-ode-hp-flash
TESTS += tests/test-sim-ode-hp-irbssfp tests/test-sim-ode-hp-bssfp
TESTS += tests/test-sim-multi-relaxation
TESTS += tests/test-sim-ode-stm-bssfp tests/test-sim-ode-stm-irbssfp
TESTS += tests/test-sim-ode-stm-flash tests/test-sim-ode-stm-irflash
TESTS += tests/test-sim-ode-rot-bssfp tests/test-sim-ode-rot-irbssfp
TESTS += tests/test-sim-ode-rot-flash tests/test-sim-ode-rot-irflash
TESTS += tests/test-sim-split-dim-mag tests/test-sim-split-dim-deriv
TESTS += tests/test-sim-ode-stm-flash-te-eq-trf-eq-tr
TESTS += tests/test-sim-ode-deriv-r1 tests/test-sim-ode-deriv-r2 tests/test-sim-ode-deriv-b1 tests/test-sim-ode-stm-deriv
