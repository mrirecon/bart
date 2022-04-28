tests/test-sim-to-signal-irflash: sim cabs mip spow fmac signal nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --sim ir-flash,tr=0.0041,te=0.0025,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=8,bwtp=4 -1 3:3:1 -2 1:1:1 _simu.ra ;\
	$(TOOLDIR)/cabs _simu.ra _simu_abs.ra								;\
	$(TOOLDIR)/mip 32 _simu_abs.ra max.ra								;\
	$(TOOLDIR)/spow -- -1 max.ra scale.ra								;\
	$(TOOLDIR)/fmac _simu.ra scale.ra simu.ra 							;\
	$(TOOLDIR)/signal -I -F -r0.0041 -e0.00258 -f8 -n1000 -1 3:3:1 -2 1:1:1 signal.ra		;\
	$(TOOLDIR)/nrmse -t 0.003 simu.ra signal.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-to-signal-flash: sim cabs mip spow fmac signal nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --sim flash,tr=0.0041,te=0.0025,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=8,bwtp=4 -1 3:3:1 -2 1:1:1 _simu.ra ;\
	$(TOOLDIR)/cabs _simu.ra _simu_abs.ra								;\
	$(TOOLDIR)/mip 32 _simu_abs.ra max.ra								;\
	$(TOOLDIR)/spow -- -1 max.ra scale.ra								;\
	$(TOOLDIR)/fmac _simu.ra scale.ra simu.ra 							;\
	$(TOOLDIR)/signal -F -r0.0041 -e0.00258 -f8 -n1000 -1 3:3:1 -2 1:1:1 signal.ra		;\
	$(TOOLDIR)/nrmse -t 0.003 simu.ra signal.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-to-signal-irbSSFP: sim signal nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --sim ir-bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 sim.ra ;\
	$(TOOLDIR)/signal -I -B -r0.0045 -e0.00225 -f45 -n1000 -1 3:3:1 -2 1:1:1 signal.ra		;\
	$(TOOLDIR)/nrmse -t 0.003 sim.ra signal.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-sim-ode-hp-irflash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --sim ir-flash,tr=0.0041,te=0.0025,nrep=1000,pinv,ipl=0,ppl=0,trf=0.0001,fa=8,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --sim ir-flash,tr=0.0041,te=0.0025,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=8,bwtp=4 -1 3:3:1 -2 1:1:1 simu_hp.ra ;\
	$(TOOLDIR)/nrmse -t 0.0006 simu_hp.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-hp-flash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --sim flash,tr=0.0041,te=0.0025,nrep=1000,pinv,ipl=0,ppl=0,trf=0.0001,fa=8,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --sim flash,tr=0.0041,te=0.0025,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=8,bwtp=4 -1 3:3:1 -2 1:1:1 simu_hp.ra ;\
	$(TOOLDIR)/nrmse -t 0.0006 simu_hp.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-hp-irbssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --sim ir-bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0.0001,ppl=0,trf=0,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --sim ir-bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_hp.ra ;\
	$(TOOLDIR)/nrmse -t 0.00015 simu_hp.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-hp-bssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --sim bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0.0001,ppl=0,trf=0,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --sim bssfp,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0,ppl=0,trf=0,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_hp.ra ;\
	$(TOOLDIR)/nrmse -t 0.00015 simu_hp.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-sim-multi-relaxation: sim slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --sim bssfp,tr=0.0045,te=0.00225,nrep=1000,ipl=0.0001,ppl=0,trf=0,fa=45,bwtp=4 -1 3:3:4 -2 1:1:5 simu.ra ;\
	$(TOOLDIR)/slice 6 0 7 0 simu.ra slice1.ra				;\
	$(TOOLDIR)/slice 6 3 7 4 simu.ra slice2.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 slice1.ra slice2.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-bssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --sim bssfp,ODE,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0.01,ppl=0.00225,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --sim bssfp,STM,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0.01,ppl=0.00225,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_stm.ra ;\
	$(TOOLDIR)/nrmse -t 0.003 simu_stm.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-irbssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --sim ir-bssfp,ODE,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0.01,ppl=0.00225,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --sim ir-bssfp,STM,tr=0.0045,te=0.00225,nrep=1000,pinv,ipl=0.01,ppl=0.00225,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_stm.ra ;\
	$(TOOLDIR)/nrmse -t 0.003 simu_stm.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-flash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --sim flash,ODE,tr=0.003,te=0.0017,nrep=1000,pinv,ipl=0.01,ppl=0,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --sim flash,STM,tr=0.003,te=0.0017,nrep=1000,pinv,ipl=0.01,ppl=0,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_stm.ra ;\
	$(TOOLDIR)/nrmse -t 0.001 simu_stm.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-stm-irflash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim --sim ir-flash,ODE,tr=0.003,te=0.0017,nrep=1000,pinv,ipl=0.01,ppl=0,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_ode.ra ;\
	$(TOOLDIR)/sim --sim ir-flash,STM,tr=0.003,te=0.0017,nrep=1000,pinv,ipl=0.01,ppl=0,trf=0.001,fa=45,bwtp=4 -1 3:3:1 -2 1:1:1 simu_stm.ra ;\
	$(TOOLDIR)/nrmse -t 0.001 simu_stm.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-sim-to-signal-irflash tests/test-sim-to-signal-flash
TESTS += tests/test-sim-to-signal-irbSSFP
TESTS += tests/test-sim-ode-hp-irflash tests/test-sim-ode-hp-flash
TESTS += tests/test-sim-ode-hp-irbssfp tests/test-sim-ode-hp-bssfp
TESTS += tests/test-sim-multi-relaxation
TESTS += tests/test-sim-ode-stm-bssfp tests/test-sim-ode-stm-irbssfp
TESTS += tests/test-sim-ode-stm-flash tests/test-sim-ode-stm-irflash