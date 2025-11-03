
tests/test-mobafit-r2s: phantom signal fmac index scale extract mobafit slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/phantom -x16 -c circ.ra				;\
	$(TOOLDIR)/signal -G -n8 -1 3:3:1 -2 0.02:0.02:1 signal_p1.ra	;\
	$(TOOLDIR)/extract 5 1 8 signal_p1.ra signal.ra			;\
	$(TOOLDIR)/fmac circ.ra signal.ra echoes.ra			;\
	$(TOOLDIR)/index 5 8 tmp1.ra					;\
	$(TOOLDIR)/scale 1.6 tmp1.ra tmp2.ra				;\
	$(TOOLDIR)/extract 5 1 8 tmp2.ra TE.ra				;\
	$(TOOLDIR)/mobafit -G -m3 TE.ra echoes.ra reco.ra		;\
	$(TOOLDIR)/slice 6 1 reco.ra R2S.ra				;\
	$(TOOLDIR)/phantom -x16 -c circ.ra				;\
	$(TOOLDIR)/fmac R2S.ra circ.ra masked.ra			;\
	$(TOOLDIR)/scale -- 0.05 circ.ra ref.ra				;\
	$(TOOLDIR)/nrmse -t 0.00001 ref.ra masked.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-mobafit-wfr2s: phantom signal fmac index scale extract mobafit saxpy cabs spow ones slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)                      ;\
	$(TOOLDIR)/phantom -x16 -c circ.ra                                ;\
	$(TOOLDIR)/signal -G --fat -n8 -1 3:3:1 -2 0.02:0.02:1 signal_p1.ra  ;\
	$(TOOLDIR)/extract 5 1 8 signal_p1.ra signal.ra                   ;\
	$(TOOLDIR)/fmac circ.ra signal.ra echoes.ra                       ;\
	$(TOOLDIR)/index 5 8 tmp1.ra                                      ;\
	$(TOOLDIR)/scale 1.6 tmp1.ra tmp2.ra                              ;\
	$(TOOLDIR)/extract 5 1 8 tmp2.ra TE.ra                            ;\
	$(TOOLDIR)/mobafit -G -m1 TE.ra echoes.ra reco.ra                 ;\
	$(TOOLDIR)/slice 6 0 reco.ra W.ra                                 ;\
	$(TOOLDIR)/slice 6 1 reco.ra F.ra                                 ;\
	$(TOOLDIR)/slice 6 2 reco.ra R2S.ra                               ;\
	$(TOOLDIR)/slice 6 3 reco.ra fB0.ra                               ;\
	$(TOOLDIR)/saxpy 1 W.ra F.ra temp_inphase.ra                      ;\
	$(TOOLDIR)/cabs temp_inphase.ra temp_inphase_abs.ra               ;\
	$(TOOLDIR)/ones 2 16 16 ones.ra                                   ;\
	$(TOOLDIR)/saxpy 0.000001 ones.ra temp_inphase_abs.ra temp_wf.ra  ;\
	$(TOOLDIR)/spow -- -1. temp_wf.ra temp_deno.ra                    ;\
	$(TOOLDIR)/cabs F.ra temp_F_abs.ra                                ;\
	$(TOOLDIR)/fmac temp_F_abs.ra temp_deno.ra fatfrac.ra             ;\
	$(TOOLDIR)/phantom -x16 -c circ.ra                                ;\
	$(TOOLDIR)/fmac fatfrac.ra circ.ra fatfrac_masked.ra              ;\
	$(TOOLDIR)/scale -- 0.2 circ.ra fatfrac_ref.ra                    ;\
	$(TOOLDIR)/nrmse -t 0.00001 fatfrac_ref.ra fatfrac_masked.ra      ;\
	$(TOOLDIR)/fmac R2S.ra circ.ra R2S_masked.ra                      ;\
	$(TOOLDIR)/scale -- 0.05 circ.ra R2S_ref.ra                       ;\
	$(TOOLDIR)/nrmse -t 0.00001 R2S_ref.ra R2S_masked.ra         	  ;\
	$(TOOLDIR)/fmac fB0.ra circ.ra fB0_masked.ra                      ;\
	$(TOOLDIR)/scale -- 0.02 circ.ra fB0_ref.ra                       ;\
	$(TOOLDIR)/nrmse -t 0.000001 fB0_ref.ra fB0_masked.ra            ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-mobafit-r2: phantom signal reshape fmac index mobafit slice nrmse index extract invert
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/phantom -x32 -T -b tubes.ra				;\
	$(TOOLDIR)/signal -250:160:11 -S -e10 -n16 sig.ra		;\
	$(TOOLDIR)/reshape 192 11 1 sig.ra sig2.ra			;\
	$(TOOLDIR)/fmac -s 64 tubes.ra sig2.ra x.ra			;\
	$(TOOLDIR)/index 5 16 te.ra					;\
	$(TOOLDIR)/mobafit -T te.ra x.ra fit.ra				;\
	$(TOOLDIR)/slice 6 0 fit.ra x0.ra				;\
	$(TOOLDIR)/slice 6 1 fit.ra x1.ra				;\
	$(TOOLDIR)/phantom -x32 -T r0.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 r0.ra x0.ra			;\
	$(TOOLDIR)/index 6 16 t2.ra					;\
	$(TOOLDIR)/extract 6 5 16 t2.ra t2b.ra 				;\
	$(TOOLDIR)/invert t2b.ra r2.ra					;\
	$(TOOLDIR)/fmac -s 64 tubes.ra r2.ra r1.ra 			;\
	$(TOOLDIR)/nrmse -t 0.000001 r1.ra x1.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

$(TESTS_OUT)/basis_irll.ra: signal reshape extract transpose squeeze svd
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/signal -F -I -1 0.1:10:20 -3 0.1:1.:10 -r0.41 -n30 sig.ra	;\
	$(TOOLDIR)/reshape 192 200 1 sig.ra dicc1			;\
	$(TOOLDIR)/squeeze dicc1 dicc					;\
	$(TOOLDIR)/svd -e dicc U S V					;\
	$(TOOLDIR)/extract 1 0 4 U basis				;\
	$(TOOLDIR)/transpose 1 6 basis basis1				;\
	$(TOOLDIR)/transpose 0 5 basis1 $@				;\
	rm *.ra *.hdr *.cfl ; cd .. ; rmdir $(TESTS_TMP)

tests/test-mobafit-irll: phantom signal reshape fmac index mobafit slice nrmse index extract invert scale saxpy ones
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/phantom -x32 -T -b tubes.ra				;\
	$(TOOLDIR)/signal -F -I -1 2:2:1 -3.8:.8:11 -r0.41 -n30 sig.ra	;\
	$(TOOLDIR)/reshape 192 11 1 sig.ra sig2.ra			;\
	$(TOOLDIR)/fmac -s 64 tubes.ra sig2.ra x.ra			;\
	$(TOOLDIR)/index 5 30 ti1.ra					;\
	$(TOOLDIR)/ones 6 1 1 1 1 1 30 ones.ra				;\
	$(TOOLDIR)/saxpy 0.5 ones.ra ti1.ra ti2.ra          		;\
	$(TOOLDIR)/scale 0.41 ti2.ra ti.ra				;\
	$(TOOLDIR)/mobafit --init=.6:1.:.8 -L ti.ra x.ra fit.ra		;\
	$(TOOLDIR)/slice 6 0 fit.ra x0.ra				;\
	$(TOOLDIR)/slice 6 1 fit.ra x1.ra				;\
	$(TOOLDIR)/slice 6 2 fit.ra x2.ra				;\
	$(TOOLDIR)/phantom -x32 -T r1.ra				;\
	$(TOOLDIR)/nrmse -t 0.00001 r1.ra x1.ra				;\
	$(TOOLDIR)/scale .8 r1.ra r0.ra					;\
	$(TOOLDIR)/nrmse -t 0.00001 r0.ra x0.ra				;\
	$(TOOLDIR)/scale .5 r1.ra r2.ra					;\
	$(TOOLDIR)/nrmse -t 0.00001 r2.ra x2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mobafit-irll_bas1: $(TESTS_OUT)/basis_irll.ra phantom signal reshape fmac index mobafit slice nrmse index extract invert scale saxpy ones
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/phantom -x32 -T -b tubes.ra				;\
	$(TOOLDIR)/signal -F -I -1 2:2:1 -3.8:.8:11 -r0.41 -n30 sig.ra	;\
	$(TOOLDIR)/reshape 192 11 1 sig.ra sig2.ra			;\
	$(TOOLDIR)/fmac -s 64 tubes.ra sig2.ra x.ra			;\
	$(TOOLDIR)/index 5 30 ti1.ra					;\
	$(TOOLDIR)/ones 6 1 1 1 1 1 30 ones.ra				;\
	$(TOOLDIR)/saxpy 0.5 ones.ra ti1.ra ti2.ra          		;\
	$(TOOLDIR)/scale 0.41 ti2.ra ti.ra				;\
	$(TOOLDIR)/mobafit -B$(TESTS_OUT)/basis_irll.ra  --init=.6:1.:.8 -L ti.ra x.ra fit.ra	;\
	$(TOOLDIR)/slice 6 0 fit.ra x0.ra				;\
	$(TOOLDIR)/slice 6 1 fit.ra x1.ra				;\
	$(TOOLDIR)/slice 6 2 fit.ra x2.ra				;\
	$(TOOLDIR)/phantom -x32 -T r1.ra				;\
	$(TOOLDIR)/nrmse -t 0.00001 r1.ra x1.ra				;\
	$(TOOLDIR)/scale .8 r1.ra r0.ra					;\
	$(TOOLDIR)/nrmse -t 0.00001 r0.ra x0.ra				;\
	$(TOOLDIR)/scale .5 r1.ra r2.ra					;\
	$(TOOLDIR)/nrmse -t 0.00001 r2.ra x2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mobafit-irll_bas2: $(TESTS_OUT)/basis_irll.ra phantom signal reshape fmac index mobafit slice nrmse index extract invert scale saxpy ones
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/phantom -x32 -T -b tubes.ra				;\
	$(TOOLDIR)/signal -F -I -1 2:2:1 -3.8:.8:11 -r0.41 -n30 sig.ra	;\
	$(TOOLDIR)/reshape 192 11 1 sig.ra sig2.ra			;\
	$(TOOLDIR)/fmac -s 64 tubes.ra sig2.ra x.ra			;\
	$(TOOLDIR)/index 5 30 ti1.ra					;\
	$(TOOLDIR)/ones 6 1 1 1 1 1 30 ones.ra				;\
	$(TOOLDIR)/saxpy 0.5 ones.ra ti1.ra ti2.ra          		;\
	$(TOOLDIR)/scale 0.41 ti2.ra ti.ra				;\
	$(TOOLDIR)/fmac -s 32 x.ra $(TESTS_OUT)/basis_irll.ra coef.ra	;\
	$(TOOLDIR)/mobafit -B$(TESTS_OUT)/basis_irll.ra --init=.6:1.:.8 -L ti.ra coef.ra fit.ra	;\
	$(TOOLDIR)/slice 6 0 fit.ra x0.ra				;\
	$(TOOLDIR)/slice 6 1 fit.ra x1.ra				;\
	$(TOOLDIR)/slice 6 2 fit.ra x2.ra				;\
	$(TOOLDIR)/phantom -x32 -T r1.ra				;\
	$(TOOLDIR)/nrmse -t 0.00001 r1.ra x1.ra				;\
	$(TOOLDIR)/scale .8 r1.ra r0.ra					;\
	$(TOOLDIR)/nrmse -t 0.00001 r0.ra x0.ra				;\
	$(TOOLDIR)/scale .5 r1.ra r2.ra					;\
	$(TOOLDIR)/nrmse -t 0.00001 r2.ra x2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mobafit-gpu: phantom signal reshape fmac index mobafit slice nrmse index extract invert
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/phantom -x32 -T -b tubes.ra				;\
	$(TOOLDIR)/signal -250:160:11 -S -e10 -n16 sig.ra		;\
	$(TOOLDIR)/reshape 192 11 1 sig.ra sig2.ra			;\
	$(TOOLDIR)/fmac -s 64 tubes.ra sig2.ra x.ra			;\
	$(TOOLDIR)/index 5 16 te.ra					;\
	$(TOOLDIR)/mobafit -g -T te.ra x.ra fit.ra			;\
	$(TOOLDIR)/slice 6 0 fit.ra x0.ra				;\
	$(TOOLDIR)/slice 6 1 fit.ra x1.ra				;\
	$(TOOLDIR)/phantom -x32 -T r0.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 r0.ra x0.ra			;\
	$(TOOLDIR)/index 6 16 t2.ra					;\
	$(TOOLDIR)/extract 6 5 16 t2.ra t2b.ra 				;\
	$(TOOLDIR)/invert t2b.ra r2.ra					;\
	$(TOOLDIR)/fmac -s 64 tubes.ra r2.ra r1.ra 			;\
	$(TOOLDIR)/nrmse -t 0.000001 r1.ra x1.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mobafit-ir_t1: mobafit nrmse transpose scale vec join
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)										;\
	$(TOOLDIR)/vec +1.929238e-01+0.000000e+00i	+1.255160e-01+0.000000e+00i	+8.905754e-02+0.000000e+00i	+5.494139e-02+0.000000e+00i	+2.563298e-02+0.000000e+00i	+6.914235e-02+0.000000e+00i	+2.460453e-01+0.000000e+00i	+2.540495e-01+0.000000e+00i	+2.561918e-01+0.000000e+00i	+2.552134e-01+0.000000e+00i d0.ra		;\
	$(TOOLDIR)/vec +2.552905e-01+0.000000e+00i	+4.081513e+00+0.000000e+00i	+7.058685e-01+0.000000e+00i r0.ra	;\
	$(TOOLDIR)/vec +1.844473e-01+0.000000e+00i	+9.110585e-02+0.000000e+00i	+4.486995e-02+0.000000e+00i	+3.315099e-03+0.000000e+00i	+3.353683e-02+0.000000e+00i	+1.363250e-01+0.000000e+00i	+2.706290e-01+0.000000e+00i	+2.735427e-01+0.000000e+00i	+2.733905e-01+0.000000e+00i	+2.728096e-01+0.000000e+00i d1.ra		;\
	$(TOOLDIR)/vec +2.723165e-01+0.000000e+00i	+5.029132e+00+0.000000e+00i	+6.884007e-01+0.000000e+00i r1.ra	;\
	$(TOOLDIR)/join 1 d0.ra d1.ra td0.ra											;\
	$(TOOLDIR)/transpose 0 5 td0.ra td1.ra											;\
	$(TOOLDIR)/transpose 1 0 td1.ra testdata.ra										;\
	$(TOOLDIR)/join 1 r0.ra r1.ra tr0.ra 											;\
	$(TOOLDIR)/transpose 0 6 tr0.ra tr1.ra 											;\
	$(TOOLDIR)/transpose 1 0 tr1.ra testresults.ra 										;\
	$(TOOLDIR)/vec 35 75 100 125 150 250 1000 1500 2000 3000 test_echotimes_ms_o.ra 					;\
	$(TOOLDIR)/transpose 0 5 test_echotimes_ms_o.ra test_echotimes_ms.ra 							;\
	$(TOOLDIR)/scale -- 0.001 test_echotimes_ms.ra test_echotimes_s.ra 							;\
	$(TOOLDIR)/mobafit -I -a --init 1:1:1 --min-flag 2 --min=0:0.1 test_echotimes_s.ra testdata.ra testfit_R1.ra 		;\
	$(TOOLDIR)/nrmse -t 0.00001 testresults.ra testfit_R1.ra 								;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mobafit-sim: phantom sim repmat fmac index mobafit slice nrmse scale
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/phantom -x32 -T -b tubes.ra				;\
	$(TOOLDIR)/sim --ODE --seq IR-BSSFP,TR=0.004,TE=0.002,Nrep=100,ipl=0.01,isp=0.005,ppl=0.002,Trf=0.001,FA=45,BWTP=4 -1 1:1:1 -2 0.1:0.1:1 sim0.ra ;\
	$(TOOLDIR)/repmat 6 11 sim0.ra sim.ra				;\
	$(TOOLDIR)/fmac -s 64 tubes.ra sim.ra ph_img.ra	;\
	$(TOOLDIR)/index 5 100 te.ra					;\
	$(TOOLDIR)/mobafit -S --sim ODE -i5 --init=0.8:1:9:1 --seq IR-BSSFP,TR=0.004,TE=0.002,Nrep=100,ipl=0.01,isp=0.005,ppl=0.002,Trf=0.001,FA=45,BWTP=4 --scale=1:1:1:0:0:0:0:0:0 te.ra ph_img.ra fit.ra ;\
	$(TOOLDIR)/slice 6 0 fit.ra fit_r1.ra				;\
	$(TOOLDIR)/slice 6 2 fit.ra fit_r2.ra				;\
	$(TOOLDIR)/phantom -x32 -T r1.ra				;\
	$(TOOLDIR)/nrmse -t 0.0001 r1.ra fit_r1.ra			;\
	$(TOOLDIR)/scale 10 r1.ra r2.ra					;\
	$(TOOLDIR)/nrmse -t 0.00015 r2.ra fit_r2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mobafit-bmc-sim: phantom sim slice repmat fmac index mobafit nrmse scale
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/phantom -x16 -T -b tubes.ra				;\
	$(TOOLDIR)/sim --ODE --seq IR-BSSFP,Nrep=100,ipl=0.01,isp=0.005,ppl=0.002,Trf=0.001,TR=0.004,TE=0.002,FA=45,BWTP=4 --BMC --pool P=3,T1=1.:1:1:1,T2=1e-4:1e-4:1e-4:1e-4,k=10.:20:20:20,M0=0.1:0.2:0.2:0.2 -1 1:1:1 -2 0.1:0.1:1 sim0.ra ;\
	$(TOOLDIR)/slice 8 0 sim0.ra sim.ra				;\
	$(TOOLDIR)/repmat 6 11 sim.ra sim2.ra				;\
	$(TOOLDIR)/fmac -s 64 tubes.ra sim2.ra ph_img.ra		;\
	$(TOOLDIR)/index 5 100 te.ra					;\
	$(TOOLDIR)/mobafit -S --sim ODE,BMC -i3 --init=1:1:10:1:1:1:1e4:1e4:10:20:0.1:0.2.:0.2:0:0 --seq IR-BSSFP,TR=0.004,TE=0.002,ipl=0.01,isp=0.005,ppl=0.002,Nrep=100,Trf=0.001,FA=45,BWTP=4 --pool P=3 --scale=1:0:1:0:0:0:0:0:0:0:0:0:0:0 te.ra ph_img.ra fit.ra ;\
	$(TOOLDIR)/slice 6 0 fit.ra r1.ra				;\
	$(TOOLDIR)/phantom -x16 -T ph.ra				;\
	$(TOOLDIR)/fmac ph.ra r1.ra r1p.ra					;\
	$(TOOLDIR)/nrmse -t 0.001 ph.ra r1p.ra				;\
	$(TOOLDIR)/slice 6 2 fit.ra r2.ra				;\
	$(TOOLDIR)/fmac ph.ra r2.ra r2p.ra					;\
	$(TOOLDIR)/scale 10 ph.ra ph2.ra				;\
	$(TOOLDIR)/nrmse -t 0.001 r2p.ra ph2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mobafit-sim-om: sim slice ones index mobafit scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/sim --ODE --seq IR-BSSFP,Nrep=100,ipl=0.01,isp=0.005,ppl=0.002,Trf=0.001,TR=0.004,TE=0.002,FA=45,BWTP=4 --BMC --pool P=3,T1=1.:1:1:1,T2=1e-4:1e-4:1e-4:1e-4,k=20.:10.:0:0,M0=0.3:0.2:0.2:0.2,Om=900:0:0:0 -1 1:1:1 -2 0.1:0.1:1 sim0.ra ;\
	$(TOOLDIR)/slice 8 0 sim0.ra sim.ra				;\
	$(TOOLDIR)/index 5 100 te.ra					;\
	$(TOOLDIR)/mobafit -S --sim ODE,BMC -i6 --init=1:1:10:1:1:1:1e4:1e4:20:10:0.3:0.2.:0.2:600:0 --seq IR-BSSFP,TR=0.004,TE=0.002,ipl=0.01,isp=0.005,ppl=0.002,Nrep=200,Trf=0.001,FA=45,BWTP=4 --pool P=3 --scale=0:0:0:0:0:0:0:0:0:0:0:0:1:0 te.ra sim.ra fit.ra ;\
	$(TOOLDIR)/slice 6 12 fit.ra om_fit.ra				;\
	$(TOOLDIR)/ones 1 1 one.ra					;\
	$(TOOLDIR)/scale 900 one.ra om_ref.ra				;\
	$(TOOLDIR)/nrmse -t 3 om_fit.ra om_ref.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mobafit-sim-pool2: sim slice index mobafit ones scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/sim --ODE --seq IR-BSSFP,Nrep=100,ipl=0.01,isp=0.009,ppl=0.011,Trf=0.001,TR=0.004,TE=0.002,FA=45,BWTP=4 --BMC --pool P=2,T1=1.:1:1:1,T2=1e-4:1e-4:1e-4:1e-4,k=20.:0:0:0,M0=0.3:0.:0.:0.,Om=0:0:0:0 -1 1:1:1 -2 0.1:0.1:1 sim0.ra ;\
	$(TOOLDIR)/slice 8 0 sim0.ra sim.ra				;\
	$(TOOLDIR)/index 5 100 te.ra					;\
	$(TOOLDIR)/mobafit -S --sim ODE,BMC -i5 --init=1:1:10:1:1:0.92e4:18:0.26:0 --seq IR-BSSFP,TR=0.004,TE=0.002,ipl=0.01,isp=0.009,ppl=0.011,Nrep=100,Trf=0.001,FA=45,BWTP=4 --pool P=2 --scale=0:0:0:0:0:1:1:1:0 te.ra sim.ra fit.ra ;\
	$(TOOLDIR)/slice 6 5 fit.ra r2_fit.ra				;\
	$(TOOLDIR)/slice 6 6 fit.ra k_fit.ra				;\
	$(TOOLDIR)/slice 6 7 fit.ra m0_fit.ra				;\
	$(TOOLDIR)/ones 1 1 one.ra					;\
	$(TOOLDIR)/scale 1e4 one.ra r2_ref.ra				;\
	$(TOOLDIR)/scale 20 one.ra k_ref.ra				;\
	$(TOOLDIR)/scale 0.3 one.ra m0_ref.ra				;\
	$(TOOLDIR)/nrmse -t 1 r2_fit.ra r2_ref.ra			;\
	$(TOOLDIR)/nrmse -t 0.2 k_fit.ra k_ref.ra			;\
	$(TOOLDIR)/nrmse -t 0.02 m0_fit.ra m0_ref.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mobafit-sim-pool3: sim slice index mobafit ones scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/sim --ODE --seq IR-BSSFP,Nrep=100,ipl=0.01,isp=0.009,ppl=0.011,Trf=0.001,TR=0.004,TE=0.002,FA=45,BWTP=4 --BMC --pool P=3,T1=1.:1:1:1,T2=1e-3:1e-3:0:0,k=15.:15:0:0,M0=0.3:0.2:0:0 -1 1:1:1 -2 0.1:0.1:1 sim0.ra ;\
	$(TOOLDIR)/slice 8 0 sim0.ra sim.ra				;\
	$(TOOLDIR)/index 5 100 te.ra					;\
	$(TOOLDIR)/mobafit -S --sim ODE,BMC -i5 --init=1:1:10:1:1:0.6:1e3:1e3:15:13:0.3:0.14.:0:0 --seq IR-BSSFP,TR=0.004,TE=0.002,ipl=0.01,isp=0.009,ppl=0.011,Nrep=100,Trf=0.001,FA=45,BWTP=4 --pool P=3 --scale=0:0:0:0:0:1:0:0:0:1:0:1:0:0 te.ra sim.ra fit.ra ;\
	$(TOOLDIR)/slice 6 5 fit.ra r1_fit.ra				;\
	$(TOOLDIR)/slice 6 9 fit.ra k_fit.ra				;\
	$(TOOLDIR)/slice 6 11 fit.ra m0_fit.ra				;\
	$(TOOLDIR)/ones 1 1 one.ra					;\
	$(TOOLDIR)/scale 15 one.ra k_ref.ra				;\
	$(TOOLDIR)/scale 0.2 one.ra m0_ref.ra				;\
	$(TOOLDIR)/nrmse -t 0.01 r1_fit.ra one.ra			;\
	$(TOOLDIR)/nrmse -t 0.2 k_fit.ra k_ref.ra			;\
	$(TOOLDIR)/nrmse -t 0.01 m0_fit.ra m0_ref.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-mobafit-irll-lm: phantom signal reshape fmac index mobafit slice threshold nrmse index extract invert scale saxpy ones
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/phantom -x32 -T -b tubes.ra								;\
	$(TOOLDIR)/signal -F -I -1 2:2:1 -3.8:.8:11 -r0.41 -n30 sig.ra				        ;\
	$(TOOLDIR)/reshape 192 11 1 sig.ra sig2.ra							;\
	$(TOOLDIR)/fmac -s 64 tubes.ra sig2.ra x.ra							;\
	$(TOOLDIR)/index 5 30 ti1.ra									;\
	$(TOOLDIR)/ones 6 1 1 1 1 1 30 ones.ra								;\
	$(TOOLDIR)/saxpy 0.5 ones.ra ti1.ra ti2.ra          						;\
	$(TOOLDIR)/scale 0.41 ti2.ra ti.ra								;\
	$(TOOLDIR)/mobafit --init=.6:1.:.8 --levenberg-marquardt -L ti.ra x.ra fit.ra		        ;\
	$(TOOLDIR)/slice 6 0 fit.ra x0.ra								;\
	$(TOOLDIR)/slice 6 1 fit.ra x1.ra								;\
	$(TOOLDIR)/slice 6 2 fit.ra x2.ra								;\
	$(TOOLDIR)/threshold -B 0.1 x0.ra mask.ra							;\
	$(TOOLDIR)/fmac  mask.ra x0.ra x0.ra								;\
	$(TOOLDIR)/fmac  mask.ra x1.ra x1.ra								;\
	$(TOOLDIR)/fmac  mask.ra x2.ra x2.ra								;\
	$(TOOLDIR)/phantom -x32 -T r1.ra								;\
	$(TOOLDIR)/nrmse -t 0.00001 r1.ra x1.ra								;\
	$(TOOLDIR)/scale .8 r1.ra r0.ra									;\
	$(TOOLDIR)/nrmse -t 0.00001 r0.ra x0.ra								;\
	$(TOOLDIR)/scale .5 r1.ra r2.ra									;\
	$(TOOLDIR)/nrmse -t 0.00001 r2.ra x2.ra								;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-mobafit-r2s tests/test-mobafit-wfr2s
TESTS += tests/test-mobafit-r2
TESTS += tests/test-mobafit-irll
TESTS += tests/test-mobafit-irll-lm
TESTS += tests/test-mobafit-ir_t1
TESTS += tests/test-mobafit-sim-om
TESTS += tests/test-mobafit-sim-pool2 tests/test-mobafit-sim-pool3

TESTS_SLOW += tests/test-mobafit-sim
TESTS_SLOW += tests/test-mobafit-bmc-sim
TESTS_GPU += tests/test-mobafit-gpu

