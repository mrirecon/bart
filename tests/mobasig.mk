tests/test-mobasig-ir: mobafit mobasig nrmse transpose scale index signal
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/signal -F -I -1 1:1:1 -r0.1 -n100 signal.ra			;\
	$(TOOLDIR)/index --end 1 100 indexTE.ra					;\
	$(TOOLDIR)/scale 0.1 indexTE.ra TEs.ra					;\
	$(TOOLDIR)/transpose 1 5 TEs.ra TEs.ra					;\
	$(TOOLDIR)/mobafit -I  --init 1:1:1  TEs.ra signal.ra testfit_R1.ra	;\
	$(TOOLDIR)/mobasig -I testfit_R1.ra TEs.ra forward_data.ra		;\
	$(TOOLDIR)/nrmse -t 0.00001 signal.ra forward_data.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-mobasig-irll: phantom signal reshape fmac index ones zeros cabs saxpy scale mobafit mobasig zexp nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/ones 7 1 1 1 1 1 1 3 ones_param.ra			;\
	$(TOOLDIR)/ones 6 1 1 1 1 1 1 ones_ti.ra			;\
	$(TOOLDIR)/scale 0.693147 ones_ti.ra ones_t1.ra			;\
	$(TOOLDIR)/mobasig -L ones_param.ra ones_t1.ra sig_ti.ra	;\
	$(TOOLDIR)/zexp sig_ti.ra exp_sig_ti.ra				;\
	$(TOOLDIR)/nrmse -t 0.00001 ones_ti.ra exp_sig_ti.ra 		;\
	$(TOOLDIR)/zeros 6 1 1 1 1 1 1 zeros_ti.ra			;\
	$(TOOLDIR)/mobasig -L ones_param.ra zeros_ti.ra sig_0.ra	;\
	$(TOOLDIR)/cabs sig_0.ra sig_0.ra				;\
	$(TOOLDIR)/nrmse -t 0.00001 ones_ti.ra sig_0.ra 		;\
	$(TOOLDIR)/scale 1e5 ones_ti.ra big_ti.ra			;\
	$(TOOLDIR)/mobasig -L ones_param.ra big_ti.ra sig_mss.ra	;\
	$(TOOLDIR)/nrmse -t 0.00001 ones_ti.ra sig_mss.ra 		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-mobasig-irll-fit: phantom signal reshape fmac index ones saxpy scale mobafit mobasig nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/phantom -x32 -T -b tubes.ra				;\
	$(TOOLDIR)/signal -F -I -1 2:2:1 -3.8:.8:11 -r0.41 -n30 sig.ra	;\
	$(TOOLDIR)/reshape 192 11 1 sig.ra sig2.ra			;\
	$(TOOLDIR)/fmac -s 64 tubes.ra sig2.ra x.ra			;\
	$(TOOLDIR)/index 5 30 ti1.ra					;\
	$(TOOLDIR)/ones 6 1 1 1 1 1 30 ones.ra				;\
	$(TOOLDIR)/saxpy 0.5 ones.ra ti1.ra ti2.ra			;\
	$(TOOLDIR)/scale 0.41 ti2.ra ti.ra				;\
	$(TOOLDIR)/mobafit --init=.6:1.:.8 -L ti.ra x.ra fit.ra		;\
	$(TOOLDIR)/mobasig -L fit.ra ti.ra y.ra				;\
	$(TOOLDIR)/nrmse -t 0.00001 x.ra y.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mobasig-mpl-fit: sim slice index vec transpose repmat saxpy scale flip mobafit mobasig nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/sim --T1 1:1:1 --T2 0.1:0.1:1 --BMC --split-dim --seq CEST,Nrep=101,Trf=0.1 --pool P=2,T1=1:1:1:1,T2=1e-4:1e-4:1e-4:1e-4,M0=0.1:0.1:0.1:0.1,k=200:10:10:10,Om=3:3:3:3 --CEST b1=1,b0=3,max=5,min=-5,n_p=20,t_d=0.01 mag.ra 		;\
	$(TOOLDIR)/slice 0 2 mag.ra magZ.ra				;\
	$(TOOLDIR)/slice 8 0 magZ.ra spectra.ra 			;\
	$(TOOLDIR)/index --end 5 101 indexOffres.ra 			;\
	$(TOOLDIR)/vec 50 onres.ra					;\
	$(TOOLDIR)/transpose 0 5 onres.ra onres.ra			;\
	$(TOOLDIR)/repmat 5 101 onres.ra onres.ra 			;\
	$(TOOLDIR)/saxpy -- -1 indexOffres.ra onres.ra Offsets.ra 	;\
	$(TOOLDIR)/scale 0.1 Offsets.ra Offsets.ra 			;\
	$(TOOLDIR)/flip 5 Offsets.ra Offsets.ra 			;\
	$(TOOLDIR)/mobafit  -i5 --liniter=10 -M 2 --levenberg-marquardt --init=1.:1.:2.:0.:0.1:1.:3 --min=0.:0.2:0.4:-1.:0.:0.1:2 --max=1.3:1.:4.:1.:0.4:4.:5. --min-flag=128 --max-flag=128 Offsets.ra spectra.ra param_fit.ra 	;\
	$(TOOLDIR)/mobasig -M param_fit.ra Offsets.ra signal.ra 	;\
	$(TOOLDIR)/nrmse -t 0.01 spectra.ra signal.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mobasig-r2: phantom signal reshape fmac index mobafit mobasig slice nrmse index extract invert
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/phantom -x32 -T -b tubes.ra				;\
	$(TOOLDIR)/signal -250:160:11 -S -e10 -n16 sig.ra		;\
	$(TOOLDIR)/reshape 192 11 1 sig.ra sig2.ra			;\
	$(TOOLDIR)/fmac -s 64 tubes.ra sig2.ra x.ra			;\
	$(TOOLDIR)/index 5 16 te.ra					;\
	$(TOOLDIR)/mobafit -T -i20 te.ra x.ra fit.ra			;\
	$(TOOLDIR)/mobasig -T fit.ra te.ra forward.ra 			;\
	$(TOOLDIR)/nrmse -t 0.0001 forward.ra x.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mobasig-wfr2s: phantom signal fmac index scale extract mobafit saxpy cabs spow ones slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x16 -c circ.ra						;\
	$(TOOLDIR)/signal -G --fat -n8 -1 3:3:1 -2 0.02:0.02:1 signal_p1.ra		;\
	$(TOOLDIR)/extract 5 1 8 signal_p1.ra signal.ra					;\
	$(TOOLDIR)/fmac circ.ra signal.ra echoes.ra 					;\
	$(TOOLDIR)/index 5 8 tmp1.ra							;\
	$(TOOLDIR)/scale 1.6 tmp1.ra tmp2.ra						;\
	$(TOOLDIR)/extract 5 1 8 tmp2.ra TE.ra						;\
	$(TOOLDIR)/mobafit -G -m1 TE.ra echoes.ra reco.ra				;\
	$(TOOLDIR)/mobasig -G -m1 reco.ra TE.ra forward.ra				;\
	$(TOOLDIR)/nrmse -t 0.0001 forward.ra echoes.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
	
TESTS += tests/test-mobasig-ir
TESTS += tests/test-mobasig-irll
TESTS += tests/test-mobasig-irll-fit
TESTS += tests/test-mobasig-r2
TESTS += tests/test-mobasig-wfr2s

TESTS_SLOW += tests/test-mobasig-mpl-fit
