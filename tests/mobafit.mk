
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

TESTS += tests/test-mobafit-r2s tests/test-mobafit-wfr2s
TESTS += tests/test-mobafit-r2
TESTS += tests/test-mobafit-irll

TESTS_GPU += tests/test-mobafit-gpu
