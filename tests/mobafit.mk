
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


TESTS += tests/test-mobafit-r2s tests/test-mobafit-wfr2s
