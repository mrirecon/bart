
tests/test-pixelfit-T2: scale phantom signal fmac index extract pixelfit slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	                  ;\
	$(TOOLDIR)/phantom -x16 -c circ.ra                 ;\
	$(TOOLDIR)/signal -G -n7 -1 3:3:1 -2 0.02:0.02:1 signal.ra        ;\
	$(TOOLDIR)/fmac circ.ra signal.ra echoes.ra             ;\
	$(TOOLDIR)/index 5 8 tmp1.ra                                      ;\
	$(TOOLDIR)/scale 1.6 tmp1.ra tmp2.ra                              ;\
	$(TOOLDIR)/extract 5 1 8 tmp2.ra TE.ra                            ;\
	$(TOOLDIR)/pixelfit -G -m3 echoes.ra TE.ra reco.ra   ;\
	$(TOOLDIR)/slice 6 1 reco.ra R2S.ra                               ;\
	$(TOOLDIR)/phantom -x16 -c circ.ra                                 ;\
	$(TOOLDIR)/fmac R2S.ra circ.ra masked.ra                     ;\
	$(TOOLDIR)/scale -- 0.05 circ.ra ref.ra                             ;\
	$(TOOLDIR)/nrmse -t 0.005 ref.ra masked.ra                         ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-pixelfit-T2