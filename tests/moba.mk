


tests/test-moba-t1: phantom signal fft ones index scale moba looklocker fmac nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -n100 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 100 psf.ra					;\
	$(TOOLDIR)/index 5 100 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -L -i11 -f1 -C100 --other pinit=1:1:2:1 --scale_data=5000. --scale_psf=1000. --normalize_scaling -p psf.ra k_space.ra TI.ra reco.ra	;\
	$(TOOLDIR)/looklocker -t0.1 -D0. reco.ra T1.ra		    			;\
	$(TOOLDIR)/fmac T1.ra circ.ra masked.ra		    				;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.005 masked.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-t1-tv: phantom signal fft ones index scale moba looklocker fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -n100 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 100 psf.ra					;\
	$(TOOLDIR)/index 5 100 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -L -i11 -rT:3:0:0.001 -f1 -C100 --other tvscale=1.:1.:0.:0.,pinit=1:1:2:1 --scale_data=5000. --scale_psf=1000. --normalize_scaling -p psf.ra k_space.ra TI.ra reco.ra	;\
	$(TOOLDIR)/looklocker -t0.1 -D0. reco.ra T1.ra		    			;\
	$(TOOLDIR)/fmac T1.ra circ.ra masked.ra		    				;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.001 masked.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-moba-t1-gpu: phantom signal fmac fft ones index scale moba nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -n100 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 100 psf.ra					;\
	$(TOOLDIR)/index 5 100 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -L -i11 -f1 -C100 --other pinit=1:1:2:1 --scale_data=5000. --scale_psf=1000. --normalize_scaling -p psf.ra k_space.ra TI.ra reco.ra	;\
	$(TOOLDIR)/moba -g -L -i11 -f1 -C100 --other pinit=1:1:2:1 --scale_data=5000. --scale_psf=1000. --normalize_scaling -p psf.ra k_space.ra TI.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.0005 reco2.ra reco.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-moba-t1-magn: phantom signal fft ones index scale moba normalize slice fmac nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -n100 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 100 psf.ra					;\
	$(TOOLDIR)/index 5 100 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -L -i11 -f1 -C30 --other pinit=1:1:2:1 --scale_data=5000. --scale_psf=1000. --normalize_scaling -p psf.ra k_space.ra TI.ra reco.ra sens.ra	;\
	$(TOOLDIR)/normalize 8 sens.ra norm.ra						;\
	$(TOOLDIR)/slice 6 0 reco.ra magn.ra						;\
	$(TOOLDIR)/fmac magn.ra norm.ra reco2.ra		    			;\
	$(TOOLDIR)/nrmse -s -t 0.001 circ.ra reco2.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-t1-sms: phantom signal repmat fft ones index scale moba looklocker fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 circ.ra 		                  		;\
	$(TOOLDIR)/repmat 13 3 circ.ra circ2.ra		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -n100 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ2.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 8195 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 16 16 16 1 1 1 100 1 1 1 1 1 1 1 3 1 1 psf.ra			;\
	$(TOOLDIR)/index 5 100 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -M -L -i11 -f1 -C150 --other pinit=1:1:2:1 --scale_data=3000. --scale_psf=600. --normalize_scaling -p psf.ra k_space.ra TI.ra reco.ra		;\
	$(TOOLDIR)/looklocker -t0.1 -D0. reco.ra T1.ra		    			;\
	$(TOOLDIR)/fmac T1.ra circ2.ra masked.ra		 			;\
	$(TOOLDIR)/scale -- 1.12 circ2.ra ref.ra		    			;\
	$(TOOLDIR)/nrmse -t 0.008 masked.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-t1-no-IR: phantom signal fft ones index scale moba looklocker fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -F -r0.005 -n300 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 300 psf.ra					;\
	$(TOOLDIR)/index 5 300 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -L -i11 -f1 -C30 --other pinit=1:2:1:1 --scale_data=5000. --scale_psf=1000. --normalize_scaling -p psf.ra k_space.ra TI.ra reco.ra		;\
	$(TOOLDIR)/looklocker -t0.1 -D0. reco.ra T1.ra		    			;\
	$(TOOLDIR)/fmac T1.ra circ.ra masked.ra		    				;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.005 masked.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-t1-nonCartesian: traj transpose phantom signal nufft fft ones index scale moba looklocker resize fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x16 -y1 -r -D -G -s7 -t300 _traj.ra  		                ;\
	$(TOOLDIR)/transpose 5 10 _traj.ra _traj2.ra    		                ;\
	$(TOOLDIR)/scale 0.5 _traj2.ra traj.ra   	    		                ;\
	$(TOOLDIR)/phantom -k -c -t traj.ra basis_geom.ra    	    		        ;\
	$(TOOLDIR)/signal -F -I -r0.005 -n300 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac -s 64 basis_geom.ra signal.ra data.ra				;\
 	$(TOOLDIR)/ones 16 1 16 1 1 1 300 1 1 1 1 1 1 1 1 1 1 ones.ra	   		;\
	$(TOOLDIR)/nufft -x 16:16:1 -a _traj2.ra ones.ra pattern.ra	   		;\
	$(TOOLDIR)/fft -u 3 pattern.ra psf.ra				   		;\
	$(TOOLDIR)/nufft -x 16:16:1 -a _traj2.ra data.ra zerofill_reco.ra  		;\
	$(TOOLDIR)/fft -u 3 zerofill_reco.ra k_space.ra					;\
	$(TOOLDIR)/index 5 300 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -L -l1 -i11 -C30 -j0.01 --other pinit=1:2:1:1 --scale_data=5000. --scale_psf=1000. --normalize_scaling -p psf.ra k_space.ra TI.ra reco.ra	;\
	$(TOOLDIR)/looklocker -t0.1 -D0. reco.ra T1.ra		    			;\
	$(TOOLDIR)/resize -c 0 8 1 8 T1.ra T1_crop.ra					;\
	$(TOOLDIR)/phantom -x8 -c circ.ra						;\
	$(TOOLDIR)/fmac T1_crop.ra circ.ra masked.ra	    				;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.02 masked.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-t1-nufft: traj transpose phantom signal nufft fft ones index scale moba fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x16 -y1 -r -D -G -s7 -t300 traj2.ra  		                ;\
	$(TOOLDIR)/transpose 5 10 traj2.ra traj2T.ra    		                ;\
	$(TOOLDIR)/scale 0.5 traj2T.ra traj.ra   	    		                ;\
	$(TOOLDIR)/phantom -k -c -t traj.ra basis_geom.ra    	    		        ;\
	$(TOOLDIR)/signal -F -I -r0.005 -n300 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac -s 64 basis_geom.ra signal.ra data.ra				;\
 	$(TOOLDIR)/ones 16 1 16 1 1 1 300 1 1 1 1 1 1 1 1 1 1 ones.ra	   		;\
	$(TOOLDIR)/nufft -x 16:16:1 -a traj2T.ra ones.ra pattern.ra	   		;\
	$(TOOLDIR)/fft -u 3 pattern.ra psf.ra				   		;\
	$(TOOLDIR)/nufft -x 16:16:1 -a traj2T.ra data.ra zerofill_reco.ra  		;\
	$(TOOLDIR)/fft -u 3 zerofill_reco.ra k_space.ra					;\
	$(TOOLDIR)/index 5 300 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -L -l1 -i11 -C30 -j0.01 --scale_data=5000. --scale_psf=1000. --normalize_scaling -p psf.ra k_space.ra TI.ra reco.ra	;\
	$(TOOLDIR)/moba -L -l1 -i11 -C30 -j0.01 --scale_data=5000. --scale_psf=1000. --normalize_scaling -o1.0 -t traj.ra data.ra TI.ra reco2.ra ;\
	$(TOOLDIR)/nrmse -t 0.00007 reco.ra reco2.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-t2: phantom signal fmac fft ones index scale moba slice invert nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -T -e0.01 -n16 -1 1.25:1.25:1 -2 0.09:0.09:1 signal.ra  	;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 16 psf.ra						;\
	$(TOOLDIR)/index 5 16 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.01 tmp1.ra TE.ra                  	       			;\
	$(TOOLDIR)/moba -F -i10 -f1 -C30 --scale_data=5000. --scale_psf=1000. --normalize_scaling -d4 -p psf.ra k_space.ra TE.ra reco.ra	;\
	$(TOOLDIR)/slice 6 1 reco.ra R2.ra						;\
	$(TOOLDIR)/invert R2.ra T2.ra							;\
	$(TOOLDIR)/phantom -x16 -c circ.ra						;\
	$(TOOLDIR)/fmac T2.ra circ.ra masked.ra						;\
	$(TOOLDIR)/scale -- 0.9 circ.ra ref.ra						;\
	$(TOOLDIR)/nrmse -t 0.0008 masked.ra ref.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-meco-noncart-r2s: traj scale phantom signal fmac index extract moba slice resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	                  ;\
	$(TOOLDIR)/traj -x16 -y15 -r -D -E -e7 -c traj2.ra                ;\
	$(TOOLDIR)/scale 0.5 traj2.ra traj.ra                             ;\
	$(TOOLDIR)/phantom -k -c -t traj.ra basis_geom.ra                 ;\
	$(TOOLDIR)/signal -G -n8 -1 3:3:1 -2 0.02:0.02:1 signal_p1.ra     ;\
	$(TOOLDIR)/extract 5 1 8 signal_p1.ra signal.ra                   ;\
	$(TOOLDIR)/fmac -s 64 basis_geom.ra signal.ra data.ra             ;\
	$(TOOLDIR)/index 5 8 tmp1.ra                                      ;\
	$(TOOLDIR)/scale 1.6 tmp1.ra tmp2.ra                              ;\
	$(TOOLDIR)/extract 5 1 8 tmp2.ra TE.ra                            ;\
	$(TOOLDIR)/moba -G -m3 -rQ:1 -rS:0 -rW:3:64:1 -i10 -C100 -u0.0001 -R3 -o1.5 -k --kfilter-2 -t traj.ra data.ra TE.ra reco.ra   ;\
	$(TOOLDIR)/slice 6 1 reco.ra R2S.ra                               ;\
	$(TOOLDIR)/resize -c 0 8 1 8 R2S.ra R2S_crop.ra                   ;\
	$(TOOLDIR)/phantom -x8 -c circ.ra                                 ;\
	$(TOOLDIR)/fmac R2S_crop.ra circ.ra masked.ra                     ;\
	$(TOOLDIR)/scale -- 50 circ.ra ref.ra                             ;\
	$(TOOLDIR)/nrmse -t 0.008 ref.ra masked.ra                        ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-meco-noncart-wfr2s: traj scale phantom signal fmac index extract moba slice resize saxpy cabs spow nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	                  ;\
	$(TOOLDIR)/traj -x16 -y15 -r -D -E -e7 -c traj2.ra                ;\
	$(TOOLDIR)/scale 0.5 traj2.ra traj.ra                             ;\
	$(TOOLDIR)/phantom -k -c -t traj.ra basis_geom.ra                 ;\
	$(TOOLDIR)/signal -G --fat -n8 -1 3:3:1 -2 0.02:0.02:1 signal_p1.ra  ;\
	$(TOOLDIR)/extract 5 1 8 signal_p1.ra signal.ra                   ;\
	$(TOOLDIR)/fmac -s 64 basis_geom.ra signal.ra data.ra             ;\
	$(TOOLDIR)/index 5 8 tmp1.ra                                      ;\
	$(TOOLDIR)/scale 1.6 tmp1.ra tmp2.ra                              ;\
	$(TOOLDIR)/extract 5 1 8 tmp2.ra TE.ra                            ;\
	$(TOOLDIR)/moba -G -m1 -rQ:1 -rS:0 -rW:3:64:1 -i10 -C100 -u0.0001 -R3 -o1.5 -k --kfilter-2 -t traj.ra data.ra TE.ra reco.ra   ;\
	$(TOOLDIR)/resize -c 0 8 1 8 reco.ra reco_crop.ra                 ;\
	$(TOOLDIR)/slice 6 0 reco_crop.ra W.ra                            ;\
	$(TOOLDIR)/slice 6 1 reco_crop.ra F.ra                            ;\
	$(TOOLDIR)/slice 6 2 reco_crop.ra R2S.ra                          ;\
	$(TOOLDIR)/slice 6 3 reco_crop.ra fB0.ra                          ;\
	$(TOOLDIR)/saxpy 1 W.ra F.ra temp_inphase.ra                      ;\
	$(TOOLDIR)/cabs temp_inphase.ra temp_inphase_abs.ra               ;\
	$(TOOLDIR)/spow -- -1. temp_inphase_abs.ra temp_deno.ra           ;\
	$(TOOLDIR)/cabs F.ra temp_F_abs.ra                                ;\
	$(TOOLDIR)/fmac temp_F_abs.ra temp_deno.ra fatfrac.ra             ;\
	$(TOOLDIR)/phantom -x8 -c circ.ra                                 ;\
	$(TOOLDIR)/fmac fatfrac.ra circ.ra fatfrac_masked.ra              ;\
	$(TOOLDIR)/scale -- 0.20 circ.ra fatfrac_ref.ra                   ;\
	$(TOOLDIR)/nrmse -t 0.02 fatfrac_ref.ra fatfrac_masked.ra         ;\
	$(TOOLDIR)/fmac R2S.ra circ.ra R2S_masked.ra                      ;\
	$(TOOLDIR)/scale -- 50 circ.ra R2S_ref.ra                         ;\
	$(TOOLDIR)/nrmse -t 0.008 R2S_ref.ra R2S_masked.ra                ;\
	$(TOOLDIR)/fmac fB0.ra circ.ra fB0_masked.ra                      ;\
	$(TOOLDIR)/scale -- 20 circ.ra fB0_ref.ra                         ;\
	$(TOOLDIR)/nrmse -t 0.0003 fB0_ref.ra fB0_masked.ra               ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-bloch-irflash-psf: phantom signal fft ones index scale moba slice fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -f8 -n100 -1 1.25:1.25:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 100 psf.ra					;\
	$(TOOLDIR)/index 5 100 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba --bloch --sim STM --seq IR-FLASH,TR=0.005,TE=0.003,FA=6,Trf=0.00001,BWTP=4,pinv,ipl=0,ppl=0 --other pscale=1:1:1:1,pinit=3:1:1:1 -i11 -C300 -s0.95 -R3 -f1 -o1 -j0.001 --scale_data=5000. --scale_psf=1000. --normalize_scaling -p psf.ra k_space.ra TI.ra reco.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra r1map.ra						;\
	$(TOOLDIR)/phantom -x16 -c circ.ra						;\
	$(TOOLDIR)/fmac r1map.ra circ.ra masked.ra	    				;\
	$(TOOLDIR)/scale -- 0.8 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.014 masked.ra ref.ra			    		;\
	$(TOOLDIR)/slice 6 3 reco.ra famap.ra						;\
	$(TOOLDIR)/fmac famap.ra circ.ra famasked.ra	    				;\
	$(TOOLDIR)/scale -- 1.333 circ.ra faref.ra			    		;\
	$(TOOLDIR)/nrmse -t 0.007 famasked.ra faref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-bloch-irflash-traj: traj repmat phantom signal fmac ones scale index moba slice spow fmac resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
	$(TOOLDIR)/repmat 5 1000 _traj.ra traj2.ra	;\
	$(TOOLDIR)/scale 0.5 traj2.ra traj.ra						;\
	$(TOOLDIR)/phantom -c -k -t traj.ra circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -f8 -n1000 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra k_space.ra					;\
	$(TOOLDIR)/index 5 1000 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba --bloch --sim STM --seq IR-FLASH,TR=0.005,TE=0.003,FA=6,Trf=0.00001,BWTP=4,pinv,ipl=0,ppl=0 --other pscale=1:1:1:1,pinit=3:1:1:1 -i11 -C300 -s0.95 -d3 -R3 -o1 -j0.001 --scale_data=5000. --scale_psf=1000. --normalize_scaling -t traj.ra k_space.ra TI.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra r1map.ra						;\
	$(TOOLDIR)/spow -- -1. r1map.ra t1map.ra						;\
	$(TOOLDIR)/phantom -x8 -c circ2.ra						;\
	$(TOOLDIR)/resize -c 0 16 1 16 circ2.ra circ.ra					;\
	$(TOOLDIR)/fmac t1map.ra circ.ra masked.ra	    				;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.012 masked.ra ref.ra			    		;\
	$(TOOLDIR)/slice 6 3 reco.ra famap.ra						;\
	$(TOOLDIR)/fmac famap.ra circ.ra famasked.ra	    				;\
	$(TOOLDIR)/scale -- 1.333 circ.ra faref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.0016 famasked.ra faref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-bloch-irflash-traj-fixfa: traj repmat phantom signal fmac scale index moba slice spow fmac resize nrmse ones
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
	$(TOOLDIR)/repmat 5 1000 _traj.ra traj2.ra	;\
	$(TOOLDIR)/scale 0.5 traj2.ra traj.ra						;\
	$(TOOLDIR)/phantom -c -k -t traj.ra circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -f8 -n1000 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra k_space.ra					;\
	$(TOOLDIR)/index 5 1000 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba --bloch --sim STM --seq IR-FLASH,TR=0.005,TE=0.003,FA=8,Trf=0.00001,BWTP=4,pinv,ipl=0,ppl=0 --other pscale=1:1:1:0,pinit=3:1:1:1 -i11 -C30 -s0.95 -R3 -o1 -j0.001 --scale_data=5000. --scale_psf=1000. --normalize_scaling -t traj.ra k_space.ra TI.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra r1map.ra						;\
	$(TOOLDIR)/spow -- -1. r1map.ra t1map.ra						;\
	$(TOOLDIR)/phantom -x8 -c circ2.ra						;\
	$(TOOLDIR)/resize -c 0 16 1 16 circ2.ra circ.ra					;\
	$(TOOLDIR)/fmac t1map.ra circ.ra masked.ra	    				;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.012 masked.ra ref.ra			    		;\
	$(TOOLDIR)/slice 6 3 reco.ra b1map.ra						;\
	$(TOOLDIR)/ones 2 16 16 ones.ra						;\
	$(TOOLDIR)/scale 8 b1map.ra b1maps.ra                    	       		;\
	$(TOOLDIR)/scale 8 ones.ra fa.ra                    	       		;\
	$(TOOLDIR)/nrmse -t 0.00001 fa.ra b1maps.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-bloch-irflash-r2fix: traj repmat scale phantom signal fmac index moba slice ones nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
	$(TOOLDIR)/repmat 5 1000 _traj.ra traj2.ra	;\
	$(TOOLDIR)/scale 0.5 traj2.ra traj.ra						;\
	$(TOOLDIR)/phantom -c -k -t traj.ra circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -f8 -n1000 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra k_space.ra					;\
	$(TOOLDIR)/index 5 1000 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba --bloch --sim STM --seq IR-FLASH,TR=0.005,TE=0.003,FA=8,Trf=0.00001,BWTP=4,pinv,ipl=0,ppl=0 --other pscale=1:1:1:1,pinit=3:1:1:1 -i11 -C30 -s0.95 -R3 -o1 -j0.001 --scale_data=5000. --scale_psf=1000. --normalize_scaling -t traj.ra k_space.ra TI.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 2 reco.ra r2map.ra						;\
	$(TOOLDIR)/ones 2 16 16 ones.ra						;\
	$(TOOLDIR)/nrmse -t 0.00001 ones.ra r2map.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-t1-phy-psf: phantom signal fft ones index scale moba slice spow fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -n300 -1 1.25:1.25:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 300 psf.ra					;\
	$(TOOLDIR)/index 5 300 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -P -i11 -C250 -s0.95 -f1 -R3 -o1 -j0.001 --scale_data=5000. --scale_psf=1000. --normalize_scaling -d3 --seq TR=0.005 -p psf.ra k_space.ra TI.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 1 reco.ra r1map.ra						;\
	$(TOOLDIR)/phantom -x16 -c circ.ra						;\
	$(TOOLDIR)/fmac r1map.ra circ.ra masked.ra	    				;\
	$(TOOLDIR)/scale -- 0.8 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.005 masked.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-t1-phy-traj: traj repmat scale phantom signal fmac index moba slice resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x12 -y12 _traj.ra                  				;\
	$(TOOLDIR)/repmat 5 300 _traj.ra traj2.ra					;\
	$(TOOLDIR)/scale 0.5 traj2.ra traj.ra						;\
	$(TOOLDIR)/phantom -k -c -t traj.ra basis_geom.ra				;\
	$(TOOLDIR)/signal -F -I -r0.005 -f8 -n300 -1 1.25:1.25:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac basis_geom.ra signal.ra k_space.ra				;\
	$(TOOLDIR)/index 5 300 tmp1.ra							;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra						;\
	$(TOOLDIR)/moba -P -i11 -C250 -s0.95 -f1 -R3 -o1 -j0.001 --scale_data=5000. --scale_psf=1000. --normalize_scaling -d3 --seq TR=0.005 -t traj.ra k_space.ra TI.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 1 reco.ra r1map.ra						;\
	$(TOOLDIR)/phantom -x6 -c circ2.ra						;\
	$(TOOLDIR)/resize -c 0 12 1 12 circ2.ra circ.ra					;\
	$(TOOLDIR)/fmac r1map.ra circ.ra r1masked.ra	    				;\
	$(TOOLDIR)/scale -- 0.8 circ.ra r1ref.ra			    		;\
	$(TOOLDIR)/nrmse -t 0.009 r1masked.ra r1ref.ra			    		;\
	$(TOOLDIR)/slice 6 2 reco.ra famap.ra						;\
	$(TOOLDIR)/fmac famap.ra circ.ra famasked.ra	    				;\
	$(TOOLDIR)/scale -- 8 circ.ra faref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.02 famasked.ra faref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-bloch-irbssfp-psf: traj repmat phantom signal fmac fft ones index moba spow slice scale resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -B -I -r 0.0045 -e 0.00225 -f45 -n 1000 -1 1.25:1.25:1 -2 0.1:0.1:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 1000 psf.ra					;\
	$(TOOLDIR)/index 5 1000 dummy_ti.ra 	;\
	$(TOOLDIR)/moba --bloch --sim STM --seq IR-BSSFP,TR=0.0045,TE=0.00225,FA=45,Trf=0.00001,BWTP=4,pinv,ipl=0,ppl=0.00225 --other pscale=1:1:1:0,pinit=3:1:1:1 -i11 -C300 -s0.95 -R3 -o1 -j0.001 --scale_data=5000. --scale_psf=1000. --normalize_scaling -p psf.ra k_space.ra dummy_ti.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra r1map.ra				;\
	$(TOOLDIR)/phantom -x 8 -c ref.ra				;\
	$(TOOLDIR)/resize -c 0 16 1 16 ref.ra ref2.ra					;\
	$(TOOLDIR)/fmac r1map.ra ref2.ra masked_r1.ra				;\
	$(TOOLDIR)/scale -- 0.8 ref2.ra ref3.ra				;\
	$(TOOLDIR)/nrmse -t 0.014 masked_r1.ra ref3.ra				;\
	$(TOOLDIR)/slice 6 2 reco.ra r2map.ra				;\
	$(TOOLDIR)/fmac r2map.ra ref2.ra masked_r2.ra				;\
	$(TOOLDIR)/scale -- 10 ref2.ra ref4.ra				;\
	$(TOOLDIR)/nrmse -t 0.01 masked_r2.ra ref4.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-bloch-irbssfp-traj: traj repmat phantom signal fmac index moba spow slice scale resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
	$(TOOLDIR)/repmat 5 1000 _traj.ra traj2.ra	;\
	$(TOOLDIR)/scale 0.5 traj2.ra traj.ra						;\
	$(TOOLDIR)/phantom -c -k -t traj.ra basis_geom.ra				;\
	$(TOOLDIR)/signal -B -I -r 0.0045 -e 0.00225 -f45 -n 1000 -1 1.25:1.25:1 -2 0.1:0.1:1 basis_simu.ra	;\
	$(TOOLDIR)/fmac basis_geom.ra basis_simu.ra k_space.ra		;\
	$(TOOLDIR)/index 5 1000 dummy_ti.ra 	;\
	$(TOOLDIR)/moba --bloch --sim STM --seq IR-BSSFP,TR=0.0045,TE=0.00225,FA=45,Trf=0.00001,BWTP=4,pinv,ipl=0,ppl=0.00225 --other pscale=1:1:1:0,pinit=3:1:1:1 -i11 -C300 -s0.95 -R3 -o1 -j0.001 --scale_data=5000. --scale_psf=1000. --normalize_scaling -t traj.ra k_space.ra dummy_ti.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra r1map.ra				;\
	$(TOOLDIR)/phantom -x 8 -c ref.ra				;\
	$(TOOLDIR)/resize -c 0 16 1 16 ref.ra ref2.ra					;\
	$(TOOLDIR)/fmac r1map.ra ref2.ra masked_r1.ra				;\
	$(TOOLDIR)/scale -- 0.8 ref2.ra ref3.ra				;\
	$(TOOLDIR)/nrmse -t 0.014 masked_r1.ra ref3.ra				;\
	$(TOOLDIR)/slice 6 2 reco.ra r2map.ra				;\
	$(TOOLDIR)/fmac r2map.ra ref2.ra masked_r2.ra				;\
	$(TOOLDIR)/scale -- 10 ref2.ra ref4.ra				;\
	$(TOOLDIR)/nrmse -t 0.01 masked_r2.ra ref4.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-bloch-irbssfp-traj-input-b1: traj repmat phantom signal fmac index moba spow slice scale resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
	$(TOOLDIR)/repmat 5 1000 _traj.ra traj2.ra	;\
	$(TOOLDIR)/scale 0.5 traj2.ra traj.ra						;\
	$(TOOLDIR)/phantom -c -k -t traj.ra basis_geom.ra				;\
	$(TOOLDIR)/signal -B -I -r 0.0045 -e 0.00225 -f45 -n 1000 -1 1.25:1.25:1 -2 0.1:0.1:1 basis_simu.ra	;\
	$(TOOLDIR)/fmac basis_geom.ra basis_simu.ra k_space.ra		;\
	$(TOOLDIR)/index 5 1000 dummy_ti.ra 	;\
	$(TOOLDIR)/ones 2 16 16 ones.ra				;\
	$(TOOLDIR)/scale -- 45 ones.ra b1map.ra				;\
	$(TOOLDIR)/moba --bloch --sim STM --seq IR-BSSFP,TR=0.0045,TE=0.00225,FA=1,Trf=0.00001,BWTP=4,pinv,ipl=0,ppl=0.00225 --other pscale=1:1:1:0,pinit=3:1:1:1,b1map=b1map.ra -i11 -C300 -s0.95 -R3 -o1 -j0.001 --scale_data=5000. --scale_psf=1000. --normalize_scaling -t traj.ra k_space.ra dummy_ti.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra r1map.ra				;\
	$(TOOLDIR)/phantom -x 8 -c ref.ra				;\
	$(TOOLDIR)/resize -c 0 16 1 16 ref.ra ref2.ra					;\
	$(TOOLDIR)/fmac r1map.ra ref2.ra masked_r1.ra				;\
	$(TOOLDIR)/scale -- 0.8 ref2.ra ref3.ra				;\
	$(TOOLDIR)/nrmse -t 0.014 masked_r1.ra ref3.ra				;\
	$(TOOLDIR)/slice 6 2 reco.ra r2map.ra				;\
	$(TOOLDIR)/fmac r2map.ra ref2.ra masked_r2.ra				;\
	$(TOOLDIR)/scale -- 10 ref2.ra ref4.ra				;\
	$(TOOLDIR)/nrmse -t 0.01 masked_r2.ra ref4.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-bloch-irbssfp-traj-av-spokes: traj repmat phantom signal fmac index moba spow slice scale resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
	$(TOOLDIR)/repmat 5 100 _traj.ra traj2.ra	;\
	$(TOOLDIR)/scale 0.5 traj2.ra traj.ra						;\
	$(TOOLDIR)/phantom -c -k -t traj.ra basis_geom.ra				;\
	$(TOOLDIR)/signal -B -I -r 0.0045 -e 0.00225 -f45 -n 100 -1 1.25:1.25:1 -2 0.1:0.1:1 --av-spokes 10 basis_simu.ra	;\
	$(TOOLDIR)/fmac basis_geom.ra basis_simu.ra k_space.ra		;\
	$(TOOLDIR)/index 5 100 dummy_ti.ra 	;\
	$(TOOLDIR)/moba --bloch --sim STM --seq IR-BSSFP,TR=0.0045,TE=0.00225,FA=45,Trf=0.00001,BWTP=4,pinv,ipl=0,ppl=0.00225,av-spokes=10 --other pscale=1:1:1:0,pinit=3:1:1:1 -i11 -C150 -s0.95 -R3 -o1 -j0.001 --scale_data=5000. --scale_psf=1000. --normalize_scaling -t traj.ra k_space.ra dummy_ti.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra r1map.ra				;\
	$(TOOLDIR)/phantom -x 8 -c ref.ra				;\
	$(TOOLDIR)/resize -c 0 16 1 16 ref.ra ref2.ra					;\
	$(TOOLDIR)/fmac r1map.ra ref2.ra masked_r1.ra				;\
	$(TOOLDIR)/scale -- 0.8 ref2.ra ref3.ra				;\
	$(TOOLDIR)/nrmse -t 0.014 masked_r1.ra ref3.ra				;\
	$(TOOLDIR)/slice 6 2 reco.ra r2map.ra				;\
	$(TOOLDIR)/fmac r2map.ra ref2.ra masked_r2.ra				;\
	$(TOOLDIR)/scale -- 10 ref2.ra ref4.ra				;\
	$(TOOLDIR)/nrmse -t 0.01 masked_r2.ra ref4.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-bloch-irbssfp-traj-slice-profile: traj repmat scale phantom sim fmac index moba slice resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
	$(TOOLDIR)/repmat 5 100 _traj.ra traj2.ra	;\
	$(TOOLDIR)/scale 0.5 traj2.ra traj.ra						;\
	$(TOOLDIR)/phantom -c -k -t traj.ra basis_geom.ra				;\
	$(TOOLDIR)/sim --seq IR-BSSFP,TR=0.0045,TE=0.00225,Nrep=100,pinv,ipl=0,ppl=0.00225,Trf=0.001,FA=45,BWTP=4,av-spokes=10,slice-thickness=0.04,sl-grad=0.01,Nspins=61 -1 1.25:1.25:1 -2 0.1:0.1:1 basis_simu.ra ;\
	$(TOOLDIR)/fmac basis_geom.ra basis_simu.ra k_space.ra		;\
	$(TOOLDIR)/index 5 100 dummy_ti.ra 	;\
	$(TOOLDIR)/moba --bloch --sim STM --seq IR-BSSFP,TR=0.0045,TE=0.00225,FA=45,Trf=0.001,BWTP=4,pinv,ipl=0,ppl=0.00225,av-spokes=10,slice-thickness=0.02,sl-grad=0.01,Nspins=11 --other pscale=1:1:1:0,pinit=3:1:1:1 -i11 -C250 --scale_data=5000. --scale_psf=1000. --normalize_scaling -s0.95 -R3 -o1 -j0.001 -t traj.ra k_space.ra dummy_ti.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra r1map.ra				;\
	$(TOOLDIR)/phantom -x 8 -c ref.ra				;\
	$(TOOLDIR)/resize -c 0 16 1 16 ref.ra ref2.ra					;\
	$(TOOLDIR)/fmac r1map.ra ref2.ra masked_r1.ra				;\
	$(TOOLDIR)/scale -- 0.8 ref2.ra ref3.ra				;\
	$(TOOLDIR)/nrmse -t 0.001 masked_r1.ra ref3.ra				;\
	$(TOOLDIR)/slice 6 2 reco.ra r2map.ra				;\
	$(TOOLDIR)/fmac r2map.ra ref2.ra masked_r2.ra				;\
	$(TOOLDIR)/scale -- 10 ref2.ra ref4.ra				;\
	$(TOOLDIR)/nrmse -t 0.01 masked_r2.ra ref4.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS_SLOW += tests/test-moba-t1 tests/test-moba-t1-sms tests/test-moba-t1-no-IR
TESTS_SLOW += tests/test-moba-t1-magn tests/test-moba-t1-nonCartesian tests/test-moba-t1-nufft
TESTS_SLOW += tests/test-moba-t2 tests/test-moba-t1-tv
TESTS_SLOW += tests/test-moba-meco-noncart-r2s tests/test-moba-meco-noncart-wfr2s
TESTS_SLOW += tests/test-moba-bloch-irflash-psf tests/test-moba-bloch-irflash-traj tests/test-moba-bloch-irflash-traj-fixfa tests/test-moba-bloch-irflash-r2fix
TESTS_SLOW += tests/test-moba-t1-phy-psf tests/test-moba-t1-phy-traj
TESTS_SLOW += tests/test-moba-bloch-irbssfp-psf tests/test-moba-bloch-irbssfp-traj tests/test-moba-bloch-irbssfp-traj-input-b1 tests/test-moba-bloch-irbssfp-traj-av-spokes
TESTS_SLOW += tests/test-moba-bloch-irbssfp-traj-slice-profile

TESTS_GPU += tests/test-moba-t1-gpu

