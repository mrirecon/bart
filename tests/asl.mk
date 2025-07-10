
# Creates a phantom ASL dataset
# PWI is created by multiplying slices of a brain phantom with the gray and white matter signals
# Control - PWI = Label
# Control and label image are denoised, PWI is calculated and compared to original
tests/test-asl-denoise: phantom slice signal fmac saxpy scale repmat join noise denoise nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/phantom --BRAIN p.ra									;\
	$(TOOLDIR)/phantom --BRAIN -b pb.ra								;\
	$(TOOLDIR)/slice 6 2 pb.ra gm.ra								;\
	$(TOOLDIR)/slice 6 3 pb.ra wm.ra								;\
	$(TOOLDIR)/signal -A -r0.1 -n10 -1 1.4:1.4:1 -6 60:60:1 -a0.5 --acquisition-only -l0.98 s_gm.ra	;\
	$(TOOLDIR)/signal -A -r0.1 -n10 -1 1.4:1.4:1 -6 20:20:1 -a0.5 --acquisition-only -l0.82 s_wm.ra	;\
	$(TOOLDIR)/fmac gm.ra s_gm.ra pwi_gm.ra								;\
	$(TOOLDIR)/scale 3 pwi_gm.ra pwi_gm.ra								;\
	$(TOOLDIR)/fmac wm.ra s_wm.ra pwi_wm.ra								;\
	$(TOOLDIR)/saxpy -- 4. pwi_wm.ra pwi_gm.ra pwi.ra						;\
	$(TOOLDIR)/repmat 6 4 pwi.ra pwi.ra								;\
	$(TOOLDIR)/repmat 5 10 p.ra c.ra								;\
	$(TOOLDIR)/repmat 6 4 c.ra c.ra									;\
	$(TOOLDIR)/saxpy -- -1. pwi.ra c.ra l.ra							;\
	$(TOOLDIR)/join 8 c.ra l.ra o.ra								;\
	$(TOOLDIR)/noise -n 0.00001 o.ra on.ra								;\
	$(TOOLDIR)/denoise --asl -i50 -C10 --theta 1:2.5 --tvscales 1:1:1:15 -RT:99:0:0.0002 on.ra or.ra	;\
	$(TOOLDIR)/slice 8 0 or.ra c_r.ra								;\
	$(TOOLDIR)/slice 8 1 or.ra l_r.ra								;\
	$(TOOLDIR)/saxpy -- -1. l_r.ra c_r.ra pwi_r.ra							;\
	$(TOOLDIR)/nrmse -t 0.08 pwi_r.ra pwi.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS_SLOW += tests/test-asl-denoise


