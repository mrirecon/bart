


# create phantom

$(OUTDIR)/shepplogan.ra: phantom
	$(TOOLDIR)/phantom $@ 


# basic 2D FFT

$(OUTDIR)/shepplogan_fft.ra: fft $(OUTDIR)/shepplogan.ra
	$(TOOLDIR)/fft 7 $(OUTDIR)/shepplogan.ra $@ 


tests/test-fft-basic: scale fft nrmse $(OUTDIR)/shepplogan_fft.ra $(OUTDIR)/shepplogan.ra
	$(TOOLDIR)/scale 16384 $(OUTDIR)/shepplogan.ra $(TMPDIR)/shepploganS.ra
	$(TOOLDIR)/fft -i 7 $(OUTDIR)/shepplogan_fft.ra $(TMPDIR)/shepplogan2.ra
	$(TOOLDIR)/nrmse -t 0.000001 $(TMPDIR)/shepploganS.ra $(TMPDIR)/shepplogan2.ra
	rm $(TMPDIR)/*.ra
	touch $@ 
	

# unitary FFT

$(OUTDIR)/shepplogan_fftu.ra: fft $(OUTDIR)/shepplogan.ra
	$(TOOLDIR)/fft -u 7 $(OUTDIR)/shepplogan.ra $@ 

tests/test-fft-unitary: fft nrmse $(OUTDIR)/shepplogan.ra $(OUTDIR)/shepplogan_fftu.ra
	$(TOOLDIR)/fft -u -i 7 $(OUTDIR)/shepplogan_fftu.ra $(TMPDIR)/shepplogan2u.ra
	$(TOOLDIR)/nrmse -t 0.000001 $(OUTDIR)/shepplogan.ra $(TMPDIR)/shepplogan2u.ra
	rm $(TMPDIR)/*.ra
	touch $@ 
	

TESTS += tests/test-fft-basic tests/test-fft-unitary


