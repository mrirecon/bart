
$(TESTS_OUT)/tetrahedron_cfl.ra: stl
	$(TOOLDIR)/stl --model=TET $@

$(TESTS_OUT)/ascii_enc.stl:
	echo -e "solid name\nfacet normal 0 0 1\nouter loop\nvertex 1 0 0\nvertex 0 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid name\n" > $@

tests/test-stl-rw-tetrahedron: nrmse stl $(TESTS_OUT)/tetrahedron_cfl.ra $(TESTS_OUT)/ascii_enc.stl
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/stl --input $(TESTS_OUT)/tetrahedron_cfl.ra tet.stl	;\
	$(TOOLDIR)/stl --input tet.stl tet_cfl          		;\
	$(TOOLDIR)/nrmse -t 0 $(TESTS_OUT)/tetrahedron_cfl.ra tet_cfl	;\
	$(TOOLDIR)/stl --input $(TESTS_OUT)/ascii_enc.stl test_cfl	;\
	rm *.cfl *.hdr *.stl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-stl-rw-tetrahedron

