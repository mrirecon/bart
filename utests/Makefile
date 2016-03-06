# Requires CUnit testing framework
# macports: port install cunit
#
test_md_zwavg: test_md_zwavg.c test_md_zwavg.h
	gcc -fopenmp -L../lib/ -L/opt/local/lib -I../src/ -lcunit -lnum -lmisc test_md_zwavg.c test_md_zwavg.h -o test_md_zwavg

clean:
	rm test_md_zwavg
