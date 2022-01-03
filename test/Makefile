ifeq ($(PREFIX),)
	PREFIX := $(CONDA_PREFIX)
endif

capi_test: capi_test

capi_test: capi_test.c
	gcc -std=c99 capi_test.c -lssu -L${PREFIX}/lib -Wl,-rpath,${PREFIX}/lib -o capi_test
	export LD_LIBRARY_PATH="${PREFIX}/lib":"./capi_test"

clean:
	-rm -f *.o capi_test

