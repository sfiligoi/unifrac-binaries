.PHONY: test_binaries test

CFLAGS := 
LDFLAGS :=

ifeq ($(PREFIX),)
	PREFIX := $(CONDA_PREFIX)
endif

test: capi_test
	export LD_LIBRARY_PATH="${PREFIX}/lib":"./capi_test"

test_binaries: capi_test

capi_test: capi_test.c
	$(CC) -std=c99 capi_test.c -I../src -lssu -L${PREFIX}/lib -Wl,-rpath,${PREFIX}/lib -o capi_test

clean:
	-rm -f *.o capi_test

