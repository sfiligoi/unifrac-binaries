.PHONY: test_binaries test

CFLAGS := 
LDFLAGS :=

ifeq ($(PREFIX),)
	PREFIX := $(CONDA_PREFIX)
endif

test: capi_test
	./capi_test 1

test_binaries: capi_test

capi_test: capi_test.c
	$(CC) -std=c99 -O0 -g capi_test.c -I../src -lssu -L${PREFIX}/lib -Wl,-rpath,${PREFIX}/lib -o capi_test

clean:
	-rm -f *.o capi_test

