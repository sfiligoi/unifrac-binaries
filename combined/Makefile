.PHONY: all main install clean

all: main install

main: libssu.so


ifeq ($(PREFIX),)
    PREFIX := $(CONDA_PREFIX)
endif

libssu.o: libssu.c
	$(CC) -c libssu.c -fPIC

libssu.so: libssu.o
	$(CC) -shared -o libssu.so libssu.o -fPIC -ldl

install: libssu.so
	rm -f ${PREFIX}/lib//libssu.so; cp libssu.so ${PREFIX}/lib/
	rm -f ${PREFIX}/bin/ssu; cp ssu ${PREFIX}/bin/

clean:
	rm -f libssu.o libssu.so
