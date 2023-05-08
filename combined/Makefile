.PHONY: all api main install clean

all: api install

api: libssu.so

# no-op, just for completeness
main:


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
	rm -f ${PREFIX}/bin/faithpd; cp faithpd ${PREFIX}/bin/

clean:
	rm -f libssu.o libssu.so
