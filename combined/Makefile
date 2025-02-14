.PHONY: all api main install clean

all: api install

api: libssu.so

# no-op, just for completeness
main:


ifeq ($(PREFIX),)
    PREFIX := $(CONDA_PREFIX)
endif

libssu.o: libssu.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c libssu.c -fPIC

libssu.so: libssu.o
	$(CC) -shared -o libssu.so libssu.o -fPIC -ldl $(LDFLAGS)

install: libssu.so
	rm -f ${PREFIX}/lib/libssu.so; cp libssu.so ${PREFIX}/lib/

clean:
	rm -f libssu.o libssu.so
