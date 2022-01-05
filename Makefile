.PHONY: all main api test_binaries test install clean

all: api main install

main:
	cd src && make main

api:
	cd src && make api

test_binaries:
	cd src && make test_binaries
	cd test && make test_binaries

test:
	cd src && make test
	cd test && make test

install:
	cd src && make install

clean:
	-cd test && make clean
	-cd src && make clean
