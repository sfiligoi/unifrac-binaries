.PHONY: main api test install clean

main:
	cd src && make main

api:
	cd src && make api

test:
	cd src && make test
	cd test && make test

install:
	cd src && make install

clean:
	-cd test && make clean
	-cd src && make clean
