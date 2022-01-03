
main:
	cd src && make main

api:
	cd src && make api

test: api
	cd src && make test
	cd test && make test

clean:
	-cd test && make clean
	-cd src && make clean
