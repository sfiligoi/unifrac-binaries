.PHONY: test clean all

# Note: This Makefile will NOT properly work with the -j option 

PLATFORM := $(shell uname -s)
COMPILER := $(shell ($(CXX) -v 2>&1) | tr A-Z a-z )

ifeq ($(PLATFORM),Darwin)
# no GPU support for MacOS
all:
	make api
	make install
	make main
	make install_main
	make test_binaries

else
# Linux with optional GPU support

ifndef NOGPU

all: 
	make all_cpu_basic
	make all_nv_avx2
	make all_nv
	make all_combined
	make test_binaries

else

all: 
	make all_cpu_basic
	make all_combined
	make test_binaries

endif

all_cpu_basic:
	make api_cpu_basic
	make install_cpu_basic

all_nv: 
	make api_nv
	make install_nv

all_nv_avx2: 
	make api_nv_avx2
	make install_nv_avx2

all_combined:
	make api_combined
	make install_combined
	make main
	make install_main

endif

clean:
	-cd test && make clean
	-cd src && make clean
	-cd combined && make clean

########### api

api:
	cd src && make clean && make api

api_cpu_basic:
	export BUILD_VARIANT=cpu_basic ; export BUILD_FULL_OPTIMIZATION=False ; cd src && make clean && make api

api_nv:
	. ./setup_nv_h5.sh; export BUILD_VARIANT=nv ; export BUILD_FULL_OPTIMIZATION=False ; cd src && make clean && make api

api_nv_avx2:
	. ./setup_nv_h5.sh; export BUILD_VARIANT=nv_avx2 ; export BUILD_FULL_OPTIMIZATION=True ; cd src && make clean && make api

api_combined:
	cd combined && make clean && make api

########### main

main:
	cd src && make clean && make main

install_main:
	cd src && make install

########### install

install:
	cd src && make install_lib

install_cpu_basic:
	export BUILD_VARIANT=cpu_basic ; export BUILD_FULL_OPTIMIZATION=False ; cd src && make install_lib

install_nv:
	. ./setup_nv_h5.sh; export BUILD_VARIANT=nv ; export BUILD_FULL_OPTIMIZATION=False ; cd src && make install_lib

install_nv_avx2:
	. ./setup_nv_h5.sh; export BUILD_VARIANT=nv_avx2 ; export BUILD_FULL_OPTIMIZATION=True ; cd src && make install_lib

install_combined:
	cd combined && make install

########### test

test_binaries:
	cd src && make clean && make test_binaries
	cd test && make clean && make test_binaries

test:
	cd src && make test
	cd test && make test

