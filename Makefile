.PHONY: test clean all

# Note: This Makefile will NOT properly work with the -j option 

PLATFORM := $(shell uname -s)
COMPILER := $(shell ($(CXX) -v 2>&1) | tr A-Z a-z )

ifeq ($(PLATFORM),Darwin)
all: api install main install_main test_binaries

else

# Note: important that all_nv is after all_cpu_basic and all_nv_avx2 for tests to work
all: all_cpu_basic all_nv_avx2 all_nv all_combined test_binaries

all_cpu_basic: api_cpu_basic install_cpu_basic

all_nv: api_nv install_nv

all_nv_avx2: api_nv_avx2 install_nv_avx2

all_combined: api_combined install_combined main install_main

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

