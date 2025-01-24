.PHONY: test clean all clean_install

# Note: This Makefile will NOT properly work with the -j option 

PLATFORM := $(shell uname -s)
COMPILER := $(shell ($(CXX) -v 2>&1) | tr A-Z a-z )

ifeq ($(PLATFORM),Darwin)
# no GPU support for MacOS
all:
	$(MAKE) api
	$(MAKE) install
	$(MAKE) main
	$(MAKE) install_main
	$(MAKE) test_binaries

clean:
	-cd test && $(MAKE) clean
	-cd src && $(MAKE) clean

clean_install:
	-cd src && $(MAKE) clean_install

else
# Linux with optional GPU support

ifndef NOGPU

all: 
	$(MAKE) all_cpu_basic
	$(MAKE) all_nv_avx2
	$(MAKE) all_nv
	$(MAKE) all_combined
	$(MAKE) test_binaries

clean:
	-cd test && $(MAKE) clean
	-export BUILD_VARIANT=cpu_basic; cd src && $(MAKE) clean
	-export BUILD_VARIANT=nv; cd src && $(MAKE) clean
	-export BUILD_VARIANT=nv_avx2; cd src && $(MAKE) clean
	-cd combined && $(MAKE) clean

clean_install:
	-export BUILD_VARIANT=cpu_basic; cd src && $(MAKE) clean_install
	-export BUILD_VARIANT=nv; cd src && $(MAKE) clean_install
	-export BUILD_VARIANT=nv_avx2; cd src && $(MAKE) clean_install
	-cd combined && $(MAKE) clean_install

else

all: 
	$(MAKE) all_cpu_basic
	$(MAKE) all_combined
	$(MAKE) test_binaries

clean:
	-cd test && $(MAKE) clean
	-export BUILD_VARIANT=cpu_basic; cd src && $(MAKE) clean
	-cd combined && $(MAKE) clean

clean_install:
	-export BUILD_VARIANT=cpu_basic; cd src && $(MAKE) clean_install
	-cd combined && $(MAKE) clean_install

endif

all_cpu_basic:
	$(MAKE) api_cpu_basic
	$(MAKE) install_cpu_basic

all_nv: 
	$(MAKE) api_nv
	$(MAKE) install_nv

all_nv_avx2: 
	$(MAKE) api_nv_avx2
	$(MAKE) install_nv_avx2

all_combined:
	$(MAKE) api_combined
	$(MAKE) install_combined
	$(MAKE) main
	$(MAKE) install_main

endif

########### api

api:
	cd src && $(MAKE) clean && $(MAKE) api

api_cpu_basic:
	export BUILD_VARIANT=cpu_basic ; export BUILD_FULL_OPTIMIZATION=False ; cd src && $(MAKE) clean && $(MAKE) api

api_nv:
	. ./setup_nv_h5.sh; export BUILD_VARIANT=nv ; export BUILD_FULL_OPTIMIZATION=False ; cd src && $(MAKE) clean && $(MAKE) api

api_nv_avx2:
	. ./setup_nv_h5.sh; export BUILD_VARIANT=nv_avx2 ; export BUILD_FULL_OPTIMIZATION=True ; cd src && $(MAKE) clean && $(MAKE) api

api_combined:
	cd combined && $(MAKE) clean && $(MAKE) api

########### main

main:
	cd src && $(MAKE) clean && $(MAKE) main

install_main:
	cd src && $(MAKE) install

########### install

install:
	cd src && $(MAKE) install_lib

install_cpu_basic:
	export BUILD_VARIANT=cpu_basic ; export BUILD_FULL_OPTIMIZATION=False ; cd src && $(MAKE) install_lib

install_nv:
	. ./setup_nv_h5.sh; export BUILD_VARIANT=nv ; export BUILD_FULL_OPTIMIZATION=False ; cd src && $(MAKE) install_lib

install_nv_avx2:
	. ./setup_nv_h5.sh; export BUILD_VARIANT=nv_avx2 ; export BUILD_FULL_OPTIMIZATION=True ; cd src && $(MAKE) install_lib

install_combined:
	cd combined && $(MAKE) install

########### test

test_binaries:
	cd src && $(MAKE) clean && $(MAKE) test_binaries
	cd test && $(MAKE) clean && $(MAKE) test_binaries

test:
	cd src && $(MAKE) test
	cd test && $(MAKE) test

