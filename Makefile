.PHONY: all main api test_binaries test install clean all_combined all_basic all_gpu all_avx2

ifeq ($(PLATFORM),Darwin)
all: api main install

else
all: all_combined

endif

clean:
	-cd test && make clean
	-cd src && make clean

all_combined: all_cpu_basic all_nv all_nv_avx2

all_cpu_basic: api_cpu_basic main_cpu_basic install_cpu_basic

all_nv: api_nv main_nv install_nv

all_nv_avx2: api_nv_avx2 main_nv_avx2 install_nv_avx2

########### api

api:
	cd src && make api

api_cpu_basic:
	export BUILD_VARIANT=cpu_basic ; export BUILD_FULL_OPTIMIZATION=False ; cd src && make clean && make api

api_nv:
	source ./setup_nv_h5.sh; export BUILD_VARIANT=nv ; export BUILD_FULL_OPTIMIZATION=False ; cd src && make clean && make api

api_nv_avx2:
	source ./setup_nv_h5.sh; export BUILD_VARIANT=nv_avx2 ; export BUILD_FULL_OPTIMIZATION=True ; cd src && make clean && make api

########### main

main:
	cd src && make main

main_cpu_basic:
	export BUILD_VARIANT=cpu_basic ; export BUILD_FULL_OPTIMIZATION=False ; cd src && make main

main_nv:
	source ./setup_nv_h5.sh; export BUILD_VARIANT=nv ; export BUILD_FULL_OPTIMIZATION=False ; cd src && make main

main_nv_avx2:
	source ./setup_nv_h5.sh; export BUILD_VARIANT=nv_avx2 ; export BUILD_FULL_OPTIMIZATION=True ; cd src && make main

########### install

install:
	cd src && make install

install_cpu_basic:
	export BUILD_VARIANT=cpu_basic ; export BUILD_FULL_OPTIMIZATION=False ; cd src && make install

install_nv:
	source ./setup_nv_h5.sh; export BUILD_VARIANT=nv ; export BUILD_FULL_OPTIMIZATION=False ; cd src && make install

install_nv_avx2:
	source ./setup_nv_h5.sh; export BUILD_VARIANT=nv_avx2 ; export BUILD_FULL_OPTIMIZATION=True ; cd src && make install

########### test

test_binaries:
	cd src && make test_binaries
	cd test && make test_binaries

test:
	cd src && make test
	cd test && make test

