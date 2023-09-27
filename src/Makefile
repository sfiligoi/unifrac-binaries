.PHONY: all main api test test_binaries install clean rapi_test

CXX := h5c++

PLATFORM := $(shell uname -s)
COMPILER := $(shell ($(CXX) -v 2>&1) | tr A-Z a-z )

ifeq ($(BUILD_VARIANT),)
	SSU = ssu
	FPD = faithpd
else
	SSU = ssu_$(BUILD_VARIANT)
	FPD = faithpd_$(BUILD_VARIANT)
endif

ifdef DEBUG
	ifneq (,$(findstring pgi,$(COMPILER)))
		OPT = -g
	else
		OPT = -O0 -DDEBUG=1 --debug -g -ggdb
	endif
else
	ifneq (,$(findstring pgi,$(COMPILER)))
		OPT = -fast
	else
	  ifneq (,$(findstring gcc,$(COMPILER)))
		OPT = -Ofast
		TGTFLAGS = -fwhole-program
	  else
		OPT = -O3
	  endif
	endif
endif

ifeq ($(PREFIX),)
	PREFIX := $(CONDA_PREFIX)
endif

BLASLIB=-llapacke -lcblas

ifeq ($(PLATFORM),Darwin)
	AVX2 := $(shell sysctl -a | grep -c AVX2)
	LDDFLAGS = -dynamiclib -install_name @rpath/lib$(SSU).so
else
	AVX2 := $(shell grep "^flags" /proc/cpuinfo | head -n 1 | grep -c avx2)
	LDDFLAGS = -shared
	BLASLIB += -lgfortran -lquadmath
endif

EXEFLAGS =

ifneq (,$(findstring pgi,$(COMPILER)))
	MPFLAG =  -mp
else
	MPFLAG = -fopenmp
endif

LDDFLAGS += $(MPFLAG)
CPPFLAGS += $(MPFLAG)

UNIFRAC_FILES = unifrac_internal.o unifrac_cmp_cpu.o

ifndef NOGPU
	ifneq (,$(findstring pgi,$(COMPILER)))
		CPPFLAGS += -DUNIFRAC_ENABLE_ACC=1
                UNIFRAC_FILES += unifrac_cmp_acc.o
		ACCCPPFLAGS += -acc
	        ifeq ($(PERFORMING_CONDA_BUILD),True)
	            ACCCPPFLAGS += -gpu=ccall
		else
	            ACCCPPFLAGS += -gpu=ccnative
		    # optional info
		    ACCCPPFLAGS += -Minfo=accel
                endif
	        LDDFLAGS += -shlib -acc -Bstatic_pgi
	        EXEFLAGS += -acc -Bstatic_pgi
	endif
endif

ifneq ($(BUILD_FULL_OPTIMIZATION),False)
    ifneq (,$(findstring pgi,$(COMPILER)))
	ifeq ($(PERFORMING_CONDA_BUILD),True)
		# enable avx2
		CPPFLAGS += -tp=haswell -Mvect=simd:256
	endif
    else
	ifeq ($(PERFORMING_CONDA_BUILD),True)
	   ifeq ($(PLATFORM),Darwin)
		CPPFLAGS += -mpopcnt
	   else
		# enable avx2
		CPPFLAGS += -march=haswell -mtune=znver2
	   endif
	else
		CPPFLAGS += -march=native -mtune=native
	endif
    endif
else
    ifneq (,$(findstring pgi,$(COMPILER)))
	ifeq ($(PERFORMING_CONDA_BUILD),True)
		# PGI does not support anything older than sandybridge
		CPPFLAGS += -tp=sandybridge -Mvect=simd:256
	endif
    else
	ifeq ($(PERFORMING_CONDA_BUILD),True)
		CPPFLAGS += -mtune=core2
	else
		CPPFLAGS += -march=native -mtune=native
	endif
    endif
endif

ifeq (,$(findstring pgi,$(COMPILER)))
    ifneq ($(PERFORMING_CONDA_BUILD),True)
	# basically, not gcc
	CPPFLAGS += -Wextra -Wno-unused-parameter
    endif
endif


CPPFLAGS += -Wall  -std=c++17 -pedantic -I. $(OPT) -fPIC -L$(PREFIX)/lib

all: api main install

main: $(SSU) $(FPD)

api: lib$(SSU).so

test_binaries: test_su test_ska test_api

test_su: test_su.cpp tree.o tsv.o test_su.cpp biom.o biom_inmem.o biom_subsampled.o unifrac.o skbio_alt.o api.o $(UNIFRAC_FILES)
	$(CXX) $(CPPFLAGS) $(EXEFLAGS) test_su.cpp -o test_su tree.o biom.o biom_inmem.o biom_subsampled.o tsv.o $(UNIFRAC_FILES) unifrac.o skbio_alt.o api.o -llz4 $(BLASLIB) -lpthread
test_ska: test_ska.cpp tree.o tsv.o test_su.cpp biom.o biom_inmem.o biom_subsampled.o unifrac.o skbio_alt.o api.o $(UNIFRAC_FILES)
	$(CXX) $(CPPFLAGS) $(EXEFLAGS) test_ska.cpp -o test_ska skbio_alt.o tree.o biom.o biom_inmem.o biom_subsampled.o tsv.o $(UNIFRAC_FILES) unifrac.o api.o -llz4 $(BLASLIB) -lpthread
test_api: test_api.cpp tree.o tsv.o test_su.cpp biom.o biom_inmem.o biom_subsampled.o unifrac.o skbio_alt.o api.o $(UNIFRAC_FILES)
	$(CXX) $(CPPFLAGS) $(EXEFLAGS) test_api.cpp -o test_api tree.o biom.o biom_inmem.o biom_subsampled.o tsv.o $(UNIFRAC_FILES) unifrac.o skbio_alt.o api.o -llz4 $(BLASLIB) -lpthread

$(SSU): su.cpp tree.o biom.o biom_inmem.o biom_subsampled.o tsv.o unifrac.o cmd.o skbio_alt.o api.o $(UNIFRAC_FILES)
	$(CXX) $(CPPFLAGS) $(EXEFLAGS) su.cpp -o $(SSU) tree.o biom.o biom_inmem.o biom_subsampled.o tsv.o $(UNIFRAC_FILES) unifrac.o cmd.o skbio_alt.o api.o -lhdf5_cpp -llz4 $(BLASLIB) -lpthread

$(FPD): faithpd.cpp tree.o biom.o biom_inmem.o biom_subsampled.o tsv.o unifrac.o cmd.o skbio_alt.o api.o $(UNIFRAC_FILES)
	$(CXX) $(CPPFLAGS) $(EXEFLAGS) faithpd.cpp -o $(FPD) tree.o biom.o biom_inmem.o biom_subsampled.o tsv.o $(UNIFRAC_FILES) unifrac.o cmd.o skbio_alt.o api.o -lhdf5_cpp -llz4 $(BLASLIB) -lpthread

lib$(SSU).so: tree.o biom.o biom_inmem.o biom_subsampled.o tsv.o unifrac.o cmd.o skbio_alt.o api.o $(UNIFRAC_FILES)
	$(CXX) $(LDDFLAGS) -o lib$(SSU).so tree.o biom.o biom_inmem.o biom_subsampled.o tsv.o $(UNIFRAC_FILES) unifrac.o cmd.o skbio_alt.o api.o -lc -llz4 $(BLASLIB) -L$(PREFIX)/lib -noshlib -lhdf5_cpp -lhdf5_hl_cpp -lhdf5_hl -lhdf5

api.o: api.cpp api.hpp unifrac.hpp skbio_alt.hpp biom.hpp biom_inmem.hpp biom_subsampled.hpp tree.hpp tsv.hpp
	$(CXX) $(CPPFLAGS) api.cpp -c -o api.o -fPIC

unifrac_cmp_cpu.o: unifrac_cmp.cpp unifrac_cmp.hpp unifrac_internal.hpp unifrac.hpp unifrac_task.cpp unifrac_task.hpp biom_interface.hpp tree.hpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

unifrac_cmp_acc.o: unifrac_cmp.cpp unifrac_cmp.hpp unifrac_internal.hpp unifrac.hpp unifrac_task.cpp unifrac_task.hpp biom_interface.hpp tree.hpp
	$(CXX) $(CPPFLAGS) $(ACCCPPFLAGS) -c $< -o $@

%.o: %.cpp %.hpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

test: test_binaries
	./test_su
	./test_ska
	./test_api

install: lib$(SSU).so $(SSU) $(FPD)
	rm -f ${PREFIX}/lib//lib$(SSU).so; cp lib$(SSU).so ${PREFIX}/lib/
	rm -f ${PREFIX}/bin/$(SSU); cp $(SSU) ${PREFIX}/bin/
	rm -f ${PREFIX}/bin/$(FPD); cp $(FPD) ${PREFIX}/bin/
	mkdir -p ${PREFIX}/include/unifrac
	rm -f ${PREFIX}/include/unifrac/task_parameters.hpp; cp task_parameters.hpp ${PREFIX}/include/unifrac/
	rm -f ${PREFIX}/include/unifrac/api.hpp; cp api.hpp ${PREFIX}/include/unifrac/
	rm -f ${PREFIX}/include/unifrac/status_enum.hpp; cp status_enum.hpp ${PREFIX}/include/unifrac/

rapi_test: main
	mkdir -p ~/.R
	if [ -e ~/.R/Makevars ] ; \
	then \
		echo "WARNING: OVERWRITING ~/.R/Makevars" ; \
		echo "The original Makevars file has been copied to ~/.R/Makevars" ;\
		cp ~/.R/Makevars Makevars-original ; \
	fi;
	echo CXX1X=h5c++ > ~/.R/Makevars
	echo CXX=h5c++ >> ~/.R/Makevars 
	echo CC=h5c++ >> ~/.R/Makevars
	echo CFLAGS=  >> ~/.R/Makevars
	echo CXXFLAGS=  >> ~/.R/Makevars
	echo LDFLAGS= -llz4 $(BLASLIB) >> ~/.R/Makevars
	rm -f *.o
	Rscript --verbose R_interface/rapi_test.R
	rm -f *.o

clean:
	-rm -f *.o $(SSU) $(FPD) test_su test_ska test_api lib$(SSU).so

