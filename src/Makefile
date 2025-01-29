.PHONY: all main api api_acc test test_binaries install install_lib clean rapi_test

PLATFORM := $(shell uname -s)
COMPILER := $(shell (h5c++ -v 2>&1) | tr A-Z a-z )

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

UNIFRAC_FILES = unifrac_internal.o unifrac_task_cpu.o unifrac_cmp_cpu.o
UNIFRAC_ACC_SHLIBS =

ifndef NOGPU
   # assuming NV only GPU-enabled compiler available
   NV_COMPILER := $(shell ($(NV_CXX) -v 2>&1) | tr A-Z a-z )
   ifeq ($(BUILD_OFFLOAD),ompgpu)
	CPPFLAGS += -DUNIFRAC_ENABLE_OMPGPU=1
	UNIFRAC_FILES += unifrac_task_ompgpu.o unifrac_cmp_ompgpu.o
	ifneq (,$(findstring pgi,$(COMPILER)))
		OMPGPUCXX=$(NV_CXX)
		UNIFRAC_ACC_SHLIBS += libssu_ompgpu.so
		OMPGPUCPPFLAGS += -tp=px -mp=gpu -DOMPGPU=1 -noacc
	        ifeq ($(PERFORMING_CONDA_BUILD),True)
	            OMPGPUCPPFLAGS += -gpu=ccall
		else
	            OMPGPUCPPFLAGS += -gpu=ccnative
		    # optional info
		    OMPGPUCPPFLAGS += -Minfo=accel
                endif
		OMPGPULDDFLAGS := $(NV_LDFLAGS) -shared -mp=gpu -Bstatic_pgi
	endif
   else 
     ifeq ($(BUILD_OFFLOAD),acc)
	CPPFLAGS += -DUNIFRAC_ENABLE_ACC=1
	UNIFRAC_FILES += unifrac_task_acc.o unifrac_cmp_acc.o
	ifneq (,$(findstring pgi,$(NV_COMPILER)))
		ACCCXX=$(NV_CXX)
		UNIFRAC_ACC_SHLIBS += libssu_acc.so
		ACCCPPFLAGS += -tp=px -acc
	        ifeq ($(PERFORMING_CONDA_BUILD),True)
	            ACCCPPFLAGS += -gpu=ccall
		else
	            ACCCPPFLAGS += -gpu=ccnative
		    # optional info
		    ACCCPPFLAGS += -Minfo=accel
                endif
		ACCLDDFLAGS := $(NV_LDFLAGS) -shared -acc -Bstatic_pgi
	endif
     endif
   endif
endif

ifeq ($(BUILD_FULL_OPTIMIZATION),False)
    ifneq (,$(findstring pgi,$(COMPILER)))
	CPPFLAGS += -tp=px
    else
	ifeq ($(PLATFORM),Darwin)
         	# no special tuning for the MacOS at this point
	  	CPPFLAGS += 
	else
		# assuming x86 for now... may want to check for architecture
		# x86-64 the most generic option avaialable in modern gcc
		CPPFLAGS += -march=x86-64 -mtune=generic
	endif
    endif
else
  ifneq ($(BUILD_FULL_OPTIMIZATION),)
    ifneq (,$(findstring pgi,$(COMPILER)))
	CPPFLAGS += -tp=$(BUILD_FULL_OPTIMIZATION)
    else
	CPPFLAGS += -march=$(BUILD_FULL_OPTIMIZATION)
        ifneq ($(BUILD_TUNE_OPTIMIZATION),)
	  CPPFLAGS += -mtune=$(BUILD_TUNE_OPTIMIZATION)
        endif
    endif
  else
    # no preference from the user
    # change default only if not performing CONDA_BUILD
    ifeq ($(PERFORMING_CONDA_BUILD),True)
	ifeq ($(PLATFORM),Darwin)
         	# no special tuning for the MacOS at this point
	  	CPPFLAGS += 
	else
		CPPFLAGS += -march=native -mtune=native
	endif
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
ACCCPPFLAGS += -std=c++17 -I. $(OPT) -fPIC -L$(PREFIX)/lib
OMPGPUCPPFLAGS += -std=c++17 -I. $(OPT) -fPIC -L$(PREFIX)/lib

all: api install_lib main install

main: $(SSU) $(FPD)

api: lib$(SSU).so

api_acc: $(UNIFRAC_ACC_SHLIBS)

test_binaries: test_su test_ska test_api test_su_api

# first test the internal structures
test_su: test_su.cpp api.hpp tree.o tsv.o biom.o biom_inmem.o biom_subsampled.o unifrac.o skbio_alt.o api.o $(UNIFRAC_FILES)
	h5c++ $(CPPFLAGS) $(EXEFLAGS) test_su.cpp -o $@ tree.o biom.o biom_inmem.o biom_subsampled.o tsv.o $(UNIFRAC_FILES) unifrac.o skbio_alt.o api.o -llz4 $(BLASLIB) -lpthread
# then exercise the optimized code through the API interface
test_su_api: test_su.cpp test_helper.hpp api.hpp $(PREFIX)/lib/lib$(SSU).so
	h5c++ $(CPPFLAGS) $(EXEFLAGS) -DAPI_ONLY=1 test_su.cpp -o $@ -l$(SSU)

# Create an archive, so we do not need dependencies for functions we do not use
test_ska_deps.a: tree.o tsv.o biom.o biom_inmem.o biom_subsampled.o skbio_alt.o api.o
	rm -f $@
	ar -r $@ $^
 
# Not invoking the unifrac acc code, so we do not need to use the shlib
test_ska: test_ska.cpp skbio_alt.hpp api.hpp test_helper.hpp biom.hpp test_ska_deps.a
	h5c++ $(CPPFLAGS) $(EXEFLAGS) test_ska.cpp -o test_ska test_ska_deps.a -llz4 $(BLASLIB) -lpthread

test_api: test_api.cpp api.hpp test_helper.hpp biom.hpp $(PREFIX)/lib/lib$(SSU).so
	h5c++ $(CPPFLAGS) $(EXEFLAGS) test_api.cpp -o $@ -l$(SSU)

$(SSU): su.cpp api.hpp biom.hpp $(PREFIX)/lib/lib$(SSU).so
	h5c++ $(CPPFLAGS) $(EXEFLAGS) su.cpp -o $@ -l$(SSU)

$(FPD): faithpd.cpp api.hpp $(PREFIX)/lib/lib$(SSU).so
	h5c++ $(CPPFLAGS) $(EXEFLAGS) faithpd.cpp -o $@ -l$(SSU)

lib$(SSU).so: tree.o biom.o biom_inmem.o biom_subsampled.o tsv.o unifrac.o cmd.o skbio_alt.o api.o $(UNIFRAC_FILES)
	h5c++ $(LDDFLAGS) -o lib$(SSU).so tree.o biom.o biom_inmem.o biom_subsampled.o tsv.o $(UNIFRAC_FILES) unifrac.o cmd.o skbio_alt.o api.o -lc -llz4 $(BLASLIB) -L$(PREFIX)/lib -noshlib -lhdf5_cpp -lhdf5_hl_cpp -lhdf5_hl -lhdf5

api.o: api.cpp api.hpp unifrac.hpp skbio_alt.hpp biom.hpp biom_inmem.hpp biom_subsampled.hpp tree.hpp tsv.hpp
	h5c++ $(CPPFLAGS) api.cpp -c -o api.o -fPIC


unifrac_task_noclass_cpu.cpp: unifrac_task_impl.hpp generate_unifrac_task_noclass.py
	# Use the same compiler for CPU-based code
	# So no need for futher levels of indirection
	python3 generate_unifrac_task_noclass.py cpu direct > $@

unifrac_task_api_acc.h: unifrac_task_impl.hpp generate_unifrac_task_noclass.py
	python3 generate_unifrac_task_noclass.py acc api_h > $@
unifrac_task_api_acc.cpp: unifrac_task_impl.hpp unifrac_task_api_acc.h generate_unifrac_task_noclass.py
	python3 generate_unifrac_task_noclass.py acc api > $@
unifrac_task_noclass_acc.cpp: unifrac_task_impl.hpp unifrac_task_api_acc.h generate_unifrac_task_noclass.py
	python3 generate_unifrac_task_noclass.py acc indirect > $@

unifrac_task_api_ompgpu.h: unifrac_task_impl.hpp generate_unifrac_task_noclass.py
	python3 generate_unifrac_task_noclass.py ompgpu api_h > $@
unifrac_task_api_ompgpu.cpp: unifrac_task_impl.hpp unifrac_task_api_ompgpu.h generate_unifrac_task_noclass.py
	python3 generate_unifrac_task_noclass.py ompgpu api > $@
unifrac_task_noclass_ompgpu.cpp: unifrac_task_impl.hpp unifrac_task_api_ompgpu.h generate_unifrac_task_noclass.py
	python3 generate_unifrac_task_noclass.py ompgpu indirect > $@

unifrac_task_cpu.o: unifrac_task_noclass_cpu.cpp unifrac_task_noclass.hpp unifrac_task_impl.hpp
	h5c++ $(CPPFLAGS) -DSUCMP_NM=su_cpu -c $< -o $@
# no separate task_api 

# create symlink to make R (and any autodeduct system) happy
unifrac_task.cpp: unifrac_task_noclass_cpu.cpp
	ln -s unifrac_task_noclass_cpu.cpp unifrac_task.cpp

unifrac_task_acc.o: unifrac_task_noclass_acc.cpp unifrac_task_noclass.hpp unifrac_task_api_acc.h
	h5c++ $(CPPFLAGS) -DSUCMP_NM=su_acc -c $< -o $@
unifrac_task_api_acc.o: unifrac_task_api_acc.cpp unifrac_task_api_acc.h unifrac_task_impl.hpp
	${ACCCXX} $(ACCCPPFLAGS) -DSUCMP_NM=su_acc -c $< -o $@

libssu_acc.so: unifrac_task_api_acc.o
	${ACCCXX} $(ACCLDDFLAGS) -o $@ unifrac_task_api_acc.o

unifrac_task_ompgpu.o: unifrac_task_noclass_ompgpu.cpp unifrac_task_noclass.hpp unifrac_task_api_ompgpu.h
	h5c++ $(CPPFLAGS) -DSUCMP_NM=su_ompgpu -c $< -o $@
unifrac_task_api_ompgpu.o: unifrac_task_api_ompgpu.cpp unifrac_task_api_ompgpu.h unifrac_task_impl.hpp
	${OMPGPUCXX} $(OMPGPUCPPFLAGS) -DSUCMP_NM=su_ompgpu -c $< -o $@

libssu_ompgpu.so: unifrac_task_api_ompgpu.o
	${OMPGPUCXX} $(OMPGPULDDFLAGS) -o $@ unifrac_task_api_ompgpu.o


unifrac_cmp_cpu.o: unifrac_cmp.cpp unifrac_cmp.hpp unifrac_internal.hpp unifrac.hpp unifrac_task.hpp unifrac_task_noclass.hpp biom_interface.hpp tree.hpp
	h5c++ $(CPPFLAGS) -DSUCMP_NM=su_cpu -c $< -o $@

unifrac_cmp_acc.o: unifrac_cmp.cpp unifrac_cmp.hpp unifrac_internal.hpp unifrac.hpp unifrac_task.hpp unifrac_task_noclass.hpp biom_interface.hpp tree.hpp
	h5c++ $(CPPFLAGS) -DSUCMP_NM=su_acc -c $< -o $@
unifrac_cmp_ompgpu.o: unifrac_cmp.cpp unifrac_cmp.hpp unifrac_internal.hpp unifrac.hpp unifrac_task.hpp unifrac_task_noclass.hpp biom_interface.hpp tree.hpp
	h5c++ $(CPPFLAGS) -DSUCMP_NM=su_ompgpu -c $< -o $@

%.o: %.cpp %.hpp
	h5c++ $(CPPFLAGS) -c $< -o $@

test: test_binaries
	./test_su
	./test_ska
	./test_api
	./test_su_api

install_lib: lib$(SSU).so
	for f in lib$(SSU).so ; do rm -f "${PREFIX}/lib/$${f}"; cp "$${f}" ${PREFIX}/lib/; done


install_lib_acc: $(UNIFRAC_ACC_SHLIBS)
	for f in $(UNIFRAC_ACC_SHLIBS); do rm -f "${PREFIX}/lib/$${f}"; cp "$${f}" ${PREFIX}/lib/; done


install: $(SSU) $(FPD)
	rm -f ${PREFIX}/bin/$(SSU); cp $(SSU) ${PREFIX}/bin/
	rm -f ${PREFIX}/bin/$(FPD); cp $(FPD) ${PREFIX}/bin/
	mkdir -p ${PREFIX}/include/unifrac
	rm -f ${PREFIX}/include/unifrac/task_parameters.hpp; cp task_parameters.hpp ${PREFIX}/include/unifrac/
	rm -f ${PREFIX}/include/unifrac/api.hpp; cp api.hpp ${PREFIX}/include/unifrac/
	rm -f ${PREFIX}/include/unifrac/status_enum.hpp; cp status_enum.hpp ${PREFIX}/include/unifrac/

rapi_test: unifrac_task.cpp tree.o biom.o biom_inmem.o biom_subsampled.o tsv.o unifrac.o cmd.o skbio_alt.o api.o $(UNIFRAC_FILES)
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
	-rm -f *.o $(SSU) $(FPD) *.a test_su test_ska test_api test_su_api lib$(SSU).so unifrac_task_noclass_*.cpp unifrac_task_api_*.h unifrac_task.cpp $(UNIFRAC_ACC_SHLIBS)

clean_install:
	-(cd ${PREFIX}/lib && rm -f lib$(SSU).so $(UNIFRAC_ACC_SHLIBS))
	-(cd ${PREFIX}/bin && rm -f $(SSU) $(FPD))
	-(cd ${PREFIX}/include/unifrac && rm -f task_parameters.hpp api.hpp status_enum.hpp)

