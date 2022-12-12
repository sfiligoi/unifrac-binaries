#!/bin/bash

#
# This is a helper script for installing the NVIDIA HPC SDK 
# needed to compile a GPU-enabled version of unifrac.
#
# Note: The script currently assumes Linux_x86_64 platform.
#

if [ "x${SYSROOT_DIR}" == "x" ]; then
  SYSROOT_DIR=${CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot/usr/lib64
fi

# Create GCC symbolic links
# since NVIDIA HPC SDK does not use the env variables
if [ "x${GCC}" == "x" ]; then
  echo "ERROR: GCC not defined"
  exit 1
fi

# usually $CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-
EXE_PREFIX=`echo "$GCC" |sed 's/gcc$//g'`

echo "GCC pointing to ${EXE_PREFIX}gcc"
ls -l ${EXE_PREFIX}gcc

mkdir conda_nv_bins
(cd conda_nv_bins && for f in \
  ar as c++ cc cpp g++ gcc ld nm ranlib strip; \
  do \
    ln -s ${EXE_PREFIX}${f} ${f}; \
  done )

export PATH=$PWD/conda_nv_bins:$PATH

# Install the NVIDIA HPC SDK

# This link may need to be updated, as new compiler versions are released
# Note: Verified that it works with v21.7
if [ "x${NV_URL}" == "x" ]; then
  NV_URL=https://developer.download.nvidia.com/hpc-sdk/21.7/nvhpc_2021_217_Linux_x86_64_cuda_multi.tar.gz
fi

echo "Downloading the NVIDIA HPC SDK"
# Defaults to using curl
# set USE_CURL=N if you want to use aria2 or wget
if [ "x${USE_CURL}" == "x" ]; then
  # defaults to using inline untarrring
  # set INLINE_CURL=N if you want a temp copy of file on disk
  if [ "x${INLINE_CURL}" == "xN" ]; then
    curl "${NV_URL}" -o nvhpc.tgz
    tar xpzf nvhpc.tgz
    rm -f nvhpc.tgz
  else
    # Do not unpack things we do not use for unifrac
    curl -s "${NV_URL}" | tar xpzf - --exclude '*libcublas*' --exclude '*libcufft*' --exclude '*libcusparse*' --exclude '*libcusolver*' --exclude '*libcurand*' --exclude '*profilers*' --exclude '*comm_libs*' --exclude '*/doc/*' --exclude '*/plugin*'
  fi
elif [ "x${USE_ARIA2}" == "x" ]; then
  aria2c "${NV_URL}"
  tar xpzf nvhpc_*.tar.gz
  rm -f nvhpc_*.tar.gz
else
  wget "${NV_URL}"
  tar xpzf nvhpc_*.tar.gz
  rm -f nvhpc_*.tar.gz
fi

echo "Installing NVIDIA HPC SDK"

# must patch the install scripts to find the right gcc
for f in nvhpc_*/install_components/install nvhpc_*/install_components/*/*/compilers/bin/makelocalrc nvhpc_*/install_components/install_cuda; do
  sed -i -e "s#PATH=/#PATH=$PWD/conda_nv_bins:/#g" $f
done


export NVHPC_INSTALL_DIR=$PWD/hpc_sdk
export NVHPC_SILENT=true

(cd nvhpc_*; ./install)

# create helper scripts
mkdir setup_scripts
cat > setup_scripts/setup_nv_hpc_bins.sh << EOF
PATH=$PWD/conda_nv_bins:`ls -d $PWD/hpc_sdk/*/202*/compilers/bin`:\$PATH

# pgc++ does not define it, but gcc libraries expect it
# also remove the existing conda flags, which are not compatible
export CPPFLAGS=-D__GCC_ATOMIC_TEST_AND_SET_TRUEVAL=0
export CXXFLAGS=\${CPPFLAGS}
export CFLAGS=\${CPPFLAGS}

unset DEBUG_CPPFLAGS
unset DEBUG_CXXFLAGS
unset DEBUG_CFLAGS

EOF

# h5c++ patch
mkdir conda_h5
cp $CONDA_PREFIX/bin/h5c++ conda_h5/

# This works on linux with gcc ..
sed -i \
  "s#x86_64-conda.*-linux-gnu-c++#pgc++ -I`ls -d $NVHPC_INSTALL_DIR/*/202*/compilers/include`#g" \
  conda_h5/h5c++ 
sed -i \
  's#H5BLD_CXXFLAGS=".*"#H5BLD_CXXFLAGS=" -fvisibility-inlines-hidden -std=c++17 -fPIC -O2 -I${includedir}"#g'  \
  conda_h5/h5c++
sed -i \
  's#H5BLD_CPPFLAGS=".*"#H5BLD_CPPFLAGS=" -I${includedir} -DNDEBUG -D_FORTIFY_SOURCE=2 -O2"#g' \
  conda_h5/h5c++
sed -i \
  's#H5BLD_LDFLAGS=".*"#H5BLD_LDFLAGS=" -L${prefix}/x86_64-conda-linux-gnu/sysroot/usr/lib64/ -L${libdir} -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,-rpath,\\\\\\$ORIGIN/../x86_64-conda-linux-gnu/sysroot/usr/lib64/ -Wl,-rpath,\\\\\\$ORIGIN -Wl,-rpath,\\\\\\$ORIGIN/../lib -Wl,-rpath,${prefix}/x86_64-conda-linux-gnu/sysroot/usr/lib64/ -Wl,-rpath,${libdir}"#g' \
 conda_h5/h5c++

# patch localrc to find crt1.o
for f in ${NVHPC_INSTALL_DIR}/*/202*/compilers/bin/localrc; do
  echo "set DEFSTDOBJDIR=${SYSROOT_DIR};" >> $f
  #echo "====localrc $f ===="
  #cat $f
  #echo "===="
done

cat > setup_nv_h5.sh  << EOF
source $PWD/setup_scripts/setup_nv_hpc_bins.sh

PATH=${PWD}/conda_h5:\$PATH
EOF

# we don't need the install dir anymore
rm -fr nvhpc_*

echo "Setup script avaiabile in $PWD/setup_nv_h5.sh"
