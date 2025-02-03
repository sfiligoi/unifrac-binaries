#!/bin/bash

#
# This is a helper script for installing the AMD clang-based compiler
# needed to compile a GPU-enabled version of unifrac.
#
# Note: The script currently assumes Linux_x86_64 platform.
#

# Note:
# for the time being, it reuqires sudo access
# as we do not support static linking, and thus must have abspaths

#
# based on https://github.com/ROCm/aomp/blob/aomp-dev/docs/INSTALL.md
#

export AOMP_VERSION=20.0-1

if [ -f /usr/lib/aomp_${AOMP_VERSION}/llvm/bin/amdclang++ ]; then
 dist=`cat /etc/os-release |awk '/^NAME=/{split($1,a,"="); split(a[2],b,"\""); print b[2]}'`
 osver=`cat /etc/os-release |awk '/^VERSION_ID=/{split($1,a,"="); split(a[2],b,"\""); print b[2]}'`
 if [ "x$dist" = "xUbuntu" ]; then
  if [ "x$osver" == "x22.04" ]; then
    wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_${AOMP_VERSION}/aomp_Ubuntu2204_${AOMP_VERSION}_amd64.deb
    echo "[INFO] Executing sudo dpkg -i aomp_Ubuntu2204_${AOMP_VERSION}_amd64.deb"
    echo "Please insert password, if you want to install aomp"
    sudo dpkg -i aomp_Ubuntu2204_${AOMP_VERSION}_amd64.deb
    rm -f aomp_Ubuntu2404_${AOMP_VERSION}_amd64.deb
  elif [ "x$osver" == "x24.04" ]; then
    wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_${AOMP_VERSION}/aomp_Ubuntu2404_${AOMP_VERSION}_amd64.deb
    echo "[INFO] Executing sudo dpkg -i aomp_Ubuntu2404_${AOMP_VERSION}_amd64.deb"
    echo "Please insert password, if you want to install aomp"
    sudo dpkg -i aomp_Ubuntu2404_${AOMP_VERSION}_amd64.deb
    rm -f aomp_Ubuntu2404_${AOMP_VERSION}_amd64.deb
  else
   echo "Unsupported $dist Linux version $osver"
   exit 1
  fi
 else
  echo "Unsupported Linux distribution $dist"
  exit 1
 fi
else
    echo "[INFO] Found existing /usr/lib/aomp_${AOMP_VERSION}/llvm/bin/amdclang++"
fi


cat > setup_amd_compiler.sh  << EOF
export PATH=/usr/lib/aomp_${AOMP_VERSION}/:/usr/lib/aomp_${AOMP_VERSION}/bin/:/usr/lib/aomp_${AOMP_VERSION}/llvm/bin:\$PATH

export AMD_CXX=amdclang++

# no special  flags needed in most cases
export AMD_CPPFLAGS=
export AMD_CXXFLAGS=
export AMD_CFLAGS=

export AMD_LDFLAGS=
EOF

# we don't need the install dir anymore
rm -fr nvhpc_*

echo "Setup script avaiabile in $PWD/setup_amd_compiler.sh"
