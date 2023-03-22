#!/usr/bin/env python

# Requires h5py library
# Usage:
#  compare_unifrac_pcoa.py fname_single fname_multi nmulti elements precision
#
# Example:
#  compare_unifrac_pcoa.py test500.unweighted_fp32.f.h5 a.h5 3 0.1
#
import h5py
import sys

fname1=sys.argv[1]
fname2=sys.argv[2]
mcount=int(sys.argv[3])
els=int(sys.argv[4])
prec=float(sys.argv[5])

d1= h5py.File(fname1)['pcoa_eigvals'][:els]
for i in range(mcount):
  d2= h5py.File(fname2)['pcoa_eigvals:%i'%i][:els]

  l1=len(d1)
  l2=len(d2)
  if (l1!=l2):
    print("Multi %i has the wrong number of eigvals"%i)
    sys.exit(1)

  m=max(abs((d1[:]-d2[:])/d1[:]))
  if (m>prec):
    print("Diff too large for multi %i: %e>%e "%(i,m,prec))
    print("Raw data:",d1,d2, abs((d1[:]-d2[:])/d1[:]))
    sys.exit(2)

print("Files match within precision")
