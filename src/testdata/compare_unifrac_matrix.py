#!/usr/bin/env python

# Requires h5py library
# Usage:
#  compare_unifrac_matrix.py fname1 fname2 precisio
#
# Example:
#  compare_unifrac_matrix.py test500.unweighted_fp32.f.h5 a.h5 1.e-5
#
import h5py
import sys

fname1=sys.argv[1]
fname2=sys.argv[2]
prec=float(sys.argv[3])

d1= h5py.File(fname1)['matrix'][:,:]
d2= h5py.File(fname2)['matrix'][:,:]

l1=len(d1)
l2=len(d2)
if (l1!=l2):
  print("The two files do not have the same number of rows")
  sys.exit(1)

for r in range(l1):
  m=max(abs(d1[r,:]-d2[r,:]))
  if (m>prec):
    print("Diff too large at row %i: %e>%e"%(r,m,prec))
    sys.exit(2)
print("Files match within precision")
