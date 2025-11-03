#!/usr/bin/env python

# Requires h5py library
# Usage:
#  compare_unifrac_dm.py fname1 fname2 precisio
#
# Example:
#  compare_unifrac_dm.py test.absquant.wn.dm a.dm 1.e-5
#
import sys

fname1=sys.argv[1]
fname2=sys.argv[2]
prec=float(sys.argv[3])

with open(fname1,"r") as fd:
  lines1=fd.readlines()

with open(fname2,"r") as fd:
  lines2=fd.readlines()

l1=len(lines1)
l2=len(lines2)
if (l1!=l2):
  print("The two files do not have the same number of rows")
  sys.exit(1)

if lines1[0]!=lines2[1]:
  larr1 = lines1[0].strip().split()
  larr2 = lines2[0].strip().split()
  narr1=len(larr1)
  narr2=len(larr2)
  if narr1!=narr2:
    print("The two files do not have the same number of columns")
    sys.exit(1)
 
  for i in range(narr1):
    if larr1[0]!=larr2[0]:
      print("The two files do not have the same header at column %i",i)
      sys.exit(1)

for r in range(1,l1):
  larr1 = lines1[r].strip().split()
  larr2 = lines2[r].strip().split()
  narr1=len(larr1)
  narr2=len(larr2)
  if narr1!=narr2:
    print("The two files do not have the same number of columns at row %i",r)
    sys.exit(1)
  if larr1[0]!=larr2[0]:
    print("The two files do not have the same header at row %i",r)
    sys.exit(1)
 
  for i in range(1,narr1):
     m=float(larr1[i])-float(larr2[i])
     if (abs(m)>prec):
       print("Diff too large at row %i col %i: %e>%e"%(r,i,m,prec))
       sys.exit(2)

print("Files match within precision")
