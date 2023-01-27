#!/usr/bin/env python

# Requires h5py library
# Usage:
#  compare_unifrac_stats.py fname n_groups n_perms value pvalue precision
#
# Example:
#  compare_unifrac_stats.py a.h5 PERMANOVA 
#
import h5py
import sys

fname1=sys.argv[1]
n_groups=int(sys.argv[2])
n_perms=int(sys.argv[3])
value=float(sys.argv[4])
pvalue=float(sys.argv[5])
prec=float(sys.argv[6])

d= h5py.File(fname1)

if 'stat_n_groups' not in d.keys():
  print("No stats found")
  sys.exit(1)


if (d['stat_n_groups'][0]!=n_groups):
  print("n_groups does not match:", d['stat_n_groups'][0]!=n_groups)
  sys.exit(1)
if (d['stat_n_permutations'][0]!=n_perms):
  print("n_perms does not match:",d['stat_n_permutations'][0]!=n_perms)
  sys.exit(1)

t1=d['stat_values'][0]
t2=value
m=abs((t1-t2)/t2)
if (m>prec):
    print("Value diff too large: %e>%e "%(m,prec))
    print("Raw data:",t1,t2)
    sys.exit(2)

t1=d['stat_pvalues'][0]
t2=pvalue
m=abs((t1-t2)/t2)
if (m>prec):
    print("P-Value diff too large: %e>%e "%(m,prec))
    print("Raw data:",t1,t2)
    sys.exit(2)

print("File matches within precision")
