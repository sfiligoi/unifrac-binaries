#!/usr/bin/env python

# Requires h5py library
# Usage:
#  compare_unifrac_stats_multi.py fname n_stats n_groups n_perms value pvalue prec_value prec_pvalue
#
# Example:
#  compare_unifrac_stats.py a.h5 10 5 99 1.001112 0.456  0.05 0.5
#
import h5py
import sys

fname1=sys.argv[1]
n_stats=int(sys.argv[2])
n_groups=int(sys.argv[3])
n_perms=int(sys.argv[4])
value=float(sys.argv[5])
pvalue=float(sys.argv[6])
precv=float(sys.argv[7])
precp=float(sys.argv[8])

d= h5py.File(fname1)

if 'stat_n_groups' not in d.keys():
  print("No stats found")
  sys.exit(1)

idxs=range(n_stats)

for i in idxs:
  if (d['stat_n_groups'][i]!=n_groups):
    print("n_groups does not match for %i:"%i, d['stat_n_groups'][i],n_groups)
    sys.exit(1)
  if (d['stat_n_permutations'][i]!=n_perms):
    print("n_perms does not match for %i:"%i,d['stat_n_permutations'][i],n_perms)
    sys.exit(1)

  t1=d['stat_values'][i]
  t2=value
  m=abs((t1-t2)/t2)
  if (m>precv):
    print("Value diff too large for %i: %e>%e "%(i,m,precv))
    print("Raw data:",t1,t2)
    sys.exit(2)

  t1=d['stat_pvalues'][i]
  t2=pvalue
  m=abs((t1-t2)/t2)
  if (m>precp):
    print("P-Value diff too large for %i: %e>%e "%(i,m,precp))
    print("Raw data:",t1,t2)
    sys.exit(2)

print("File matches within precision")
