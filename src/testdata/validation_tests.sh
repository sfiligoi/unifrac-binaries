#!/bin/bash -x
#
# Complete validation test suite
#
# Note:  python must be present in the path
#

set -e

# weighted_unnormalized
time ssu -m weighted_unnormalized_fp32 -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_fp32 -o t1.h5
./compare_unifrac_matrix.py test500.weighted_unnormalized_fp32.h5 t1.h5 1.e-5
./compare_unifrac_pcoa.py test500.weighted_unnormalized_fp32.h5 t1.h5 3 0.1
rm -f t1.h5

# retry with default precision handling
time ssu -m weighted_unnormalized -i test500.biom  -t test500.tre --pcoa 4  -r hdf5 -o t1.h5
./compare_unifrac_matrix.py test500.weighted_unnormalized_fp32.h5 t1.h5 1.e-5
./compare_unifrac_pcoa.py test500.weighted_unnormalized_fp32.h5 t1.h5 3 0.1
rm -f t1.h5
time ssu -f -m weighted_unnormalized_fp32 -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_fp32 -o t1.h5
# matrrix will be different, but PCOA similar
./compare_unifrac_pcoa.py test500.weighted_unnormalized_fp32.h5 t1.h5 3 0.1
rm -f t1.h5
time ssu -m weighted_unnormalized_fp64 -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_fp64 -o t1.h5
# minimal precision loss between fp32 and fp64
./compare_unifrac_matrix.py test500.weighted_unnormalized_fp32.h5 t1.h5 1.e-5
./compare_unifrac_pcoa.py test500.weighted_unnormalized_fp32.h5 t1.h5 3 0.1
rm -f t1.h5
# weighted_normalized
time ssu -f -m weighted_normalized -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_fp32 -o t1.h5
./compare_unifrac_matrix.py test500.weighted_normalized_fp32.f.h5 t1.h5 1.e-5
./compare_unifrac_pcoa.py test500.weighted_normalized_fp32.f.h5 t1.h5 3 0.1
rm -f t1.h5
time ssu -f -m weighted_normalized_fp64 -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_fp64 -o t1.h5
./compare_unifrac_matrix.py test500.weighted_normalized_fp32.f.h5 t1.h5 1.e-5
./compare_unifrac_pcoa.py test500.weighted_normalized_fp32.f.h5 t1.h5 3 0.1
rm -f t1.h5
# unweighted_unnormalized
time ssu -f -m unweighted_unnormalized -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_fp32 -o t1.h5
./compare_unifrac_matrix.py test500.unweighted_unnormalized_fp32.f.h5 t1.h5 1.e-4
rm -f t1.h5
# unweighted
time ssu -f -m unweighted -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_fp32 -o t1.h5
./compare_unifrac_matrix.py test500.unweighted_fp32.f.h5 t1.h5 1.e-5
./compare_unifrac_pcoa.py test500.unweighted_fp32.f.h5 t1.h5 3 0.1
rm -f t1.h5
time ssu -f -m unweighted_fp64 -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_fp64 -o t1.h5
./compare_unifrac_matrix.py test500.unweighted_fp32.f.h5 t1.h5 1.e-5
./compare_unifrac_pcoa.py test500.unweighted_fp32.f.h5 t1.h5 3 0.1
ls -l t1.h5
rm -f t1.h5
# hdf5 without distance matrix, just PCoA
echo "hdf5_nodist"
time ssu -f -m unweighted_fp64 -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_nodist -o t1.h5
./compare_unifrac_pcoa.py test500.unweighted_fp32.f.h5 t1.h5 3 0.1
ls -l t1.h5
rm -f t1.h5
time ssu -f -m unweighted_fp32 -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_nodist -o t1.h5
./compare_unifrac_pcoa.py test500.unweighted_fp32.f.h5 t1.h5 3 0.1
ls -l t1.h5
rm -f t1.h5
# permanova
echo "permanova"
time ssu -m unweighted -i test500.biom  -t test500.tre --pcoa 4  -r hdf5 --permanova 999 -g test500.tsv -c empo_2 -o t1.h5
# compare to values given by skbio.stats.distance.permanova
python compare_unifrac_stats.py t1.h5 5 999 1.001112 0.456  0.001 0.1
ls -l t1.h5
rm -f t1.h5
time ssu -m unweighted -i test500.biom  -t test500.tre --pcoa 4  -r hdf5 --permanova 99 -g test500.tsv -c empo_2 -o t1.h5
python compare_unifrac_stats.py t1.h5 5 99 1.001112 0.456  0.001 0.25
ls -l t1.h5
rm -f t1.h5
time ssu -m weighted_unnormalized_fp32 -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_nodist -g test500.tsv -c empo_3 -o t1.h5
# compare to values given by skbio.stats.distance.permanova
python compare_unifrac_stats.py t1.h5 17 999 0.890697 0.865 0.001 0.1
ls -l t1.h5
rm -f t1.h5
# partials
echo "partials"
ssu -f -m unweighted_fp32 -i test500.biom  -t test500.tre --mode partial-report --n-partials 2
time ssu -f -m unweighted_fp32 -i test500.biom  -t test500.tre --mode partial --start 0 --stop 125 -o t1.partial.1
time ssu -f -m unweighted_fp32 -i test500.biom  -t test500.tre --mode partial --start 125 --stop 250 -o t1.partial.2
ls -l t1.partial*
ssu -f -m unweighted_fp32 -i test500.biom  -t test500.tre --mode check-partial --partial-pattern 't1.partial.*'
time ssu -f -m unweighted_fp64 -i test500.biom  -t test500.tre --pcoa 4  --mode merge-partial --partial-pattern 't1.partial.*' -r hdf5_fp64 -o t1.h5
./compare_unifrac_matrix.py test500.unweighted_fp32.f.h5 t1.h5 1.e-5
./compare_unifrac_pcoa.py test500.unweighted_fp32.f.h5 t1.h5 3 0.1
ls -l t1.h5
rm -f t1.h5
time ssu -f -m unweighted_fp32 -i test500.biom  -t test500.tre --pcoa 4  --mode merge-partial --partial-pattern 't1.partial.*' -r hdf5_fp32 -o t1.h5
./compare_unifrac_matrix.py test500.unweighted_fp32.f.h5 t1.h5 1.e-5
./compare_unifrac_pcoa.py test500.unweighted_fp32.f.h5 t1.h5 3 0.1
ls -l t1.h5
rm -f t1.h5
time ssu -f -m unweighted_fp32 -i test500.biom  -t test500.tre --pcoa 4  --mode merge-partial --partial-pattern 't1.partial.*' -r hdf5_nodist -o t1.h5
./compare_unifrac_pcoa.py test500.unweighted_fp32.f.h5 t1.h5 3 0.1
ls -l t1.h5
rm -f t1.h5
rm -f t1.partial.*
# subsample
echo "subsample"
time ssu -f -m unweighted -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_fp32 --subsample-depth 100 -o t1.h5
./compare_unifrac_pcoa.py test500.unweighted_fp32.f.h5 t1.h5 3 0.3
rm -f t1.h5
time ssu -f -m unweighted -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_fp32 --subsample-depth 100 --subsample-replacement with -o t1.h5
./compare_unifrac_pcoa.py test500.unweighted_fp32.f.h5 t1.h5 3 0.3
rm -f t1.h5
time ssu -f -m unweighted -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_fp32 --subsample-depth 100 --subsample-replacement without -o t1.h5
./compare_unifrac_pcoa.py test500.unweighted_fp32.f.h5 t1.h5 3 0.3
rm -f t1.h5
time ssu -m unweighted -i test500.biom  -t test500.tre --pcoa 4  -r hdf5_nodist --permanova 99 -g test500.tsv -c empo_2 --subsample-depth 100 -o t1.h5
python compare_unifrac_stats.py t1.h5 5 99 1.001112 0.456  0.05 0.6
ls -l t1.h5
rm -f t1.h5
# multi
echo "multi"
time ssu -f -m unweighted -i test500.biom  -t test500.tre --pcoa 4 --mode multi --subsample-depth 100 --n-subsamples 10 -o t1.h5
./compare_unifrac_pcoa_multi.py test500.unweighted_fp32.f.h5 t1.h5 10 3 0.3
ls -l t1.h5
rm -f t1.h5
time ssu -m unweighted -i test500.biom  -t test500.tre --pcoa 4  --mode multi --n-subsamples 10 --permanova 99 -g test500.tsv -c empo_2 --subsample-depth 100 -o t1.h5
python compare_unifrac_stats_multi.py t1.h5 10 5 99 1.001112 0.456  0.08 0.6
ls -l t1.h5
rm -f t1.h5
time ssu -m unweighted -i test500.biom  -t test500.tre --pcoa 4  --mode multi --n-subsamples 10 --permanova 99 -g test500.tsv -c empo_2 --subsample-depth 100 --subsample-replacement without -o t1.h5
python compare_unifrac_stats_multi.py t1.h5 10 5 99 1.001112 0.456  0.08 0.6
ls -l t1.h5
rm -f t1.h5

