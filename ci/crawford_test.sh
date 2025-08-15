#!/bin/bash

set -e

if [[ "$(uname -s)" == "Linux" ]]; 
then
  MD5=md5sum
else
  MD5='md5 -r'
fi

ssu -i crawford.biom -t crawford.tre -o test.dm -m unweighted

# test only 4 decimal places, as rounding is not unreasonable
cat test.dm |awk '{printf("%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",$1,$2,$3,$4,$5,$6,$7,$8,$9,$10)}' > test.dm.f4
cat test.ssu.exp |awk '{printf("%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",$1,$2,$3,$4,$5,$6,$7,$8,$9,$10)}' > test.ssu.exp.f4

exp=$($MD5 test.dm.f4 | awk '{ print $1 }')
obs=$($MD5 test.ssu.exp.f4 | awk '{ print $1 }')
python -c "assert '${obs}' == '${exp}'"

ssu -i crawford.biom -t crawford.tre -o test.vaw.dm -m unweighted --vaw

# test only 4 decimal places, as rounding is not unreasonable
cat test.vaw.dm |awk '{printf("%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",$1,$2,$3,$4,$5,$6,$7,$8,$9,$10)}' > test.vaw.dm.f4
cat test.ssu_vaw.exp |awk '{printf("%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",$1,$2,$3,$4,$5,$6,$7,$8,$9,$10)}' > test.ssu_vaw.exp.f4

exp=$($MD5 test.dm.f4 | awk '{ print $1 }')
obs=$($MD5 test.ssu.exp.f4 | awk '{ print $1 }')
python -c "assert '${obs}' == '${exp}'"

ssu -i crawford.biom -t crawford.tre -o test.dm.start0.stop3 -m unweighted --mode partial --start 0 --stop 3
ssu -i crawford.biom -t crawford.tre -o test.dm.start3.stop5 -m unweighted --mode partial --start 3 --stop 5
ssu -i crawford.biom -t crawford.tre -o test.dm.partial --mode merge-partial --partial-pattern "test.dm.start*"

exp=$($MD5 test.dm | awk '{ print $1 }')
obs=$($MD5 test.dm.partial | awk '{ print $1 }')
python -c "assert '${obs}' == '${exp}'"

faithpd -i crawford.biom -t crawford.tre -o test.faith.obs
tail -n +2 test.faith.obs > test.faith.header-removed.obs
exp1=$($MD5 test.faith.exp | awk '{ print $1 }')
obs1=$($MD5 test.faith.header-removed.obs | awk '{ print $1 }')
python -c "assert '${obs1}' == '${exp1}'"

echo "All tests succeeded"
