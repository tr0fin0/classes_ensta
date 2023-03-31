#!/bin/sh
## pour cortex A07
# $GEM5/build/ARM/gem5.fast $GEM5/configs/example/se.py -n $n -c  test_omp -o "$n $m" --cpu-type=arm_detailed --caches --l2cache --num-l2caches=1 --l1d_size="32kB" --l1i_size="32kB" --l2_size="512kB" --cacheline_size=64 --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=16 --o3-width=$w
# taille de matrice
M=( 1 2 4 8 16 32 64 128 )
# nombre de thread = nombre de cpu
N=( 1 2 4 8 16 32 64)

rm RES_Q5.log
for m in ${M[@]}
do
    for n in ${N[@]}
    do
        if ((n<=m))
        then
            /home/g/gac/ES201/tools/TP5/gem5-stable/build/ARM/gem5.fast /home/g/gac/ES201/tools/TP5/gem5-stable/configs/example/se.py -n $n -c  /home/g/gac/ES201/tools/TP5/test_omp -o "$n $m" --cpu-type=arm_detailed --caches --l2cache --num-l2caches=1 --l1d_size="32kB" --l1i_size="32kB" --l2_size="512kB" --cacheline_size=32 --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=8
            numCycle=$(sed -n 's/system\.cpu0*\.numCycles\s\+\([0-9]\+\).*/\1/p' m5out/stats.txt)
            simInsts=$(sed -n 's/sim_insts\s\+\([0-9]\+\).*/\1/p' m5out/stats.txt)
            echo "$m,$n,$w,$numCycle,$simInsts" >> RES_Q5.log
        fi
    done
done