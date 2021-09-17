#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value

outPath=out-scanSep4b
cellName=witness2c
#cellName=practice10c

for m in  4 ; do
#for m in 4 ; do
    #for lr in   5e-4 1e-3 2e-3 5e-3 ; do 
    for lr in   1e-3  ; do 

	out=$outPath/m${m}_lr${lr}
	jobId="$lr ${m}xIPU"
	echo start lr=$lr  m=$m job=$jobId  out= $out  

	for j  in `seq 1 2`; do
	    echo try=$j
	    mkdir -p $out
	    time poprun --num-instances=$m --num-replicas=$m    ./train_replica.py --design hpar_gc4  --outPath $out --initLR $lr --jobId "$jobId" --cellName $cellName >&${out}/log.train
	    date
	    if grep 'M:done'  ${out}/log.train; then
		echo success  for try=$j  out= $out 
		break
	    else
		echo FAIL for try=$j   out= $out 
		mv ${out}  ${out}-crash${j}
		sleep 5
		pkill python
		sleep 5
		echo "do vipu reset ..."
		vipu -H localhost reset partition lr66-poplar3-part-16

	    fi
	    sleep 10  # may be it will help w/  'poplar_unknown_runtime_error': IPUDevice: still waiting for host sync after: 301 seconds
	    #exit
	done  # loop over trials    
    #exit
done

done
