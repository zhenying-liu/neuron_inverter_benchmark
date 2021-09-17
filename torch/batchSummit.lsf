#!/bin/bash
#BSUB -P AST153  # Peter
#BSUB -W 00:30
#-BSUB -W 2:00
#BSUB -nnodes 21
#BSUB -J lsf-ni
#BSUB -alloc_flags "nvme smt4"
# - - - - - End LSF directives and begin shell commands

#1arrIdx=${LSB_JOBINDEX}
#1jobId=${LSB_JOBID}_${LSB_JOBINDEX}
jobId=${LSB_JOBID}
nprocspn=6 # 6 is correct

#determine number of nodes and total procs
nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
nprocs=$(( ${nnodes} * ${nprocspn} ))

G=$[ ${nnodes} * ${nprocspn} ]
echo S: job=${LSB_JOBID}  G=$G  N=${nnodes} 

#echo S:iniLR=${NEUINV_INIT_LR}:
# grab some variables from environment - if dedefined
[[ -z "${NEUINV_INIT_LR}" ]] && initLRstr=" --initLR 1e-2 " || initLRstr=" --initLR ${NEUINV_INIT_LR} "
[[ -z "${NEUINV_JOBID}" ]] && jobId=$LSB_JOBID || jobId="G${G}_${NEUINV_JOBID}"

wrkSufix=scanLR/G${G}_lr${NEUINV_INIT_LR}

env |grep NI

#cellName=practice10c  # has 4.8M training samples
cellName=witness2c  # has 1M training samples
design=hpar_gcref  # reference jobs for Graphcore

wrkDir0=/gpfs/alpine/world-shared/nro106/balewski/neurInv/sept/
wrkDir=$wrkDir0/$wrkSufix

echo "S:cellName=$cellName  initLRstr=$initLRstr"
date
echo S:my job  JID=$LSB_JOBID wrkSufix=$wrkSufix 

# load modules
module load open-ce/1.2.0-py37-0
# to run inside conda with my privately compiled Apex
conda activate  /ccs/proj/ast153/balewski/conda_envs/torch_apex120py37

python -V
pwd

export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

echo "S: nprocs=$nprocs  nnodes=$nnodes jobId=$jobId"

codeList="  train_dist.py  toolbox/ batchSummit.lsf launch-smt4.sh  hpar*.yaml  "

outPath=$wrkDir/out
mkdir -p $outPath
cp -rp $codeList  $wrkDir
cd  $wrkDir
echo lsfPWD=`pwd`

echo "starting  jobId=$jobId neurInv 2021-08 " `date` " outPath="$outPath

CMD="python -u train_dist.py  --facility summit --cellName $cellName  --outPath ./out --design $design --jobId $jobId  $initLRstr "

echo S:CMD=$CMD
# if multi_h5:
#spare:  train_dist_mh5.py  --globalTrainSamples $globNTS 

jsrun -n${nnodes} -a${nprocspn} -c42 -g6 -r1 --smpiargs off  --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0  toolbox/launch-smt4.sh "$CMD"  >& log.train

echo 'S:done' 

# bsub  batchSummit.lsf

# notes for job array
# bsub -J 'lsf-pitch[11-14]' batchTrainOntra4.lsf
# bkill 520324[11]
# bjobs


