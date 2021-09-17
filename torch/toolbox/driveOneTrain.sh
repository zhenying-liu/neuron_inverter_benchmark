#!/bin/bash 

if [ ${SLURM_PROCID} -eq 0 ] ; then
    echo D: job=${SLURM_JOBID} `hostname` isShifter=`env|grep  SHIFTER_RUNTIME`
    cat /etc/*release |grep PRETTY_NAME
    free -g
    echo D: num-cpus:`nproc --all`
    nvidia-smi --list-gpus
    python -V
    python -c 'import torch; print("D: pytorch:",torch.__version__)'
    echo D: survey-end
    #nvidia-smi -l 5 >&L.smi_${SLURM_JOBID} &
fi

# the task command executed here:
${CMD}


# from Xander
#!/bin/bash
#while true;
#do nvidia-smi --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory --format=csv >> gpu_utillization.log; sleep 1; 
#done
