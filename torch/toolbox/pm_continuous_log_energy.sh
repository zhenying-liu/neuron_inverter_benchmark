#!/bin/bash
if [[  $NERSC_HOST != perlmutter ]]   ; then
    echo  continuous energy counter runs only on PM, skip
    exit
fi
appname="${1:-myJob}"
niter="${1:-10}"
tsleep=3

#echo started continuous energy logging  tsleep=$tsleep  niter=$niter

echo "ENERGY_LOG,date,unix_millisec,hostname,node_governor,node_ene_J,cpu_ene_J,memory_ene_J,gpu0_ene_J,gpu1_ene_J,gpu2_ene_J,gpu3_ene_J,freshness,appname"

for i in `seq 1 ${niter}` ; do
    echo "ENERGY_LOG,`date '+%Y-%m-%dT%H:%M:%S'`,`date +%s%3N`,`hostname`,"`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`","`cat /sys/cray/pm_counters/energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/cpu_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/memory_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/accel0_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/accel1_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/accel2_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/accel3_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/freshness | cut -d' ' -f1`","$1
    sleep  $tsleep
done
