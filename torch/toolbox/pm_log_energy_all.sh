#!/bin/bash
if [[  $NERSC_HOST != perlmutter ]]   ; then
    echo energy counter runs only on PM, skip
    exit
fi



# unixtme in milliseconds: date +%s%3N

appname=$1
state=$2
head="${3:-skip}"
if [[ $head != skip ]] ; then
    echo "ENERGY_LOG,date,unix_millisec,hostname,node_governor,node_energy,cpu_energy,memory_energy,gpu0_energy,gpu1_energy,gpu2_energy,gpu3_energy,freshness,appname,state"
fi 
echo "ENERGY_LOG,`date '+%Y-%m-%dT%H:%M:%S'`,`date +%s%3N`,`hostname`,"`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`","`cat /sys/cray/pm_counters/energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/cpu_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/memory_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/accel0_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/accel1_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/accel2_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/accel3_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/freshness | cut -d' ' -f1`","$1","$2
