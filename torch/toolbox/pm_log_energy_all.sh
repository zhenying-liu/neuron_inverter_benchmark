#!/bin/bash
if [[  $NERSC_HOST != perlmutter ]]   ; then
    echo energy counter runs only on PM, skip
    exit
fi

# energy_node cpu_energy memory_energy accel0_energy accel1_energy accel2_energy accel3_energy freshness

appname=$1
state=$2

echo "ENERGY_LOG: ,"$(date '+%Y-%m-%dT%H:%M:%S.%N')","`hostname`","`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`","`cat /sys/cray/pm_counters/energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/cpu_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/memory_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/accel0_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/accel1_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/accel2_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/accel3_energy | cut -d' ' -f1`","`cat /sys/cray/pm_counters/freshness | cut -d' ' -f1`","$1","$2
