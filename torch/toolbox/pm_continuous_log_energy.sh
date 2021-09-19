#!/bin/bash
if [[  $NERSC_HOST != perlmutter ]]   ; then
    echo  continuous energy counter runs only on PM, skip
    exit
fi
csvName=energy_log.csv
tsleep=4
echo started continuous energy logging to $csvName  tsleep=$tsleep
appname="${1:-myJob}"
./toolbox/pm_log_energy_all.sh $appname start x  >&$csvName
sleep 1
while [ true ] ; do
    ./toolbox/pm_log_energy_all.sh $appname   >>$csvName
    sleep  $tsleep
done
