#!/bin/bash

if [[ $# < 2 ]]
then
    echo "usage: test_tc_detect_app.sh [app prefix] [data root] "     \
         "[mpi exec] [test cores]"
    exit -1
fi

app_prefix=${1}
data_root=${2}

if [[ $# -eq 4 ]]
then
    mpi_exec=${3}
    test_cores=${4}
    launcher="${mpi_exec} -n ${test_cores}"
fi

set -x

# run the app
${launcher} ${app_prefix}/teca_detect_nodes                           \
    --input_file "${app_prefix}/../test/HighResMIP_TC_test.mcf"       \
    --sea_level_pressure PSL --500mb_height Z1000 --300mb_height Z200 \
    --surface_wind_u UBOT --surface_wind_v VBOT
