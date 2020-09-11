#!/bin/bash

if [[ $# -ne 3 ]]
then
    echo "usage: test_table_diff_app.sh [app_prefix] " \
         "[dataset 1] [dataset 2]"
    exit -1
fi

app_prefix=${1}
dataset1=${2}
dataset2=${3}

set -x

${app_prefix}/teca_table_diff ${dataset1} ${dataset2}
