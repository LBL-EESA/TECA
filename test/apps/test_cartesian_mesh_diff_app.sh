#!/bin/bash

if [[ $# -lt 4 ]]
then
    echo "usage: test_cartesian_mesh_diff_app.sh [app_prefix] " \
         "[dataset 1] [dataset 2] [array 1] ... [array n]"
    exit -1
fi

app_prefix=${1}
dataset1=${2}
dataset2=${3}
arrays=${@:4}

set -x

${app_prefix}/teca_cartesian_mesh_diff                          \
    --reference_dataset ${dataset1} --test_dataset ${dataset2}  \
    --arrays ${arrays}
