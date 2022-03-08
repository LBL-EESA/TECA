#!/bin/bash

if [[ $# -ne 1 ]]
then
    echo "usage: test_lapse_rate_app.sh [app prefix]"
    exit -1
fi

app_prefix=${1}

output_prefix="test_lapse_rate_app_output"

set -x

# run the app
#${app_prefix}/teca_lapse_rate --help
${app_prefix}/teca_lapse_rate                                      \
    --input_file "${app_prefix}/../test/ERA5_lapse_rate_test.mcf"  \
    --output_file "${output_prefix}_%t%.nc" --verbose --no_inline_reduction

# check if number of outputs are correct
output_list=($(ls ${output_prefix}*))
output_len=${#output_list[@]}

if [[ ${output_len} -ne 1 ]]
then
    echo "error: unexpected number of lapse_rate outputs"
    exit 1
fi

# check if outputs aren't empty
for i in ${output_list[@]}
do
    if [[ ! -s $i ]]
    then
        echo "error: file '$i' is empty"
        exit 1
    fi
done

# clean up
rm ${output_prefix}*
