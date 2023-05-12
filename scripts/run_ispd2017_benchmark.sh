#!/bin/bash
# File              : run_ispd2017_benchmark.sh
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 08.31.2021
# Last Modified Date: 09.16.2021
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
# Usage:
#   Run this script under `install` directory.

host=$(hostname)
date_str="${host}_$(date +"%Y.%m.%d_%H.%M.%S")"
log_dir="../ispd2017_log/log_${date_str}"
echo "host: $host"
echo "log_dir = ${log_dir}"

read -p "Description: "
desp=$REPLY

mkdir -p "${log_dir}/results"
echo "desp: $desp"
echo $desp > "${log_dir}/description.txt"

for i in $(seq -f '%02g' 1 13)
do
	configuration_file_path="./unittest/regression/ispd2017/CLK-FPGA${i}.json"
	screen_output_file_path="${log_dir}/CLK-FPGA${i}.log"
	echo "================================"
	echo "Running Benchmark ${configuration_file_path}"
	python ./openparf.py --config ${configuration_file_path} 2>&1 | tee ${screen_output_file_path}
	if [ $? -eq 0 ]; then
	  cp -r "./clkresults/CLK-FPGA${i}" "${log_dir}/results"
	else
	  echo "Benchmark ${configuration_file_path} failed."
	fi
done
