#!/bin/bash

# 项目根目录
PROJECT_ROOT="/mnt/disk1/decom/VLATest"

models=("$1")

datasets=("$2")

output_root=$3

timeout_duration="1h"  # Adjust the timeout duration as needed

for data in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo "Running ${model} on ${data}"
    data_name=$(basename "${data%.json}")
    output_dir="${output_root}/${data_name}/${model}_2024/"

    # Check if the output directory exists and if all folders have log.json
    if [ ! -d "$output_dir" ] || [ "$(find "$output_dir" -type d | wc -l)" -lt 1000 ] || [ "$(find "$output_dir" -name 'log.json' | wc -l)" -ne 1000 ]; then
      while true; do
        # Run the Python script with a timeout
        cd "${PROJECT_ROOT}/experiments"
        timeout "${timeout_duration}" bash -c "PYTHONPATH=${PROJECT_ROOT} python3 openVLA.py -s 2024 -m '${model}' -d ${data} -o ${output_dir}"

        # Re-check if the output directory has the expected folders and log.json files
        log_count=$(find "$output_dir" -name 'log.json' 2>/dev/null | wc -l)

        if [ "$log_count" -eq 1000 ]; then
          echo "${model} on ${data} completed successfully with all log.json files."
          break
        else
          echo "Script failed, timed out, or got stuck; restarting ${model} on ${data} from ${log_count}..."
          pkill -f run_fuzzer.py  # Ensure the process is killed if timeout didn't do it
          sleep 5  # Wait a bit before retrying
        fi
      done
    else
      echo "Skipping ${model} on ${data}, already completed and verified."
    fi
  done
done

#./run_openVLA.sh openvla-7b ../data/t-grasp_n-100_o-0_s-170912623.json  ../newresult
#./run_openVLA.sh openvla-7b 输入  输出 